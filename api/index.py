import os
import json
import time
import math
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from whitenoise import WhiteNoise  # <--- NEW IMPORT

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# --- WHITENOISE CONFIGURATION (FIXES CSS/JS ON RAILWAY) ---
# This forces the app to serve static files from your static folder
app.wsgi_app = WhiteNoise(app.wsgi_app, root=STATIC_DIR, prefix='static/')

# Folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Files & Secrets
CSV_PATH = os.path.join(BASE_DIR, 'survey.csv')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LOCAL_TZ = ZoneInfo("America/New_York") 
CHUNK_SIZE = 60 # Minutes per study session

# --- LIBRARIES ---
try:
    from sklearn.linear_model import ElasticNet
except ImportError:
    sys.exit("CRITICAL: scikit-learn missing. Check requirements.txt")

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
except ImportError:
    genai = None

try:
    from icalendar import Calendar as ICalLoader
    from ics import Calendar as IcsCalendar, Event as IcsEvent
    import recurring_ical_events
except ImportError:
    sys.exit("CRITICAL: Calendar libs missing. Check requirements.txt")

# ==========================================
# 1. SCHEDULER LOGIC (Integrated v2.0)
# ==========================================

def parse_user_ics_busy_times(ics_path, start_date, end_date):
    """Reads user's uploaded ICS and returns busy blocks."""
    busy = []
    if not ics_path or not os.path.exists(ics_path):
        return busy

    try:
        with open(ics_path, 'rb') as f:
            cal = ICalLoader.from_ical(f.read())
        
        # Get events within window
        events = recurring_ical_events.of(cal).between(start_date, end_date)
        for ev in events:
            dtstart = ev.get('DTSTART').dt
            dtend = ev.get('DTEND').dt
            
            # Normalize to datetime (handle all-day events which are dates)
            if not isinstance(dtstart, datetime): 
                dtstart = datetime.combine(dtstart, dt_time.min).replace(tzinfo=LOCAL_TZ)
            if not isinstance(dtend, datetime):
                dtend = datetime.combine(dtend, dt_time.max).replace(tzinfo=LOCAL_TZ)
            
            # Ensure timezone awareness
            if dtstart.tzinfo is None: dtstart = dtstart.replace(tzinfo=LOCAL_TZ)
            else: dtstart = dtstart.astimezone(LOCAL_TZ)
            
            if dtend.tzinfo is None: dtend = dtend.replace(tzinfo=LOCAL_TZ)
            else: dtend = dtend.astimezone(LOCAL_TZ)
            
            busy.append((dtstart, dtend))
    except Exception as e:
        print(f"ICS Parse Error: {e}")
    
    return busy

def generate_free_blocks(start_date, end_date, preferences, busy_blocks):
    """Calculates available study slots based on Work Windows and Busy Blocks."""
    free_blocks = []
    current_day = start_date.date()
    end_date_date = end_date.date()
    
    # Sort busy blocks
    busy_blocks.sort(key=lambda x: x[0])

    while current_day <= end_date_date:
        is_weekend = current_day.weekday() >= 5
        
        # Get Work Window from Frontend Preferences
        if is_weekend:
            s_str = preferences.get('weekendStart', '10:00')
            e_str = preferences.get('weekendEnd', '20:00')
        else:
            s_str = preferences.get('weekdayStart', '09:00')
            e_str = preferences.get('weekdayEnd', '22:00')

        # Parse times
        try:
            sh, sm = map(int, s_str.split(':'))
            eh, em = map(int, e_str.split(':'))
        except:
            sh, sm, eh, em = 9, 0, 21, 0

        # Define the "Work Window" for this specific day
        day_start = datetime.combine(current_day, dt_time(sh, sm)).replace(tzinfo=LOCAL_TZ)
        day_end = datetime.combine(current_day, dt_time(eh, em)).replace(tzinfo=LOCAL_TZ)

        # Subtract Busy Blocks from the Work Window
        current_pointer = day_start
        
        # Filter busy blocks relevant to this day window
        day_busy = [b for b in busy_blocks if b[1] > day_start and b[0] < day_end]

        for b_start, b_end in day_busy:
            # Clip the busy block to the work window
            b_start = max(b_start, day_start)
            b_end = min(b_end, day_end)

            if b_start > current_pointer:
                dur = (b_start - current_pointer).total_seconds() / 60
                if dur >= 30: # Minimum slot size (30 mins)
                    free_blocks.append({'start': current_pointer, 'end': b_start, 'duration': dur})
            current_pointer = max(current_pointer, b_end)

        # Capture remaining time after last busy event
        if current_pointer < day_end:
            dur = (day_end - current_pointer).total_seconds() / 60
            if dur >= 30:
                free_blocks.append({'start': current_pointer, 'end': day_end, 'duration': dur})

        current_day += timedelta(days=1)
    
    return pd.DataFrame(free_blocks)

def run_scheduler_logic(courses, preferences, user_ics_path, output_filename):
    """
    Core logic: Takes predicted courses, finds free time, and creates schedule.
    """
    now = datetime.now(LOCAL_TZ)
    end_horizon = now + timedelta(days=90) # Schedule 3 months out

    # 1. Get Free Time
    busy = parse_user_ics_busy_times(user_ics_path, now, end_horizon)
    free_df = generate_free_blocks(now, end_horizon, preferences, busy)

    if free_df.empty:
        print("Scheduler: No free time found.")
        return None

    # 2. Prepare Sessions (Break assignments into chunks based on predicted hours)
    sessions = []
    for c in courses:
        try:
            d_str = c.get('date')
            if not d_str: continue
            
            # Parse Due Date
            try:
                due_dt = datetime.strptime(d_str, '%Y-%m-%d').replace(tzinfo=LOCAL_TZ)
            except:
                continue 
                
            due_dt = due_dt.replace(hour=23, minute=59) # End of due day

            # Get Predicted Hours from ML Model
            hours = float(c.get('predicted_hours', 1.0))
            if hours <= 0: hours = 0.5

            total_mins = int(hours * 60)
            num_chunks = math.ceil(total_mins / CHUNK_SIZE)

            for i in range(num_chunks):
                dur = min(CHUNK_SIZE, total_mins - (i*CHUNK_SIZE))
                sessions.append({
                    'name': c['name'],
                    'due': due_dt,
                    'duration': dur,
                    'uid': f"{c['name']}_{i}"
                })
        except Exception as e:
            print(f"Error prepping course {c.get('name')}: {e}")

    # Sort assignments by Due Date (Earliest Deadline First)
    sessions.sort(key=lambda x: x['due'])

    # 3. Allocate Sessions to Free Blocks
    scheduled_events = []
    
    for sess in sessions:
        # Find valid blocks
        valid = free_df[
            (free_df['start'] >= now) & 
            (free_df['end'] <= sess['due']) & 
            (free_df['duration'] >= sess['duration'])
        ]

        if not valid.empty:
            # Greedy: Pick the very first available slot
            idx = valid.index[0]
            block = free_df.loc[idx]

            start_t = block['start']
            end_t = start_t + timedelta(minutes=sess['duration'])

            scheduled_events.append({
                'name': f"Study: {sess['name']}",
                'begin': start_t,
                'end': end_t
            })

            # Consume time from the block
            new_start = end_t
            new_dur = (block['end'] - new_start).total_seconds() / 60
            
            if new_dur >= 30:
                free_df.at[idx, 'start'] = new_start
                free_df.at[idx, 'duration'] = new_dur
            else:
                free_df.drop(idx, inplace=True) # Block used up

    # 4. Generate Output ICS
    cal = IcsCalendar()
    for ev in scheduled_events:
        e = IcsEvent()
        e.name = ev['name']
        e.begin = ev['begin']
        e.end = ev['end']
        cal.events.add(e)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    with open(output_path, 'w') as f:
        f.writelines(cal.serialize_iter())
    
    return output_filename

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def standardize_time(time_str):
    if not time_str: return None
    import re
    clean = re.split(r'\s*[-–]\s*|\s+to\s+', str(time_str))[0].strip()
    for fmt in ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p"]:
        try: return datetime.strptime(clean, fmt).strftime("%I:%M %p")
        except ValueError: continue
    if clean.isdigit():
        val = int(clean)
        if 8 <= val <= 11: return f"{val:02d}:00 AM"
        if 1 <= val <= 6:  return f"{val:02d}:00 PM"
        if val == 12:      return "12:00 PM"
    return clean

def map_pdf_category(cat):
    c = str(cat).lower()
    if 'reading' in c: return 'readings'
    if 'writing' in c: return 'essay'
    if 'exam' in c: return 'p_set'
    if 'project' in c: return 'research_paper'
    return 'p_set'

# ==========================================
# 3. PDF PARSER & ML MODEL
# ==========================================

def parse_syllabus(file_path):
    if not genai or not GEMINI_API_KEY: return None
    try:
        class AssignmentItem(BaseModel):
            date: str = Field(description="YYYY-MM-DD")
            time: Optional[str] = Field(description="Deadline time or null")
            assignment_name: str = Field(description="Name")
            category: str = Field(description="Category")
            description: str = Field(description="Details")

        class SyllabusResponse(BaseModel):
            metadata: dict = Field(description="Metadata")
            assignments: List[AssignmentItem]

        client = genai.Client(api_key=GEMINI_API_KEY)
        file_upload = client.files.upload(file=file_path)
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
        if file_upload.state.name != "ACTIVE": return None

        prompt = "Extract assignments (Dates YYYY-MM-DD), readings, and deliverables."
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[file_upload, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SyllabusResponse)
        )
        data = response.parsed
        rows = []
        c_name = getattr(data.metadata, 'course_name', 'Parsed Course')
        for item in data.assignments:
            rows.append({"Course": c_name, "Date": item.date, "Time": item.time, "Category": item.category, "Assignment": item.assignment_name, "Description": item.description})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

model = None
model_columns = []
def initialize_model():
    global model, model_columns
    if not ElasticNet or not os.path.exists(CSV_PATH): return
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={'What year are you? ': 'year', 'What is your major/concentration?': 'major', 'What type of assignment was it?': 'assignment_type', 'Approximately how long did it take (in hours)': 'time_spent_hours'})
        for col in ['year', 'assignment_type', 'external_resources', 'work_location', 'worked_in_group', 'submitted_in_person']:
            if col in df.columns: df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int, drop_first=True)
        df = df.select_dtypes(include=[np.number])
        if 'time_spent_hours' in df.columns:
            X = df.drop('time_spent_hours', axis=1)
            y = df['time_spent_hours']
            clf = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
            clf.fit(X, y)
            model = clf
            model_columns = list(X.columns)
            print("✅ Model Trained.")
    except Exception as e: print(f"Training Failed: {e}")

initialize_model()

# ==========================================
# 4. ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def home():
    return render_template('mains.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        # 1. Parse JSON data from form
        data_str = request.form.get('data')
        if not data_str: return jsonify({'error': 'No data provided'}), 400
        
        req_data = json.loads(data_str)
        survey = req_data.get('survey', {})
        courses = req_data.get('courses', [])
        preferences = req_data.get('preferences', {})

        # 2. Handle PDF Uploads (Syllabus Parsing)
        uploaded_pdfs = request.files.getlist('pdfs')
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf.filename == '': continue
                filename = secure_filename(pdf.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf.save(path)
                
                # Parse with Gemini
                df_parsed = parse_syllabus(path)
                if df_parsed is not None and not df_parsed.empty:
                    for _, row in df_parsed.iterrows():
                        courses.append({
                            'name': f"{row['Course']}: {row['Assignment']}",
                            'type': map_pdf_category(row['Category']),
                            'date': row['Date']
                        })

        # 3. Handle ICS Upload (Busy Times)
        user_ics_path = None
        ics_file = request.files.get('ics')
        if ics_file and ics_file.filename != '':
            filename = secure_filename(ics_file.filename)
            user_ics_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            ics_file.save(user_ics_path)

        # 4. Predict Times (ML Model)
        for c in courses:
            if model and model_columns:
                # Build input vector matching training columns
                input_data = {col: 0 for col in model_columns}
                
                # Encode Year
                y_col = f"year_{survey.get('year')}"
                if y_col in input_data: input_data[y_col] = 1
                
                # Encode Type
                t_col = f"assignment_type_{c.get('type')}"
                if t_col in input_data: input_data[t_col] = 1

                # Predict
                pred = model.predict(pd.DataFrame([input_data]))[0]
                c['predicted_hours'] = round(max(0.5, pred), 1)
            else:
                c['predicted_hours'] = 2.0 # Fallback

        # 5. Run Scheduler
        output_filename = f"schedule_{int(time.time())}.ics"
        result_file = run_scheduler_logic(courses, preferences, user_ics_path, output_filename)

        if not result_file:
            return jsonify({'error': 'Could not generate schedule (no free time?)'}), 400

        # Return Success
        return jsonify({
            'message': 'Success',
            'courses': courses,
            'ics_url': f"/download/{result_file}"
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
