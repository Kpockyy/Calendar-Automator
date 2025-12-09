import os
import json
import time
import re
import sys
import math
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- DEFENSIVE IMPORTS (Prevents Crash on Startup) ---
errors = []

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    pd = None
    errors.append(f"Pandas/Numpy missing: {e}")

try:
    from sklearn.linear_model import ElasticNet
except ImportError as e:
    ElasticNet = None
    errors.append(f"Scikit-learn missing: {e}")

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
except ImportError as e:
    genai = None
    BaseModel = object # Dummy
    errors.append(f"Google GenAI missing: {e}")

try:
    from icalendar import Calendar as ICalLoader
    from ics import Calendar as IcsCalendar, Event as IcsEvent
    import recurring_ical_events
except ImportError as e:
    ICalLoader = None
    errors.append(f"Calendar libs missing: {e}")

# ============================================
# 1. CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, 'templates') if os.path.exists(os.path.join(BASE_DIR, 'templates')) else None
static_dir = os.path.join(BASE_DIR, 'static') if os.path.exists(os.path.join(BASE_DIR, 'static')) else None

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Files & Secrets
CSV_PATH = os.path.join(BASE_DIR, 'survey.csv')
TRAINING_PKL = os.path.join(DATA_FOLDER, "training_data.pkl")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LOCAL_TZ = ZoneInfo("America/New_York") 
CHUNK_SIZE = 60 

# ============================================
# 2. DATA MODELS (Safe Definition)
# ============================================

if genai:
    class AssignmentItem(BaseModel):
        date: str = Field(description="YYYY-MM-DD")
        time: Optional[str] = Field(description="Deadline time or null")
        assignment_name: str = Field(description="Name")
        category: str = Field(description="Category")
        description: str = Field(description="Details")

    class SyllabusResponse(BaseModel):
        metadata: dict = Field(description="Metadata")
        assignments: List[AssignmentItem]

# ============================================
# 3. HELPER FUNCTIONS
# ============================================

def standardize_time(time_str):
    if not time_str: return None
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

# ============================================
# 4. SCHEDULER LOGIC
# ============================================

def parse_user_ics_busy_times(ics_path, start_date, end_date):
    if not ICalLoader: return [] # Safety check
    busy = []
    if not ics_path or not os.path.exists(ics_path):
        return busy

    try:
        with open(ics_path, 'rb') as f:
            cal = ICalLoader.from_ical(f.read())
        
        events = recurring_ical_events.of(cal).between(start_date, end_date)
        for ev in events:
            dtstart = ev.get('DTSTART').dt
            dtend = ev.get('DTEND').dt
            
            # Normalize to datetime with timezone
            if not isinstance(dtstart, datetime): # Is pure date
                dtstart = datetime.combine(dtstart, dt_time.min).replace(tzinfo=LOCAL_TZ)
            if not isinstance(dtend, datetime):
                dtend = datetime.combine(dtend, dt_time.max).replace(tzinfo=LOCAL_TZ)
            
            if dtstart.tzinfo is None: dtstart = dtstart.replace(tzinfo=LOCAL_TZ)
            else: dtstart = dtstart.astimezone(LOCAL_TZ)
            
            if dtend.tzinfo is None: dtend = dtend.replace(tzinfo=LOCAL_TZ)
            else: dtend = dtend.astimezone(LOCAL_TZ)
            
            busy.append((dtstart, dtend))
    except Exception as e:
        print(f"ICS Parse Error: {e}")
    
    return busy

def generate_free_blocks(start_date, end_date, preferences, busy_blocks):
    if pd is None: return [] # Safety check
    
    free_blocks = []
    current_day = start_date.date()
    end_date_date = end_date.date()
    
    busy_blocks.sort(key=lambda x: x[0])

    while current_day <= end_date_date:
        is_weekend = current_day.weekday() >= 5
        
        if is_weekend:
            s_str = preferences.get('weekendStart', '10:00')
            e_str = preferences.get('weekendEnd', '20:00')
        else:
            s_str = preferences.get('weekdayStart', '09:00')
            e_str = preferences.get('weekdayEnd', '22:00')

        try:
            sh, sm = map(int, s_str.split(':'))
            eh, em = map(int, e_str.split(':'))
        except:
            sh, sm, eh, em = 9, 0, 21, 0

        day_start = datetime.combine(current_day, dt_time(sh, sm)).replace(tzinfo=LOCAL_TZ)
        day_end = datetime.combine(current_day, dt_time(eh, em)).replace(tzinfo=LOCAL_TZ)

        current_pointer = day_start
        day_busy = [b for b in busy_blocks if b[1] > day_start and b[0] < day_end]

        for b_start, b_end in day_busy:
            b_start = max(b_start, day_start)
            b_end = min(b_end, day_end)

            if b_start > current_pointer:
                dur = (b_start - current_pointer).total_seconds() / 60
                if dur >= 30:
                    free_blocks.append({'start': current_pointer, 'end': b_start, 'duration': dur})
            current_pointer = max(current_pointer, b_end)

        if current_pointer < day_end:
            dur = (day_end - current_pointer).total_seconds() / 60
            if dur >= 30:
                free_blocks.append({'start': current_pointer, 'end': day_end, 'duration': dur})

        current_day += timedelta(days=1)
    
    return pd.DataFrame(free_blocks)

def run_scheduler_logic(courses, preferences, user_ics_path, output_filename):
    if not IcsCalendar or pd is None:
        return None # Libs missing

    now = datetime.now(LOCAL_TZ)
    end_horizon = now + timedelta(days=90)

    # 1. Get Free Time
    busy = parse_user_ics_busy_times(user_ics_path, now, end_horizon)
    free_df = generate_free_blocks(now, end_horizon, preferences, busy)

    if free_df.empty:
        print("Scheduler: No free time found.")
        return None

    # 2. Prepare Sessions
    sessions = []
    for c in courses:
        try:
            d_str = c.get('date')
            if not d_str: continue
            
            due_dt = datetime.strptime(d_str, '%Y-%m-%d').replace(tzinfo=LOCAL_TZ)
            due_dt = due_dt.replace(hour=23, minute=59)

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

    sessions.sort(key=lambda x: x['due'])

    # 3. Allocate
    scheduled_events = []
    for sess in sessions:
        valid = free_df[
            (free_df['start'] >= now) & 
            (free_df['end'] <= sess['due']) & 
            (free_df['duration'] >= sess['duration'])
        ]

        if not valid.empty:
            idx = valid.index[0]
            block = free_df.loc[idx]

            start_t = block['start']
            end_t = start_t + timedelta(minutes=sess['duration'])

            scheduled_events.append({
                'name': f"Study: {sess['name']}",
                'begin': start_t,
                'end': end_t
            })

            new_start = end_t
            new_dur = (block['end'] - new_start).total_seconds() / 60
            
            if new_dur >= 30:
                free_df.at[idx, 'start'] = new_start
                free_df.at[idx, 'duration'] = new_dur
            else:
                free_df.drop(idx, inplace=True)

    # 4. Generate ICS
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

# ============================================
# 5. CORE LOGIC (Parser + ML)
# ============================================

def parse_syllabus(file_path):
    if not genai or not GEMINI_API_KEY: return None
    try:
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
    if not ElasticNet or not pd: return # Safety check
    
    if not os.path.exists(CSV_PATH):
        print(f"CSV Not Found at {CSV_PATH}")
        return
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

# ============================================
# 6. API ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    # DIAGNOSTIC HOME PAGE - Shows exactly what is missing
    return jsonify({
        "status": "Online", 
        "errors": errors, 
        "model_loaded": model is not None,
        "files_in_root": os.listdir(BASE_DIR)
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    if errors:
        return jsonify({'error': 'Server has missing dependencies', 'details': errors}), 500

    try:
        data_json = request.form.get('data')
        if not data_json: return jsonify({'error': 'No data'}), 400
        
        data = json.loads(data_json)
        survey = data.get('survey', {})
        manual_courses = data.get('courses', [])
        preferences = data.get('preferences', {}) 
        
        pdf_courses = []
        user_ics_path = None

        if 'pdfs' in request.files:
            for f in request.files.getlist('pdfs'):
                if f.filename:
                    fname = secure_filename(f.filename)
                    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    f.save(path)
                    if GEMINI_API_KEY:
                        df = parse_syllabus(path)
                        if df is not None:
                            for _, r in df.iterrows():
                                pdf_courses.append({'name': f"{r['Course']} - {r['Assignment']}", 'type': map_pdf_category(r['Category']), 'subject': survey.get('major'), 'resources': 'Google/internet', 'date': r['Date'], 'time': r['Time'], 'description': r['Description'], 'source': 'pdf_parser'})

        if 'ics' in request.files:
            f = request.files['ics']
            if f.filename:
                fname = secure_filename(f.filename)
                user_ics_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                f.save(user_ics_path)

        all_courses = manual_courses + pdf_courses
        for course in all_courses:
            predicted = 0.0
            if model:
                try:
                    input_row = pd.DataFrame(0, index=[0], columns=model_columns)
                    yr = f"year_{survey.get('year')}"
                    if yr in input_row.columns: input_row[yr] = 1
                    rt = course.get('type', '').lower()
                    mt = 'p_set'
                    if 'essay' in rt: mt = 'essay'
                    elif 'coding' in rt: mt = 'coding'
                    elif 'read' in rt: mt = 'readings'
                    ty = f"assignment_type_{mt}"
                    if ty in input_row.columns: input_row[ty] = 1
                    predicted = float(model.predict(input_row)[0])
                except: pass
            course['predicted_hours'] = max(0.5, round(predicted, 2))

        output_ics = f"study_plan_{int(time.time())}.ics"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_ics)
        
        generated_file = run_scheduler_logic(
            courses=all_courses,
            preferences=preferences,
            user_ics_path=user_ics_path,
            output_filename=output_ics
        )

        response = {
            "status": "success",
            "courses": all_courses,
            "parsed_count": len(pdf_courses)
        }
        
        if generated_file:
            response["ics_url"] = f"/download/{output_ics}"
        else:
            response["warning"] = "Could not generate schedule (no free time or parsing error)"

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
