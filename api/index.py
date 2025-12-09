from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import re
import time
from datetime import datetime
from typing import List, Optional

# --- DEFENSIVE IMPORTS ---
# We wrap these to prevent the server from crashing immediately if a lib is missing
import_errors = []

try:
    from sklearn.linear_model import ElasticNet
except ImportError as e:
    ElasticNet = None
    import_errors.append(f"sklearn: {str(e)}")

try:
    from pydantic import BaseModel, Field
except ImportError as e:
    BaseModel, Field = object, None # Dummies
    import_errors.append(f"pydantic: {str(e)}")

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    genai = None
    import_errors.append(f"google-genai: {str(e)}")


# ============================================
# 1. CONFIGURATION & PATHS
# ============================================

current_directory = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_directory, 'templates')
static_dir = os.path.join(current_directory, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Vercel Storage & Config
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'pdf', 'ics'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TRAINING_PATH = os.path.join(UPLOAD_FOLDER, "training_data.pkl")
ABOUT_PATH = os.path.join(UPLOAD_FOLDER, "about_you.pkl")
MASTER_SCHEDULE_PATH = os.path.join(UPLOAD_FOLDER, "MASTER_Schedule.pkl")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ============================================
# 2. GLOBAL CSV LOADING
# ============================================

csv_filename = 'Student Assignment Survey v2 2.csv'
csv_path = os.path.join(current_directory, csv_filename)
global_df = pd.DataFrame()
csv_status = "Not Loaded"

try:
    if os.path.exists(csv_path):
        global_df = pd.read_csv(csv_path)
        csv_status = f"Loaded {len(global_df)} rows"
        print(f"✅ CSV loaded: {csv_path}")
    else:
        csv_status = f"Missing File at {csv_path}"
        print(f"❌ CSV not found: {csv_path}")
except Exception as e:
    csv_status = f"Error: {str(e)}"
    print(f"❌ CSV Error: {e}")

# ==========================================
# 3. PDF PARSER (Defined only if imports worked)
# ==========================================

# Dummy classes if import failed to prevent NameError later
if not genai or not BaseModel:
    class MeetingSchedule: pass
    class AssignmentItem: pass
    class CourseMetadata: pass
    class SyllabusResponse: pass
else:
    class MeetingSchedule(BaseModel):
        days: List[str] = Field(description="List of days.")
        start_time: str = Field(description="Start time (e.g. '11:00 AM').")

    class AssignmentItem(BaseModel):
        date: str = Field(description="YYYY-MM-DD")
        time: Optional[str] = Field(description="11:59 PM or specific")
        assignment_name: str = Field(description="Name")
        category: str = Field(description="Category")
        description: str = Field(description="Details")

    class CourseMetadata(BaseModel):
        course_name: str = Field(description="Course Name")
        semester_year: str = Field(description="Term")
        class_meetings: List[MeetingSchedule] = Field(description="Schedule")

    class SyllabusResponse(BaseModel):
        metadata: CourseMetadata
        assignments: List[AssignmentItem]

def standardize_time(time_str):
    if not time_str: return None
    clean = re.split(r'\s*[-–]\s*|\s+to\s+', str(time_str))[0].strip()
    for fmt in ["%I:%M %p", "%I %p", "%H:%M", "%I:%M%p"]:
        try:
            return datetime.strptime(clean, fmt).strftime("%I:%M %p")
        except ValueError:
            continue
    if clean.isdigit():
        val = int(clean)
        if 8 <= val <= 11: return f"{val:02d}:00 AM"
        if 1 <= val <= 6:  return f"{val:02d}:00 PM"
        if val == 12:      return "12:00 PM"
    return clean

def resolve_time(row, schedule_map):
    existing_time = row['Time']
    if existing_time and any(char.isdigit() for char in str(existing_time)):
        return standardize_time(existing_time)
    try:
        day_name = pd.to_datetime(row['Date']).strftime('%A') 
    except:
        return "11:59 PM"
    return schedule_map.get(day_name, "11:59 PM")

def parse_syllabus(file_path):
    # Safety Check
    if not genai:
        print("❌ Parsing skipped: google-genai lib missing.")
        return None
    if not GEMINI_API_KEY:
        print("❌ Parsing skipped: No API Key.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        filename = os.path.basename(file_path)
        print(f"🤖 Processing PDF: {filename}...")

        file_upload = client.files.upload(file=file_path)
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            return None

        prompt = """
        Analyze this syllabus for Calendar Import.
        PHASE 1: METADATA (Course Name, Class Schedule with START TIME).
        PHASE 2: ASSIGNMENTS (Deliverables, Readings, Dates YYYY-MM-DD).
        """

        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[file_upload, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyllabusResponse
            )
        )
        
        data: SyllabusResponse = response.parsed
        
        schedule_map = {}
        for meeting in data.metadata.class_meetings:
            std_time = standardize_time(meeting.start_time)
            for day in meeting.days:
                for full_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if full_day.lower() in day.lower():
                        schedule_map[full_day] = std_time

        rows = []
        for item in data.assignments:
            rows.append({
                "Course": data.metadata.course_name,
                "Date": item.date,
                "Time": item.time, 
                "Category": item.category,
                "Assignment": item.assignment_name,
                "Description": item.description
            })
            
        df = pd.DataFrame(rows)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['Date'])
            df['Time'] = df.apply(lambda row: resolve_time(row, schedule_map), axis=1)

        return df
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        return None

def map_pdf_category_to_model(pdf_category):
    cat = pdf_category.lower()
    if 'reading' in cat: return 'readings'
    if 'writing' in cat: return 'essay'
    if 'presentation' in cat: return 'presentation'
    if 'project' in cat: return 'research_paper'
    if 'exam' in cat: return 'p_set' 
    return 'p_set' 


# ==========================================
# 4. ML MODEL INITIALIZATION
# ==========================================

model = None
model_columns = []
model_status = "Not Initialized"

new_column_names = {
    'What year are you? ': 'year',
    'What is your major/concentration?': 'major',
    'Second concentration? (if none, select N/A)': 'second_concentration',
    'Minor? (if none select N/A)': 'minor',
    'What class was the assignment for (Please write as said in BrightSpace)': 'class_name',
    'What field of study was the assignment in?': 'field_of_study',
    'What type of assignment was it?': 'assignment_type',
    'Approximately how long did it take (in hours)': 'time_spent_hours',
    'What was the extent of your reliance on external resources? ': 'external_resources',
    'Where did you primarily work on the assignment?': 'work_location',
    'Did you work in a group?': 'worked_in_group',
    'Did you have to submit the assignment in person (physical copy)?': 'submitted_in_person',
    'Approximately how many separate work sessions did you spend on this assignment? (1 or more)': 'work_sessions'
}

category_mapping = {
    'Accounting': 'business', 'Finance': 'business', 'Economics': 'business',
    'Business Administration': 'business', 'Management': 'business', 'Marketing': 'business',
    'International Business': 'business', 'Entrepreneurship': 'business',
    'Supply Chain Management / Logistics': 'business',
    'Management Information Systems (MIS)': 'tech_data', 'Computer Science': 'tech_data',
    'Information Technology': 'tech_data', 'Data Science': 'tech_data', 'Data Analytics': 'tech_data',
    'Computer Engineering': 'engineering', 'Software Engineering': 'engineering',
    'Electrical Engineering': 'engineering', 'Mechanical Engineering': 'engineering',
    'Industrial Engineering': 'engineering', 'Civil Engineering': 'engineering',
    'Chemical Engineering': 'engineering', 'Systems Engineering': 'engineering',
    'Biomedical Engineering': 'engineering', 'Environmental Engineering': 'engineering',
    'Mathematics': 'math', 'Statistics': 'math', 'Applied Mathematics': 'math',
    'Physics': 'natural_sciences', 'Chemistry': 'natural_sciences', 'Biology': 'natural_sciences',
    'Environmental Science': 'natural_sciences', 'Biochemistry': 'natural_sciences',
    'Neuroscience': 'natural_sciences', 'Marine Science': 'natural_sciences',
    'Environmental Studies': 'natural_sciences', 'Agriculture': 'natural_sciences', 'Forestry': 'natural_sciences',
    'Political Science': 'social_sciences_law', 'Psychology': 'social_sciences_law',
    'Sociology': 'social_sciences_law', 'Anthropology': 'social_sciences_law',
    'International Relations': 'social_sciences_law', 'Public Policy': 'social_sciences_law',
    'Geography': 'social_sciences_law', 'Criminology': 'social_sciences_law',
    'Legal Studies': 'social_sciences_law', 'Urban Studies / Planning': 'social_sciences_law',
    'Public Administration': 'social_sciences_law', 'Homeland Security': 'social_sciences_law',
    'English / Literature': 'arts_humanities', 'History': 'arts_humanities',
    'Philosophy': 'arts_humanities', 'Linguistics': 'arts_humanities',
    'Art / Art History': 'arts_humanities', 'Design / Graphic Design': 'arts_humanities',
    'Music': 'arts_humanities', 'Theatre / Performing Arts': 'arts_humanities',
    'Communications': 'arts_humanities', 'Journalism': 'arts_humanities',
    'Film / Media Studies': 'arts_humanities',
    'Nursing': 'health_education', 'Public Health': 'health_education',
    'Pre-Med / Biology (Health Sciences)': 'health_education',
    'Kinesiology / Exercise Science': 'health_education', 'Pharmacy': 'health_education',
    'Nutrition': 'health_education', 'Education': 'health_education',
    'Early Childhood Education': 'health_education', 'Secondary Education': 'health_education',
    'Human Development': 'health_education', 'Social Work': 'health_education',
}

assignment_type_mapping = {
    'Problem Set': 'p_set', 'Coding Assignment': 'coding', 'Research Paper': 'research_paper',
    'Creative Writing/Essay': 'essay', 'Presentation/Slide deck': 'presentation',
    'Modeling (financial, statistics, data)': 'modeling',
    'Discussion post/short written assignment': 'discussion',
    'Readings (textbooks or otherwise)': 'readings', 'Case Study': 'case_study'
}

external_resources_mapping = {
    'Textbook / class materials': 'class_materials', 'Google/internet': 'google',
    'AI / Chatgpt': 'ai', 'Tutoring service (Chegg, etc.)': 'tutoring_service',
    'Study group with peers': 'study_group', 'Other': 'other'
}

work_location_mapping = {
    'At home/private setting': 'home', 'School/library': 'school',
    'Other public setting (cafe, etc.)': 'public'
}

def initialize_model():
    global model
    global model_columns
    global model_status

    if ElasticNet is None:
        model_status = "Skipped (sklearn missing)"
        return
    
    if global_df.empty:
        model_status = "Skipped (CSV missing)"
        return

    try:
        survey_df = global_df.copy()
        survey_df = survey_df.rename(columns=new_column_names)
        
        survey_df['major_category'] = survey_df['major'].map(category_mapping)
        survey_df['second_concentration_category'] = survey_df['second_concentration'].map(category_mapping)
        survey_df['minor_category'] = survey_df['minor'].map(category_mapping)
        survey_df['field_of_study_category'] = survey_df['field_of_study'].map(category_mapping)
        
        survey_df['assignment_type'] = survey_df['assignment_type'].replace(assignment_type_mapping)
        survey_df['external_resources'] = survey_df['external_resources'].replace(external_resources_mapping)
        survey_df['work_location'] = survey_df['work_location'].replace(work_location_mapping)

        categorical_cols = ['year', 'major_category', 'second_concentration_category', 'minor_category', 
                            'field_of_study_category', 'assignment_type', 'external_resources', 
                            'work_location', 'worked_in_group', 'submitted_in_person']
        
        for col in categorical_cols:
            if col in survey_df.columns:
                survey_df = pd.get_dummies(survey_df, columns=[col], prefix=col, dtype=int, drop_first=True)

        if 'Timestamp' in survey_df.columns: survey_df = survey_df.drop(columns=['Timestamp'])
        drop_cols = ['major', 'second_concentration', 'minor', 'class_name', 'field_of_study', 'Who referred you to this survey?']
        survey_df = survey_df.drop(columns=[c for c in drop_cols if c in survey_df.columns])

        X = survey_df.drop('time_spent_hours', axis=1)
        y = survey_df['time_spent_hours']

        model = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
        model.fit(X, y)
        
        model_columns = list(X.columns)
        model_status = "Trained Successfully"
        print("✅ Model trained and ready.")
        
    except Exception as e:
        model_status = f"Error: {e}"
        print(f"❌ Model initialization failed: {e}")

initialize_model()


# ============================================
# 5. DATA STORE
# ============================================

class SimpleDataStore:
    def __init__(self, data_file=TRAINING_PATH, about_file=ABOUT_PATH):
        self.data_file = data_file
        self.about_file = about_file
        self.df = pd.DataFrame()
        self.about_df = pd.DataFrame()
    
    def save_submission(self, survey, courses, pdf_filenames, ics_filenames):
        timestamp = datetime.now().isoformat()
        new_row = {
            'timestamp': timestamp,
            'year': survey.get('year'),
            'major': survey.get('major'),
            'num_courses': len(courses),
            'courses_json': json.dumps(courses),
            'pdf_files': json.dumps(pdf_filenames),
            'ics_files': json.dumps(ics_filenames)
        }
        
        self.df = pd.DataFrame([new_row])
        self.df.to_pickle(self.data_file)
        return 0

    def get_dataframe(self):
        if os.path.exists(self.data_file):
            return pd.read_pickle(self.data_file)
        return pd.DataFrame()

data_store = SimpleDataStore()

# ============================================
# 6. PREDICTION LOGIC
# ============================================

def process_and_predict(survey, courses):
    global model
    global model_columns
    
    if not model or not courses:
        return [0.0] * len(courses)

    rows = []
    for course in courses:
        row = {
            'year': survey.get('year'),
            'major': survey.get('major'),
            'second_concentration': survey.get('secondConcentration', 'N/A'),
            'minor': survey.get('minor', 'N/A'),
            'work_location': survey.get('workLocation'),
            'worked_in_group': survey.get('workInGroup'),
            'submitted_in_person': survey.get('submitInPerson', 'No'),
            'assignment_type': course.get('type'),
            'field_of_study': course.get('subject'), 
            'external_resources': course.get('resources'),
            'class_name': course.get('name')
        }
        rows.append(row)

    input_df = pd.DataFrame(rows)

    if 'major' in input_df: input_df['major_category'] = input_df['major'].map(category_mapping)
    if 'second_concentration' in input_df: input_df['second_concentration_category'] = input_df['second_concentration'].map(category_mapping)
    if 'minor' in input_df: input_df['minor_category'] = input_df['minor'].map(category_mapping)
    if 'field_of_study' in input_df: input_df['field_of_study_category'] = input_df['field_of_study'].map(category_mapping)

    if 'assignment_type' in input_df: input_df['assignment_type'] = input_df['assignment_type'].replace(assignment_type_mapping)
    if 'external_resources' in input_df: input_df['external_resources'] = input_df['external_resources'].replace(external_resources_mapping)
    if 'work_location' in input_df: input_df['work_location'] = input_df['work_location'].replace(work_location_mapping)

    categorical_cols = ['year', 'major_category', 'second_concentration_category', 'minor_category', 
                        'field_of_study_category', 'assignment_type', 'external_resources', 
                        'work_location', 'worked_in_group', 'submitted_in_person']

    for col in categorical_cols:
        if col in input_df.columns:
            input_df = pd.get_dummies(input_df, columns=[col], prefix=col, dtype=int, drop_first=True)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    try:
        preds = model.predict(input_df)
        return [float(p) for p in preds]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return [0.0] * len(courses)

# ============================================
# 7. ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    # DIAGNOSTIC HOME PAGE
    return jsonify({
        "status": "API Online",
        "csv_status": csv_status,
        "model_status": model_status,
        "import_errors": import_errors,
        "gemini_key_present": bool(GEMINI_API_KEY)
    })

@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory(static_dir, filename)

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        data_json = request.form.get('data')
        if not data_json: return jsonify({'error': 'No data'}), 400
        
        data = json.loads(data_json)
        survey = data.get('survey', {})
        manual_courses = data.get('courses', [])
        
        # 1. Handle PDF
        pdf_filenames = []
        pdf_extracted_courses = []
        
        if 'pdfs' in request.files:
            files_to_parse = []
            for pdf_file in request.files.getlist('pdfs'):
                if allowed_file(pdf_file.filename):
                    filename = secure_filename(pdf_file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    pdf_file.save(filepath)
                    pdf_filenames.append(filename)
                    files_to_parse.append(filepath)

            # Only run parser if no import errors
            if files_to_parse and genai and GEMINI_API_KEY:
                all_pdf_dfs = []
                for fpath in files_to_parse:
                    df_parsed = parse_syllabus(fpath)
                    if df_parsed is not None:
                        all_pdf_dfs.append(df_parsed)
                
                if all_pdf_dfs:
                    master_pdf_df = pd.concat(all_pdf_dfs, ignore_index=True)
                    # Convert to course objects...
                    for _, row in master_pdf_df.iterrows():
                        pdf_extracted_courses.append({
                            'name': f"{row['Course']} - {row['Assignment']}",
                            'type': map_pdf_category_to_model(row['Category']),
                            'subject': survey.get('major'),
                            'resources': 'Google/internet',
                            'date': row['Date'],
                            'time': row['Time'],
                            'description': row['Description'],
                            'source': 'pdf_parser'
                        })

        # 2. Handle ICS
        ics_filenames = []
        if 'ics' in request.files:
            ics_file = request.files['ics']
            if allowed_file(ics_file.filename):
                filename = secure_filename(ics_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                ics_file.save(filepath)
                ics_filenames.append(filename)

        all_courses = manual_courses + pdf_extracted_courses
        data_store.save_submission(survey, all_courses, pdf_filenames, ics_filenames)
        
        predicted_times = process_and_predict(survey, all_courses)
        
        courses_with_predictions = []
        for i, course in enumerate(all_courses):
            c_copy = course.copy()
            if i < len(predicted_times):
                c_copy['predicted_hours'] = round(predicted_times[i], 2)
            else:
                c_copy['predicted_hours'] = 0
            courses_with_predictions.append(c_copy)
            
        return jsonify({
            'status': 'success',
            'courses': courses_with_predictions,
            'debug_info': {
                'pdf_parsed': bool(genai and GEMINI_API_KEY),
                'model_active': bool(model)
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/view-data', methods=['GET'])
def view_data():
    return jsonify({'status': 'success', 'data': data_store.get_dataframe().to_dict('records')})

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
