from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import time
from datetime import datetime
from werkzeug.utils import secure_filename

# ============================================
# 1. SETUP (LIGHTWEIGHT STARTUP)
# ============================================

current_directory = os.path.dirname(os.path.abspath(__file__))
# Check if folders exist before setting paths
template_dir = os.path.join(current_directory, 'templates') if os.path.exists(os.path.join(current_directory, 'templates')) else None
static_dir = os.path.join(current_directory, 'static') if os.path.exists(os.path.join(current_directory, 'static')) else None

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Storage Config
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'pdf', 'ics'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

TRAINING_PATH = os.path.join(UPLOAD_FOLDER, "training_data.pkl")
ABOUT_PATH = os.path.join(UPLOAD_FOLDER, "about_you.pkl")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ============================================
# 2. GLOBAL STATE (LAZY LOADED)
# ============================================
# We verify these exist, but we do NOT load them yet.
model_cache = {
    "model": None,
    "columns": [],
    "status": "Not Loaded",
    "df_rows": 0
}

# ============================================
# 3. HEAVY LIFTING FUNCTIONS (RUN ONLY ON DEMAND)
# ============================================

def get_lazy_imports():
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import ElasticNet
        return pd, np, ElasticNet, None

def get_genai_imports():
        from google import genai
        from google.genai import types
        from pydantic import BaseModel, Field
        return genai, types, BaseModel, Field, None

def train_model_lazy():
    """Loads CSV and trains model ONLY if not already done."""
    # Return cached model if it exists
    if model_cache["model"] is not None:
        return model_cache["model"], model_cache["columns"]

    pd, np, ElasticNet, err = get_lazy_imports()
    if err:
        model_cache["status"] = f"Import Error: {err}"
        return None, []

    csv_path = os.path.join(current_directory, 'survey.csv')
    if not os.path.exists(csv_path):
        model_cache["status"] = "CSV Not Found"
        return None, []

    try:
        print("⏳ Loading CSV and Training Model...")
        df = pd.read_csv(csv_path)
        model_cache["df_rows"] = len(df)
        
        # Mappings
        new_column_names = {
            'What year are you? ': 'year', 'What is your major/concentration?': 'major',
            'Second concentration? (if none, select N/A)': 'second_concentration',
            'Minor? (if none select N/A)': 'minor',
            'What field of study was the assignment in?': 'field_of_study',
            'What type of assignment was it?': 'assignment_type',
            'Approximately how long did it take (in hours)': 'time_spent_hours',
            'What was the extent of your reliance on external resources? ': 'external_resources',
            'Where did you primarily work on the assignment?': 'work_location',
            'Did you work in a group?': 'worked_in_group',
            'Did you have to submit the assignment in person (physical copy)?': 'submitted_in_person'
        }
        df = df.rename(columns=new_column_names)

        # Simplified Cleaning for stability
        # (Add your full cleaning logic here if needed, but keeping it robust for now)
        cols_to_encode = ['year', 'assignment_type', 'external_resources', 'work_location', 'worked_in_group', 'submitted_in_person']
        for col in cols_to_encode:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int, drop_first=True)

        # Drop non-numeric for training
        df_train = df.select_dtypes(include=[np.number])
        
        if 'time_spent_hours' not in df_train.columns:
            model_cache["status"] = "Target Column Missing"
            return None, []

        X = df_train.drop('time_spent_hours', axis=1)
        y = df_train['time_spent_hours']
        
        clf = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
        clf.fit(X, y)
        
        model_cache["model"] = clf
        model_cache["columns"] = list(X.columns)
        model_cache["status"] = "Active"
        print("✅ Model Trained Successfully")
        return clf, list(X.columns)

    except Exception as e:
        model_cache["status"] = f"Training Error: {str(e)}"
        print(f"❌ Training Failed: {e}")
        return None, []

# ============================================
# 4. ROUTE HANDLERS
# ============================================

@app.route('/', methods=['GET'])
def home():
    return render_template('mains.html')

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    pd, np, _, _ = get_lazy_imports() # Need pandas for data saving
    
    # 1. Parse Input
    try:
        data_json = request.form.get('data')
        if not data_json: return jsonify({'error': 'No data provided'}), 400
        data_input = json.loads(data_json)
        survey = data_input.get('survey', {})
        courses = data_input.get('courses', [])
    except Exception as e:
        return jsonify({'error': f"JSON Parse Error: {e}"}), 400

    # 2. Train Model (Lazy)
    clf, model_cols = train_model_lazy()
    
    # 3. Save Data (If pandas is available)
    if pd:
        try:
            row = {
                'timestamp': datetime.now().isoformat(),
                'year': survey.get('year'),
                'major': survey.get('major'),
                'num_courses': len(courses),
                'raw_json': data_json
            }
            pd.DataFrame([row]).to_pickle(TRAINING_PATH)
        except Exception as e:
            print(f"Save Error: {e}")

    # 4. Process Courses & Predict
    results = []
    
    # --- PDF PARSING LOGIC (Lazy) ---
    # Only try to parse if files exist and Key is set
    genai, types, BaseModel, _, _ = get_genai_imports()
    if 'pdfs' in request.files and genai and GEMINI_API_KEY:
        # (Insert simplified PDF logic here or skip for safety check)
        pass 

    # --- PREDICTION LOGIC ---
    for course in courses:
        predicted = 0
        if clf and pd:
            try:
                # Create a mini dataframe for one row
                # We simply map the input fields to the model columns
                # For robustness, we create a zero-filled DF matching model columns
                input_row = pd.DataFrame(0, index=[0], columns=model_cols)
                
                # Simple mapping example (expand this to match your specific one-hot encoding logic)
                # If the user says "year" is "2026", we look for column "year_2026"
                yr_col = f"year_{survey.get('year')}"
                if yr_col in input_row.columns: input_row[yr_col] = 1
                
                type_col = f"assignment_type_{course.get('type')}" # Needs your mapping logic
                if type_col in input_row.columns: input_row[type_col] = 1

                predicted = float(clf.predict(input_row)[0])
            except Exception as e:
                print(f"Prediction row error: {e}")
                predicted = 0
        
        # Attach result
        c_out = course.copy()
        c_out['predicted_hours'] = round(predicted, 2)
        results.append(c_out)

    return jsonify({
        "status": "success",
        "model_status": model_cache["status"],
        "courses": results
    })

@app.route('/api/view-data', methods=['GET'])
def view_data():
    pd, _, _, _ = get_lazy_imports()
    if pd and os.path.exists(TRAINING_PATH):
        try:
            return jsonify(pd.read_pickle(TRAINING_PATH).to_dict('records'))
        except:
            return jsonify([])
    return jsonify([])

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

# Helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
