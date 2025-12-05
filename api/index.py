from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import PyPDF2
from icalendar import Calendar
import pandas as pd

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

UPLOAD_FOLDER = 'template'
ALLOWED_EXTENSIONS = {'pdf', 'ics'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Base directory (folder where app.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_PATH = os.path.join(BASE_DIR, "training_data.pkl")
ABOUT_PATH = os.path.join(BASE_DIR, "about_you.pkl")

# ============================================
# DATA STORAGE (LATEST ONLY – OVERWRITE)
# ============================================

class SimpleDataStore:
    def __init__(self, data_file=TRAINING_PATH, about_file=ABOUT_PATH):
        self.data_file = data_file
        self.about_file = about_file
        self.df = self._load_df(self.data_file)
        self.about_df = self._load_df(self.about_file)
    
    def _load_df(self, path):
        """Load existing data or create new empty DataFrame."""
        if os.path.exists(path):
            print(f"📊 Loading existing data from {path}")
            return pd.read_pickle(path)
        else:
            print(f"📊 Creating new empty DataFrame for {path}")
            return pd.DataFrame()
    
    def save_submission(self, survey, courses, pdf_data, calendar_events):
        """
        Save ONE submission as the ONLY row in:
          - training_data.pkl (full features)
          - about_you.pkl (About You only)
        Every click on 'Generate My Schedule' replaces previous data.
        """
        time_prefs = survey.get('preferredWorkingTime', []) or []
        
        # ---------- FULL ROW (main training data) ----------
        timestamp = datetime.now().isoformat()
        new_row = {
            'timestamp': timestamp,
            'year': survey.get('year'),
            'major': survey.get('major'),
            'work_in_group': survey.get('workInGroup'),
            'work_location': survey.get('workLocation'),

            # Preferred time ranking (1–3)
            'preferred_time_1': time_prefs[0].get('time') if len(time_prefs) > 0 else None,
            'preferred_time_2': time_prefs[1].get('time') if len(time_prefs) > 1 else None,
            'preferred_time_3': time_prefs[2].get('time') if len(time_prefs) > 2 else None,

            # NEW: min/max work time (raw strings like "08:00", "22:00")
            'min_work_time': survey.get('minWorkTime'),
            'max_work_time': survey.get('maxWorkTime'),

            'num_courses': len(courses),
            'num_pdfs': len(pdf_data),
            'num_calendar_events': len(calendar_events),

            # Course aggregates
            'has_stem_courses': any(c.get('classMajor') == 'stem' for c in courses),
            'has_business_courses': any(c.get('classMajor') == 'business' for c in courses),
            'has_writing': any(c.get('type') == 'writing' for c in courses),
            'has_lab': any(c.get('type') == 'lab' for c in courses),
            'total_sessions': sum(self._parse_sessions(c.get('sessions')) for c in courses),
            'uses_ai': any(c.get('resources') in ['ai', 'mixed'] for c in courses),
            'uses_textbook': any(c.get('resources') == 'textbook' for c in courses),

            # Raw JSON storage (all course details preserved here)
            'courses_json': json.dumps(courses),
            'pdf_files': json.dumps([p.get('filename') for p in pdf_data]),
        }
        
        # ---------- ABOUT YOU ROW (subset only) ----------
        about_row = {
            'timestamp': timestamp,
            'year': new_row['year'],
            'major': new_row['major'],
            'work_in_group': new_row['work_in_group'],
            'work_location': new_row['work_location'],
            'preferred_time_1': new_row['preferred_time_1'],
            'preferred_time_2': new_row['preferred_time_2'],
            'preferred_time_3': new_row['preferred_time_3'],
            'min_work_time': new_row['min_work_time'],
            'max_work_time': new_row['max_work_time'],
        }
        
        # 🔁 OVERWRITE both DataFrames with a single row
        self.df = pd.DataFrame([new_row])
        self.about_df = pd.DataFrame([about_row])
        
        # Save to disk (these overwrite existing pickles)
        self.df.to_pickle(self.data_file)
        self.about_df.to_pickle(self.about_file)
        
        print("✅ Saved latest submission only (training_data & about_you each have 1 row).")
        return 0  # only row index is 0
    
    def _parse_sessions(self, sessions_str):
        """Convert session ranges to approximate numeric values."""
        if not sessions_str:
            return 0
        mapping = {'1-2': 1.5, '3-5': 4, '6-10': 8, '10+': 12}
        return mapping.get(sessions_str, 0)
    
    def get_dataframe(self):
        """Full training data (1 row, latest submission)."""
        return self.df.copy()
    
    def get_about_dataframe(self):
        """About You data (1 row, latest submission)."""
        return self.about_df.copy()
    
    def export_to_csv(self,
                      training_csv='training_data.csv',
                      about_csv='about_you.csv'):
        """Export both tables to CSV."""
        self.df.to_csv(os.path.join(BASE_DIR, training_csv), index=False)
        self.about_df.to_csv(os.path.join(BASE_DIR, about_csv), index=False)
        return training_csv, about_csv

# Initialize global datastore
data_store = SimpleDataStore()

# ============================================
# HELPERS
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += (page.extract_text() or "")
            return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def parse_ics_file(ics_path):
    try:
        with open(ics_path, 'rb') as file:
            cal = Calendar.from_ical(file.read())
            events = []
            for component in cal.walk():
                if component.name == "VEVENT":
                    dtstart = component.get('dtstart')
                    dtend = component.get('dtend')
                    events.append({
                        'summary': str(component.get('summary')),
                        'start': dtstart.dt.isoformat() if dtstart else None,
                        'end': dtend.dt.isoformat() if dtend else None,
                        'description': str(component.get('description', ''))
                    })
            return events
    except Exception as e:
        print(f"❌ Error parsing ICS: {e}")
        return []

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return send_from_directory('static', 'mains.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        data_json = request.form.get('data')
        if not data_json:
            return jsonify({'error': 'No data provided'}), 400
        
        data = json.loads(data_json)
        survey = data.get('survey', {})
        courses = data.get('courses', [])

        # PDFs
        pdf_contents = []
        if 'pdfs' in request.files:
            for pdf_file in request.files.getlist('pdfs'):
                if allowed_file(pdf_file.filename):
                    filename = secure_filename(pdf_file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    pdf_file.save(filepath)
                    text = extract_text_from_pdf(filepath)
                    pdf_contents.append({
                        'filename': filename,
                        'preview': text[:500]
                    })
        
        # ICS
        calendar_events = []
        if 'ics' in request.files:
            ics_file = request.files['ics']
            if allowed_file(ics_file.filename):
                filename = secure_filename(ics_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                ics_file.save(filepath)
                calendar_events = parse_ics_file(filepath)
        
        # Save submission (overwrites both training_data & about_you)
        submission_id = data_store.save_submission(
            survey,
            courses,
            pdf_contents,
            calendar_events
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Saved latest submission (old data replaced).',
            'submission_id': submission_id,
            'total_records': len(data_store.df),  # 1 if anything saved
            'survey': survey,
            'courses_count': len(courses),
            'pdfs_processed': len(pdf_contents),
            'calendar_events_count': len(calendar_events),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"❌ Error in /api/generate-schedule: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/view-data', methods=['GET'])
def view_data():
    """Full training data (flattened submission)."""
    df = data_store.get_dataframe()
    return jsonify({
        'status': 'success',
        'total_records': len(df),
        'columns': list(df.columns),
        'data': df.to_dict('records')
    })


@app.route('/api/view-about', methods=['GET'])
def view_about():
    """About You table only."""
    about_df = data_store.get_about_dataframe()
    return jsonify({
        'status': 'success',
        'total_records': len(about_df),
        'columns': list(about_df.columns),
        'data': about_df.to_dict('records')
    })


@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    training_csv, about_csv = data_store.export_to_csv()
    return jsonify({
        'status': 'success',
        'message': 'Exported CSVs.',
        'training_csv': training_csv,
        'about_csv': about_csv,
        'records_training': len(data_store.df),
        'records_about': len(data_store.about_df)
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    df = data_store.get_dataframe()
    if len(df) == 0:
        return jsonify({'message': 'No data yet'})
    
    return jsonify({
        'total_records': len(df),  # effectively 1
        'majors': df['major'].value_counts().to_dict() if 'major' in df.columns else {},
        'avg_courses': float(df['num_courses'].mean()) if 'num_courses' in df.columns and len(df) > 0 else 0,
        'common_location': df['work_location'].mode()[0]
            if 'work_location' in df.columns and len(df['work_location'].mode()) > 0
            else None,
        'total_pdfs_uploaded': int(df['num_pdfs'].sum()) if 'num_pdfs' in df.columns else 0
    })


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'records_training': len(data_store.df),
        'records_about': len(data_store.about_df)
    })

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🚀 Server running on http://localhost:{port}")
    print(f"📄 Training data file: {data_store.data_file}")
    print(f"📄 About-you file:    {data_store.about_file}")
    print(f"📈 Current records (training): {len(data_store.df)}")
    print(f"📈 Current records (about):    {len(data_store.about_df)}\n")
    app.run(host="0.0.0.0", port=port, debug=True)

# source venv/bin/activate

