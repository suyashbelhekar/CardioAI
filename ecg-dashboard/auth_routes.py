"""
Authentication and patient management routes.
Imported by app.py after the app object is created.
"""

from flask import render_template, request, session, redirect, url_for
from functools import wraps
import store


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'doctor_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def register(app):
    """Register all auth + patient routes onto the Flask app."""

    # ── Auth ──────────────────────────────────────────────────────────────────

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            doc = store.authenticate_doctor(
                request.form.get('username', '').strip(),
                request.form.get('password', '')
            )
            if doc:
                session['doctor_id']    = doc['id']
                session['doctor_name']  = doc['full_name']
                session['doctor_email'] = doc['email']
                return redirect('/')
            return render_template('login.html', error='Invalid username or password')
        return render_template('login.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            pw  = request.form.get('password', '')
            cpw = request.form.get('confirm_password', '')
            if pw != cpw:
                return render_template('signup.html', error='Passwords do not match')
            if len(pw) < 6:
                return render_template('signup.html', error='Password must be at least 6 characters')
            did = store.create_doctor(
                username       = request.form.get('username', '').strip(),
                email          = request.form.get('email', '').strip(),
                password       = pw,
                full_name      = request.form.get('full_name', '').strip(),
                specialization = request.form.get('specialization', ''),
                license_number = request.form.get('license_number', ''),
                phone          = request.form.get('phone', ''),
            )
            if did:
                return redirect('/login?success=1')
            return render_template('signup.html', error='Username or email already exists')
        return render_template('signup.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect('/login')

    # ── Patients ──────────────────────────────────────────────────────────────

    @app.route('/patients')
    @login_required
    def patients_list():
        from datetime import datetime, timedelta
        patients = store.get_patients(session['doctor_id'])
        cutoff   = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        return render_template('patients.html',
            patients        = patients,
            recent_count    = sum(1 for p in patients if p['created_at'][:10] >= cutoff),
            high_risk_count = sum(1 for p in patients
                                  if p['ecg_records'] and p['ecg_records'][0]['risk_level'] == 'High'),
            total_records   = sum(len(p['ecg_records']) for p in patients),
        )

    @app.route('/patients/add', methods=['GET', 'POST'])
    @login_required
    def add_patient():
        if request.method == 'POST':
            store.add_patient(
                doctor_id       = session['doctor_id'],
                full_name       = request.form.get('full_name', '').strip(),
                dob             = request.form.get('dob', ''),
                gender          = request.form.get('gender', ''),
                address         = request.form.get('address', ''),
                mobile_no       = request.form.get('mobile_no', ''),
                email           = request.form.get('email', ''),
                disease         = request.form.get('disease', ''),
                medical_history = request.form.get('medical_history', ''),
            )
            return redirect('/patients')
        return render_template('add_patient.html')

    @app.route('/patients/<patient_id>')
    @login_required
    def patient_profile(patient_id):
        patient = store.get_patient(patient_id)
        if not patient or patient['doctor_id'] != session['doctor_id']:
            return redirect('/patients')
        return render_template('patient_profile.html',
            patient      = patient,
            improvements = store.get_improvements(patient),
            latest_record= patient['ecg_records'][0] if patient['ecg_records'] else None,
        )
