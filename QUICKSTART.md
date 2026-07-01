# Quick Start Guide - ECG Arrhythmia Detection System

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize Database
```bash
cd ecg-dashboard
python init_db.py
```

You'll see:
```
✓ Demo doctor created successfully!
  Username: doctor
  Password: doctor123
```

### Step 3: Start the Server
```bash
python app.py
```

### Step 4: Login
1. Open browser: `http://127.0.0.1:5000`
2. Login with:
   - **Username**: `doctor`
   - **Password**: `doctor123`

---

## 📋 First Steps After Login

### 1. Add Your First Patient
1. Click **"Patients"** in sidebar
2. Click **"Add New Patient"** button
3. Fill in patient details:
   - Full Name (required)
   - Mobile Number (required)
   - Disease, Address, Medical History (optional)
4. Click **"Save Patient"**

### 2. Analyze an ECG
1. Click **"Analyze ECG"** in sidebar
2. Select a patient from dropdown (optional)
3. Either:
   - Upload your own CSV file, OR
   - Click **"Use Sample ECG"** to test
4. Wait for analysis (10-30 seconds)
5. View results with:
   - Beat classifications
   - Arrhythmia diagnosis
   - Risk assessment
   - HRV metrics

### 3. View Patient Profile
1. Go to **"Patients"** → **"Patient List"**
2. Click **"View Profile"** on any patient
3. See:
   - Patient information
   - Medical history
   - ECG records history
   - Health improvements over time
   - Last appointment details

### 4. Generate PDF Report
1. After analyzing an ECG, click **"Report"**
2. Fill in patient details (if not linked)
3. Click **"Download PDF Report"**
4. Professional medical report is generated

---

## 🎯 Common Workflows

### Workflow 1: New Patient Visit
```
1. Add Patient → Fill details → Save
2. Analyze ECG → Select patient → Upload file
3. View Results → Review diagnosis
4. Generate Report → Download PDF
5. View Patient Profile → Check history
```

### Workflow 2: Follow-up Visit
```
1. Patients → Find patient → View Profile
2. Check previous ECG records
3. Analyze ECG → Select patient → Upload new file
4. View Results → Compare with previous
5. Patient Profile → See improvements
```

### Workflow 3: Quick Analysis (No Patient)
```
1. Analyze ECG → Don't select patient
2. Upload file or use sample
3. View Results
4. Generate Report (add patient info manually)
```

---

## 📊 Understanding Results

### Beat Classifications
- **N (Normal)**: Regular heartbeat - Low risk
- **S (Supraventricular)**: Extra beat from atria - Moderate risk
- **V (Ventricular)**: Extra beat from ventricles - High risk
- **F (Fusion)**: Mixed beat pattern - Moderate risk
- **Q (Unknown/Paced)**: Pacemaker or unclear - Low risk

### Arrhythmia Types
- **NSR**: Normal Sinus Rhythm ✅
- **AFib**: Atrial Fibrillation ⚠️ High Risk
- **VT**: Ventricular Tachycardia ⚠️ High Risk
- **PVC**: Premature Ventricular Contractions ⚠️ Moderate
- **SVT**: Supraventricular Tachycardia ⚠️ Moderate
- **SBR**: Sinus Bradycardia (slow) ℹ️ Mild
- **STach**: Sinus Tachycardia (fast) ℹ️ Mild

### Risk Levels
- 🟢 **Low**: Normal or minor variations
- 🟡 **Moderate**: Requires monitoring
- 🔴 **High**: Urgent medical attention needed

---

## 🔧 Troubleshooting

### Can't Login?
- Make sure you ran `python init_db.py`
- Use credentials: `doctor` / `doctor123`
- Or create new account via signup page

### Upload Fails?
- Check CSV format: must have `signal` or `time,signal` columns
- Minimum 10 seconds of data recommended
- Sampling rate: 360 Hz preferred

### No Results Showing?
- Make sure analysis completed (check for success message)
- Try refreshing the page
- Check browser console for errors

### Database Errors?
```bash
# Reset database
cd ecg-dashboard
rm ecg_dashboard.db
python init_db.py
```

---

## 📁 Sample CSV Format

### Option 1: With Time Column
```csv
time,signal
0.0000,0.145
0.0028,0.150
0.0056,0.142
0.0083,0.138
...
```

### Option 2: Signal Only
```csv
signal
0.145
0.150
0.142
0.138
...
```

The system auto-detects the format!

---

## 🎓 Tips for Best Results

1. **Recording Length**: Use at least 10 seconds for accurate arrhythmia detection
2. **Sampling Rate**: 360 Hz is optimal (MIT-BIH standard)
3. **Signal Quality**: Clean signals produce better results
4. **Patient Linking**: Always link ECGs to patients to track improvements
5. **Regular Monitoring**: Upload multiple ECGs over time to see trends

---

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review sample ECG files in `ecg-dashboard/input_samples/`
- Check the [Known Issues](#known-issues) section in README

---

## 🔐 Security Notes

- Change default password after first login
- Use strong passwords for production
- Database is stored locally in `ecg_dashboard.db`
- Session timeout: 24 hours
- Passwords are hashed with SHA-256

---

## ⚠️ Medical Disclaimer

This system is for **research and educational purposes only**. 
- NOT for clinical diagnosis
- NOT a substitute for professional medical advice
- Always consult qualified healthcare professionals

---

**Ready to start? Run `python app.py` and visit http://127.0.0.1:5000** 🚀
