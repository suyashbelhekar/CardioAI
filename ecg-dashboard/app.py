import os, io, json, base64, uuid, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
import tensorflow as tf
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm
import tempfile
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'doctor_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


import store          # JSON-file storage
import recovery as rec_model  # Recovery time prediction

app = Flask(__name__)
app.secret_key = 'ecg-arrhythmia-secret-2024'

RESULTS_DIR = os.path.join(tempfile.gettempdir(), 'ecg_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(data):
    # Clean up result files older than 1 hour
    try:
        cutoff = datetime.now().timestamp() - 3600
        for f in os.listdir(RESULTS_DIR):
            fp = os.path.join(RESULTS_DIR, f)
            if os.path.getmtime(fp) < cutoff:
                os.remove(fp)
    except Exception:
        pass
    rid  = str(uuid.uuid4())
    path = os.path.join(RESULTS_DIR, f'{rid}.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    return rid

def load_results(rid):
    if not rid:
        return None
    path = os.path.join(RESULTS_DIR, f'{rid}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_ecg_model.keras')
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), 'sample_ecg.csv')

CLASSES = ['N', 'S', 'V', 'F', 'Q']
CLASS_LABELS = {
    'N': 'Normal Beat',
    'S': 'Supraventricular Ectopic',
    'V': 'Ventricular Ectopic',
    'F': 'Fusion Beat',
    'Q': 'Unknown / Paced'
}
CLASS_COLORS = {
    'N': '#22c55e', 'S': '#f59e0b', 'V': '#ef4444', 'F': '#8b5cf6', 'Q': '#94a3b8'
}
RISK_MAP  = {'N': 'Low', 'S': 'Moderate', 'V': 'High', 'F': 'Moderate', 'Q': 'Low'}

WIN_BEATS = 15
SEVERITY_ORDER = {'Normal':0,'Mild':1,'Moderate':2,'High':3}

ARRHYTHMIA_META = {
    'NSR':      {'name':'Normal Sinus Rhythm',                      'severity':'Normal',   'color':'#22c55e',
                 'description':'Heart beats in a normal, regular rhythm from the sinus node.',
                 'advice':'No arrhythmia detected. Continue routine monitoring as recommended.'},
    'SBR':      {'name':'Sinus Bradycardia',                        'severity':'Mild',     'color':'#3b82f6',
                 'description':'Heart rate below 60 bpm with regular rhythm from the sinus node.',
                 'advice':'Common in athletes. Consult a cardiologist if accompanied by dizziness or fatigue.'},
    'STach':    {'name':'Sinus Tachycardia',                        'severity':'Mild',     'color':'#f59e0b',
                 'description':'Heart rate above 100 bpm with regular rhythm.',
                 'advice':'Often caused by stress, fever, or dehydration. Seek evaluation if persistent.'},
    'PAC':      {'name':'Premature Atrial Contractions (PAC)',      'severity':'Mild',     'color':'#f59e0b',
                 'description':'Extra beats originating in the atria before the normal beat.',
                 'advice':'Usually benign. Reduce caffeine and alcohol. Consult a doctor if frequent.'},
    'PVC':      {'name':'Premature Ventricular Contractions (PVC)', 'severity':'Moderate', 'color':'#ef4444',
                 'description':'Extra beats originating in the ventricles.',
                 'advice':'Seek cardiology evaluation if PVCs are frequent or cause symptoms.'},
    'Bigeminy': {'name':'Ventricular Bigeminy',                     'severity':'Moderate', 'color':'#ef4444',
                 'description':'Every other beat is a premature ventricular contraction (N-V-N-V pattern).',
                 'advice':'Cardiology evaluation recommended. May require Holter monitoring.'},
    'SVT':      {'name':'Supraventricular Tachycardia (SVT)',       'severity':'Moderate', 'color':'#f59e0b',
                 'description':'Rapid heart rate originating above the ventricles.',
                 'advice':'Medical evaluation required. Treatment may include medication or catheter ablation.'},
    'AFib':     {'name':'Atrial Fibrillation (AFib)',               'severity':'High',     'color':'#dc2626',
                 'description':'Irregular, chaotic atrial activity with highly variable RR intervals.',
                 'advice':'Seek prompt medical attention. AFib increases stroke risk. Anticoagulation may be needed.'},
    'VT':       {'name':'Ventricular Tachycardia (VT)',             'severity':'High',     'color':'#dc2626',
                 'description':'Rapid rhythm originating in the ventricles â€” potentially life-threatening.',
                 'advice':'Urgent medical evaluation required. May require ICD or antiarrhythmic therapy.'},
    'Block':    {'name':'Heart Block / Conduction Defect',          'severity':'Moderate', 'color':'#8b5cf6',
                 'description':'Delayed or blocked electrical conduction through the heart.',
                 'advice':'Cardiology evaluation needed. Pacemaker may be required for high-degree block.'},
    'Paced':    {'name':'Paced Rhythm',                             'severity':'Normal',   'color':'#94a3b8',
                 'description':'Heartbeats are being generated by an implanted pacemaker.',
                 'advice':'Continue regular pacemaker follow-up with your cardiologist.'},
}


def extract_window_features(beat_seq, rr_seq, fs=360):
    from collections import Counter as C
    n = len(beat_seq); counts = C(beat_seq); total = max(n,1)
    pct_N=counts.get('N',0)/total; pct_S=counts.get('S',0)/total
    pct_V=counts.get('V',0)/total; pct_F=counts.get('F',0)/total; pct_Q=counts.get('Q',0)/total
    rr=np.array(rr_seq,dtype=np.float64); rr_ms=rr/fs*1000
    mean_rr=float(np.mean(rr_ms)); std_rr=float(np.std(rr_ms))
    rmssd=float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms)>1 else 0
    hr=60000/mean_rr if mean_rr>0 else 75; cv_rr=std_rr/mean_rr if mean_rr>0 else 0
    min_rr=float(np.min(rr_ms)); max_rr=float(np.max(rr_ms)); rr_range=max_rr-min_rr
    rr_diff=np.abs(np.diff(rr_ms)) if len(rr_ms)>1 else np.array([0])
    pnn50=float(np.mean(rr_diff>50)); pnn20=float(np.mean(rr_diff>20))
    irr=float(np.mean(rr_diff/(rr_ms[:-1]+1e-6)>0.20)) if len(rr_ms)>1 else 0
    alt=sum(1 for i in range(1,n-1) if beat_seq[i]!=beat_seq[i-1] and beat_seq[i]==beat_seq[i+1])/max(n-2,1)
    tri=sum(1 for i in range(2,n) if beat_seq[i]==beat_seq[i-2] and beat_seq[i]!=beat_seq[i-1])/max(n-2,1)
    if len(rr_ms)>3:
        rr_norm=(rr_ms-rr_ms.mean())/(rr_ms.std()+1e-8)
        hist,_=np.histogram(rr_norm,bins=5,density=True); hist=hist[hist>0]
        rr_ent=float(-np.sum(hist*np.log(hist+1e-10)))
    else: rr_ent=0.0
    mcv=cur=0
    for b in beat_seq: cur=cur+1 if b=='V' else 0; mcv=max(mcv,cur)
    mcs=cur=0
    for b in beat_seq: cur=cur+1 if b=='S' else 0; mcs=max(mcs,cur)
    mcn=cur=0
    for b in beat_seq: cur=cur+1 if b=='N' else 0; mcn=max(mcn,cur)
    trans=sum(1 for i in range(1,n) if beat_seq[i]!=beat_seq[i-1])/max(n-1,1)
    sv_r=pct_S/(pct_V+1e-6); vs_r=pct_V/(pct_S+1e-6)
    abn=(counts.get('S',0)+counts.get('V',0)+counts.get('F',0))/total
    return [pct_N,pct_S,pct_V,pct_F,pct_Q,
            mean_rr,std_rr,rmssd,hr,cv_rr,
            min_rr,max_rr,rr_range,pnn50,pnn20,irr,
            alt,tri,rr_ent,
            mcv/WIN_BEATS,mcs/WIN_BEATS,mcn/WIN_BEATS,
            trans,sv_r,vs_r,abn,
            counts.get('V',0),counts.get('S',0),counts.get('N',0),n]


def diagnose_arrhythmia(beat_preds, peak_positions, input_fs=360):
    """
    RR-first arrhythmia diagnosis.
    Rules calibrated on verified MIT-BIH segments.
    """
    from collections import Counter as C
    counts = C(beat_preds)
    total  = max(len(beat_preds), 1)
    pct_v  = counts.get('V', 0) / total
    pct_s  = counts.get('S', 0) / total
    pct_q  = counts.get('Q', 0) / total

    def make(cls, conf=0.0):
        meta = ARRHYTHMIA_META.get(cls, ARRHYTHMIA_META['NSR'])
        return [{'id': cls, 'name': meta['name'], 'severity': meta['severity'],
                 'color': meta['color'], 'description': meta['description'],
                 'advice': meta['advice'], 'confidence': conf}]

    # â”€â”€ RR metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(peak_positions) >= 2:
        rr_ms   = np.diff(peak_positions) / input_fs * 1000
        mean_rr = float(np.mean(rr_ms))
        std_rr  = float(np.std(rr_ms))
        cv_rr   = std_rr / mean_rr if mean_rr > 0 else 0
        hr      = 60000 / mean_rr if mean_rr > 0 else 75
        rr_diff = np.abs(np.diff(rr_ms)) if len(rr_ms) > 1 else np.array([0.0])
        irreg   = float(np.mean(rr_diff / (rr_ms[:-1] + 1e-6) > 0.20)) if len(rr_ms) > 1 else 0
        pnn50   = float(np.mean(rr_diff > 50)) if len(rr_diff) > 0 else 0
    else:
        mean_rr = 800; std_rr = 0; cv_rr = 0; hr = 75; irreg = 0; pnn50 = 0

    # â”€â”€ Bigeminy: strict alternating RR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(peak_positions) >= 8:
        rr_arr = np.diff(peak_positions) / input_fs * 1000
        if len(rr_arr) >= 6:
            ev = rr_arr[::2]; od = rr_arr[1::2]
            if (len(ev) > 2 and len(od) > 2 and
                    abs(np.mean(ev) - np.mean(od)) > 100 and
                    np.std(ev) < 60 and np.std(od) < 60):
                return make('Bigeminy', 88.0)

    # â”€â”€ AFib: very high CV + very high irreg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # AFib: CV=0.665, irreg=0.90, pnn50=0.95
    if cv_rr > 0.50 and irreg > 0.70:
        return make('AFib', 90.0)

    # â”€â”€ VT: high CV + moderate-high irreg (fast or mixed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # VT: CV=0.859, irreg=0.56, HR=98
    if cv_rr > 0.50 and irreg > 0.40:
        return make('VT', 85.0)
    if mean_rr < 400 and cv_rr < 0.15:   # HR>150, regular
        return make('VT', 82.0)

    # â”€â”€ PVC: moderate-high CV + high irreg (compensatory pause) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PVC: CV=0.314, irreg=0.78, pnn50=0.81
    if cv_rr > 0.20 and irreg > 0.50:
        return make('PVC', 82.0)

    # â”€â”€ PAC: moderate CV + moderate irreg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PAC: CV=0.315, irreg=0.37, pnn50=0.80
    if cv_rr > 0.20 and irreg > 0.20 and pnn50 > 0.50:
        return make('PAC', 75.0)

    # â”€â”€ Paced: moderate CV (0.08-0.18) + low irreg + normal HR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Paced: CV=0.134, irreg=0.14, HR=73
    if pct_q > 0.30:
        return make('Paced', 85.0)
    if 0.07 < cv_rr < 0.18 and irreg < 0.20 and 55 < hr < 90:
        return make('Paced', 72.0)

    # â”€â”€ SVT: higher HR + moderate CV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SVT: HR=93, CV=0.120
    if hr > 85 and cv_rr > 0.08:
        return make('SVT', 72.0)

    # â”€â”€ Heart Block: slow + very regular â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Block: HR=57, CV=0.047, irreg=0.00
    if hr < 65 and cv_rr < 0.08 and irreg < 0.05:
        return make('Block', 75.0)

    # â”€â”€ Sinus Bradycardia: slow HR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SBR: HR=50, CV=0.361 (high due to pauses)
    if hr < 60:
        return make('SBR', 72.0)

    # â”€â”€ Sinus Tachycardia: fast + regular â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # STach: HR=110, CV=0.040
    if hr > 100 and cv_rr < 0.10:
        return make('STach', 75.0)

    # â”€â”€ NSR: normal HR + low variability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # NSR: HR=73, CV=0.028
    if 60 <= hr <= 100 and cv_rr < 0.08:
        return make('NSR', 82.0)

    # â”€â”€ Fallback: ML classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(beat_preds) < WIN_BEATS + 1:
        if hr > 100:  return make('STach', 55.0)
        elif hr < 60: return make('SBR', 55.0)
        else:         return make('NSR', 55.0)

    feats = []
    for i in range(len(beat_preds) - WIN_BEATS):
        rr = np.diff(peak_positions[i:i+WIN_BEATS+1]) * (360.0 / input_fs)
        feats.append(extract_window_features(beat_preds[i:i+WIN_BEATS], rr, fs=360))

    X = np.array(feats, dtype=np.float32)
    if arr_gbm is not None:
        mean_probs = (0.6 * arr_gbm.predict_proba(X) + 0.4 * arr_rf.predict_proba(X)).mean(axis=0)
    else:
        mean_probs = arr_clf.predict_proba(X).mean(axis=0)

    diagnoses = []
    for idx in np.argsort(mean_probs)[::-1]:
        conf = float(mean_probs[idx])
        if conf < 0.03 and len(diagnoses) >= 1:
            break
        cls  = arr_encoder.classes_[idx]
        meta = ARRHYTHMIA_META.get(cls, {})
        diagnoses.append({
            'id': cls, 'name': meta.get('name', cls),
            'severity': meta.get('severity', 'Unknown'),
            'color': meta.get('color', '#94a3b8'),
            'description': meta.get('description', ''),
            'advice': meta.get('advice', ''),
            'confidence': round(conf * 100, 1),
        })

    diagnoses.sort(key=lambda x: (SEVERITY_ORDER.get(x['severity'], 0), x['confidence']), reverse=True)
    return diagnoses if diagnoses else make('NSR', 50.0)

BEFORE    = 90
AFTER     = 110
FS        = 360
TARGET_LEN = BEFORE + AFTER


model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Beat model loaded")

# Load Stage-2 arrhythmia classifier (ensemble)
ARR_CLF_PATH = os.path.join(os.path.dirname(__file__), '..', 'arrhythmia_classifier.pkl')
with open(ARR_CLF_PATH, 'rb') as f:
    _arr = pickle.load(f)
# Support both old (single model) and new (ensemble) format
if 'gbm' in _arr:
    arr_gbm     = _arr['gbm']
    arr_rf      = _arr['rf']
    arr_encoder = _arr['encoder']
    arr_clf     = None
    print("[OK] Arrhythmia ensemble classifier loaded")
else:
    arr_clf     = _arr['model']
    arr_gbm     = None
    arr_rf      = None
    arr_encoder = _arr['encoder']
    print("[OK] Arrhythmia classifier loaded")


# â”€â”€â”€ SIGNAL PROCESSING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def infer_fs(time_arr):
    """Estimate sampling rate from time column."""
    if len(time_arr) < 2:
        return FS
    diffs = np.diff(time_arr[:min(100, len(time_arr))])
    dt    = np.median(diffs)
    return round(1.0 / dt) if dt > 0 else FS


def detect_peaks(signal, fs=360):
    """
    Local-adaptive R-peak detector.
    Computes a sliding-window threshold so beats of varying amplitude
    are all detected regardless of global signal level.
    Returns (peak_indices, filtered_signal).
    """
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.ndimage import uniform_filter1d

    sig = signal.astype(np.float64)

    # 1. Bandpass 0.5â€“40 Hz â€” removes baseline wander & HF noise
    nyq  = fs / 2.0
    b, a = butter(2, [0.5 / nyq, min(40.0 / nyq, 0.99)], btype='band')
    filtered = filtfilt(b, a, sig)

    # 2. Auto-flip inverted signals
    if np.abs(np.min(filtered)) > np.abs(np.max(filtered)):
        filtered = -filtered

    # 3. Square the signal to amplify peaks and suppress noise
    squared = filtered ** 2

    # 4. Sliding-window local threshold
    #    Window = 1.5 s, step = 0.15 s â€” adapts to local amplitude changes
    win   = int(1.5 * fs)
    step  = int(0.15 * fs)
    n     = len(squared)
    local_thresh = np.zeros(n)

    for start in range(0, n, step):
        end = min(start + win, n)
        # threshold = 40% of local max in this window
        local_thresh[start:end] = np.maximum(
            local_thresh[start:end],
            0.40 * np.max(squared[start:end])
        )

    # Smooth the threshold curve so it doesn't jump abruptly
    local_thresh = uniform_filter1d(local_thresh, size=win)

    # 5. Find peaks above local threshold
    min_dist = int(0.2 * fs)   # 200 ms minimum between beats
    peaks, _ = find_peaks(squared, distance=min_dist)

    # Keep only peaks that exceed their local threshold
    peaks = np.array([p for p in peaks if squared[p] > local_thresh[p]])

    # 6. Safety fallback â€” if still too few, use simple global threshold
    expected_min = max(3, int(len(signal) / fs / 1.5))
    if len(peaks) < expected_min:
        global_thresh = 0.10 * np.max(squared)
        peaks, _ = find_peaks(squared, height=global_thresh, distance=min_dist)

    # 7. Remove outliers: drop peaks < 10% of median peak height
    if len(peaks) > 0:
        heights = squared[peaks]
        median_h = np.median(heights)
        peaks = peaks[heights >= 0.10 * median_h]

    return peaks.tolist(), filtered.astype(np.float32)


def resample_beat(beat, target_len=TARGET_LEN):
    """Resample beat segment to exactly target_len samples."""
    from scipy.signal import resample as sp_resample
    return sp_resample(beat, target_len).astype(np.float32)


def predict_beats(signal_arr, input_fs=360):
    peaks, filtered = detect_peaks(signal_arr, fs=input_fs)

    # Scale window to input fs, then resample to 200 samples for model
    before = int(BEFORE * input_fs / FS)
    after  = int(AFTER  * input_fs / FS)

    beats, positions = [], []
    for peak in peaks:
        start, end = peak - before, peak + after
        if start < 0 or end > len(filtered):
            continue
        beat = filtered[start:end]
        # Resample to model's expected 200-sample window
        if len(beat) != TARGET_LEN:
            beat = resample_beat(beat, TARGET_LEN)
        # Per-beat z-score normalization â€” identical to training
        std = beat.std()
        if std < 1e-6:          # flat/dead segment â€” skip
            continue
        beat = (beat - beat.mean()) / (std + 1e-8)
        beats.append(beat.astype(np.float32))
        positions.append(int(peak))

    if not beats:
        return [], [], [], []

    X     = np.array(beats, dtype=np.float32)[..., np.newaxis]
    probs = model.predict(X, verbose=0)

    # Post-process: if model is very uncertain (max prob < 40%), use
    # morphology heuristics to correct obvious PVC misclassifications
    preds       = []
    confidences = []
    for i, p in enumerate(probs):
        cls  = CLASSES[np.argmax(p)]
        conf = float(np.max(p))
        # Heuristic: check beat width and amplitude asymmetry
        # PVCs have wider QRS and opposite-polarity T wave
        beat = beats[i]
        above_half = np.sum(beat > 0.5)   # width proxy
        t_area = np.sum(beat[120:180])     # T-wave region
        r_area = np.sum(beat[70:110])      # R-wave region
        t_inverted = (t_area < 0) and (r_area > 0)
        wide_qrs   = above_half > 25

        if conf < 0.50 and wide_qrs and t_inverted:
            cls  = 'V'   # likely PVC
            conf = 0.55
        elif conf < 0.50 and wide_qrs:
            cls  = 'V'
            conf = 0.45

        preds.append(cls)
        confidences.append(conf)

    return positions, preds, confidences, probs.tolist()


def compute_hrv(peaks, fs=360):
    if len(peaks) < 2:
        return {'mean_rr': 0, 'sdnn': 0, 'rmssd': 0, 'hr': 0}
    rr = np.diff(peaks) / fs * 1000
    return {
        'mean_rr': round(float(np.mean(rr)), 1),
        'sdnn':    round(float(np.std(rr)), 1),
        'rmssd':   round(float(np.sqrt(np.mean(np.diff(rr)**2))), 1) if len(rr) > 1 else 0,
        'hr':      round(60000 / float(np.mean(rr)), 1) if np.mean(rr) > 0 else 0
    }


# â”€â”€â”€ CHART HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def make_ecg_plot(time, signal, peaks, preds):
    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=90)
    ax.plot(time, signal, color='#3b82f6', linewidth=0.8, alpha=0.9)
    for i, peak in enumerate(peaks):
        if peak < len(time):
            c = CLASS_COLORS.get(preds[i], '#94a3b8')
            ax.axvline(x=time[peak], color=c, alpha=0.4, linewidth=1)
            ax.scatter(time[peak], signal[peak], color=c, s=30, zorder=5)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude (mV)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_pie_chart(beat_counts):
    labels = [f"{k} â€” {CLASS_LABELS[k]}" for k in beat_counts]
    sizes  = list(beat_counts.values())
    clrs   = [CLASS_COLORS[k] for k in beat_counts]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=90)
    wedges, _, autotexts = ax.pie(sizes, labels=None, colors=clrs,
                                   autopct='%1.1f%%', startangle=90,
                                   pctdistance=0.75,
                                   wedgeprops=dict(width=0.55))
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=1)
    ax.set_title('Beat Distribution', fontsize=11, fontweight='bold', pad=10)
    fig.patch.set_facecolor('#ffffff')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_rr_plot(peaks, fs=360):
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / fs * 1000
    fig, ax = plt.subplots(figsize=(7, 3), dpi=90)
    ax.plot(rr, color='#8b5cf6', linewidth=1.2, marker='o', markersize=3)
    ax.axhline(np.mean(rr), color='#ef4444', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(rr):.0f} ms')
    ax.fill_between(range(len(rr)), rr, alpha=0.15, color='#8b5cf6')
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlabel('Beat Index', fontsize=9)
    ax.set_ylabel('RR Interval (ms)', fontsize=9)
    ax.set_title('RR Interval Tachogram', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_beat_heatmap(signal_arr, peaks, preds, input_fs=360):
    """
    3-panel beat visualization for clinical review:
    Panel 1 â€” Overlaid beat waveforms colored by class (morphology comparison)
    Panel 2 â€” Beat matrix heatmap (raw amplitude per beat, time on X, beat index on Y)
    Panel 3 â€” Anomaly score per beat (Euclidean distance from normal template)
    """
    from scipy.signal import resample as sp_resample
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    BEFORE_S = int(90 * input_fs / 360)
    AFTER_S  = int(110 * input_fs / 360)
    TARGET   = 200

    beat_matrix, valid_preds = [], []
    for i, peak in enumerate(peaks):
        start, end = peak - BEFORE_S, peak + AFTER_S
        if start < 0 or end > len(signal_arr):
            continue
        beat = signal_arr[start:end].astype(np.float64)
        if len(beat) != TARGET:
            beat = sp_resample(beat, TARGET)
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        beat_matrix.append(beat)
        valid_preds.append(preds[i])

    if len(beat_matrix) < 2:
        return None

    beat_matrix = np.array(beat_matrix)   # (n_beats, 200)
    n_beats     = len(beat_matrix)
    x_ms        = np.linspace(-250, 305, TARGET)

    cls_colors  = {'N':'#22c55e','S':'#f59e0b','V':'#ef4444','F':'#8b5cf6','Q':'#94a3b8'}
    cls_alpha   = {'N': 0.25,    'S': 0.85,    'V': 0.90,    'F': 0.85,   'Q': 0.60}

    # Normal template
    n_mask = np.array([p == 'N' for p in valid_preds])
    normal_tpl = beat_matrix[n_mask].mean(axis=0) if n_mask.sum() > 0 else beat_matrix.mean(axis=0)

    # Anomaly score = RMS deviation from normal template per beat
    anomaly = np.sqrt(((beat_matrix - normal_tpl) ** 2).mean(axis=1))

    # â”€â”€ Figure layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig = plt.figure(figsize=(14, 11), dpi=90)
    fig.patch.set_facecolor('#ffffff')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2.5, 3, 1.5],
                            width_ratios=[20, 1],
                            hspace=0.45, wspace=0.04)

    ax1  = fig.add_subplot(gs[0, 0])   # overlaid waveforms
    ax2  = fig.add_subplot(gs[1, 0])   # heatmap
    cax  = fig.add_subplot(gs[1, 1])   # colorbar
    ax3  = fig.add_subplot(gs[2, 0])   # anomaly score

    # â”€â”€ Panel 1: Overlaid beat waveforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Draw normal beats first (background), then abnormal on top
    for cls_order in ['N', 'Q', 'F', 'S', 'V']:
        for i, pred in enumerate(valid_preds):
            if pred != cls_order:
                continue
            lw  = 0.6 if pred == 'N' else 1.4
            ax1.plot(x_ms, beat_matrix[i],
                     color=cls_colors[pred],
                     alpha=cls_alpha[pred],
                     linewidth=lw)

    # Normal template as bold reference line
    ax1.plot(x_ms, normal_tpl, color='#1e293b', linewidth=2.0,
             linestyle='--', label='Normal template', zorder=10)

    # ECG landmark lines
    for xv, lbl in [(-60,'Q'), (0,'R'), (55,'S'), (155,'T')]:
        ax1.axvline(xv, color='#cbd5e1', linewidth=0.8, linestyle=':', zorder=0)
        ax1.text(xv, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 3,
                 lbl, ha='center', va='bottom', fontsize=7.5, color='#94a3b8')

    ax1.set_xlim(x_ms[0], x_ms[-1])
    ax1.set_xlabel('Time relative to R-peak (ms)', fontsize=9)
    ax1.set_ylabel('Amplitude (Ïƒ)', fontsize=9)
    ax1.set_title('Beat Morphology Overlay  â€”  All beats superimposed by class',
                  fontsize=10, fontweight='bold')
    ax1.set_facecolor('#f8fafc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25)

    legend_els = [Line2D([0],[0], color='#1e293b', lw=2, linestyle='--', label='Normal template')]
    legend_els += [Patch(facecolor=cls_colors[k], label=f'{k} â€” {CLASS_LABELS[k]}')
                   for k in ['N','S','V','F','Q'] if k in set(valid_preds)]
    ax1.legend(handles=legend_els, fontsize=7.5, loc='upper right',
               framealpha=0.9, ncol=min(len(legend_els), 4))

    # â”€â”€ Panel 2: Beat matrix heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vmax = float(np.percentile(np.abs(beat_matrix), 97))
    im   = ax2.imshow(beat_matrix, aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax,
                      extent=[x_ms[0], x_ms[-1], n_beats - 0.5, -0.5],
                      interpolation='bilinear')

    # Highlight abnormal beat rows
    for i, pred in enumerate(valid_preds):
        if pred != 'N':
            c = cls_colors.get(pred, '#94a3b8')
            ax2.add_patch(plt.Rectangle(
                (x_ms[0], i - 0.5), x_ms[-1] - x_ms[0], 1,
                linewidth=1.5, edgecolor=c, facecolor='none', zorder=3))

    # Beat class labels on right margin
    for i, pred in enumerate(valid_preds):
        c = cls_colors.get(pred, '#94a3b8')
        ax2.text(x_ms[-1] + 8, i, pred, ha='left', va='center',
                 fontsize=6.5, fontweight='bold', color=c, clip_on=False)

    # ECG landmark lines
    for xv, lbl in [(-60,'Q'), (0,'R'), (55,'S'), (155,'T')]:
        ax2.axvline(xv, color='white', linewidth=0.7, linestyle=':', alpha=0.6)

    ax2.set_xlim(x_ms[0], x_ms[-1])
    ax2.set_xlabel('Time relative to R-peak (ms)', fontsize=9)
    ax2.set_ylabel('Beat Index', fontsize=9)
    ax2.set_title('Beat Amplitude Heatmap  â€”  Red = high, Blue = low amplitude',
                  fontsize=10, fontweight='bold')
    ax2.set_facecolor('#f8fafc')

    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Amplitude (Ïƒ)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # â”€â”€ Panel 3: Anomaly score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bar_colors = [cls_colors.get(p, '#94a3b8') for p in valid_preds]
    bars = ax3.bar(range(n_beats), anomaly, color=bar_colors, width=0.8, zorder=3)

    # Threshold line at mean + 1.5 std of normal beats
    if n_mask.sum() > 1:
        thr = anomaly[n_mask].mean() + 1.5 * anomaly[n_mask].std()
        ax3.axhline(thr, color='#ef4444', linewidth=1.2, linestyle='--',
                    label=f'Anomaly threshold (Î¼+1.5Ïƒ = {thr:.2f})', zorder=4)
        ax3.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

        # Shade above threshold
        ax3.fill_between([-0.5, n_beats - 0.5], thr,
                         max(anomaly.max() * 1.1, thr * 1.2),
                         color='#fef2f2', alpha=0.5, zorder=0)

    ax3.set_xlim(-0.5, n_beats - 0.5)
    ax3.set_xlabel('Beat Index', fontsize=9)
    ax3.set_ylabel('Anomaly Score', fontsize=9)
    ax3.set_title('Per-Beat Anomaly Score  â€”  Distance from normal template',
                  fontsize=10, fontweight='bold')
    ax3.set_facecolor('#f8fafc')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, axis='y', alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# â”€â”€â”€ ROUTES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/analyze')
@login_required
def analyze():
    patients = store.get_patients(session['doctor_id'])
    return render_template('analyze.html', patients=patients)

@app.route('/results')
def results():
    rid  = session.get('result_id')
    data = load_results(rid)
    if not data:
        return render_template('analyze.html', error="No results yet. Please upload an ECG first.")
    patient_id = session.get('last_patient_id')
    patient    = store.get_patient(patient_id) if patient_id else None
    return render_template('results.html', data=data, patient=patient)

@app.route('/report')
def report():
    rid  = session.get('result_id')
    data = load_results(rid)
    if not data:
        return render_template('analyze.html', error="No results yet.")
    return render_template('report.html', data=data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files and request.files['file'].filename:
            f        = request.files['file']
            df       = pd.read_csv(io.StringIO(f.read().decode('utf-8')))
            filename = f.filename
        else:
            df       = pd.read_csv(SAMPLE_CSV)
            filename = 'sample_ecg.csv'

        # Auto-detect signal column
        signal_col = None
        for c in ['signal', 'Signal', 'SIGNAL', 'amplitude', 'Amplitude',
                  'value', 'Value', 'ecg', 'ECG', 'mlii', 'MLII', 'v5', 'V5']:
            if c in df.columns:
                signal_col = c
                break
        if signal_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(f"No numeric signal column found. Columns: {list(df.columns)}")
            signal_col = numeric_cols[-1]

        # Auto-detect time column
        time_col = None
        for c in ['time', 'Time', 'TIME', 't', 'timestamp', 'Timestamp', 'seconds']:
            if c in df.columns:
                time_col = c
                break

        signal   = df[signal_col].dropna().values.astype(np.float32)
        time_arr = df[time_col].values if time_col else np.arange(len(signal)) / FS

        # Infer actual sampling rate from time column
        input_fs = infer_fs(time_arr) if time_col else FS
        time     = time_arr.tolist()

        print(f"  Signal length: {len(signal)}, Detected FS: {input_fs} Hz")

        # Warn if recording is too short for reliable analysis
        recording_secs = len(signal) / input_fs
        min_recommended = 10.0
        short_warning = recording_secs < min_recommended

        positions, preds, confidences, all_probs = predict_beats(signal, input_fs=input_fs)

        beat_counts = {c: preds.count(c) for c in CLASSES if preds.count(c) > 0}
        dominant    = max(beat_counts, key=beat_counts.get) if beat_counts else 'N'
        hrv         = compute_hrv(positions, fs=input_fs)
        duration    = round(float(time[-1]) - float(time[0]), 2) if len(time) > 1 else 0

        ecg_img  = make_ecg_plot(time, signal.tolist(), positions, preds)
        pie_img  = make_pie_chart(beat_counts) if beat_counts else None
        rr_img   = make_rr_plot(positions, fs=input_fs)
        heat_img = make_beat_heatmap(signal, positions, preds, input_fs=input_fs)

        risk_score = sum(1 for p in preds if p == 'V') / max(len(preds), 1) * 100

        # Clinical arrhythmia diagnosis from ML classifier
        diagnoses = diagnose_arrhythmia(preds, positions, input_fs=input_fs)
        primary   = diagnoses[0] if diagnoses else {}

        # Risk level: driven by primary diagnosis severity (ML model output)
        # This ensures the risk banner matches the diagnosis banner
        _sev_map = {'High': 'High', 'Moderate': 'Moderate',
                    'Mild': 'Low', 'Normal': 'Low', 'Unknown': 'Low'}
        if primary:
            risk_level = _sev_map.get(primary.get('severity', 'Normal'), 'Low')
        else:
            risk_level = 'High' if risk_score > 20 else ('Moderate' if risk_score > 5 else 'Low')

        results = {
            'filename':       filename,
            'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration':       duration,
            'input_fs':       input_fs,
            'total_beats':    len(preds),
            'dominant':       dominant,
            'dominant_label': CLASS_LABELS.get(dominant, ''),
            'beat_counts':    beat_counts,
            'risk_level':     risk_level,
            'risk_score':     round(risk_score, 1),
            'short_warning':  short_warning,
            'recording_secs': round(recording_secs, 1),
            'hrv':            hrv,
            'diagnoses':      diagnoses,
            'primary':        primary,
            'peaks':          positions,
            'predictions':    preds,
            'confidences':    [round(c * 100, 1) for c in confidences],
            'all_probs':      [[round(p * 100, 1) for p in row] for row in all_probs],
            'ecg_img':        ecg_img,
            'pie_img':        pie_img,
            'rr_img':         rr_img,
            'heat_img':       heat_img,
            'time':           time,
            'signal':         signal.tolist(),
            'classes':        CLASSES,
            'class_labels':   CLASS_LABELS,
            'class_colors':   CLASS_COLORS,
        }

        # Recovery time prediction (non-blocking)
        try:
            recovery = rec_model.predict(
                hrv=hrv, beat_counts=beat_counts,
                total_beats=len(preds), risk_score=risk_score,
                diagnoses=diagnoses, risk_level=risk_level,
            )
            results['recovery'] = recovery
        except Exception as _re:
            print(f"[WARNING] Recovery prediction failed: {_re}")
            results['recovery'] = None

        rid = save_results(results)
        session['result_id'] = rid
        session['last_patient_id'] = request.form.get('patient_id', '').strip() or None

        # Save to patient profile if one was selected
        patient_id = request.form.get('patient_id', '').strip()
        if patient_id:
            store.add_ecg_record(
                patient_id     = patient_id,
                filename       = filename,
                total_beats    = len(preds),
                dominant_rhythm= dominant,
                risk_level     = risk_level,
                risk_score     = risk_score,
                diagnoses      = diagnoses,
                hrv            = hrv,
                beat_counts    = beat_counts,
                recovery_days  = results['recovery']['days'] if results.get('recovery') is not None else None,
            )

        return jsonify({'status': 'ok'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/download_report')
def download_report():
    rid  = session.get('result_id')
    data = load_results(rid)
    if not data:
        return "No results", 404

    # Patient details from query params
    pt = {
        'name':    request.args.get('pt_name', '').strip(),
        'phone':   request.args.get('pt_phone', '').strip(),
        'address': request.args.get('pt_address', '').strip(),
        'history': request.args.get('pt_history', '').strip(),
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(tmp.name, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    normal_style = styles['Normal']
    h2_style = ParagraphStyle('h2', fontSize=12, fontName='Helvetica-Bold',
                               textColor=colors.HexColor('#1e293b'), spaceAfter=8, spaceBefore=12)
    muted = ParagraphStyle('muted', fontSize=8, textColor=colors.HexColor('#94a3b8'))

    # â”€â”€ Clinic Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    header_data = [[
        Paragraph('<b><font size=16 color="#1e40af">CardioAI</font></b><br/>'
                  '<font size=8 color="#94a3b8">ECG Arrhythmia Detection System Â· MIT-BIH Â· AAMI Standard</font>',
                  normal_style),
        Paragraph(f'<font size=8 color="#94a3b8">REPORT GENERATED</font><br/>'
                  f'<b>{data["timestamp"]}</b><br/>'
                  f'<font size=8 color="#94a3b8">File: {data["filename"]}</font>',
                  ParagraphStyle('right', alignment=2, fontSize=9))
    ]]
    ht = Table(header_data, colWidths=[10*cm, 6*cm])
    ht.setStyle(TableStyle([
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW',   (0,0), (-1,0),  2, colors.HexColor('#1e40af')),
        ('BOTTOMPADDING',(0,0),(-1,0),  12),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Patient Information Block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pt['name'] or pt['phone'] or pt['address']:
        story.append(Paragraph("PATIENT INFORMATION",
                                ParagraphStyle('sec', fontSize=8, fontName='Helvetica-Bold',
                                               textColor=colors.HexColor('#94a3b8'),
                                               spaceBefore=4, spaceAfter=6)))
        pt_rows = [
            ['Full Name',  pt['name']  or 'â€”', 'Mobile', pt['phone']  or 'â€”'],
            ['Address',    pt['address'] or 'â€”', '', ''],
        ]
        if pt['history']:
            pt_rows.append(['Medical History', pt['history'], '', ''])

        pt_table = Table(pt_rows, colWidths=[3*cm, 7*cm, 2.5*cm, 3.5*cm])
        pt_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('FONTNAME',    (0,0), (0,-1),  'Helvetica-Bold'),
            ('FONTNAME',    (2,0), (2,-1),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 9),
            ('TEXTCOLOR',   (0,0), (0,-1),  colors.HexColor('#94a3b8')),
            ('TEXTCOLOR',   (2,0), (2,-1),  colors.HexColor('#94a3b8')),
            ('TEXTCOLOR',   (1,0), (1,-1),  colors.HexColor('#1e293b')),
            ('TEXTCOLOR',   (3,0), (3,-1),  colors.HexColor('#1e293b')),
            ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING',     (0,0), (-1,-1), 8),
            ('SPAN',        (1,1), (3,1)),   # address spans
        ]))
        if pt['history']:
            pt_table.setStyle(TableStyle([('SPAN', (1,2), (3,2))]))
        story.append(pt_table)
        story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("ECG Arrhythmia Analysis Report",
                            ParagraphStyle('title', fontSize=14, fontName='Helvetica-Bold',
                                           textColor=colors.HexColor('#1e293b'), spaceAfter=12)))

    # â”€â”€ Arrhythmia Diagnoses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if data.get('diagnoses'):
        story.append(Paragraph("Arrhythmia Diagnosis", h2_style))
        for d in data['diagnoses']:
            sev_colors = {'High':'#fef2f2','Moderate':'#fffbeb','Mild':'#eff6ff','Normal':'#f0fdf4'}
            bg = sev_colors.get(d['severity'], '#f8fafc')
            diag_rows = [
                [Paragraph(f'<b>{d["name"]}</b>', normal_style),
                 Paragraph(f'<b>{d["severity"]}</b>  {d.get("confidence","")}{"%" if d.get("confidence") else ""}',
                           ParagraphStyle('sev', fontSize=9, textColor=colors.HexColor(d['color'])))],
                [Paragraph(d['description'], ParagraphStyle('desc', fontSize=9, textColor=colors.HexColor('#475569'))), ''],
                [Paragraph(f'<b>Advice:</b> {d["advice"]}',
                           ParagraphStyle('adv', fontSize=9, textColor=colors.HexColor('#334155'))), ''],
            ]
            dt = Table(diag_rows, colWidths=[12*cm, 4*cm])
            dt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg)),
                ('LINEAFTER',  (0,0), (0,-1),  2, colors.HexColor(d['color'])),
                ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
                ('PADDING',    (0,0), (-1,-1), 8),
                ('SPAN',       (0,1), (1,1)),
                ('SPAN',       (0,2), (1,2)),
            ]))
            story.append(dt)
            story.append(Spacer(1, 0.25*cm))
        story.append(Spacer(1, 0.3*cm))

    # â”€â”€ Clinical Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph("Clinical Summary", h2_style))
    summary_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['Total Beats',      str(data['total_beats']),   'Heart Rate',    f"{data['hrv'].get('hr','N/A')} bpm"],
        ['Duration',         f"{data['duration']} s",    'Mean RR',       f"{data['hrv'].get('mean_rr','N/A')} ms"],
        ['Dominant Rhythm',  f"{data['dominant']} â€” {data['dominant_label']}", 'SDNN', f"{data['hrv'].get('sdnn','N/A')} ms"],
        ['Risk Level',       data['risk_level'],          'RMSSD',         f"{data['hrv'].get('rmssd','N/A')} ms"],
        ['Sampling Rate',    f"{data.get('input_fs',360)} Hz", '', ''],
    ]
    st = Table(summary_data, colWidths=[4*cm, 4.5*cm, 4*cm, 3.5*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (2,1), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1), (0,-1), colors.HexColor('#64748b')),
        ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor('#64748b')),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',       (0,0), (-1,-1), 8),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Beat Classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph("Beat Classification Summary", h2_style))
    beat_data = [['Class', 'Label', 'Count', 'Percentage', 'Risk']]
    total = data['total_beats']
    for cls, cnt in data['beat_counts'].items():
        pct  = round(cnt / total * 100, 1)
        risk = {'N':'Low','S':'Moderate','V':'High','F':'Moderate','Q':'Low'}.get(cls,'')
        beat_data.append([cls, CLASS_LABELS.get(cls,''), str(cnt), f"{pct}%", risk])
    bt = Table(beat_data, colWidths=[2*cm, 6.5*cm, 2.5*cm, 3*cm, 2*cm])
    bt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f1f5f9'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',       (0,0), (-1,-1), 7),
    ]))
    story.append(bt)
    story.append(Spacer(1, 0.4*cm))

    # â”€â”€ ECG Image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if data.get('ecg_img'):
        story.append(Paragraph("ECG Signal with Beat Annotations", h2_style))
        img = Image(io.BytesIO(base64.b64decode(data['ecg_img'])), width=15*cm, height=3.5*cm)
        story.append(img)
        story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Pie + RR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    row_imgs = []
    if data.get('pie_img'):
        row_imgs.append(Image(io.BytesIO(base64.b64decode(data['pie_img'])), width=7*cm, height=5.5*cm))
    if data.get('rr_img'):
        row_imgs.append(Image(io.BytesIO(base64.b64decode(data['rr_img'])), width=7*cm, height=5.5*cm))
    if row_imgs:
        story.append(Table([row_imgs], colWidths=[7.5*cm]*len(row_imgs)))
        story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Beat Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if data.get('heat_img'):
        story.append(Paragraph("Beat Morphology Heatmap", h2_style))
        story.append(Paragraph(
            "Each row represents one beat. Color shows deviation from the normal template "
            "(red = above normal, blue = below normal). Abnormal beats are marked with dashed lines.",
            ParagraphStyle('sub', fontSize=8, textColor=colors.HexColor('#64748b'), spaceAfter=6)))
        hmap = Image(io.BytesIO(base64.b64decode(data['heat_img'])), width=15*cm, height=7*cm)
        story.append(hmap)
        story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Disclaimer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI model for research and educational purposes only. "
        "It is not a substitute for professional medical diagnosis or clinical evaluation. "
        "Always consult a qualified cardiologist for medical decisions.",
        ParagraphStyle('disc', fontSize=8, textColor=colors.HexColor('#94a3b8'),
                       borderPadding=8, backColor=colors.HexColor('#f8fafc'))))

    doc.build(story)
    return send_file(tmp.name, as_attachment=True,
                     download_name=f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf')
import auth_routes; auth_routes.register(app)


# ── Human-in-the-Loop Correction API ─────────────────────────────────────────
_corrections = []  # in-memory log, session lifetime only

@app.route('/api/correct', methods=['POST'])
def api_correct():
    if 'doctor_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated.'}), 401
    import re as _re
    body     = request.get_json(silent=True) or {}
    ai_pred  = _re.sub(r'<[^>]+>', '', str(body.get('ai_prediction',     '')).strip())[:120]
    doc_corr = _re.sub(r'<[^>]+>', '', str(body.get('doctor_correction', '')).strip())[:80]

    if not doc_corr:
        return jsonify({'status': 'error', 'message': 'Correction cannot be empty.'}), 400

    entry = {
        'ai_prediction':     ai_pred,
        'doctor_correction': doc_corr,
        'doctor_id':         session.get('doctor_id', ''),
        'result_id':         session.get('result_id', ''),
        'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    _corrections.append(entry)

    # Persist correction into the saved result JSON
    rid = session.get('result_id')
    if rid:
        try:
            saved = load_results(rid)
            if saved:
                saved['doctor_correction']    = doc_corr
                saved['doctor_correction_ts'] = entry['timestamp']
                path = os.path.join(RESULTS_DIR, f'{rid}.json')
                with open(path, 'w') as _f:
                    json.dump(saved, _f)
        except Exception:
            pass

    return jsonify({
        'status':            'ok',
        'ai_prediction':     ai_pred,
        'doctor_correction': doc_corr,
        'timestamp':         entry['timestamp'],
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

