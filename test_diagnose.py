"""Direct test of diagnose_arrhythmia on all 10 sample files."""
import os, sys, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

SAMPLES_DIR = 'ecg-dashboard/input_samples'
EXPECTED = {
    'NSR_Normal_Sinus_Rhythm':    'NSR',
    'PVC_Premature_Ventricular':  'PVC',
    'VT_Ventricular_Tachycardia': 'VT',
    'AFib_Atrial_Fibrillation':   'AFib',
    'PAC_Premature_Atrial':       'PAC',
    'SVT_Supraventricular_Tachy': 'SVT',
    'Paced_Rhythm':               'Paced',
    'Block_Heart_Block':          'Block',
    'SBR_Sinus_Bradycardia':      'SBR',
    'STach_Sinus_Tachycardia':    'STach',
}

def detect_peaks_local(signal, fs=360):
    sig = signal.astype(np.float64)
    nyq = fs / 2.0
    b, a = butter(2, [0.5/nyq, min(40/nyq, 0.99)], btype='band')
    filt = filtfilt(b, a, sig)
    if abs(filt.min()) > abs(filt.max()): filt = -filt
    sq = filt ** 2
    win = int(1.5*fs); step = int(0.15*fs); n = len(sq)
    lt = np.zeros(n)
    for s in range(0, n, step):
        e = min(s+win, n); lt[s:e] = np.maximum(lt[s:e], 0.40*np.max(sq[s:e]))
    lt = uniform_filter1d(lt, size=win)
    peaks, _ = find_peaks(sq, distance=int(0.2*fs))
    peaks = np.array([p for p in peaks if sq[p] > lt[p]])
    if len(peaks) < 3:
        peaks, _ = find_peaks(sq, height=0.10*np.max(sq), distance=int(0.2*fs))
    return peaks.tolist()

def diagnose_rr(peaks, input_fs, pct_v=0, pct_s=0, pct_q=0):
    """Replicate the RR-first diagnosis logic from app.py"""
    if len(peaks) < 2:
        return 'NSR', 50.0
    rr_ms   = np.diff(peaks) / input_fs * 1000
    mean_rr = float(np.mean(rr_ms))
    std_rr  = float(np.std(rr_ms))
    cv_rr   = std_rr / mean_rr if mean_rr > 0 else 0
    hr      = 60000 / mean_rr if mean_rr > 0 else 75
    rr_diff = np.abs(np.diff(rr_ms)) if len(rr_ms) > 1 else np.array([0.0])
    irreg   = float(np.mean(rr_diff / (rr_ms[:-1]+1e-6) > 0.20)) if len(rr_ms) > 1 else 0
    pnn50   = float(np.mean(rr_diff > 50)) if len(rr_diff) > 0 else 0

    if cv_rr > 0.50 and irreg > 0.70:                   return 'AFib',    90.0
    if cv_rr > 0.50 and irreg > 0.40:                   return 'VT',      85.0
    if cv_rr > 0.20 and irreg > 0.50:                   return 'PVC',     82.0
    if cv_rr > 0.20 and irreg > 0.20 and pnn50 > 0.50:  return 'PAC',     75.0
    if len(peaks) >= 8:
        rr_arr = np.diff(peaks) / input_fs * 1000
        if len(rr_arr) >= 6:
            ev = rr_arr[::2]; od = rr_arr[1::2]
            if (len(ev)>2 and len(od)>2 and
                    abs(np.mean(ev)-np.mean(od))>100 and
                    np.std(ev)<60 and np.std(od)<60):
                return 'Bigeminy', 85.0
    if cv_rr > 0.15 and pct_v > 0.20:                   return 'VT',      75.0
    if pct_q > 0.30:                                     return 'Paced',   85.0
    if 0.07 < cv_rr < 0.18 and irreg < 0.20 and 55 < hr < 90: return 'Paced', 72.0
    if hr > 85 and cv_rr > 0.08:                         return 'SVT',     72.0
    if hr < 65 and cv_rr < 0.08 and irreg < 0.05:       return 'Block',   75.0
    if hr < 60:                                          return 'SBR',     72.0
    if hr > 100 and cv_rr < 0.10:                        return 'STach',   72.0
    if 60 <= hr <= 100 and cv_rr < 0.10:                 return 'NSR',     80.0
    return 'NSR', 50.0

print(f"\n{'File':<42} {'Expected':>8} {'Got':>10} {'HR':>6} {'CV':>6} {'irr':>5} {'Match'}")
print('-' * 85)

correct = 0
for fname in sorted(os.listdir(SAMPLES_DIR)):
    if not fname.endswith('.csv'): continue
    key      = fname.replace('.csv', '')
    expected = EXPECTED.get(key, '?')

    df  = pd.read_csv(os.path.join(SAMPLES_DIR, fname))
    sig = df['signal'].values.astype(np.float32)
    dt  = np.median(np.diff(df['time'].values[:100]))
    fs  = round(1.0 / dt)

    peaks = detect_peaks_local(sig, fs=fs)
    got, conf = diagnose_rr(peaks, fs)

    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs * 1000
        hr = 60000 / np.mean(rr); cv = np.std(rr)/np.mean(rr)
        rd = np.abs(np.diff(rr)) if len(rr)>1 else np.array([0.0])
        irr = float(np.mean(rd/(rr[:-1]+1e-6)>0.20)) if len(rr)>1 else 0
    else:
        hr=0; cv=0; irr=0

    match = 'OK' if got == expected else 'WRONG'
    if got == expected: correct += 1
    print(f"  {key:<42} {expected:>8} {got:>10} {hr:6.1f} {cv:6.3f} {irr:5.2f}  {match}")

print(f"\nResult: {correct}/10 correct ({correct*10}%)")
