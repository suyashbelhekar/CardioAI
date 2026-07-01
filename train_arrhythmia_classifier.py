"""
Stage 2: Arrhythmia-level classifier — improved version
Uses richer features + XGBoost for better accuracy across all 11 classes.
"""

import wfdb
import numpy as np
import os
import pickle
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

DATA_PATH = 'Dataset/mit-bih-arrhythmia-database-1.0.0/'
FS        = 360
WIN_BEATS = 15   # wider window for better rhythm context

AAMI_MAP = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V','!':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q',
}

RHYTHM_MAP = {
    '(N':'NSR','(NSR':'NSR','(SBR':'SBR','(SVTA':'SVT','(SVT':'SVT',
    '(VT':'VT','(VFL':'VT','(AFIB':'AFib','(AFL':'AFib',
    '(B':'Block','(BII':'Block','(P':'Paced','(PACE':'Paced',
    '(T':'SVT','(AB':'SVT','(IVR':'VT','(NOD':'NSR','(PREX':'NSR',
}

# ─── RICH FEATURE EXTRACTION ──────────────────────────────────────────────────
def extract_features(beat_seq, rr_seq, fs=360):
    from collections import Counter as C
    n      = len(beat_seq)
    counts = C(beat_seq)
    total  = max(n, 1)

    # Beat type ratios
    pct_N = counts.get('N',0)/total
    pct_S = counts.get('S',0)/total
    pct_V = counts.get('V',0)/total
    pct_F = counts.get('F',0)/total
    pct_Q = counts.get('Q',0)/total

    # RR interval statistics
    rr     = np.array(rr_seq, dtype=np.float64)
    rr_ms  = rr / fs * 1000
    mean_rr = float(np.mean(rr_ms))
    std_rr  = float(np.std(rr_ms))
    rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms)>1 else 0
    hr      = 60000/mean_rr if mean_rr>0 else 75
    cv_rr   = std_rr/mean_rr if mean_rr>0 else 0
    min_rr  = float(np.min(rr_ms))
    max_rr  = float(np.max(rr_ms))
    rr_range = max_rr - min_rr

    # RR irregularity metrics
    rr_diff = np.abs(np.diff(rr_ms)) if len(rr_ms)>1 else np.array([0])
    pnn50   = float(np.mean(rr_diff > 50)) if len(rr_diff)>0 else 0
    pnn20   = float(np.mean(rr_diff > 20)) if len(rr_diff)>0 else 0
    rr_irreg = float(np.mean(rr_diff/(rr_ms[:-1]+1e-6) > 0.20)) if len(rr_ms)>1 else 0

    # Rhythm pattern features
    alt_score = sum(1 for i in range(1,n-1)
                    if beat_seq[i]!=beat_seq[i-1] and beat_seq[i]==beat_seq[i+1]) / max(n-2,1)
    tri_score = sum(1 for i in range(2,n)
                    if beat_seq[i]==beat_seq[i-2] and beat_seq[i]!=beat_seq[i-1]) / max(n-2,1)

    # Consecutive run lengths
    max_cv=cur=0
    for b in beat_seq: cur=cur+1 if b=='V' else 0; max_cv=max(max_cv,cur)
    max_cs=cur=0
    for b in beat_seq: cur=cur+1 if b=='S' else 0; max_cs=max(max_cs,cur)
    max_cn=cur=0
    for b in beat_seq: cur=cur+1 if b=='N' else 0; max_cn=max(max_cn,cur)

    # Transition counts (how often beat type changes)
    transitions = sum(1 for i in range(1,n) if beat_seq[i]!=beat_seq[i-1])
    trans_rate  = transitions / max(n-1, 1)

    # RR entropy (irregularity measure for AFib detection)
    if len(rr_ms) > 3:
        rr_norm = (rr_ms - rr_ms.mean()) / (rr_ms.std() + 1e-8)
        hist, _ = np.histogram(rr_norm, bins=5, density=True)
        hist    = hist[hist > 0]
        rr_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    else:
        rr_entropy = 0.0

    # Ratio features
    sv_ratio = pct_S / (pct_V + 1e-6)
    vs_ratio = pct_V / (pct_S + 1e-6)
    abnormal_ratio = (counts.get('S',0) + counts.get('V',0) + counts.get('F',0)) / total

    return [
        pct_N, pct_S, pct_V, pct_F, pct_Q,
        mean_rr, std_rr, rmssd, hr, cv_rr,
        min_rr, max_rr, rr_range,
        pnn50, pnn20, rr_irreg,
        alt_score, tri_score, rr_entropy,
        max_cv/WIN_BEATS, max_cs/WIN_BEATS, max_cn/WIN_BEATS,
        trans_rate, sv_ratio, vs_ratio, abnormal_ratio,
        counts.get('V',0), counts.get('S',0), counts.get('N',0),
        n,  # window size
    ]

N_FEATURES = 30

# ─── DATA EXTRACTION ──────────────────────────────────────────────────────────
def extract_record_windows(rec):
    try:
        signal, _ = wfdb.rdsamp(DATA_PATH + rec)
        ann = wfdb.rdann(DATA_PATH + rec, 'atr')
    except Exception as e:
        print(f"  Skipping {rec}: {e}")
        return [], []

    # Build rhythm label array
    rhythm_labels = ['NSR'] * len(signal)
    current = 'NSR'
    for i, sym in enumerate(ann.symbol):
        if sym == '+' and i < len(ann.aux_note):
            note = ann.aux_note[i].strip().rstrip('\x00')
            current = RHYTHM_MAP.get(note, current)
        rhythm_labels[ann.sample[i]] = current

    # Forward-fill
    cur = 'NSR'
    for i in range(len(rhythm_labels)):
        if rhythm_labels[i] != 'NSR' or i == 0:
            cur = rhythm_labels[i]
        rhythm_labels[i] = cur

    # Get beat annotations
    beats = [(s, AAMI_MAP.get(sym)) for s, sym in zip(ann.sample, ann.symbol)
             if AAMI_MAP.get(sym)]

    if len(beats) < WIN_BEATS + 1:
        return [], []

    X, y = [], []
    for i in range(len(beats) - WIN_BEATS):
        window_beats   = [b[1] for b in beats[i:i+WIN_BEATS]]
        window_samples = [b[0] for b in beats[i:i+WIN_BEATS+1]]
        rr_intervals   = np.diff(window_samples)

        mid = beats[i + WIN_BEATS//2][0]
        rhythm = rhythm_labels[min(mid, len(rhythm_labels)-1)]

        # Override with beat-pattern rules when rhythm annotation is missing
        v_r = window_beats.count('V') / WIN_BEATS
        s_r = window_beats.count('S') / WIN_BEATS
        n_r = window_beats.count('N') / WIN_BEATS
        rr_ms = np.array(rr_intervals) / FS * 1000
        hr    = 60000 / np.mean(rr_ms) if np.mean(rr_ms) > 0 else 75
        cv    = np.std(rr_ms) / np.mean(rr_ms) if np.mean(rr_ms) > 0 else 0

        if rhythm == 'NSR':
            if v_r >= 0.5:                          rhythm = 'VT'
            elif s_r >= 0.4 and hr > 100:           rhythm = 'SVT'
            elif cv > 0.15 and s_r > 0.15:          rhythm = 'AFib'
            elif v_r >= 0.3:                        rhythm = 'PVC'
            elif s_r >= 0.15:                       rhythm = 'PAC'
            elif v_r > 0:
                alt = sum(1 for j in range(1,WIN_BEATS-1)
                          if window_beats[j]!=window_beats[j-1]
                          and window_beats[j]==window_beats[j+1])
                rhythm = 'Bigeminy' if alt >= WIN_BEATS//3 else 'PVC'
            elif hr < 55 and n_r > 0.85:            rhythm = 'SBR'
            elif hr > 105 and n_r > 0.85:           rhythm = 'STach'

        feats = extract_features(window_beats, rr_intervals)
        X.append(feats)
        y.append(rhythm)

    return X, y


print("Extracting rhythm windows from all records...")
all_records = sorted([f.replace('.hea','') for f in os.listdir(DATA_PATH) if f.endswith('.hea')])

X_all, y_all = [], []
for rec in all_records:
    Xr, yr = extract_record_windows(rec)
    X_all.extend(Xr)
    y_all.extend(yr)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all)

print(f"\nTotal windows: {len(X_all)}")
print("Class distribution:")
for cls, cnt in sorted(Counter(y_all).items(), key=lambda x: -x[1]):
    print(f"  {cls:12} : {cnt:6}  ({cnt/len(y_all)*100:.1f}%)")

# ─── BALANCE ──────────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y_all)
print(f"\nClasses: {list(le.classes_)}")

counts  = Counter(y_all)
min_cnt = max(min(counts.values()), 100)
target  = min(max(counts.values()), min_cnt * 10)

X_bal, y_bal = [], []
for cls in le.classes_:
    idx   = np.where(y_all == cls)[0]
    Xc    = X_all[idx]
    yc    = y_enc[idx]
    n     = len(idx)
    if n < target:
        Xc, yc = resample(Xc, yc, replace=True, n_samples=target, random_state=42)
    elif n > target:
        Xc, yc = resample(Xc, yc, replace=False, n_samples=target, random_state=42)
    X_bal.append(Xc);  y_bal.append(yc)

X_bal = np.vstack(X_bal)
y_bal = np.concatenate(y_bal)
print(f"\nBalanced: {X_bal.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.20,
                                            random_state=42, stratify=y_bal)

# ─── ENSEMBLE MODEL ───────────────────────────────────────────────────────────
print("\nTraining ensemble (GBM + RandomForest)...")

gbm = GradientBoostingClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=5, random_state=42, verbose=1
)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, random_state=42
)

# Train both
gbm.fit(X_tr, y_tr)
rf.fit(X_tr, y_tr)

# Soft voting ensemble
gbm_probs = gbm.predict_proba(X_te)
rf_probs  = rf.predict_proba(X_te)
ensemble_probs = 0.6 * gbm_probs + 0.4 * rf_probs
y_pred_ens = np.argmax(ensemble_probs, axis=1)

print(f"\nGBM alone:      {accuracy_score(y_te, gbm.predict(X_te))*100:.1f}%")
print(f"RF alone:       {accuracy_score(y_te, rf.predict(X_te))*100:.1f}%")
print(f"Ensemble:       {accuracy_score(y_te, y_pred_ens)*100:.1f}%")

print("\nClassification Report (Ensemble):")
print(classification_report(y_te, y_pred_ens, target_names=le.classes_))

# Save
with open('arrhythmia_classifier.pkl', 'wb') as f:
    pickle.dump({'gbm': gbm, 'rf': rf, 'encoder': le,
                 'n_features': N_FEATURES, 'win_beats': WIN_BEATS}, f)

print("\n[OK] Saved arrhythmia_classifier.pkl")
print(f"     Classes: {list(le.classes_)}")
