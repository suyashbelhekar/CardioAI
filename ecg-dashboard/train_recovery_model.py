"""
Train a Random Forest Regressor to predict recovery time (days)
from ECG analysis features.

Run once:  python ecg-dashboard/train_recovery_model.py
Output:    ecg-dashboard/recovery_model.pkl
"""

import os, pickle, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Feature order (must match predict_recovery_time) ─────────────────────────
FEATURES = [
    'avg_hr', 'sdnn', 'rmssd', 'mean_rr',
    'pct_N', 'pct_S', 'pct_V', 'pct_F', 'pct_Q',
    'total_beats', 'risk_score',
    # arrhythmia type one-hot
    'is_NSR', 'is_AFib', 'is_VT', 'is_PVC', 'is_SVT',
    'is_SBR', 'is_STach', 'is_PAC', 'is_Block', 'is_Paced', 'is_Bigeminy',
]

# ── Clinical recovery ranges (days) ──────────────────────────────────────────
RECOVERY_RANGES = {
    'VT':       (10, 20),
    'AFib':     (7,  15),
    'Block':    (7,  14),
    'SVT':      (5,  10),
    'Bigeminy': (4,  8),
    'PVC':      (2,  5),
    'PAC':      (1,  4),
    'STach':    (1,  3),
    'SBR':      (1,  3),
    'Paced':    (0,  2),
    'NSR':      (0,  1),
}

ALL_TYPES = list(RECOVERY_RANGES.keys())
rng = np.random.default_rng(42)


def _sample_row(arrhythmia_type):
    lo, hi = RECOVERY_RANGES[arrhythmia_type]
    recovery = rng.uniform(lo, hi)

    # Simulate plausible ECG features per arrhythmia type
    if arrhythmia_type == 'VT':
        hr      = rng.uniform(120, 200)
        sdnn    = rng.uniform(5,  25)
        rmssd   = rng.uniform(5,  20)
        pct_V   = rng.uniform(0.4, 0.9)
        pct_N   = 1 - pct_V
        risk_sc = pct_V * 100
    elif arrhythmia_type == 'AFib':
        hr      = rng.uniform(80, 160)
        sdnn    = rng.uniform(60, 150)
        rmssd   = rng.uniform(50, 130)
        pct_V   = rng.uniform(0.0, 0.1)
        pct_N   = rng.uniform(0.5, 0.9)
        risk_sc = rng.uniform(0, 10)
    elif arrhythmia_type == 'PVC':
        hr      = rng.uniform(60, 100)
        sdnn    = rng.uniform(20, 60)
        rmssd   = rng.uniform(15, 50)
        pct_V   = rng.uniform(0.05, 0.3)
        pct_N   = 1 - pct_V
        risk_sc = pct_V * 100
    elif arrhythmia_type == 'SVT':
        hr      = rng.uniform(100, 180)
        sdnn    = rng.uniform(10, 40)
        rmssd   = rng.uniform(8,  35)
        pct_V   = rng.uniform(0.0, 0.05)
        pct_N   = rng.uniform(0.6, 0.95)
        risk_sc = rng.uniform(0, 5)
    elif arrhythmia_type == 'NSR':
        hr      = rng.uniform(55, 95)
        sdnn    = rng.uniform(30, 80)
        rmssd   = rng.uniform(25, 70)
        pct_V   = rng.uniform(0.0, 0.02)
        pct_N   = rng.uniform(0.95, 1.0)
        risk_sc = rng.uniform(0, 2)
    else:
        hr      = rng.uniform(50, 130)
        sdnn    = rng.uniform(15, 70)
        rmssd   = rng.uniform(10, 60)
        pct_V   = rng.uniform(0.0, 0.15)
        pct_N   = rng.uniform(0.7, 1.0)
        risk_sc = pct_V * 100

    pct_S = rng.uniform(0, max(0, 1 - pct_N - pct_V - 0.02))
    pct_F = rng.uniform(0, 0.02)
    pct_Q = max(0, 1 - pct_N - pct_V - pct_S - pct_F)
    mean_rr = 60000 / max(hr, 1)
    total   = int(rng.uniform(50, 500))

    row = [
        hr, sdnn, rmssd, mean_rr,
        pct_N, pct_S, pct_V, pct_F, pct_Q,
        total, risk_sc,
    ]
    # one-hot arrhythmia type
    for t in ALL_TYPES:
        row.append(1.0 if t == arrhythmia_type else 0.0)

    return row, recovery


def generate_dataset(n=4000):
    X, y = [], []
    per_type = n // len(ALL_TYPES)
    for atype in ALL_TYPES:
        for _ in range(per_type):
            row, rec = _sample_row(atype)
            X.append(row)
            y.append(rec)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train():
    print("Generating synthetic training data...")
    X, y = generate_dataset(4000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"MAE on test set: {mae:.2f} days")

    out_path = os.path.join(os.path.dirname(__file__), 'recovery_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES, 'types': ALL_TYPES}, f)
    print(f"Model saved → {out_path}")


if __name__ == '__main__':
    train()
