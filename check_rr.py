import os, numpy as np, pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

def get_rr(fname):
    df  = pd.read_csv(fname)
    sig = df['signal'].values.astype(np.float32)
    dt  = np.median(np.diff(df['time'].values[:100]))
    fs  = round(1.0 / dt)
    s64 = sig.astype(np.float64)
    nyq = fs / 2
    b, a = butter(2, [0.5/nyq, min(40/nyq, 0.99)], btype='band')
    filt = filtfilt(b, a, s64)
    if abs(filt.min()) > abs(filt.max()): filt = -filt
    sq   = filt ** 2
    win  = int(1.5 * fs); step = int(0.15 * fs); n = len(sq)
    lt   = np.zeros(n)
    for s in range(0, n, step):
        e = min(s + win, n)
        lt[s:e] = np.maximum(lt[s:e], 0.40 * np.max(sq[s:e]))
    lt = uniform_filter1d(lt, size=win)
    peaks, _ = find_peaks(sq, distance=int(0.2 * fs))
    peaks = np.array([p for p in peaks if sq[p] > lt[p]])
    if len(peaks) < 3:
        peaks, _ = find_peaks(sq, height=0.10 * np.max(sq), distance=int(0.2 * fs))
    if len(peaks) < 2:
        return 0, 0, 0, 0, 0, len(peaks)
    rr    = np.diff(peaks) / fs * 1000
    hr    = 60000 / np.mean(rr)
    cv    = np.std(rr) / np.mean(rr)
    rd    = np.abs(np.diff(rr)) if len(rr) > 1 else np.array([0.0])
    irreg = float(np.mean(rd / (rr[:-1] + 1e-6) > 0.20)) if len(rr) > 1 else 0
    pnn50 = float(np.mean(rd > 50)) if len(rd) > 0 else 0
    return hr, cv, irreg, pnn50, float(np.mean(rr)), len(peaks)

d = 'ecg-dashboard/input_samples'
print(f"{'File':<42} {'pk':>4} {'HR':>6} {'CV':>6} {'irr':>5} {'p50':>5} {'mRR':>7}")
print('-' * 78)
for f in sorted(os.listdir(d)):
    if not f.endswith('.csv'): continue
    hr, cv, irreg, pnn50, mrr, np_ = get_rr(os.path.join(d, f))
    print(f"{f[:40]:<42} {np_:4d} {hr:6.1f} {cv:6.3f} {irreg:5.2f} {pnn50:5.2f} {mrr:7.1f}")
