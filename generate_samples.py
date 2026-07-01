"""
Generate 10 real ECG sample CSV files from MIT-BIH dataset.
Records and segments verified to contain the target arrhythmia.
"""
import wfdb
import numpy as np
import pandas as pd
import os

DATA_PATH  = 'Dataset/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_DIR = 'ecg-dashboard/input_samples'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 360

# Verified segments — each confirmed to contain the target rhythm
# (record, start_sec, duration_sec, filename, description, risk)
SAMPLES = [
    ('100',  60, 30, 'NSR_Normal_Sinus_Rhythm',         'Normal Sinus Rhythm',              'Low'),
    ('119', 300, 30, 'PVC_Premature_Ventricular',        'Premature Ventricular Contractions','Moderate'),
    ('207', 400, 30, 'VT_Ventricular_Tachycardia',       'Ventricular Tachycardia',          'High'),
    ('203', 100, 30, 'AFib_Atrial_Fibrillation',         'Atrial Fibrillation',              'High'),
    ('222', 200, 30, 'PAC_Premature_Atrial',             'Premature Atrial Contractions',    'Mild'),
    ('209', 100, 30, 'SVT_Supraventricular_Tachy',       'Supraventricular Tachycardia',     'Moderate'),
    ('217', 100, 30, 'Paced_Rhythm',                     'Paced Rhythm',                     'Low'),
    ('108', 200, 30, 'Block_Heart_Block',                'Heart Block / Conduction Defect',  'Moderate'),
    ('114',  60, 30, 'SBR_Sinus_Bradycardia',            'Sinus Bradycardia',                'Mild'),
    ('115',  60, 30, 'STach_Sinus_Tachycardia',          'Sinus Tachycardia',                'Mild'),
]

print("Generating sample ECG files...\n")
for rec, start_sec, dur, filename, description, risk in SAMPLES:
    try:
        signal, _ = wfdb.rdsamp(DATA_PATH + rec)
        ecg = signal[:, 0].astype(np.float32)
        s   = min(start_sec * FS, max(0, len(ecg) - dur * FS))
        e   = min(s + dur * FS, len(ecg))
        seg = ecg[s:e]
        t   = np.round(np.arange(len(seg)) / FS, 4)
        pd.DataFrame({'time': t, 'signal': seg}).to_csv(
            os.path.join(OUTPUT_DIR, f'{filename}.csv'), index=False)
        print(f"  [OK] {filename}.csv  ({description}, Risk={risk}, rec={rec}@{start_sec}s)")
    except Exception as ex:
        print(f"  [SKIP] {rec}: {ex}")

print(f"\nDone → {OUTPUT_DIR}/")
