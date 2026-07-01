"""Test all 10 sample files against the running dashboard and print results."""
import os, json, requests

SAMPLES_DIR = 'ecg-dashboard/input_samples'
BASE_URL    = 'http://127.0.0.1:5000'

# Expected diagnosis per file
EXPECTED = {
    'NSR_Normal_Sinus_Rhythm':         'NSR',
    'PVC_Premature_Ventricular':        'PVC',
    'VT_Ventricular_Tachycardia':       'VT',
    'AFib_Atrial_Fibrillation':         'AFib',
    'PAC_Premature_Atrial':             'PAC',
    'SVT_Supraventricular_Tachy':       'SVT',
    'Paced_Rhythm':                     'Paced',
    'Block_Heart_Block':                'Block',
    'SBR_Sinus_Bradycardia':            'SBR',
    'STach_Sinus_Tachycardia':          'STach',
}

import tempfile, pickle, numpy as np, pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

# Load models directly to test without HTTP overhead
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import app functions directly
sys.path.insert(0, 'ecg-dashboard')
import importlib.util
spec = importlib.util.spec_from_file_location("app", "ecg-dashboard/app.py")
app_mod = importlib.util.load_from_spec = None

# Use requests instead
session = requests.Session()

print(f"\n{'File':<42} {'Expected':>8} {'Got':>12} {'Conf':>6} {'Risk':>10} {'Match'}")
print('-' * 85)

correct = 0
for fname in sorted(os.listdir(SAMPLES_DIR)):
    if not fname.endswith('.csv'):
        continue
    key = fname.replace('.csv', '')
    expected = EXPECTED.get(key, '?')

    fpath = os.path.join(SAMPLES_DIR, fname)
    with open(fpath, 'rb') as f:
        resp = session.post(f'{BASE_URL}/predict',
                            files={'file': (fname, f, 'text/csv')},
                            timeout=60)

    if resp.status_code != 200 or resp.json().get('status') != 'ok':
        print(f"  {fname:<42} ERROR: {resp.text[:60]}")
        continue

    # Get results page to extract diagnosis
    r2 = session.get(f'{BASE_URL}/results', timeout=10)
    # Parse primary diagnosis from HTML
    html = r2.text
    import re
    # Extract diagnosis name from the primary banner
    m = re.search(r'font-size:1\.1rem.*?>(.*?)</span>', html, re.DOTALL)
    diag_name = m.group(1).strip() if m else '?'

    # Also get confidence and risk from HTML
    conf_m = re.search(r'(\d+\.\d+)% confidence', html)
    conf   = conf_m.group(1) if conf_m else '?'
    risk_m = re.search(r'Overall Risk: (\w+)', html)
    risk   = risk_m.group(1) if risk_m else '?'

    # Map display name back to ID
    name_to_id = {
        'Normal Sinus Rhythm': 'NSR',
        'Premature Ventricular Contractions (PVC)': 'PVC',
        'Ventricular Tachycardia (VT)': 'VT',
        'Atrial Fibrillation (AFib)': 'AFib',
        'Premature Atrial Contractions (PAC)': 'PAC',
        'Supraventricular Tachycardia (SVT)': 'SVT',
        'Paced Rhythm': 'Paced',
        'Heart Block / Conduction Defect': 'Block',
        'Sinus Bradycardia': 'SBR',
        'Sinus Tachycardia': 'STach',
        'Ventricular Bigeminy': 'Bigeminy',
    }
    got_id = name_to_id.get(diag_name, diag_name[:12])
    match  = 'OK' if got_id == expected else 'WRONG'
    if got_id == expected:
        correct += 1

    print(f"  {key:<42} {expected:>8} {got_id:>12} {conf:>6} {risk:>10}  {match}")

print(f"\nAccuracy: {correct}/10 = {correct*10}%")
