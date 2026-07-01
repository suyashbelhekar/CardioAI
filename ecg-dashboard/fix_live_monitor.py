"""
Quick fix for live ECG monitor - adds global ecg_buffer declarations
Run this after starting the server to patch the routes
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app
from app import app, ecg_buffer

# Verify buffer exists
print(f"[FIX] ecg_buffer exists: {ecg_buffer is not None}")
print(f"[FIX] ecg_buffer type: {type(ecg_buffer)}")
print(f"[FIX] ecg_buffer length: {len(ecg_buffer)}")

# Patch the route functions to use global
import app as app_module

# Store original functions
original_ecg_push = app_module.ecg_push
original_ecg_stream = app_module.ecg_stream  
original_ecg_buffer_get = app_module.ecg_buffer_get
original_ecg_clear = app_module.ecg_clear

# Create patched versions
def patched_ecg_push():
    from flask import request, jsonify
    from datetime import datetime
    global ecg_buffer
    d     = request.get_json(silent=True) or {}
    value = d.get('value', 0)
    ts    = d.get('ts', datetime.now().isoformat())
    ecg_buffer.append({'ts': ts, 'value': float(value)})
    if len(ecg_buffer) > 2000:
        ecg_buffer.pop(0)
    return jsonify({'status': 'ok', 'buffered': len(ecg_buffer)})

def patched_ecg_buffer_get():
    from flask import request, jsonify
    global ecg_buffer
    n = int(request.args.get('n', 1800))
    return jsonify(ecg_buffer[-n:])

def patched_ecg_clear():
    from flask import jsonify
    global ecg_buffer
    ecg_buffer.clear()
    return jsonify({'status': 'cleared'})

# Replace in module
app_module.ecg_push = patched_ecg_push
app_module.ecg_buffer_get = patched_ecg_buffer_get
app_module.ecg_clear = patched_ecg_clear

print("[FIX] Routes patched successfully!")
print("[FIX] Test the endpoints now")
