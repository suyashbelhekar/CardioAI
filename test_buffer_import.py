import sys
sys.path.insert(0, 'ecg-dashboard')

print("Importing app module...")
import app

print(f"Buffer exists: {hasattr(app, 'ecg_buffer')}")
if hasattr(app, 'ecg_buffer'):
    print(f"Buffer type: {type(app.ecg_buffer)}")
    print(f"Buffer length: {len(app.ecg_buffer)}")
    print("SUCCESS: ecg_buffer is accessible!")
else:
    print("FAIL: ecg_buffer not found in module")
