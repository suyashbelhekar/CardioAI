"""
Initialize the database and create a demo doctor account
"""

import database as db

print("Initializing database...")
db.init_db()

print("\nCreating demo doctor account...")
doctor_id = db.create_doctor(
    username="doctor",
    email="doctor@cardioai.com",
    password="doctor123",
    full_name="Dr. John Smith",
    specialization="Cardiologist",
    license_number="MED-12345",
    phone="+1 (555) 123-4567"
)

if doctor_id:
    print(f"✓ Demo doctor created successfully!")
    print(f"  Username: doctor")
    print(f"  Password: doctor123")
    print(f"  Doctor ID: {doctor_id}")
else:
    print("✗ Doctor account already exists or creation failed")

print("\nDatabase initialization complete!")
print("You can now run the application with: python app.py")
