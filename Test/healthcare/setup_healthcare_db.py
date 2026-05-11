import pandas as pd
import psycopg2
from psycopg2 import extras
import configparser
import os
import random
import csv
import json
from datetime import datetime

# Configuration
CSV_PATH = '../healthcare_dataset.csv'
SCHEMA_PATH = 'healthcare_schema.sql'
CONFIG_PATH = '../config/database.ini'
OUTPUT_DIR = 'output'

def get_db_config(filename=CONFIG_PATH, section='postgresql'):
    if not os.path.exists(filename):
        return None
    parser = configparser.ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        return None
    return db

def clean_name(name):
    if pd.isna(name): return name
    return name.strip().title()

def generate_vitals():
    """Generate dummy vitals for JSONB field."""
    vitals = {
        "blood_pressure": f"{random.randint(100, 150)}/{random.randint(60, 95)}",
        "heart_rate": random.randint(60, 110),
        "temperature": round(random.uniform(97.0, 102.0), 1),
        "respiratory_rate": random.randint(12, 20)
    }
    return vitals

def setup_database():
    print("Reading CSV data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Cleaning
    print("Cleaning data...")
    df['Name'] = df['Name'].apply(clean_name)
    df['Doctor'] = df['Doctor'].apply(clean_name)
    df['Hospital'] = df['Hospital'].apply(clean_name)
    
    # Data processing (Normalize)
    print("Normalizing entities...")
    
    # 1. Hospital Groups
    hospital_group_names = ["HealthFirst Network", "Unity Medical Group", "State General Systems", "Private Care Alliance", "Sub-Network A"]
    hospital_groups = [
        [1, "HealthFirst Network", None],
        [2, "Unity Medical Group", None],
        [3, "State General Systems", None],
        [4, "Private Care Alliance", None],
        [5, "Sub-Network A", 1] # Nested
    ]
    
    # 2. Hospitals
    hospitals_unique = df['Hospital'].unique()
    hospitals_data = []
    hospital_map = {}
    for i, h in enumerate(hospitals_unique, 1):
        g_id = random.randint(1, 5)
        hospitals_data.append([i, h, f"{random.randint(1, 999)} Medical Way", g_id])
        hospital_map[h] = i
        
    # 3. Doctors
    doctors_unique = df[['Doctor', 'Hospital']].drop_duplicates()
    doctors_data = []
    doctor_map = {}
    for i, (_, row) in enumerate(doctors_unique.iterrows(), 1):
        spec = random.choice(["Cardiology", "Internal Medicine", "Pediatrics", "Oncology", "Neurology"])
        h_id = hospital_map[row['Hospital']]
        doctors_data.append([i, row['Doctor'], spec, h_id])
        doctor_map[(row['Doctor'], row['Hospital'])] = i
        
    # 4. Patients
    patients_unique_df = df[['Name', 'Age', 'Gender', 'Blood Type']].drop_duplicates(subset=['Name'])
    patients_data = []
    patient_map = {}
    for i, (_, row) in enumerate(patients_unique_df.iterrows(), 1):
        # We'll add parent_id later
        patients_data.append([i, row['Name'], row['Age'], row['Gender'], row['Blood Type'], None])
        patient_map[row['Name']] = i
        
    # Add recursive links (Family tree)
    num_patients = len(patients_data)
    for i in range(min(500, num_patients)):
        if random.random() > 0.7:
            parent_id = random.randint(1, num_patients)
            if parent_id != patients_data[i][0]:
                patients_data[i][5] = parent_id
                
    # 5. Medications
    meds_unique = df['Medication'].unique()
    medications_data = []
    medication_map = {}
    for i, m in enumerate(meds_unique, 1):
        medications_data.append([i, m])
        medication_map[m] = i
        
    # 6. Admissions & Prescriptions
    admissions_data = []
    prescriptions_data = []
    print("Processing admissions...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        p_id = patient_map[row['Name']]
        h_id = hospital_map[row['Hospital']]
        d_id = doctor_map[(row['Doctor'], row['Hospital'])]
        vitals = generate_vitals()
        
        admissions_data.append([
            i, p_id, d_id, h_id, row['Date of Admission'], row['Discharge Date'],
            row['Admission Type'], row['Medical Condition'], row['Insurance Provider'],
            row['Billing Amount'], row['Room Number'], row['Test Results'], 
            json.dumps(vitals)
        ])
        
        # Prescription
        m_id = medication_map[row['Medication']]
        prescriptions_data.append([i, i, m_id, f"{random.randint(1, 3)} tabs", "Once daily"])

    # Try to connect and insert
    params = get_db_config()
    conn = None
    connected = False
    if params:
        try:
            print("Attempting to connect to database...")
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            
            print("Connected! Creating schema and inserting data...")
            with open(SCHEMA_PATH, 'r') as f:
                cur.execute(f.read())
                
            # Helper for bulk insert
            def bulk_insert(table, data, columns):
                query = f"INSERT INTO {table} ({','.join(columns)}) VALUES %s"
                extras.execute_values(cur, query, data)

            bulk_insert("hospital_groups", hospital_groups, ["group_id", "name", "parent_group_id"])
            bulk_insert("hospitals", hospitals_data, ["hospital_id", "name", "location", "group_id"])
            bulk_insert("doctors", doctors_data, ["doctor_id", "name", "specialization", "hospital_id"])
            bulk_insert("patients", patients_data, ["patient_id", "name", "age", "gender", "blood_type", "parent_id"])
            bulk_insert("medications", medications_data, ["medication_id", "name"])
            bulk_insert("admissions", admissions_data, [
                "admission_id", "patient_id", "doctor_id", "hospital_id", "admission_date", "discharge_date", 
                "admission_type", "medical_condition", "insurance_provider", "billing_amount", 
                "room_number", "test_results", "vitals"
            ])
            bulk_insert("prescriptions", prescriptions_data, ["prescription_id", "admission_id", "medication_id", "dosage", "frequency"])
            
            conn.commit()
            print("Success! Database populated directly.")
            connected = True
        except Exception as e:
            print(f"Database connection or insertion failed: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    if not connected:
        print("\nCreating CSV exports for manual import...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        def export_to_csv(filename, data, headers):
            path = os.path.join(OUTPUT_DIR, filename)
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(data)
            return path

        export_to_csv("hospital_groups.csv", hospital_groups, ["group_id", "name", "parent_group_id"])
        export_to_csv("hospitals.csv", hospitals_data, ["hospital_id", "name", "location", "group_id"])
        export_to_csv("doctors.csv", doctors_data, ["doctor_id", "name", "specialization", "hospital_id"])
        export_to_csv("patients.csv", patients_data, ["patient_id", "name", "age", "gender", "blood_type", "parent_id"])
        export_to_csv("medications.csv", medications_data, ["medication_id", "name"])
        export_to_csv("admissions.csv", admissions_data, [
            "admission_id", "patient_id", "doctor_id", "hospital_id", "admission_date", "discharge_date", 
            "admission_type", "medical_condition", "insurance_provider", "billing_amount", 
            "room_number", "test_results", "vitals"
        ])
        export_to_csv("prescriptions.csv", prescriptions_data, ["prescription_id", "admission_id", "medication_id", "dosage", "frequency"])
        
        # Create a helper SQL script for COPY commands
        with open(os.path.join(OUTPUT_DIR, "import_data.sql"), "w") as f:
            f.write(f"\\i ../{SCHEMA_PATH}\n")
            tables = ["hospital_groups", "hospitals", "doctors", "patients", "medications", "admissions", "prescriptions"]
            for table in tables:
                f.write(f"\\copy {table} FROM '{table}.csv' WITH (FORMAT csv, HEADER true);\n")
        
        print(f"Done! Files generated in {os.path.abspath(OUTPUT_DIR)}")
        print("You can run 'psql -d <your_db> -f output/import_data.sql' to load the data.")

if __name__ == '__main__':
    setup_database()
