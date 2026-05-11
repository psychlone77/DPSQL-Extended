-- Healthcare Dataset Schema

-- 1. Hospital Groups (Recursive Organization Hierarchy)
CREATE TABLE IF NOT EXISTS hospital_groups (
    group_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    parent_group_id INTEGER REFERENCES hospital_groups(group_id)
);

-- 2. Hospitals
CREATE TABLE IF NOT EXISTS hospitals (
    hospital_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    group_id INTEGER REFERENCES hospital_groups(group_id)
);

-- 3. Doctors
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    specialization TEXT,
    hospital_id INTEGER REFERENCES hospitals(hospital_id)
);

-- 4. Patients (Recursive Family Hierarchy)
CREATE TABLE IF NOT EXISTS patients (
    patient_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    blood_type TEXT,
    parent_id INTEGER REFERENCES patients(patient_id)
);

-- 5. Admissions (Includes JSONB for Nested Queries)
CREATE TABLE IF NOT EXISTS admissions (
    admission_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    hospital_id INTEGER REFERENCES hospitals(hospital_id),
    admission_date DATE,
    discharge_date DATE,
    admission_type TEXT,
    medical_condition TEXT,
    insurance_provider TEXT,
    billing_amount DECIMAL(15, 2),
    room_number INTEGER,
    test_results TEXT,
    vitals JSONB -- Nested data: e.g., {"bp": "120/80", "pulse": 72, "temp": 98.6}
);

-- 6. Medications
CREATE TABLE IF NOT EXISTS medications (
    medication_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

-- 7. Prescriptions (Transactional linkage)
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id SERIAL PRIMARY KEY,
    admission_id INTEGER REFERENCES admissions(admission_id),
    medication_id INTEGER REFERENCES medications(medication_id),
    dosage TEXT,
    frequency TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_admission_patient ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admission_doctor ON admissions(doctor_id);
CREATE INDEX IF NOT EXISTS idx_admission_hospital ON admissions(hospital_id);
CREATE INDEX IF NOT EXISTS idx_patient_parent ON patients(parent_id);
CREATE INDEX IF NOT EXISTS idx_hospital_group ON hospitals(group_id);
CREATE INDEX IF NOT EXISTS idx_admission_vitals ON admissions USING gin (vitals);
