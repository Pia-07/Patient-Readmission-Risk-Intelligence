"""
Synthetic Patient Data Generator
=================================
Generates realistic hospital patient records at configurable intervals.
Runs as a background thread and pushes data through ETL → prediction → warehouse.
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [GENERATOR] %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
BATCH_SIZE = 10
INTERVAL_SECONDS = 300          # 5 minutes (use 30 for demo)
API_URL = "http://127.0.0.1:8000"
RAW_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'data', 'raw', 'streaming_patients.csv')

# ── Realistic distributions ────────────────────────────────────
GENDERS = ['Male', 'Female']
RACES = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other']
RACE_PROBS = [0.55, 0.2, 0.12, 0.08, 0.05]
DIAG_CATEGORIES = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                   'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
AGE_BINS = [0, 20, 40, 60, 80, 100]
AGE_LABELS = ['0-20', '21-40', '41-60', '61-80', '81-100']

_counter_lock = threading.Lock()
_patient_counter = 100_000       # start IDs at 100 000 to avoid collisions


def _next_id(n: int = 1) -> list[int]:
    global _patient_counter
    with _counter_lock:
        ids = list(range(_patient_counter, _patient_counter + n))
        _patient_counter += n
    return ids


def generate_batch(n: int = BATCH_SIZE) -> pd.DataFrame:
    """Generate a batch of realistic synthetic patient records."""
    ids = _next_id(n)
    ages = np.random.randint(18, 95, n)
    age_groups = pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    df = pd.DataFrame({
        'patient_id':                ids,
        'timestamp':                 [datetime.utcnow().isoformat()] * n,
        'age':                       ages,
        'age_group':                 age_groups,
        'gender':                    np.random.choice(GENDERS, n),
        'race':                      np.random.choice(RACES, n, p=RACE_PROBS),
        'num_medications':           np.random.randint(1, 30, n),
        'num_lab_procedures':        np.random.randint(10, 80, n),
        'num_procedures':            np.random.randint(0, 6, n),
        'number_outpatient':         np.random.randint(0, 10, n),
        'number_emergency':          np.random.randint(0, 4, n),
        'number_inpatient':          np.random.randint(0, 5, n),
        'number_diagnoses':          np.random.randint(1, 15, n),
        'time_in_hospital':          np.random.randint(1, 14, n),
        'discharge_disposition_id':  np.random.randint(1, 25, n),
        'admission_type_id':         np.random.randint(1, 8, n),
        'admission_source_id':       np.random.randint(1, 20, n),
        'diag_1_category':           np.random.choice(DIAG_CATEGORIES, n),
        'diag_2_category':           np.random.choice(DIAG_CATEGORIES, n),
        'diag_3_category':           np.random.choice(DIAG_CATEGORIES, n),
        'max_glu_serum':             np.random.choice(['None', 'Norm', '>200', '>300'], n, p=[0.6, 0.15, 0.15, 0.1]),
        'A1Cresult':                 np.random.choice(['None', 'Norm', '>7', '>8'], n, p=[0.6, 0.15, 0.15, 0.1]),
        'change':                    np.random.choice(['No', 'Ch'], n, p=[0.7, 0.3]),
        'diabetesMed':               np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3]),
        'insulin':                   np.random.choice(['No', 'Up', 'Down', 'Steady'], n, p=[0.4, 0.25, 0.1, 0.25]),
        'metformin':                 np.random.choice(['No', 'Steady', 'Up', 'Down'], n, p=[0.5, 0.25, 0.15, 0.1]),
    })
    # Derived
    df['total_visits'] = df['number_inpatient'] + df['number_outpatient'] + df['number_emergency']
    return df


def _append_raw(df: pd.DataFrame):
    """Append raw records to the streaming CSV file."""
    os.makedirs(os.path.dirname(RAW_OUTPUT), exist_ok=True)
    header = not os.path.exists(RAW_OUTPUT)
    df.to_csv(RAW_OUTPUT, mode='a', index=False, header=header)


def _predict_and_store(df: pd.DataFrame):
    """Send each patient to /predict and store results in DuckDB."""
    try:
        import duckdb
        from database.schema import get_connection, create_tables
        from pipeline.bronze_ingest import _insert_bronze
    except Exception as e:
        logger.warning(f"Could not import database modules: {e}")
        return

    create_tables()
    con = get_connection()

    # Step 0: Ingest into Bronze (strictly raw)
    # We map the dict to match raw CSV column names
    bronze_records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        # Ensure we have all 50 keys (even if empty) to match Bronze schema
        full_record = {
            'encounter_id': r.get('patient_id'), # use patient_id as encounter_id for sim
            'patient_nbr': r.get('patient_id'),
            'race': r.get('race'),
            'gender': r.get('gender'),
            'age': f"[{r.get('age')}-{r.get('age')+10})", # bucketize like raw
            'weight': '?',
            'admission_type_id': r.get('admission_type_id'),
            'discharge_disposition_id': r.get('discharge_disposition_id'),
            'admission_source_id': r.get('admission_source_id'),
            'time_in_hospital': r.get('time_in_hospital'),
            'payer_code': '?',
            'medical_specialty': '?',
            'num_lab_procedures': r.get('num_lab_procedures'),
            'num_procedures': r.get('num_procedures'),
            'num_medications': r.get('num_medications'),
            'number_outpatient': r.get('number_outpatient'),
            'number_emergency': r.get('number_emergency'),
            'number_inpatient': r.get('number_inpatient'),
            'diag_1': r.get('diag_1_category'), # sim uses cat names as raw codes
            'diag_2': r.get('diag_2_category'),
            'diag_3': r.get('diag_3_category'),
            'number_diagnoses': r.get('number_diagnoses'),
            'max_glu_serum': r.get('max_glu_serum'),
            'A1Cresult': r.get('A1Cresult'),
            'metformin': r.get('metformin'),
            'repaglinide': 'No',
            'nateglinide': 'No',
            'chlorpropamide': 'No',
            'glimepiride': 'No',
            'acetohexamide': 'No',
            'glipizide': 'No',
            'glyburide': 'No',
            'tolbutamide': 'No',
            'pioglitazone': 'No',
            'rosiglitazone': 'No',
            'acarbose': 'No',
            'miglitol': 'No',
            'troglitazone': 'No',
            'tolazamide': 'No',
            'examide': 'No',
            'citoglipton': 'No',
            'insulin': r.get('insulin'),
            'glyburide-metformin': 'No',
            'glipizide-metformin': 'No',
            'glimepiride-pioglitazone': 'No',
            'metformin-rosiglitazone': 'No',
            'metformin-pioglitazone': 'No',
            'change': r.get('change'),
            'diabetesMed': r.get('diabetesMed'),
            'readmitted': 'NO'
        }
        bronze_records.append(full_record)
    
    _insert_bronze(pd.DataFrame(bronze_records))

    for _, row in df.iterrows():
        patient = row.to_dict()
        try:
            resp = requests.post(f"{API_URL}/predict", json={
                'age': int(patient['age']),
                'gender': patient['gender'],
                'race': patient['race'],
                'time_in_hospital': int(patient['time_in_hospital']),
                'num_lab_procedures': int(patient['num_lab_procedures']),
                'num_procedures': int(patient['num_procedures']),
                'num_medications': int(patient['num_medications']),
                'number_outpatient': int(patient['number_outpatient']),
                'number_emergency': int(patient['number_emergency']),
                'number_inpatient': int(patient['number_inpatient']),
                'number_diagnoses': int(patient['number_diagnoses']),
                'max_glu_serum': patient['max_glu_serum'],
                'A1Cresult': patient['A1Cresult'],
                'change': patient['change'],
                'diabetesMed': patient['diabetesMed'],
                'admission_type_id': int(patient['admission_type_id']),
                'discharge_disposition_id': int(patient['discharge_disposition_id']),
                'admission_source_id': int(patient['admission_source_id']),
                'diag_1_category': patient['diag_1_category'],
                'diag_2_category': patient['diag_2_category'],
                'diag_3_category': patient['diag_3_category'],
                'insulin': patient['insulin'],
                'metformin': patient['metformin'],
            }, timeout=5)
            pred = resp.json()
        except Exception as e:
            logger.warning(f"Prediction API error for {patient['patient_id']}: {e}")
            pred = {'risk_score': 0.5, 'risk_percentage': 50.0, 'risk_level': 'Medium'}

        pid = int(patient['patient_id'])

        # Insert into dim_patient
        try:
            con.execute("""
                INSERT OR IGNORE INTO dim_patient (patient_id, age, gender)
                VALUES (?, ?, ?)
            """, [pid, int(patient['age']), patient['gender']])
        except Exception:
            pass

        # Insert into dim_visit_metrics
        vid = pid  # 1:1 for simplicity
        try:
            con.execute("""
                INSERT OR IGNORE INTO dim_visit_metrics
                (visit_id, num_medications, num_lab_procedures, number_inpatient, time_in_hospital)
                VALUES (?, ?, ?, ?, ?)
            """, [vid, int(patient['num_medications']), int(patient['num_lab_procedures']),
                  int(patient['number_inpatient']), int(patient['time_in_hospital'])])
        except Exception:
            pass

        # Insert into fact_patient_visits
        try:
            con.execute("""
                INSERT INTO fact_patient_visits
                (patient_id, visit_id, timestamp, risk_score, risk_level)
                VALUES (?, ?, ?, ?, ?)
            """, [pid, vid, datetime.utcnow().isoformat(),
                  pred.get('risk_score', 0.5), pred.get('risk_level', 'Medium')])
        except Exception:
            pass

        # Also insert into legacy tables for backward compat with existing pages
        try:
            con.execute("""
                INSERT OR IGNORE INTO patients
                (patient_id, age, age_group, gender, race,
                 num_medications, num_lab_procedures, num_procedures,
                 number_diagnoses, time_in_hospital,
                 diag_1_category, diag_2_category, diag_3_category,
                 diabetes_med, insulin, a1c_result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [pid, int(patient['age']), patient['age_group'],
                  patient['gender'], patient['race'],
                  int(patient['num_medications']), int(patient['num_lab_procedures']),
                  int(patient['num_procedures']), int(patient['number_diagnoses']),
                  int(patient['time_in_hospital']),
                  patient['diag_1_category'], patient['diag_2_category'],
                  patient['diag_3_category'],
                  patient['diabetesMed'], patient['insulin'], patient['A1Cresult']])
        except Exception:
            pass

        try:
            con.execute("""
                INSERT OR IGNORE INTO patient_visits
                (visit_id, patient_id, admission_type_id, discharge_disposition_id,
                 admission_source_id, number_outpatient, number_emergency,
                 number_inpatient, total_visits, medication_change, high_lab_procedures)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, [vid, pid, int(patient['admission_type_id']),
                  int(patient['discharge_disposition_id']),
                  int(patient['admission_source_id']),
                  int(patient['number_outpatient']), int(patient['number_emergency']),
                  int(patient['number_inpatient']), int(patient['total_visits']),
                  patient['change'] == 'Ch',
                  int(patient['num_lab_procedures']) > 57])
        except Exception:
            pass

        try:
            con.execute("""
                INSERT INTO model_predictions
                (prediction_id, patient_id, risk_score, risk_percentage, risk_level,
                 actual_readmitted, top_factors)
                VALUES (?,?,?,?,?,?,?)
            """, [pid, pid, pred.get('risk_score', 0.5),
                  pred.get('risk_percentage', 50.0), pred.get('risk_level', 'Medium'),
                  0, 'number_inpatient,num_medications,age'])
        except Exception:
            pass

    con.close()
    logger.info(f"✅ Stored {len(df)} predictions in DuckDB")


def run_cycle():
    """Run one generate → predict → store cycle."""
    logger.info(f"Generating {BATCH_SIZE} patients...")
    batch = generate_batch(BATCH_SIZE)
    _append_raw(batch)
    logger.info(f"Raw data appended to {RAW_OUTPUT}")
    _predict_and_store(batch)


def start_background(interval: int = INTERVAL_SECONDS):
    """Start the generator as a background daemon thread."""
    def _loop():
        while True:
            try:
                run_cycle()
            except Exception as e:
                logger.error(f"Generator cycle error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    logger.info(f"🚀 Background generator started — {BATCH_SIZE} patients every {interval}s")
    return t


# ── CLI entry point ────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Patient Generator")
    parser.add_argument('--interval', type=int, default=30,
                        help='Seconds between batches (default 30 for demo)')
    parser.add_argument('--batch', type=int, default=10, help='Patients per batch')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    args = parser.parse_args()
    BATCH_SIZE = args.batch

    if args.once:
        run_cycle()
    else:
        start_background(args.interval)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Generator stopped.")
