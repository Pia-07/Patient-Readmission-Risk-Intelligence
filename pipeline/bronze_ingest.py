"""
Bronze Layer — Raw Data Ingestion
Loads raw data exactly as received and stores it in bronze_patient_visits.
No transformations are applied at this stage.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.schema import get_connection, create_tables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')


def ingest_kaggle_data(filename: str = 'diabetic_data.csv') -> int:
    """
    Load the raw Kaggle CSV and insert every row into bronze_patient_visits
    without any modification (true Bronze layer).
    Returns number of rows ingested.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        logger.warning(f"Kaggle dataset not found at {filepath}. Using synthetic data instead.")
        return ingest_synthetic_data()

    logger.info(f"Loading raw Kaggle data from {filepath}...")
    df = pd.read_csv(filepath, na_values='?')
    logger.info(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # In Bronze, we keep EVERYTHING. Data is stored exactly as in CSV.
    # We'll just clean up the hyphenated columns for DuckDB if needed, 
    # but since our schema uses double quotes, we can pass them directly.
    return _insert_bronze(df)


def ingest_synthetic_data(n: int = 5000) -> int:
    """
    Generate synthetic patient data and load into Bronze layer.
    Includes all 50 columns to match raw Kaggle schema.
    """
    logger.info(f"Generating {n} synthetic patient records for Bronze layer...")
    np.random.seed(42)

    age_buckets = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                   '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    diag_codes = ['250', '410', '428', '486', '560', '780', '820', '996', 'V58']

    # Generate all columns to match raw CSV schema
    data = {
        'encounter_id': range(2000000, 2000000 + n),
        'patient_nbr': range(100000, 100000 + n),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'age': np.random.choice(age_buckets, n),
        'weight': '?',
        'admission_type_id': np.random.randint(1, 8, n),
        'discharge_disposition_id': np.random.randint(1, 25, n),
        'admission_source_id': np.random.randint(1, 20, n),
        'time_in_hospital': np.random.randint(1, 14, n),
        'payer_code': '?',
        'medical_specialty': '?',
        'num_lab_procedures': np.random.randint(10, 100, n),
        'num_procedures': np.random.randint(0, 6, n),
        'num_medications': np.random.randint(1, 35, n),
        'number_outpatient': np.random.randint(0, 5, n),
        'number_emergency': np.random.randint(0, 5, n),
        'number_inpatient': np.random.randint(0, 5, n),
        'diag_1': np.random.choice(diag_codes, n),
        'diag_2': np.random.choice(diag_codes, n),
        'diag_3': np.random.choice(diag_codes, n),
        'number_diagnoses': np.random.randint(1, 16, n),
        'max_glu_serum': 'None',
        'A1Cresult': 'None',
        'metformin': 'No',
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
        'insulin': np.random.choice(['No', 'Up', 'Down', 'Steady'], n),
        'glyburide-metformin': 'No',
        'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No',
        'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No',
        'change': np.random.choice(['No', 'Ch'], n),
        'diabetesMed': np.random.choice(['Yes', 'No'], n),
        'readmitted': np.random.choice(['<30', '>30', 'NO'], n)
    }
    
    bronze_df = pd.DataFrame(data)

    return _insert_bronze(bronze_df)


def ingest_single_record(record: dict) -> int:
    """Ingest a single new patient record (used by real-time simulator)."""
    bronze_df = pd.DataFrame([record])
    return _insert_bronze(bronze_df)


def _insert_bronze(bronze_df: pd.DataFrame) -> int:
    """Insert a DataFrame into the bronze_patient_visits table."""
    create_tables()
    con = get_connection()

    # Clear existing bronze data for full reload
    if len(bronze_df) > 100:
        con.execute("DELETE FROM bronze_patient_visits")

    # Use DuckDB's native ability to query Pandas DataFrames
    # Quote column names to handle hyphens (e.g. "glyburide-metformin")
    quoted_cols = ', '.join([f'"{c}"' for c in bronze_df.columns])
    con.execute(f"INSERT INTO bronze_patient_visits ({quoted_cols}) SELECT {quoted_cols} FROM bronze_df")
    con.close()

    row_count = len(bronze_df)
    logger.info(f"✅ Bronze Layer: Ingested {row_count} raw records into bronze_patient_visits")
    return row_count


if __name__ == '__main__':
    count = ingest_kaggle_data()
    print(f"\nBronze ingestion complete: {count} records")
