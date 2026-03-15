"""
Silver Layer — Data Cleaning & Feature Engineering
Reads raw data from Bronze, applies transformations, and stores in silver_patient_visits.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.schema import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Age bucket → numeric midpoint mapping
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

AGE_BINS = [0, 20, 40, 60, 80, 100]
AGE_LABELS = ['0-20', '21-40', '41-60', '61-80', '81-100']


def map_diagnosis_category(code: str) -> str:
    """Map raw ICD-9 diagnosis codes to broad clinical categories."""
    if pd.isna(code) or code in ('Unknown', 'nan', '?'):
        return 'Other'

    code = str(code).strip()

    if code.startswith('E'):
        return 'External'
    if code.startswith('V'):
        return 'Supplementary'

    try:
        num = float(code)
    except ValueError:
        return 'Other'

    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    elif 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    elif 520 <= num <= 579 or num == 787:
        return 'Digestive'
    elif 250 <= num < 251:
        return 'Diabetes'
    elif 800 <= num <= 999:
        return 'Injury'
    elif 710 <= num <= 739:
        return 'Musculoskeletal'
    elif 580 <= num <= 629 or num == 788:
        return 'Genitourinary'
    elif 140 <= num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def transform_bronze_to_silver() -> int:
    """
    Read from bronze_patient_visits, clean and transform, write to silver_patient_visits.
    Returns number of rows processed.
    """
    con = get_connection(read_only=True)

    try:
        df = con.execute("SELECT * FROM bronze_patient_visits").fetchdf()
    except Exception as e:
        logger.error(f"Failed to read Bronze layer: {e}")
        return 0
    finally:
        con.close()

    if len(df) == 0:
        logger.warning("Bronze layer is empty. Nothing to transform.")
        return 0

    logger.info(f"Silver Transform: Processing {len(df)} Bronze records...")

    # ── Step 1: Remove duplicates (keep first per patient_nbr) ──
    initial = len(df)
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset='patient_nbr', keep='first')
    logger.info(f"  Deduplication: {initial} → {len(df)} rows")

    # ── Step 2: Handle missing values ──
    # Convert obvious numeric columns to numeric first
    numeric_potentials = [
        'num_medications', 'num_lab_procedures', 'num_procedures', 
        'number_inpatient', 'number_outpatient', 'number_emergency', 
        'time_in_hospital', 'number_diagnoses'
    ]
    for col in numeric_potentials:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── Step 3: Map age buckets → numeric ──
    df['age_numeric'] = df['age'].map(AGE_MAP).fillna(55).astype(int)

    # Create age groups
    df['age_group'] = pd.cut(
        df['age_numeric'], bins=AGE_BINS, labels=AGE_LABELS
    ).astype(str)

    # ── Step 4: Map diagnosis codes → categories ──
    df['diag_1_category'] = df.get('diag_1', pd.Series(['Other'] * len(df))).apply(map_diagnosis_category)
    df['diag_2_category'] = df.get('diag_2', pd.Series(['Other'] * len(df))).apply(map_diagnosis_category)
    df['diag_3_category'] = df.get('diag_3', pd.Series(['Other'] * len(df))).apply(map_diagnosis_category)

    # ── Step 5: Engineer features ──
    # Total visits
    visit_cols = ['number_inpatient', 'number_outpatient', 'number_emergency']
    df['total_visits'] = df[visit_cols].sum(axis=1)

    # Medication change (insulin or diabetes_med change)
    df['medication_change'] = (
        df.get('insulin', pd.Series(['No'] * len(df))).isin(['Up', 'Down'])
    ).astype(int)

    # High lab procedures (above 75th percentile)
    threshold = df['num_lab_procedures'].quantile(0.75)
    df['high_lab_procedures'] = (df['num_lab_procedures'] > threshold).astype(int)

    # Total medications active (simplified)
    df['total_medications_active'] = df.get('num_medications', 0)

    # Readmitted binary
    df['readmitted_binary'] = (df.get('readmitted', pd.Series(['NO'] * len(df))) == '<30').astype(int)

    # ── Step 6: Build silver DataFrame ──
    silver_df = pd.DataFrame({
        'patient_id': df['patient_nbr'].astype(int),
        'visit_id': df['encounter_id'].astype(int),
        'age': df['age_numeric'],
        'age_group': df['age_group'],
        'gender': df['gender'],
        'race': df['race'],
        'time_in_hospital': df['time_in_hospital'].astype(int),
        'num_lab_procedures': df['num_lab_procedures'].astype(int),
        'num_procedures': df['num_procedures'].astype(int),
        'num_medications': df['num_medications'].astype(int),
        'total_visits': df['total_visits'].astype(int),
        'number_inpatient': df['number_inpatient'].astype(int),
        'number_outpatient': df['number_outpatient'].astype(int),
        'number_emergency': df['number_emergency'].astype(int),
        'diag_1_category': df['diag_1_category'],
        'insulin': df.get('insulin', 'Unknown'),
        'diabetes_med': df.get('diabetesMed', 'Unknown'),
        'a1c_result': df.get('A1Cresult', 'Unknown'),
        'readmitted_binary': df['readmitted_binary'],
    })

    # ── Step 7: Write to Silver table ──
    con = get_connection()
    con.execute("DELETE FROM silver_patient_visits")
    cols = ', '.join(silver_df.columns)
    con.execute(f"INSERT INTO silver_patient_visits ({cols}) SELECT {cols} FROM silver_df")
    con.close()

    logger.info(f"✅ Silver Layer: {len(silver_df)} cleaned records in silver_patient_visits")
    return len(silver_df)


if __name__ == '__main__':
    count = transform_bronze_to_silver()
    print(f"\nSilver transform complete: {count} records")
