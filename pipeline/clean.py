"""
Data Cleaning Module
Handles missing values, duplicates, and target variable encoding.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Columns to drop (>40% missing or not useful)
DROP_COLUMNS = [
    'weight',           # ~97% missing
    'payer_code',       # ~40% missing, not clinically relevant
    'medical_specialty', # ~49% missing
    'encounter_id',     # Identifier, not a feature
    'patient_nbr',      # Will be used for dedup, then dropped
    'examide',          # Only one value
    'citoglipton',      # Only one value
]

# Diagnosis columns to simplify
DIAG_COLUMNS = ['diag_1', 'diag_2', 'diag_3']


def map_diagnosis_category(code):
    """Map ICD-9 diagnosis codes to broad categories."""
    if pd.isna(code):
        return 'Other'
    
    code = str(code)
    
    # Handle E and V codes
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    1. Remove duplicates per patient (keep first encounter)
    2. Drop high-missing and identifier columns
    3. Map diagnosis codes to categories
    4. Create binary target variable
    5. Fill remaining missing values
    """
    logger.info("Starting data cleaning...")
    initial_rows = len(df)
    
    # --- 1. Deduplicate by patient (keep first encounter) ---
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset='patient_nbr', keep='first')
        logger.info(f"Deduplication: {initial_rows} → {len(df)} rows")
    
    # --- 2. Create binary target BEFORE dropping columns ---
    # readmitted: '<30' → 1 (readmitted within 30 days), else → 0
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    logger.info(f"Target distribution:\n{df['readmitted_binary'].value_counts()}")
    
    # Drop original readmitted column
    df = df.drop(columns=['readmitted'], errors='ignore')
    
    # --- 3. Drop high-missing / identifier columns ---
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")
    
    # --- 4. Map diagnosis codes to categories ---
    for col in DIAG_COLUMNS:
        if col in df.columns:
            df[f'{col}_category'] = df[col].apply(map_diagnosis_category)
            df = df.drop(columns=[col])
    logger.info("Mapped diagnosis codes to categories")
    
    # --- 5. Handle remaining missing values ---
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    remaining_missing = df.isnull().sum().sum()
    logger.info(f"Remaining missing values: {remaining_missing}")
    logger.info(f"Cleaned dataset shape: {df.shape}")
    
    return df


if __name__ == '__main__':
    from pipeline.ingest import load_dataset
    df = load_dataset()
    df_clean = clean_data(df)
    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"\nTarget:\n{df_clean['readmitted_binary'].value_counts(normalize=True)}")
