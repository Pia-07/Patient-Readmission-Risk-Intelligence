"""
Feature Engineering Module
Creates derived features and encodes categorical variables for ML.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

# Age bucket midpoints
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# Medication columns (binary: was medication prescribed)
MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

# Columns to label-encode
LABEL_ENCODE_COLS = [
    'race', 'gender', 'age',
    'diag_1_category', 'diag_2_category', 'diag_3_category',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - total_visits
    - medication_change flag
    - high_lab_procedures indicator
    - age_numeric
    - num_medications_bin
    - medication activity encoding
    """
    logger.info("Starting feature engineering...")
    df = df.copy()
    
    # --- 1. Total visits ---
    visit_cols = ['number_inpatient', 'number_outpatient', 'number_emergency']
    existing_visit_cols = [c for c in visit_cols if c in df.columns]
    if existing_visit_cols:
        df['total_visits'] = df[existing_visit_cols].sum(axis=1)
        logger.info("Created: total_visits")
    
    # --- 2. Medication change flag ---
    if 'change' in df.columns:
        df['medication_change'] = (df['change'] == 'Ch').astype(int)
        logger.info("Created: medication_change")
    
    # --- 3. High lab procedures indicator ---
    if 'num_lab_procedures' in df.columns:
        threshold = df['num_lab_procedures'].quantile(0.75)
        df['high_lab_procedures'] = (df['num_lab_procedures'] > threshold).astype(int)
        logger.info(f"Created: high_lab_procedures (threshold={threshold})")
    
    # --- 4. Age numeric ---
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(AGE_MAP).fillna(55)
        logger.info("Created: age_numeric")
    
    # --- 5. Num medications binned ---
    if 'num_medications' in df.columns:
        df['num_medications_bin'] = pd.cut(
            df['num_medications'],
            bins=[0, 5, 10, 20, 50, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        ).astype(str)
        logger.info("Created: num_medications_bin")
    
    # --- 6. Encode medication columns to numeric ---
    for col in MEDICATION_COLS:
        if col in df.columns:
            # Map: 'No' → 0, any change → 1
            df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1)
    
    # --- 7. Total medications active ---
    existing_med_cols = [c for c in MEDICATION_COLS if c in df.columns]
    if existing_med_cols:
        df['total_medications_active'] = df[existing_med_cols].sum(axis=1)
        logger.info("Created: total_medications_active")
    
    # --- 8. Number of diagnoses filled ---
    diag_cat_cols = [c for c in df.columns if c.startswith('diag_') and c.endswith('_category')]
    if diag_cat_cols:
        df['num_diagnoses'] = (df[diag_cat_cols] != 'Other').sum(axis=1)
        logger.info("Created: num_diagnoses")
    
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns encoded DataFrame and dict of {column: LabelEncoder}.
    """
    logger.info("Encoding categorical variables...")
    df = df.copy()
    encoders = {}
    
    # Label encode specified columns
    for col in LABEL_ENCODE_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  Label-encoded: {col} ({len(le.classes_)} classes)")
    
    # Encode num_medications_bin
    if 'num_medications_bin' in df.columns:
        le = LabelEncoder()
        df['num_medications_bin'] = le.fit_transform(df['num_medications_bin'].astype(str))
        encoders['num_medications_bin'] = le
    
    # Drop any remaining object columns
    remaining_obj = df.select_dtypes(include=['object']).columns.tolist()
    if remaining_obj:
        logger.warning(f"Dropping remaining object columns: {remaining_obj}")
        df = df.drop(columns=remaining_obj)
    
    logger.info(f"Encoding complete. Final shape: {df.shape}")
    return df, encoders


if __name__ == '__main__':
    from pipeline.ingest import load_dataset
    from pipeline.clean import clean_data
    
    df = load_dataset()
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    
    print(f"\nFinal shape: {df.shape}")
    print(f"\nFeature columns:\n{list(df.columns)}")
    print(f"\nEncoders: {list(encoders.keys())}")
