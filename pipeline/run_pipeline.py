"""
Pipeline Orchestrator
Runs the full data pipeline: ingest → clean → feature engineer → save.
"""

import os
import sys
import logging
import joblib
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.ingest import load_dataset
from pipeline.clean import clean_data
from pipeline.features import engineer_features, encode_categoricals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')


def run_pipeline():
    """Execute the full data pipeline."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Step 1: Ingest
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)
    df = load_dataset()
    
    # Step 2: Clean
    logger.info("=" * 60)
    logger.info("STEP 2: Data Cleaning")
    logger.info("=" * 60)
    df = clean_data(df)
    
    # Step 3: Feature Engineering
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)
    df = engineer_features(df)
    
    # Save pre-encoded version for reference
    pre_encoded_path = os.path.join(PROCESSED_DIR, 'processed_patients_readable.csv')
    df.to_csv(pre_encoded_path, index=False)
    logger.info(f"Saved readable dataset to {pre_encoded_path}")
    
    # Step 4: Encode
    logger.info("=" * 60)
    logger.info("STEP 4: Categorical Encoding")
    logger.info("=" * 60)
    df_encoded, encoders = encode_categoricals(df)
    
    # Save encoded dataset
    encoded_path = os.path.join(PROCESSED_DIR, 'processed_patients.csv')
    df_encoded.to_csv(encoded_path, index=False)
    logger.info(f"Saved encoded dataset to {encoded_path}")
    
    # Save encoders
    encoders_path = os.path.join(PROCESSED_DIR, 'label_encoders.pkl')
    joblib.dump(encoders, encoders_path)
    logger.info(f"Saved label encoders to {encoders_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output shape: {df_encoded.shape}")
    logger.info(f"Target column: readmitted_binary")
    logger.info(f"Target distribution:\n{df_encoded['readmitted_binary'].value_counts()}")
    logger.info(f"Files saved in: {PROCESSED_DIR}")
    
    return df_encoded, encoders


if __name__ == '__main__':
    run_pipeline()
