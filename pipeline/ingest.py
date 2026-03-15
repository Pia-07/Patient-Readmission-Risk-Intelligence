"""
Data Ingestion Module
Loads the Diabetes 130-US Hospitals dataset from data/raw/
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')


def load_dataset(filename: str = 'diabetic_data.csv') -> pd.DataFrame:
    """Load the main diabetic dataset."""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        logger.info("Please download the 'Diabetes 130-US Hospitals' dataset from Kaggle")
        logger.info(f"and place '{filename}' in {RAW_DATA_DIR}/")
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath, na_values='?')
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes.value_counts()}")
    logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    return df


def load_ids_mapping(filename: str = 'IDs_mapping.csv') -> pd.DataFrame:
    """Load the IDs mapping file if available."""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"IDs mapping file not found at {filepath} — skipping.")
        return None
    
    logger.info(f"Loading IDs mapping from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"IDs mapping loaded: {df.shape[0]} rows")
    return df


if __name__ == '__main__':
    df = load_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['readmitted'].value_counts()}")
