"""
Database Seeder
Populates DuckDB tables from processed data and predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.schema import get_connection, create_tables, get_table_counts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
POWERBI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'powerbi', 'sample_data')

AGE_BINS = [0, 20, 40, 60, 80, 100]
AGE_LABELS = ['0-20', '21-40', '41-60', '61-80', '81-100']


def seed_from_processed_data():
    """Seed DB from processed data CSV (pre-encoded readable version)."""
    readable_path = os.path.join(DATA_DIR, 'processed_patients_readable.csv')
    encoded_path = os.path.join(DATA_DIR, 'processed_patients.csv')

    if os.path.exists(readable_path):
        df = pd.read_csv(readable_path)
        logger.info(f"Loaded readable data: {df.shape}")
    elif os.path.exists(encoded_path):
        df = pd.read_csv(encoded_path)
        logger.info(f"Loaded encoded data: {df.shape}")
    else:
        logger.warning("No processed data found. Generating sample data for demo...")
        seed_sample_data()
        return

    create_tables()
    con = get_connection()

    # Clear existing data
    con.execute("DELETE FROM model_predictions")
    con.execute("DELETE FROM patient_visits")
    con.execute("DELETE FROM patients")

    # Assign patient IDs
    df = df.reset_index()
    df['patient_id'] = df.index + 1

    # Add age_group
    if 'age_numeric' in df.columns:
        df['age_group'] = pd.cut(df['age_numeric'], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    else:
        df['age_group'] = 'Unknown'

    # --- Insert patients ---
    patients_cols = {
        'patient_id': 'patient_id',
        'age_numeric': 'age',
        'age_group': 'age_group',
        'gender': 'gender',
        'race': 'race',
        'num_medications': 'num_medications',
        'num_lab_procedures': 'num_lab_procedures',
        'num_procedures': 'num_procedures',
        'number_diagnoses': 'number_diagnoses',
        'time_in_hospital': 'time_in_hospital',
    }

    patients_df = pd.DataFrame()
    for src, dst in patients_cols.items():
        if src in df.columns:
            patients_df[dst] = df[src]
        else:
            patients_df[dst] = 0

    # Add optional columns
    for col in ['diag_1_category', 'diag_2_category', 'diag_3_category']:
        patients_df[col] = df[col] if col in df.columns else 'Other'

    patients_df['diabetes_med'] = df.get('diabetesMed', 'No')
    patients_df['insulin'] = df.get('insulin', 'No')
    patients_df['a1c_result'] = df.get('A1Cresult', 'None')

    patient_cols = ', '.join(patients_df.columns)
    con.execute(f"INSERT INTO patients ({patient_cols}) SELECT {patient_cols} FROM patients_df")
    logger.info(f"Inserted {len(patients_df)} patients")

    # --- Insert visits ---
    visits_df = pd.DataFrame({
        'visit_id': df['patient_id'],
        'patient_id': df['patient_id'],
        'admission_type_id': df.get('admission_type_id', 1),
        'discharge_disposition_id': df.get('discharge_disposition_id', 1),
        'admission_source_id': df.get('admission_source_id', 1),
        'number_outpatient': df.get('number_outpatient', 0),
        'number_emergency': df.get('number_emergency', 0),
        'number_inpatient': df.get('number_inpatient', 0),
        'total_visits': df.get('total_visits', 0),
        'medication_change': df.get('medication_change', False).astype(bool),
        'high_lab_procedures': df.get('high_lab_procedures', False).astype(bool),
    })

    visit_cols = ', '.join(visits_df.columns)
    con.execute(f"INSERT INTO patient_visits ({visit_cols}) SELECT {visit_cols} FROM visits_df")
    logger.info(f"Inserted {len(visits_df)} patient visits")

    # --- Generate predictions (simulate model output) ---
    np.random.seed(42)
    target_col = 'readmitted_binary' if 'readmitted_binary' in df.columns else None

    if target_col:
        # Generate risk scores correlated with actual readmission
        base = df[target_col].values * 0.4 + np.random.uniform(0, 0.5, len(df))
        risk_scores = np.clip(base, 0.01, 0.99)
    else:
        risk_scores = np.random.uniform(0.05, 0.95, len(df))

    risk_levels = pd.cut(
        risk_scores,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    predictions_df = pd.DataFrame({
        'prediction_id': df['patient_id'],
        'patient_id': df['patient_id'],
        'risk_score': np.round(risk_scores, 4),
        'risk_percentage': np.round(risk_scores * 100, 1),
        'risk_level': risk_levels.astype(str),
        'actual_readmitted': df.get(target_col, 0) if target_col else 0,
        'top_factors': 'number_inpatient,num_medications,age_numeric,number_diagnoses,time_in_hospital',
    })

    pred_cols = ', '.join(predictions_df.columns)
    con.execute(f"INSERT INTO model_predictions ({pred_cols}) SELECT {pred_cols} FROM predictions_df")
    logger.info(f"Inserted {len(predictions_df)} predictions")

    con.close()

    # --- Export for Power BI ---
    export_for_powerbi(patients_df, predictions_df)

    counts = get_table_counts()
    logger.info(f"\n✅ Database seeded successfully!")
    for table, count in counts.items():
        logger.info(f"  {table}: {count} rows")


def seed_sample_data():
    """Generate and seed sample data for demo when no processed data is available."""
    np.random.seed(42)
    n = 5000

    create_tables()
    con = get_connection()
    con.execute("DELETE FROM model_predictions")
    con.execute("DELETE FROM patient_visits")
    con.execute("DELETE FROM patients")

    ages = np.random.randint(18, 95, n)
    age_groups = pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    genders = np.random.choice(['Male', 'Female'], n)
    races = np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n,
                             p=[0.55, 0.2, 0.12, 0.08, 0.05])
    diag_categories = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                       'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']

    patients_df = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': ages,
        'age_group': age_groups,
        'gender': genders,
        'race': races,
        'num_medications': np.random.randint(1, 30, n),
        'num_lab_procedures': np.random.randint(10, 80, n),
        'num_procedures': np.random.randint(0, 6, n),
        'number_diagnoses': np.random.randint(1, 15, n),
        'time_in_hospital': np.random.randint(1, 14, n),
        'diag_1_category': np.random.choice(diag_categories, n),
        'diag_2_category': np.random.choice(diag_categories, n),
        'diag_3_category': np.random.choice(diag_categories, n),
        'diabetes_med': np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3]),
        'insulin': np.random.choice(['No', 'Up', 'Down', 'Steady'], n, p=[0.4, 0.25, 0.1, 0.25]),
        'a1c_result': np.random.choice(['None', 'Norm', '>7', '>8'], n, p=[0.6, 0.15, 0.15, 0.1]),
    })

    patient_cols = ', '.join(patients_df.columns)
    con.execute(f"INSERT INTO patients ({patient_cols}) SELECT {patient_cols} FROM patients_df")

    # Visits
    inpatient = np.random.randint(0, 5, n)
    outpatient = np.random.randint(0, 10, n)
    emergency = np.random.randint(0, 4, n)

    visits_df = pd.DataFrame({
        'visit_id': range(1, n + 1),
        'patient_id': range(1, n + 1),
        'admission_type_id': np.random.randint(1, 8, n),
        'discharge_disposition_id': np.random.randint(1, 25, n),
        'admission_source_id': np.random.randint(1, 20, n),
        'number_outpatient': outpatient,
        'number_emergency': emergency,
        'number_inpatient': inpatient,
        'total_visits': inpatient + outpatient + emergency,
        'medication_change': np.random.choice([True, False], n, p=[0.3, 0.7]),
        'high_lab_procedures': np.random.choice([True, False], n, p=[0.25, 0.75]),
    })

    visit_cols = ', '.join(visits_df.columns)
    con.execute(f"INSERT INTO patient_visits ({visit_cols}) SELECT {visit_cols} FROM visits_df")

    # Predictions
    risk_scores = np.clip(
        0.1 + ages / 200 + inpatient * 0.1 + np.random.normal(0, 0.15, n),
        0.01, 0.99
    )
    risk_levels = pd.cut(risk_scores, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

    predictions_df = pd.DataFrame({
        'prediction_id': range(1, n + 1),
        'patient_id': range(1, n + 1),
        'risk_score': np.round(risk_scores, 4),
        'risk_percentage': np.round(risk_scores * 100, 1),
        'risk_level': risk_levels.astype(str),
        'actual_readmitted': (risk_scores > 0.5).astype(int),
        'top_factors': 'number_inpatient,num_medications,age,number_diagnoses,time_in_hospital',
    })

    pred_cols = ', '.join(predictions_df.columns)
    con.execute(f"INSERT INTO model_predictions ({pred_cols}) SELECT {pred_cols} FROM predictions_df")
    con.close()

    export_for_powerbi(patients_df, predictions_df)

    counts = get_table_counts()
    logger.info(f"\n✅ Sample database seeded!")
    for table, count in counts.items():
        logger.info(f"  {table}: {count} rows")


def export_for_powerbi(patients_df, predictions_df):
    """Export CSVs for Power BI."""
    os.makedirs(POWERBI_DIR, exist_ok=True)

    # Patient risk summary
    merged = patients_df.merge(predictions_df, on='patient_id', how='left')
    summary_cols = ['patient_id', 'age', 'age_group', 'gender', 'race',
                    'num_medications', 'num_lab_procedures', 'time_in_hospital',
                    'diag_1_category', 'risk_score', 'risk_percentage', 'risk_level']
    available_cols = [c for c in summary_cols if c in merged.columns]
    merged[available_cols].to_csv(
        os.path.join(POWERBI_DIR, 'patient_risk_summary.csv'), index=False
    )

    # Risk by demographics
    if 'age_group' in merged.columns:
        risk_by_demo = merged.groupby(['age_group', 'gender'], observed=True).agg(
            avg_risk=('risk_score', 'mean'),
            patient_count=('patient_id', 'count'),
            high_risk_count=('risk_level', lambda x: (x == 'High').sum())
        ).reset_index()
        risk_by_demo.to_csv(
            os.path.join(POWERBI_DIR, 'risk_by_demographics.csv'), index=False
        )

    # Readmission trends (simulated monthly)
    months = pd.date_range('2024-01-01', periods=12, freq='MS')
    trends = pd.DataFrame({
        'month': months,
        'total_patients': np.random.randint(300, 800, 12),
        'readmissions': np.random.randint(30, 120, 12),
    })
    trends['readmission_rate'] = (trends['readmissions'] / trends['total_patients'] * 100).round(1)
    trends.to_csv(os.path.join(POWERBI_DIR, 'readmission_trends.csv'), index=False)

    logger.info(f"✅ Power BI CSVs exported to {POWERBI_DIR}")


if __name__ == '__main__':
    seed_from_processed_data()
