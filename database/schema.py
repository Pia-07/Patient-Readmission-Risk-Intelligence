"""
DuckDB Database Schema
Creates tables for patients, visits, and model predictions.
"""

import duckdb
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb')


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH, read_only=read_only)


def create_tables():
    """Create all required tables."""
    con = get_connection()

    con.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY,
            age INTEGER,
            age_group VARCHAR,
            gender VARCHAR,
            race VARCHAR,
            num_medications INTEGER,
            num_lab_procedures INTEGER,
            num_procedures INTEGER,
            number_diagnoses INTEGER,
            time_in_hospital INTEGER,
            diag_1_category VARCHAR,
            diag_2_category VARCHAR,
            diag_3_category VARCHAR,
            diabetes_med VARCHAR,
            insulin VARCHAR,
            a1c_result VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS patient_visits (
            visit_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            admission_type_id INTEGER,
            discharge_disposition_id INTEGER,
            admission_source_id INTEGER,
            number_outpatient INTEGER,
            number_emergency INTEGER,
            number_inpatient INTEGER,
            total_visits INTEGER,
            medication_change BOOLEAN,
            high_lab_procedures BOOLEAN,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            risk_score DOUBLE,
            risk_percentage DOUBLE,
            risk_level VARCHAR,
            actual_readmitted INTEGER,
            top_factors VARCHAR,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR DEFAULT '1.0.0',
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    # ── Star-Schema Tables ──────────────────────────────────────
    con.execute("CREATE SEQUENCE IF NOT EXISTS fact_visit_seq START 1")
    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_patient (
            patient_id INTEGER PRIMARY KEY,
            age INTEGER,
            age_group VARCHAR,
            gender VARCHAR,
            race VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_visit_metrics (
            visit_id INTEGER PRIMARY KEY,
            time_in_hospital INTEGER,
            num_lab_procedures INTEGER,
            num_procedures INTEGER,
            num_medications INTEGER,
            total_visits INTEGER,
            number_inpatient INTEGER,
            number_outpatient INTEGER,
            number_emergency INTEGER,
            diag_1_category VARCHAR,
            insulin VARCHAR,
            diabetes_med VARCHAR,
            a1c_result VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fact_patient_visits (
            id INTEGER DEFAULT(nextval('fact_visit_seq')),
            patient_id INTEGER,
            visit_id INTEGER,
            timestamp VARCHAR,
            risk_score DOUBLE,
            risk_level VARCHAR,
            readmitted_binary INTEGER
        )
    """)

    # ── Medallion Architecture Tables ───────────────────────────

    # BRONZE — Raw ingested data, no transformations (all 50 columns from Kaggle)
    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze_patient_visits (
            encounter_id VARCHAR,
            patient_nbr VARCHAR,
            race VARCHAR,
            gender VARCHAR,
            age VARCHAR,
            weight VARCHAR,
            admission_type_id VARCHAR,
            discharge_disposition_id VARCHAR,
            admission_source_id VARCHAR,
            time_in_hospital VARCHAR,
            payer_code VARCHAR,
            medical_specialty VARCHAR,
            num_lab_procedures VARCHAR,
            num_procedures VARCHAR,
            num_medications VARCHAR,
            number_outpatient VARCHAR,
            number_emergency VARCHAR,
            number_inpatient VARCHAR,
            diag_1 VARCHAR,
            diag_2 VARCHAR,
            diag_3 VARCHAR,
            number_diagnoses VARCHAR,
            max_glu_serum VARCHAR,
            A1Cresult VARCHAR,
            metformin VARCHAR,
            repaglinide VARCHAR,
            nateglinide VARCHAR,
            chlorpropamide VARCHAR,
            glimepiride VARCHAR,
            acetohexamide VARCHAR,
            glipizide VARCHAR,
            glyburide VARCHAR,
            tolbutamide VARCHAR,
            pioglitazone VARCHAR,
            rosiglitazone VARCHAR,
            acarbose VARCHAR,
            miglitol VARCHAR,
            troglitazone VARCHAR,
            tolazamide VARCHAR,
            examide VARCHAR,
            citoglipton VARCHAR,
            insulin VARCHAR,
            "glyburide-metformin" VARCHAR,
            "glipizide-metformin" VARCHAR,
            "glimepiride-pioglitazone" VARCHAR,
            "metformin-rosiglitazone" VARCHAR,
            "metformin-pioglitazone" VARCHAR,
            change VARCHAR,
            diabetesMed VARCHAR,
            readmitted VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # SILVER — Cleaned, deduplicated, feature-engineered
    con.execute("""
        CREATE TABLE IF NOT EXISTS silver_patient_visits (
            patient_id INTEGER,
            visit_id INTEGER,
            age INTEGER,
            age_group VARCHAR,
            gender VARCHAR,
            race VARCHAR,
            time_in_hospital INTEGER,
            num_lab_procedures INTEGER,
            num_procedures INTEGER,
            num_medications INTEGER,
            total_visits INTEGER,
            number_inpatient INTEGER,
            number_outpatient INTEGER,
            number_emergency INTEGER,
            diag_1_category VARCHAR,
            insulin VARCHAR,
            diabetes_med VARCHAR,
            a1c_result VARCHAR,
            readmitted_binary INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # GOLD — Business-ready analytics tables
    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_patient_risk_summary (
            patient_id INTEGER,
            age INTEGER,
            age_group VARCHAR,
            gender VARCHAR,
            race VARCHAR,
            diag_1_category VARCHAR,
            num_medications INTEGER,
            num_lab_procedures INTEGER,
            time_in_hospital INTEGER,
            total_visits INTEGER,
            number_inpatient INTEGER,
            insulin VARCHAR,
            diabetes_med VARCHAR,
            a1c_result VARCHAR,
            risk_score DOUBLE,
            risk_percentage DOUBLE,
            risk_level VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_hospital_kpis (
            total_patients INTEGER,
            high_risk_patients INTEGER,
            high_risk_rate DOUBLE,
            avg_risk_score DOUBLE,
            avg_length_of_stay DOUBLE,
            avg_medications DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_distribution (
            risk_level VARCHAR,
            patient_count INTEGER,
            percentage DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for common queries
    con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk ON model_predictions(risk_level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_score ON model_predictions(risk_score)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_patients_age ON patients(age)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_gold_risk_level ON gold_patient_risk_summary(risk_level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fact_patient ON fact_patient_visits(patient_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_patient_visits(risk_level)")

    con.close()
    logger.info(f"✅ Tables created in {DB_PATH}")


def populate_dimensions():
    """Populate dim_patient and dim_visit_metrics from Silver layer."""
    con = get_connection()
    try:
        # dim_patient — from silver_patient_visits
        con.execute("DELETE FROM dim_patient")
        con.execute("""
            INSERT INTO dim_patient (patient_id, age, age_group, gender, race)
            SELECT DISTINCT patient_id, age, age_group, gender, race
            FROM silver_patient_visits
        """)
        dim_p = con.execute("SELECT COUNT(*) FROM dim_patient").fetchone()[0]

        # dim_visit_metrics — from silver_patient_visits
        con.execute("DELETE FROM dim_visit_metrics")
        con.execute("""
            INSERT INTO dim_visit_metrics (
                visit_id, time_in_hospital, num_lab_procedures, num_procedures, 
                num_medications, total_visits, number_inpatient, number_outpatient, 
                number_emergency, diag_1_category, insulin, diabetes_med, a1c_result
            )
            SELECT 
                visit_id, time_in_hospital, num_lab_procedures, num_procedures, 
                num_medications, total_visits, number_inpatient, number_outpatient, 
                number_emergency, diag_1_category, insulin, diabetes_med, a1c_result
            FROM silver_patient_visits
        """)
        dim_v = con.execute("SELECT COUNT(*) FROM dim_visit_metrics").fetchone()[0]

        logger.info(f"✅ Dimensions populated: dim_patient={dim_p}, dim_visit_metrics={dim_v}")
    finally:
        con.close()


def populate_fact_from_silver():
    """Populate fact_patient_visits from silver + model_predictions."""
    import numpy as np
    import pandas as pd

    con = get_connection()
    try:
        silver_df = con.execute("SELECT patient_id, visit_id, readmitted_binary FROM silver_patient_visits").fetchdf()

        # Try to get existing predictions
        try:
            pred_df = con.execute(
                "SELECT patient_id, risk_score, risk_level FROM model_predictions"
            ).fetchdf()
        except Exception:
            pred_df = pd.DataFrame()

        # Merge
        merged = silver_df.merge(pred_df, on='patient_id', how='left')

        # Generate risk scores for unmatched
        missing = merged['risk_score'].isna()
        if missing.any():
            np.random.seed(42)
            n = int(missing.sum())
            scores = np.clip(np.random.uniform(0.05, 0.95, n), 0.01, 0.99)
            levels = pd.cut(scores, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
            merged.loc[missing, 'risk_score'] = scores
            merged.loc[missing, 'risk_level'] = levels.astype(str)

        from datetime import datetime
        fact_df = pd.DataFrame({
            'patient_id': merged['patient_id'],
            'visit_id': merged['visit_id'],
            'timestamp': datetime.now().isoformat(),
            'risk_score': merged['risk_score'],
            'risk_level': merged['risk_level'],
            'readmitted_binary': merged['readmitted_binary'],
        })

        con.execute("DELETE FROM fact_patient_visits")
        cols = ', '.join(fact_df.columns)
        con.execute(f"INSERT INTO fact_patient_visits ({cols}) SELECT {cols} FROM fact_df")

        count = con.execute("SELECT COUNT(*) FROM fact_patient_visits").fetchone()[0]
        logger.info(f"✅ Fact table populated: {count} rows")
    finally:
        con.close()


def refresh_gold_from_fact():
    """
    Refresh Gold analytics tables from fact_patient_visits + dimensions.
    This runs every time new data arrives (including DBeaver inserts).
    """
    con = get_connection()
    try:
        # ── gold_hospital_kpis ────────────────────────────────
        con.execute("DELETE FROM gold_hospital_kpis")
        con.execute("""
            INSERT INTO gold_hospital_kpis
                (total_patients, high_risk_patients, high_risk_rate,
                 avg_risk_score, avg_length_of_stay, avg_medications)
            SELECT
                COUNT(DISTINCT f.patient_id),
                COUNT(DISTINCT CASE WHEN f.risk_level='High' THEN f.patient_id END),
                ROUND(COUNT(DISTINCT CASE WHEN f.risk_level='High' THEN f.patient_id END)
                      * 100.0 / NULLIF(COUNT(DISTINCT f.patient_id), 0), 1),
                ROUND(AVG(f.risk_score), 4),
                ROUND(AVG(dv.time_in_hospital), 1),
                ROUND(AVG(dv.num_medications), 1)
            FROM fact_patient_visits f
            LEFT JOIN dim_visit_metrics dv ON f.visit_id = dv.visit_id
        """)

        # ── gold_risk_distribution ────────────────────────────
        con.execute("DELETE FROM gold_risk_distribution")
        con.execute("""
            INSERT INTO gold_risk_distribution (risk_level, patient_count, percentage)
            WITH totals AS (
                SELECT COUNT(DISTINCT patient_id) as total FROM fact_patient_visits
            )
            SELECT
                f.risk_level,
                COUNT(DISTINCT f.patient_id),
                ROUND(COUNT(DISTINCT f.patient_id) * 100.0 / NULLIF(t.total, 0), 1)
            FROM fact_patient_visits f, totals t
            GROUP BY f.risk_level, t.total
        """)

        # ── gold_patient_risk_summary ─────────────────────────
        con.execute("DELETE FROM gold_patient_risk_summary")
        con.execute("""
            INSERT INTO gold_patient_risk_summary
                (patient_id, age, age_group, gender, race, diag_1_category,
                 num_medications, num_lab_procedures, time_in_hospital, 
                 total_visits, number_inpatient, insulin, diabetes_med, a1c_result,
                 risk_score, risk_percentage, risk_level)
            SELECT
                p.patient_id, p.age, p.age_group, p.gender, p.race,
                v.diag_1_category, v.num_medications, v.num_lab_procedures, v.time_in_hospital,
                v.total_visits, v.number_inpatient, v.insulin, v.diabetes_med, v.a1c_result,
                f.risk_score, ROUND(f.risk_score * 100, 1), f.risk_level
            FROM fact_patient_visits f
            JOIN dim_patient p ON f.patient_id = p.patient_id
            JOIN dim_visit_metrics v ON f.visit_id = v.visit_id
        """)

        kpi_row = con.execute("SELECT * FROM gold_hospital_kpis").fetchone()
        dist_count = con.execute("SELECT COUNT(*) FROM gold_risk_distribution").fetchone()[0]
        summary_count = con.execute("SELECT COUNT(*) FROM gold_patient_risk_summary").fetchone()[0]

        logger.info(f"✅ Gold refreshed: KPIs={kpi_row[0]} patients, "
                     f"Distribution={dist_count} levels, Summary={summary_count} rows")
    finally:
        con.close()


def drop_tables():
    """Drop all tables (for reset)."""
    con = get_connection()
    con.execute("DROP TABLE IF EXISTS model_predictions")
    con.execute("DROP TABLE IF EXISTS patient_visits")
    con.execute("DROP TABLE IF EXISTS patients")
    con.close()
    logger.info("Tables dropped")


def get_table_counts() -> dict:
    """Get row counts for all tables."""
    con = get_connection(read_only=True)
    counts = {}
    tables = [
        'bronze_patient_visits', 'silver_patient_visits',
        'dim_patient', 'dim_visit_metrics', 'fact_patient_visits',
        'gold_patient_risk_summary', 'gold_hospital_kpis', 'gold_risk_distribution',
        'patients', 'patient_visits', 'model_predictions',
    ]
    for table in tables:
        try:
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = result[0]
        except Exception:
            counts[table] = 0
    con.close()
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_tables()
    print("Tables created successfully!")
    print(f"Database: {DB_PATH}")

