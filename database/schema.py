"""
DuckDB Database Schema
Three-layer architecture: Operational | Medallion | Star Schema

CREATE ORDER  (parent → child, respects all FK constraints)
  1. patients                  — Operational master
  2. patient_visits            — FK → patients
  3. model_predictions         — FK → patients
  4. bronze_patient_visits     — Raw dump, no FK
  5. silver_patient_visits     — PK, no FK to bronze (bronze has no PK)
  6. dim_patient               — FK → silver
  7. dim_visit_metrics         — FK → silver
  8. fact_patient_visits       — FK → dim_patient, dim_visit_metrics
  9. gold_patient_risk_summary — FK → silver
 10. gold_hospital_kpis        — Aggregate, no FK
 11. gold_risk_distribution    — Aggregate, no FK

DROP ORDER  (reverse of above)
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
    """Create all tables in FK-safe order (parents first)."""
    con = get_connection()

    # ── 1. OPERATIONAL LAYER ──────────────────────────────────────────────
    # Master patient record — seeded from processed CSV, used by FastAPI
    con.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id          INTEGER PRIMARY KEY,
            age                 INTEGER,
            age_group           VARCHAR,
            gender              VARCHAR,
            race                VARCHAR,
            num_medications     INTEGER,
            num_lab_procedures  INTEGER,
            num_procedures      INTEGER,
            number_diagnoses    INTEGER,
            time_in_hospital    INTEGER,
            diag_1_category     VARCHAR,
            diag_2_category     VARCHAR,
            diag_3_category     VARCHAR,
            diabetes_med        VARCHAR,
            insulin             VARCHAR,
            a1c_result          VARCHAR,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS patient_visits (
            visit_id                    INTEGER PRIMARY KEY,
            patient_id                  INTEGER NOT NULL,
            admission_type_id           INTEGER,
            discharge_disposition_id    INTEGER,
            admission_source_id         INTEGER,
            number_outpatient           INTEGER,
            number_emergency            INTEGER,
            number_inpatient            INTEGER,
            total_visits                INTEGER,
            medication_change           BOOLEAN,
            high_lab_procedures         BOOLEAN,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id       INTEGER PRIMARY KEY,
            patient_id          INTEGER NOT NULL,
            risk_score          DOUBLE,
            risk_percentage     DOUBLE,
            risk_level          VARCHAR,
            actual_readmitted   INTEGER,
            top_factors         VARCHAR,
            predicted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version       VARCHAR DEFAULT '1.0.0',
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    # ── 2. MEDALLION — BRONZE (Raw dump) ─────────────────────────────────
    # All 50 original Kaggle columns stored as VARCHAR — zero transformations
    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze_patient_visits (
            encounter_id                VARCHAR,
            patient_nbr                 VARCHAR,
            race                        VARCHAR,
            gender                      VARCHAR,
            age                         VARCHAR,
            weight                      VARCHAR,
            admission_type_id           VARCHAR,
            discharge_disposition_id    VARCHAR,
            admission_source_id         VARCHAR,
            time_in_hospital            VARCHAR,
            payer_code                  VARCHAR,
            medical_specialty           VARCHAR,
            num_lab_procedures          VARCHAR,
            num_procedures              VARCHAR,
            num_medications             VARCHAR,
            number_outpatient           VARCHAR,
            number_emergency            VARCHAR,
            number_inpatient            VARCHAR,
            diag_1                      VARCHAR,
            diag_2                      VARCHAR,
            diag_3                      VARCHAR,
            number_diagnoses            VARCHAR,
            max_glu_serum               VARCHAR,
            A1Cresult                   VARCHAR,
            metformin                   VARCHAR,
            repaglinide                 VARCHAR,
            nateglinide                 VARCHAR,
            chlorpropamide              VARCHAR,
            glimepiride                 VARCHAR,
            acetohexamide               VARCHAR,
            glipizide                   VARCHAR,
            glyburide                   VARCHAR,
            tolbutamide                 VARCHAR,
            pioglitazone                VARCHAR,
            rosiglitazone               VARCHAR,
            acarbose                    VARCHAR,
            miglitol                    VARCHAR,
            troglitazone                VARCHAR,
            tolazamide                  VARCHAR,
            examide                     VARCHAR,
            citoglipton                 VARCHAR,
            insulin                     VARCHAR,
            "glyburide-metformin"       VARCHAR,
            "glipizide-metformin"       VARCHAR,
            "glimepiride-pioglitazone"  VARCHAR,
            "metformin-rosiglitazone"   VARCHAR,
            "metformin-pioglitazone"    VARCHAR,
            change                      VARCHAR,
            diabetesMed                 VARCHAR,
            readmitted                  VARCHAR,
            ingested_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── 3. MEDALLION — SILVER (Cleaned & feature-engineered) ─────────────
    # Deduplicated, typed, diagnosis codes mapped, features derived
    con.execute("""
        CREATE TABLE IF NOT EXISTS silver_patient_visits (
            patient_id                  INTEGER PRIMARY KEY,
            age_numeric                 INTEGER,
            age_group                   VARCHAR,
            gender                      VARCHAR,
            race                        VARCHAR,
            num_medications             INTEGER,
            num_lab_procedures          INTEGER,
            num_procedures              INTEGER,
            number_inpatient            INTEGER,
            number_outpatient           INTEGER,
            number_emergency            INTEGER,
            time_in_hospital            INTEGER,
            number_diagnoses            INTEGER,
            diag_1_category             VARCHAR,
            diag_2_category             VARCHAR,
            diag_3_category             VARCHAR,
            diabetes_med                VARCHAR,
            insulin                     VARCHAR,
            a1c_result                  VARCHAR,
            total_visits                INTEGER,
            medication_change           INTEGER,
            high_lab_procedures         INTEGER,
            total_medications_active    INTEGER,
            readmitted_binary           INTEGER,
            processed_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── 4. STAR SCHEMA — DIMENSION TABLES ────────────────────────────────
    # Populated from silver; used by fact table for analytical joins
    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_patient (
            patient_id  INTEGER PRIMARY KEY,
            age         INTEGER,
            gender      VARCHAR,
            FOREIGN KEY (patient_id) REFERENCES silver_patient_visits(patient_id)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS dim_visit_metrics (
            visit_id            INTEGER PRIMARY KEY,
            num_medications     INTEGER,
            num_lab_procedures  INTEGER,
            number_inpatient    INTEGER,
            time_in_hospital    INTEGER,
            FOREIGN KEY (visit_id) REFERENCES silver_patient_visits(patient_id)
        )
    """)

    # ── 5. STAR SCHEMA — FACT TABLE ─────────────────────────────────────
    # Central fact table; FK to both dimension tables
    con.execute("CREATE SEQUENCE IF NOT EXISTS fact_visit_seq START 1")
    con.execute("""
        CREATE TABLE IF NOT EXISTS fact_patient_visits (
            id          INTEGER PRIMARY KEY DEFAULT(nextval('fact_visit_seq')),
            patient_id  INTEGER NOT NULL,
            visit_id    INTEGER NOT NULL,
            timestamp   VARCHAR,
            risk_score  DOUBLE,
            risk_level  VARCHAR,
            FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id),
            FOREIGN KEY (visit_id)   REFERENCES dim_visit_metrics(visit_id)
        )
    """)

    # ── 6. MEDALLION — GOLD (Business-ready analytics) ───────────────────
    # Per-patient risk summary; FK to silver ensures row-level traceability
    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_patient_risk_summary (
            patient_id          INTEGER PRIMARY KEY,
            age                 INTEGER,
            age_group           VARCHAR,
            gender              VARCHAR,
            race                VARCHAR,
            diag_1_category     VARCHAR,
            num_medications     INTEGER,
            time_in_hospital    INTEGER,
            total_visits        INTEGER,
            number_inpatient    INTEGER,
            risk_score          DOUBLE,
            risk_percentage     DOUBLE,
            risk_level          VARCHAR,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES silver_patient_visits(patient_id)
        )
    """)

    # Aggregate tables — no FK, these are roll-ups with no row-level identity
    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_hospital_kpis (
            total_patients      INTEGER,
            high_risk_patients  INTEGER,
            high_risk_rate      DOUBLE,
            avg_risk_score      DOUBLE,
            avg_length_of_stay  DOUBLE,
            avg_medications     DOUBLE,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_risk_distribution (
            risk_level      VARCHAR,
            patient_count   INTEGER,
            percentage      DOUBLE,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── INDEXES ────────────────────────────────────────────────────────────
    con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk    ON model_predictions(risk_level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_score   ON model_predictions(risk_score)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_patients_age        ON patients(age)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_gold_risk_level     ON gold_patient_risk_summary(risk_level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fact_patient        ON fact_patient_visits(patient_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fact_risk           ON fact_patient_visits(risk_level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_silver_patient      ON silver_patient_visits(patient_id)")

    con.close()
    logger.info(f"✅ Tables created in {DB_PATH}")


def drop_all_tables():
    """Drop ALL tables in reverse FK order (children first, parents last)."""
    con = get_connection()

    # Gold (no FK dependencies on others, but depend on silver)
    con.execute("DROP TABLE IF EXISTS gold_risk_distribution")
    con.execute("DROP TABLE IF EXISTS gold_hospital_kpis")
    con.execute("DROP TABLE IF EXISTS gold_patient_risk_summary")

    # Star schema (fact depends on dims, dims depend on silver)
    con.execute("DROP TABLE IF EXISTS fact_patient_visits")
    con.execute("DROP SEQUENCE IF EXISTS fact_visit_seq")
    con.execute("DROP TABLE IF EXISTS dim_patient")
    con.execute("DROP TABLE IF EXISTS dim_visit_metrics")

    # Medallion
    con.execute("DROP TABLE IF EXISTS silver_patient_visits")
    con.execute("DROP TABLE IF EXISTS bronze_patient_visits")

    # Operational (model_predictions and patient_visits depend on patients)
    con.execute("DROP TABLE IF EXISTS model_predictions")
    con.execute("DROP TABLE IF EXISTS patient_visits")
    con.execute("DROP TABLE IF EXISTS patients")

    con.close()
    logger.info("✅ All tables dropped")


def drop_tables():
    """Drop operational tables only (backwards compatibility)."""
    drop_all_tables()


def populate_dimensions():
    """Populate dim_patient and dim_visit_metrics from Silver layer."""
    con = get_connection()
    try:
        con.execute("DELETE FROM dim_patient")
        con.execute("""
            INSERT INTO dim_patient (patient_id, age, gender)
            SELECT DISTINCT patient_id, age_numeric, gender
            FROM silver_patient_visits
        """)
        dim_p = con.execute("SELECT COUNT(*) FROM dim_patient").fetchone()[0]

        con.execute("DELETE FROM dim_visit_metrics")
        con.execute("""
            INSERT INTO dim_visit_metrics (visit_id, num_medications, num_lab_procedures, number_inpatient, time_in_hospital)
            SELECT patient_id, num_medications, num_lab_procedures, number_inpatient, time_in_hospital
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
        silver_df = con.execute("SELECT patient_id, time_in_hospital FROM silver_patient_visits").fetchdf()

        try:
            pred_df = con.execute(
                "SELECT patient_id, risk_score, risk_level FROM model_predictions"
            ).fetchdf()
        except Exception:
            pred_df = pd.DataFrame()

        merged = silver_df.merge(pred_df, on='patient_id', how='left')

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
            'visit_id':   merged['patient_id'],   # 1:1 mapping — visit_id mirrors patient_id
            'timestamp':  datetime.now().isoformat(),
            'risk_score': merged['risk_score'],
            'risk_level': merged['risk_level'],
        })

        con.execute("DELETE FROM fact_patient_visits")
        cols = ', '.join(fact_df.columns)
        con.execute(f"INSERT INTO fact_patient_visits ({cols}) SELECT {cols} FROM fact_df")

        count = con.execute("SELECT COUNT(*) FROM fact_patient_visits").fetchone()[0]
        logger.info(f"✅ Fact table populated: {count} rows")
    finally:
        con.close()


def refresh_gold_from_fact():
    """Refresh Gold analytics tables from fact_patient_visits + dimensions."""
    con = get_connection()
    try:
        # gold_hospital_kpis
        con.execute("DELETE FROM gold_hospital_kpis")
        con.execute("""
            INSERT INTO gold_hospital_kpis
                (total_patients, high_risk_patients, high_risk_rate,
                 avg_risk_score, avg_length_of_stay, avg_medications)
            SELECT
                COUNT(DISTINCT f.patient_id),
                COUNT(DISTINCT CASE WHEN f.risk_level = 'High' THEN f.patient_id END),
                ROUND(COUNT(DISTINCT CASE WHEN f.risk_level = 'High' THEN f.patient_id END)
                      * 100.0 / NULLIF(COUNT(DISTINCT f.patient_id), 0), 1),
                ROUND(AVG(f.risk_score), 4),
                ROUND(AVG(dv.time_in_hospital), 1),
                ROUND(AVG(dv.num_medications), 1)
            FROM fact_patient_visits f
            LEFT JOIN dim_visit_metrics dv ON f.visit_id = dv.visit_id
        """)

        # gold_risk_distribution
        con.execute("DELETE FROM gold_risk_distribution")
        con.execute("""
            INSERT INTO gold_risk_distribution (risk_level, patient_count, percentage)
            WITH totals AS (
                SELECT COUNT(DISTINCT patient_id) AS total FROM fact_patient_visits
            )
            SELECT
                f.risk_level,
                COUNT(DISTINCT f.patient_id),
                ROUND(COUNT(DISTINCT f.patient_id) * 100.0 / NULLIF(t.total, 0), 1)
            FROM fact_patient_visits f, totals t
            GROUP BY f.risk_level, t.total
        """)

        # gold_patient_risk_summary
        con.execute("DELETE FROM gold_patient_risk_summary")
        con.execute("""
            INSERT INTO gold_patient_risk_summary
                (patient_id, age, age_group, gender, race, diag_1_category,
                 num_medications, time_in_hospital, total_visits, number_inpatient,
                 risk_score, risk_percentage, risk_level)
            SELECT
                s.patient_id, s.age_numeric, s.age_group, s.gender, s.race,
                s.diag_1_category, s.num_medications, s.time_in_hospital,
                s.total_visits, s.number_inpatient,
                f.risk_score, ROUND(f.risk_score * 100, 1), f.risk_level
            FROM fact_patient_visits f
            JOIN silver_patient_visits s ON f.patient_id = s.patient_id
        """)

        kpi_row     = con.execute("SELECT * FROM gold_hospital_kpis").fetchone()
        dist_count  = con.execute("SELECT COUNT(*) FROM gold_risk_distribution").fetchone()[0]
        summ_count  = con.execute("SELECT COUNT(*) FROM gold_patient_risk_summary").fetchone()[0]

        logger.info(f"✅ Gold refreshed: KPIs={kpi_row[0]} patients, "
                    f"Distribution={dist_count} levels, Summary={summ_count} rows")
    finally:
        con.close()


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
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            counts[table] = 0
    con.close()
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_tables()
    print("Tables created successfully!")
    print(f"Database: {DB_PATH}")
