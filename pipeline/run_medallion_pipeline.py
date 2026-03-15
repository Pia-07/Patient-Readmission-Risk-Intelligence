"""
Medallion Pipeline Orchestrator
Runs the full Bronze → Silver → Dimensions → Fact → Gold data pipeline.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.schema import (
    create_tables,
    populate_dimensions,
    populate_fact_from_silver,
    refresh_gold_from_fact,
    get_table_counts,
)
from pipeline.bronze_ingest import ingest_kaggle_data
from pipeline.silver_transform import transform_bronze_to_silver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_medallion_pipeline():
    """
    Execute the full Medallion Architecture pipeline:
      Step 1: Bronze — Raw data ingestion
      Step 2: Silver — Cleaning & feature engineering
      Step 3: Dimensions — Populate dim_patient, dim_visit_metrics
      Step 4: Fact — Populate fact_patient_visits
      Step 5: Gold — Business analytics from fact + dimensions
    """
    start = time.time()

    logger.info("=" * 70)
    logger.info("  MEDALLION ARCHITECTURE PIPELINE")
    logger.info("  Bronze → Silver → Dimensions → Fact → Gold")
    logger.info("=" * 70)

    # Ensure all tables exist
    create_tables()

    # ── Step 1: BRONZE LAYER ──────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  STEP 1: BRONZE LAYER — Raw Data Ingestion")
    logger.info("━" * 70)
    bronze_count = ingest_kaggle_data()
    logger.info(f"  → Bronze: {bronze_count} raw records ingested")

    # ── Step 2: SILVER LAYER ──────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  STEP 2: SILVER LAYER — Cleaning & Feature Engineering")
    logger.info("━" * 70)
    silver_count = transform_bronze_to_silver()
    logger.info(f"  → Silver: {silver_count} cleaned records")

    # ── Step 3: DIMENSION TABLES ──────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  STEP 3: DIMENSION TABLES — dim_patient, dim_visit_metrics")
    logger.info("━" * 70)
    populate_dimensions()

    # ── Step 4: FACT TABLE ────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  STEP 4: FACT TABLE — fact_patient_visits")
    logger.info("━" * 70)
    populate_fact_from_silver()

    # ── Step 5: GOLD LAYER ────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  STEP 5: GOLD LAYER — Business Analytics")
    logger.info("━" * 70)
    refresh_gold_from_fact()

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start
    counts = get_table_counts()
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    for table, count in counts.items():
        logger.info(f"  {table}: {count}")
    logger.info(f"  Time: {elapsed:.1f} seconds")
    logger.info("=" * 70)

    return counts


if __name__ == '__main__':
    results = run_medallion_pipeline()
