"""
Real-Time Pipeline Runner
Continuously refreshes Gold analytics from fact_patient_visits every 10 seconds.
This ensures DBeaver inserts and FastAPI predictions propagate to the dashboard.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.schema import (
    create_tables,
    refresh_gold_from_fact,
    get_table_counts,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 10  # seconds


def run_realtime_pipeline():
    """
    Continuously refresh Gold tables from fact_patient_visits.
    Any insert into fact_patient_visits (via FastAPI or DBeaver)
    will be reflected in Gold tables and then in Streamlit.
    """
    create_tables()
    logger.info("=" * 60)
    logger.info("  REAL-TIME PIPELINE STARTED")
    logger.info(f"  Refreshing Gold tables every {REFRESH_INTERVAL}s")
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info(f"\n🔄 Refresh cycle #{cycle}")

            # Refresh Gold from fact + dimensions
            refresh_gold_from_fact()

            # Log current counts
            counts = get_table_counts()
            logger.info(f"  fact_patient_visits: {counts.get('fact_patient_visits', 0)}")
            logger.info(f"  gold_hospital_kpis: {counts.get('gold_hospital_kpis', 0)}")
            logger.info(f"  gold_risk_distribution: {counts.get('gold_risk_distribution', 0)}")
            logger.info(f"  gold_patient_risk_summary: {counts.get('gold_patient_risk_summary', 0)}")

        except Exception as e:
            logger.error(f"Refresh error: {e}")

        time.sleep(REFRESH_INTERVAL)


if __name__ == '__main__':
    run_realtime_pipeline()
