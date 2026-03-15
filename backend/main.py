"""
FastAPI Backend — Patient Readmission Risk Prediction API
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.schemas import PatientInput, PredictionResponse, HealthResponse
from backend.model_loader import ModelPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    logger.info("Loading model...")
    predictor = ModelPredictor()
    if predictor.is_loaded:
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ Model not found — running in demo mode with dummy predictions")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Patient Readmission Risk Intelligence API",
    description=(
        "Predicts 30-day hospital readmission risk and provides "
        "explainable risk factors for clinical decision support."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded if predictor else False,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_readmission(patient: PatientInput):
    """
    Predict 30-day readmission risk for a patient.
    
    Returns:
    - risk_score (0.0 – 1.0)
    - risk_percentage (0 – 100)
    - risk_level (Low / Medium / High)
    - top_risk_factors with importance scores
    - clinical recommendation
    
    Also inserts the prediction into fact_patient_visits for real-time analytics.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        patient_data = patient.model_dump()
        result = predictor.predict(patient_data)

        # ── Insert into fact_patient_visits ──
        try:
            import duckdb
            from datetime import datetime
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb'
            )
            con = duckdb.connect(db_path)
            con.execute("""
                INSERT INTO fact_patient_visits (patient_id, visit_id, timestamp, risk_score, risk_level)
                VALUES (?, ?, ?, ?, ?)
            """, [
                patient_data.get('patient_id', 0),
                patient_data.get('patient_id', 0),
                datetime.now().isoformat(),
                result.get('risk_score', 0),
                result.get('risk_level', 'Unknown'),
            ])
            con.close()
            logger.info(f"  → Inserted prediction into fact_patient_visits")
        except Exception as e:
            logger.warning(f"Could not write to fact table: {e}")

        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(patients: list[PatientInput]):
    """Predict readmission risk for a batch of patients."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for patient in patients:
        try:
            result = predictor.predict(patient.model_dump())
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    return results


@app.get("/stats", tags=["Analytics"])
async def get_stats():
    """Return warehouse KPIs for dashboard consumption."""
    try:
        import duckdb
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb'
        )
        if os.path.exists(db_path):
            con = duckdb.connect(db_path, read_only=True)
            total = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            high = con.execute("SELECT COUNT(*) FROM model_predictions WHERE risk_level='High'").fetchone()[0]
            avg_risk = con.execute("SELECT AVG(risk_score) FROM model_predictions").fetchone()[0]
            avg_los = con.execute("SELECT AVG(time_in_hospital) FROM patients").fetchone()[0]
            rate = con.execute("SELECT AVG(CASE WHEN actual_readmitted=1 THEN 100.0 ELSE 0 END) FROM model_predictions").fetchone()[0]
            con.close()
            return {
                'total_patients': total,
                'high_risk_patients': high,
                'avg_risk_score': round(float(avg_risk or 0), 4),
                'avg_length_of_stay': round(float(avg_los or 0), 1),
                'readmission_rate': round(float(rate or 0), 1),
            }
    except Exception:
        pass
    return {'total_patients': 0, 'high_risk_patients': 0, 'avg_risk_score': 0, 'avg_length_of_stay': 0, 'readmission_rate': 0}


@app.get("/patients/high-risk", tags=["Patients"])
async def get_high_risk_patients():
    """
    Get patients with high risk scores from the database.
    Falls back to sample data if DB is empty.
    """
    try:
        import duckdb
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb'
        )
        
        if os.path.exists(db_path):
            con = duckdb.connect(db_path, read_only=True)
            try:
                result = con.execute("""
                    SELECT p.patient_id, p.age, p.gender,
                           mp.risk_score, mp.risk_level,
                           pv.total_visits, p.num_medications
                    FROM model_predictions mp
                    JOIN patients p ON mp.patient_id = p.patient_id
                    LEFT JOIN patient_visits pv ON p.patient_id = pv.patient_id
                    WHERE mp.risk_level = 'High'
                    ORDER BY mp.risk_score DESC
                    LIMIT 50
                """).fetchdf()
                con.close()
                return result.to_dict('records')
            except Exception:
                con.close()
        
        # Fallback sample data
        return _sample_high_risk_patients()
    except ImportError:
        return _sample_high_risk_patients()


def _sample_high_risk_patients():
    """Return sample data for demo."""
    import random
    random.seed(42)
    patients = []
    for i in range(15):
        patients.append({
            "patient_id": 10000 + i,
            "age": random.randint(55, 90),
            "gender": random.choice(["Male", "Female"]),
            "risk_score": round(random.uniform(0.6, 0.95), 3),
            "risk_level": "High",
            "total_visits": random.randint(3, 15),
            "num_medications": random.randint(10, 30),
        })
    return sorted(patients, key=lambda x: x['risk_score'], reverse=True)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
