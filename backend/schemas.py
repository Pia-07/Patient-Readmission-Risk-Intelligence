"""
Pydantic schemas for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PatientInput(BaseModel):
    """Input schema for patient prediction."""
    age: int = Field(..., ge=0, le=100, description="Patient age in years")
    gender: str = Field(..., description="Patient gender (Male/Female)")
    race: str = Field(default="Unknown", description="Patient race")
    time_in_hospital: int = Field(default=1, ge=1, description="Days in hospital")
    num_lab_procedures: int = Field(default=0, ge=0, description="Number of lab procedures")
    num_procedures: int = Field(default=0, ge=0, description="Number of procedures")
    num_medications: int = Field(default=0, ge=0, description="Number of medications")
    number_outpatient: int = Field(default=0, ge=0, description="Number of outpatient visits")
    number_emergency: int = Field(default=0, ge=0, description="Number of emergency visits")
    number_inpatient: int = Field(default=0, ge=0, description="Number of inpatient visits")
    number_diagnoses: int = Field(default=1, ge=1, description="Number of diagnoses")
    max_glu_serum: str = Field(default="None", description="Glucose serum test result")
    A1Cresult: str = Field(default="None", description="A1C test result")
    change: str = Field(default="No", description="Change in medication")
    diabetesMed: str = Field(default="No", description="Diabetes medication prescribed")
    admission_type_id: int = Field(default=1, description="Admission type")
    discharge_disposition_id: int = Field(default=1, description="Discharge disposition")
    admission_source_id: int = Field(default=1, description="Admission source")
    diag_1_category: str = Field(default="Other", description="Primary diagnosis category")
    diag_2_category: str = Field(default="Other", description="Secondary diagnosis category")
    diag_3_category: str = Field(default="Other", description="Tertiary diagnosis category")
    insulin: str = Field(default="No", description="Insulin prescribed")
    metformin: str = Field(default="No", description="Metformin prescribed")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "gender": "Female",
                "race": "Caucasian",
                "time_in_hospital": 5,
                "num_lab_procedures": 45,
                "num_procedures": 2,
                "num_medications": 15,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 2,
                "number_diagnoses": 7,
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "change": "Ch",
                "diabetesMed": "Yes",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "diag_1_category": "Diabetes",
                "diag_2_category": "Circulatory",
                "diag_3_category": "Other",
                "insulin": "Up",
                "metformin": "No"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    risk_score: float = Field(..., description="Readmission risk probability (0.0 - 1.0)")
    risk_percentage: float = Field(..., description="Risk as percentage (0 - 100)")
    risk_level: str = Field(..., description="Risk category: Low / Medium / High")
    top_risk_factors: list[dict] = Field(
        ..., description="Top contributing risk factors with importance scores"
    )
    recommendation: str = Field(..., description="Clinical recommendation based on risk level")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class HighRiskPatient(BaseModel):
    """Schema for high-risk patient record."""
    patient_id: int
    age: int
    risk_score: float
    risk_level: str
    total_visits: int
    num_medications: int
