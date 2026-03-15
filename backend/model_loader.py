"""
Model Loader
Loads the trained ML model and provides prediction + explainability utilities.
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

# Risk thresholds
RISK_THRESHOLDS = {'low': 0.3, 'high': 0.6}

# Feature name mapping for human-readable explanations
FEATURE_DISPLAY_NAMES = {
    'number_inpatient': 'Previous Inpatient Visits',
    'number_emergency': 'Emergency Visits',
    'number_outpatient': 'Outpatient Visits',
    'total_visits': 'Total Hospital Visits',
    'num_medications': 'Number of Medications',
    'num_lab_procedures': 'Lab Procedures Count',
    'num_procedures': 'Procedures Performed',
    'time_in_hospital': 'Length of Stay (Days)',
    'number_diagnoses': 'Number of Diagnoses',
    'age_numeric': 'Patient Age',
    'age': 'Age Group',
    'insulin': 'Insulin Prescribed',
    'medication_change': 'Medication Changed',
    'high_lab_procedures': 'High Lab Procedure Count',
    'total_medications_active': 'Active Medications',
    'diabetesMed': 'Diabetes Medication',
    'A1Cresult': 'A1C Test Result',
    'max_glu_serum': 'Glucose Serum Level',
    'diag_1_category': 'Primary Diagnosis',
    'diag_2_category': 'Secondary Diagnosis',
    'diag_3_category': 'Tertiary Diagnosis',
    'discharge_disposition_id': 'Discharge Disposition',
    'admission_type_id': 'Admission Type',
    'admission_source_id': 'Admission Source',
    'race': 'Race',
    'gender': 'Gender',
    'change': 'Medication Change Flag',
}


class ModelPredictor:
    """Handles model loading, prediction, and feature importance extraction."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.encoders = None
        self._load()

    def _load(self):
        """Load model, feature names, and label encoders."""
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
        encoders_path = os.path.join(DATA_DIR, 'label_encoders.pkl')

        # Load model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Prediction will use dummy model.")

        # Load feature names
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
        else:
            logger.warning("Feature names not found. Will infer from input.")

        # Load encoders
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
            logger.info(f"Label encoders loaded: {list(self.encoders.keys())}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess_input(self, patient_data: dict) -> pd.DataFrame:
        """Convert raw patient input to model-ready features."""
        df = pd.DataFrame([patient_data])

        # Derive features
        df['total_visits'] = (
            df.get('number_inpatient', 0) +
            df.get('number_outpatient', 0) +
            df.get('number_emergency', 0)
        )
        df['medication_change'] = (df.get('change', 'No') == 'Ch').astype(int)

        if 'num_lab_procedures' in df.columns:
            df['high_lab_procedures'] = (df['num_lab_procedures'] > 44).astype(int)
        else:
            df['high_lab_procedures'] = 0

        # Age mapping
        age_val = df['age'].values[0] if 'age' in df.columns else 55
        if isinstance(age_val, str):
            AGE_MAP = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            df['age_numeric'] = AGE_MAP.get(age_val, age_val)
        else:
            df['age_numeric'] = age_val

        # Medication encoding
        for col in ['insulin', 'metformin']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1)

        df['total_medications_active'] = 0
        df['num_medications_bin'] = 1  # default

        # Label encode categoricals
        if self.encoders:
            for col, le in self.encoders.items():
                if col in df.columns:
                    val = str(df[col].values[0])
                    if val in le.classes_:
                        df[col] = le.transform([val])[0]
                    else:
                        df[col] = 0  # unknown category

        # Align to model features
        if self.feature_names:
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[self.feature_names]

        # Ensure all numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        return df

    def predict(self, patient_data: dict) -> dict:
        """
        Make a prediction for a patient.
        Returns risk_score, risk_level, top_risk_factors, and recommendation.
        """
        if not self.is_loaded:
            # Return dummy prediction for demo
            return self._dummy_prediction(patient_data)

        features_df = self.preprocess_input(patient_data)
        proba = self.model.predict_proba(features_df)[0][1]

        risk_score = round(float(proba), 4)
        risk_percentage = round(risk_score * 100, 1)
        risk_level = self._get_risk_level(risk_score)
        top_factors = self._get_top_factors(features_df)
        recommendation = self._get_recommendation(risk_level)

        return {
            'risk_score': risk_score,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'top_risk_factors': top_factors,
            'recommendation': recommendation,
        }

    def _get_risk_level(self, score: float) -> str:
        if score >= RISK_THRESHOLDS['high']:
            return 'High'
        elif score >= RISK_THRESHOLDS['low']:
            return 'Medium'
        else:
            return 'Low'

    def _get_top_factors(self, features_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Extract top contributing features using model feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return [{'feature': 'Model type', 'importance': 0, 'display_name': 'N/A'}]

        names = self.feature_names or [f'feature_{i}' for i in range(len(importances))]
        top_idx = np.argsort(importances)[::-1][:top_n]

        return [
            {
                'feature': names[i],
                'importance': round(float(importances[i]), 4),
                'display_name': FEATURE_DISPLAY_NAMES.get(names[i], names[i]),
            }
            for i in top_idx
        ]

    def _get_recommendation(self, risk_level: str) -> str:
        recommendations = {
            'High': (
                "⚠️ HIGH RISK: Schedule follow-up within 7 days. "
                "Review medication plan, ensure patient education on self-management, "
                "and consider home health referral."
            ),
            'Medium': (
                "⚡ MODERATE RISK: Schedule follow-up within 14 days. "
                "Review discharge instructions and medication adherence plan."
            ),
            'Low': (
                "✅ LOW RISK: Standard discharge protocol. "
                "Schedule routine follow-up within 30 days."
            ),
        }
        return recommendations.get(risk_level, recommendations['Low'])

    def _dummy_prediction(self, patient_data: dict) -> dict:
        """Generate a plausible dummy prediction for demo when no model is loaded."""
        age = patient_data.get('age', 50)
        inpatient = patient_data.get('number_inpatient', 0)
        meds = patient_data.get('num_medications', 5)

        # Simple heuristic for demo
        base_risk = 0.1
        base_risk += min(age / 200, 0.2)
        base_risk += min(inpatient * 0.15, 0.3)
        base_risk += min(meds / 50, 0.15)
        risk_score = min(round(base_risk, 4), 0.95)

        risk_level = self._get_risk_level(risk_score)

        return {
            'risk_score': risk_score,
            'risk_percentage': round(risk_score * 100, 1),
            'risk_level': risk_level,
            'top_risk_factors': [
                {'feature': 'number_inpatient', 'importance': 0.25, 'display_name': 'Previous Inpatient Visits'},
                {'feature': 'num_medications', 'importance': 0.18, 'display_name': 'Number of Medications'},
                {'feature': 'age_numeric', 'importance': 0.15, 'display_name': 'Patient Age'},
                {'feature': 'number_diagnoses', 'importance': 0.12, 'display_name': 'Number of Diagnoses'},
                {'feature': 'time_in_hospital', 'importance': 0.10, 'display_name': 'Length of Stay (Days)'},
            ],
            'recommendation': self._get_recommendation(risk_level),
        }
