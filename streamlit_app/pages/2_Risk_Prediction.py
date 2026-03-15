"""
Page 2 — Patient Risk Prediction
Manual patient input → FastAPI /predict → risk gauge + factors.
"""

import streamlit as st
import os
import sys
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Prediction", page_icon="🔮", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.markdown("# 🔮 Patient Risk Prediction")
st.markdown("> Enter patient details to predict 30-day readmission risk")
st.markdown("---")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    with col2:
        num_medications = st.slider("Medications", 1, 40, 15)
        num_lab_procedures = st.slider("Lab Procedures", 1, 120, 45)
        num_procedures = st.slider("Procedures", 0, 6, 2)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 1)
        number_emergency = st.slider("Emergency Visits", 0, 10, 0)
    with col3:
        number_outpatient = st.slider("Outpatient Visits", 0, 20, 1)
        insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
        diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])
        change = st.selectbox("Medication Changed", ["No", "Ch"])
        diag_1 = st.selectbox("Primary Diagnosis", [
            "Circulatory", "Respiratory", "Digestive", "Diabetes",
            "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
        ])

    submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

if submitted:
    payload = {
        "age": age, "gender": gender, "race": race,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": num_diagnoses,
        "max_glu_serum": "None", "A1Cresult": "None",
        "change": change, "diabetesMed": diabetesMed,
        "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "diag_1_category": diag_1, "diag_2_category": "Other",
        "diag_3_category": "Other", "insulin": insulin, "metformin": "No"
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result = resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        result = None

    if result and 'risk_score' in result:
        st.markdown("---")

        # ── Risk Gauge ──
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            score = result['risk_score']
            color = '#48bb78' if score < 0.3 else ('#ecc94b' if score < 0.6 else '#f56565')
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['risk_percentage'],
                title={'text': "Readmission Risk", 'font': {'size': 20}},
                number={'suffix': '%', 'font': {'size': 48}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': '#c6f6d5'},
                        {'range': [30, 60], 'color': '#fefcbf'},
                        {'range': [60, 100], 'color': '#fed7d7'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 4},
                        'thickness': 0.75, 'value': result['risk_percentage']
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            level = result['risk_level']
            emoji = {'Low': '✅', 'Medium': '⚡', 'High': '⚠️'}.get(level, '❓')
            bg = {'Low': '#c6f6d5', 'Medium': '#fefcbf', 'High': '#fed7d7'}.get(level, '#eee')
            st.markdown(f"""
            <div style="background:{bg}; padding:20px; border-radius:12px;
                        text-align:center; margin-top:20px;">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="font-size:1.5rem; font-weight:700;">{level} Risk</div>
                <div style="font-size:0.9rem; opacity:0.7;">{result['risk_percentage']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("#### 📋 Recommendation")
            st.info(result.get('recommendation', 'No recommendation available'))

        # ── Feature Importance ──
        st.markdown("### 🧬 Top Risk Factors")
        factors = result.get('top_risk_factors', [])
        if factors:
            import pandas as pd
            fdf = pd.DataFrame(factors)
            fig = go.Figure(go.Bar(
                x=fdf['importance'],
                y=fdf['display_name'],
                orientation='h',
                marker_color='#2d4a7a'
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Importance",
                yaxis=dict(autorange="reversed"),
                font=dict(family="Inter")
            )
            st.plotly_chart(fig, use_container_width=True)
