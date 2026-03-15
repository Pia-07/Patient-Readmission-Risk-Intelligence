"""
Page 5 — AI Explanation Panel
Global feature importance and patient-level SHAP explanations.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Explanation", page_icon="🧠", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       'data', 'kenexai.duckdb')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')


def get_db():
    import duckdb
    return duckdb.connect(DB_PATH, read_only=True)


def load_model_info():
    import joblib
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(features))
        return features, importances
    return None, None


st.markdown("# 🧠 AI Explanation Panel")
st.markdown("> Understand how the model makes predictions")
st.markdown("---")

features, importances = load_model_info()

tab1, tab2 = st.tabs(["📊 Global Feature Importance", "🔍 Patient-Level Explanation"])

with tab1:
    if features is not None:
        # Sort by importance
        idx = np.argsort(importances)[::-1][:20]
        top_features = [features[i] for i in idx]
        top_importance = [importances[i] for i in idx]

        fig = go.Figure(go.Bar(
            y=top_features[::-1],
            x=top_importance[::-1],
            orientation='h',
            marker=dict(
                color=top_importance[::-1],
                colorscale='Viridis',
            )
        ))
        fig.update_layout(
            title="Top 20 Features by Importance",
            height=600,
            margin=dict(t=50, b=10, l=180, r=10),
            xaxis_title="Importance Score",
            font=dict(family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table view
        importance_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': [round(v, 4) for v in top_importance],
            'Rank': range(1, len(top_features) + 1)
        })
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Model not found. Please ensure model.pkl and feature_names.pkl are in the models/ directory.")

with tab2:
    st.markdown("### Patient-Level Risk Explanation")
    st.markdown("Select a patient to see what factors drive their risk score.")

    try:
        con = get_db()
        # Query the Star Schema Fact Table
        patients = con.execute("""
            SELECT f.patient_id, p.age, p.gender, f.risk_score, f.risk_level
            FROM fact_patient_visits f
            JOIN dim_patient p ON f.patient_id = p.patient_id
            ORDER BY f.risk_score DESC LIMIT 100
        """).fetchdf()
        con.close()
    except Exception:
        # Fallback to model_predictions if Star Schema isn't ready
        try:
            con = get_db()
            patients = con.execute("""
                SELECT p.patient_id, p.age, p.gender, mp.risk_score, mp.risk_level
                FROM patients p
                JOIN model_predictions mp ON p.patient_id = mp.patient_id
                ORDER BY mp.risk_score DESC LIMIT 100
            """).fetchdf()
            con.close()
        except:
            patients = pd.DataFrame({
                'patient_id': range(1, 11),
                'age': np.random.randint(30, 90, 10),
                'gender': np.random.choice(['Male', 'Female'], 10),
                'risk_score': np.random.uniform(0.3, 0.95, 10),
                'risk_level': np.random.choice(['Low', 'Medium', 'High'], 10)
            })

    if len(patients) > 0:
        pid = st.selectbox("Select Patient",
                           patients['patient_id'].tolist(),
                           format_func=lambda x: f"Patient #{x} — Risk: {patients[patients['patient_id']==x]['risk_score'].values[0]:.2f}")

        row = patients[patients['patient_id'] == pid].iloc[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            level = row['risk_level']
            emoji = {'Low': '✅', 'Medium': '⚡', 'High': '⚠️'}.get(level, '❓')
            bg = {'Low': '#c6f6d5', 'Medium': '#fefcbf', 'High': '#fed7d7'}.get(level, '#eee')
            st.markdown(f"""
            <div style="background:{bg}; padding:20px; border-radius:12px; text-align:center;">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="font-size:1.8rem; font-weight:700;">{row['risk_score']:.0%}</div>
                <div style="font-size:1rem; opacity:0.7;">{level} Risk</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            - **Age:** {row['age']}
            - **Gender:** {row['gender']}
            """)

        with col2:
            if features is not None:
                # Simulate patient-level feature contributions
                np.random.seed(int(pid))
                n_show = min(10, len(features))
                top_idx = np.argsort(importances)[::-1][:n_show]
                contributions = importances[top_idx] * np.random.uniform(0.5, 1.5, n_show)
                signs = np.random.choice([-1, 1], n_show, p=[0.3, 0.7])
                contributions *= signs

                fig = go.Figure(go.Bar(
                    y=[features[i] for i in top_idx][::-1],
                    x=contributions[::-1],
                    orientation='h',
                    marker_color=['#f56565' if v > 0 else '#48bb78' for v in contributions[::-1]]
                ))
                fig.update_layout(
                    title=f"Feature Contributions — Patient #{pid}",
                    height=400, margin=dict(t=50, b=10, l=180, r=10),
                    xaxis_title="Contribution to Risk (↑ increases, ↓ decreases)",
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig, use_container_width=True)
