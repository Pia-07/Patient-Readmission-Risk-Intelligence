"""
Page 4 — Persona-Based Dashboards
Doctor Dashboard + Hospital Admin Dashboard
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Persona Dashboards", page_icon="👥", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       'data', 'kenexai.duckdb')


def get_db():
    import duckdb
    return duckdb.connect(DB_PATH, read_only=True)


def load_patients():
    try:
        con = get_db()
        # Try Gold layer first (Medallion Architecture)
        gold_count = con.execute("SELECT COUNT(*) FROM gold_patient_risk_summary").fetchone()[0]
        if gold_count > 0:
            df = con.execute("""
                SELECT patient_id, age, age_group, gender, race,
                       diag_1_category, num_medications, time_in_hospital,
                       total_visits, number_inpatient,
                       risk_score, risk_percentage, risk_level,
                       0 as number_diagnoses
                FROM gold_patient_risk_summary
            """).fetchdf()
            con.close()
            return df

        # Fallback to raw tables
        df = con.execute("""
            SELECT p.patient_id, p.age, p.age_group, p.gender, p.race,
                   p.num_medications, p.time_in_hospital, p.number_diagnoses,
                   p.diag_1_category,
                   mp.risk_score, mp.risk_percentage, mp.risk_level,
                   pv.total_visits, pv.number_inpatient
            FROM patients p
            JOIN model_predictions mp ON p.patient_id = mp.patient_id
            LEFT JOIN patient_visits pv ON p.patient_id = pv.patient_id
        """).fetchdf()
        con.close()
        return df
    except Exception:
        return _sample_data()


def _sample_data():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'patient_id': range(1, n+1),
        'age': np.random.randint(18, 95, n),
        'age_group': np.random.choice(['0-20','21-40','41-60','61-80','81-100'], n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['Caucasian','AfricanAmerican','Hispanic','Asian','Other'], n),
        'num_medications': np.random.randint(1, 30, n),
        'time_in_hospital': np.random.randint(1, 14, n),
        'number_diagnoses': np.random.randint(1, 15, n),
        'diag_1_category': np.random.choice(['Circulatory','Respiratory','Diabetes','Other'], n),
        'risk_score': np.random.uniform(0, 1, n),
        'risk_percentage': np.random.uniform(0, 100, n),
        'risk_level': np.random.choice(['Low','Medium','High'], n),
        'total_visits': np.random.randint(0, 20, n),
        'number_inpatient': np.random.randint(0, 5, n),
    })


# ── Page ──────────────────────────────────────────────────────
st.markdown("# 👥 Persona-Based Dashboards")
st.markdown("---")

persona = st.radio("Select Persona", ["🩺 Doctor Dashboard", "🏢 Hospital Admin Dashboard"],
                   horizontal=True)

df = load_patients()

# ═══════════════════════════════════════════════════════════════
#  DOCTOR DASHBOARD
# ═══════════════════════════════════════════════════════════════
if "Doctor" in persona:
    st.markdown("## 🩺 Doctor Dashboard")
    st.markdown("> Focus on individual patient risk and treatment planning")

    # ── High Risk Patient List ──
    st.markdown("### 🚨 High Risk Patients Requiring Attention")
    high_risk = df[df['risk_level'] == 'High'].sort_values('risk_score', ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Patients", f"{len(high_risk):,}")
    c2.metric("Avg Risk Score (High)", f"{high_risk['risk_score'].mean():.2f}" if len(high_risk) > 0 else "N/A")
    c3.metric("Max Risk Score", f"{high_risk['risk_score'].max():.2f}" if len(high_risk) > 0 else "N/A")

    st.dataframe(
        high_risk[['patient_id', 'age', 'gender', 'risk_score', 'risk_level',
                   'total_visits', 'num_medications', 'diag_1_category']].head(20),
        use_container_width=True
    )

    # ── Patient Risk Explanation ──
    st.markdown("### 🧬 Patient Risk Explanations")
    if len(high_risk) > 0:
        selected_id = st.selectbox("Select Patient", high_risk['patient_id'].head(20).tolist())
        patient = df[df['patient_id'] == selected_id].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Patient #{selected_id}** | Age: {patient['age']} | Gender: {patient['gender']}
            - Risk Score: **{patient['risk_score']:.2f}** ({patient['risk_level']})
            - Medications: {patient['num_medications']}
            - Previous Visits: {patient.get('total_visits', 'N/A')}
            - Inpatient Visits: {patient.get('number_inpatient', 'N/A')}
            - Primary Diagnosis: {patient.get('diag_1_category', 'N/A')}
            """)

        with col2:
            # Simple risk factor visualization
            factors = {
                'Inpatient Visits': min(patient.get('number_inpatient', 0) / 5, 1),
                'Medications': min(patient.get('num_medications', 0) / 30, 1),
                'Age': min(patient.get('age', 50) / 100, 1),
                'Hospital Stay': min(patient.get('time_in_hospital', 1) / 14, 1),
                'Total Visits': min(patient.get('total_visits', 0) / 20, 1),
            }
            fig = go.Figure(go.Bar(
                y=list(factors.keys()),
                x=list(factors.values()),
                orientation='h',
                marker_color=['#f56565' if v > 0.6 else '#ecc94b' if v > 0.3 else '#48bb78'
                              for v in factors.values()]
            ))
            fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10),
                              xaxis_title="Normalized Risk Factor",
                              font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)


#  ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════
else:
    st.markdown("## 🏢 Hospital Admin Dashboard")
    st.markdown("> Operational analytics and resource planning")

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{len(df):,}")
    c2.metric("High Risk Rate", f"{(df['risk_level']=='High').mean():.1%}")
    c3.metric("Avg Stay (days)", f"{df['time_in_hospital'].mean():.1f}")
    c4.metric("Avg Medications", f"{df['num_medications'].mean():.1f}")

    st.markdown("---")

    # ── Readmission Trends ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Readmission Risk by Age Group")
        age_risk = df.groupby('age_group', observed=True).agg(
            avg_risk=('risk_score', 'mean'),
            high_risk_pct=('risk_level', lambda x: (x == 'High').mean() * 100)
        ).reset_index()
        fig = px.bar(age_risk, x='age_group', y='high_risk_pct',
                     color='avg_risk', color_continuous_scale='RdYlGn_r',
                     labels={'high_risk_pct': 'High Risk %', 'age_group': 'Age Group'})
        fig.update_layout(height=350, margin=dict(t=10, b=10), font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Risk by Primary Diagnosis")
        diag_risk = df.groupby('diag_1_category', observed=True).agg(
            avg_risk=('risk_score', 'mean'),
            count=('patient_id', 'count')
        ).reset_index().sort_values('avg_risk', ascending=False)
        fig = px.bar(diag_risk, x='diag_1_category', y='avg_risk',
                     color='avg_risk',
                     color_continuous_scale='RdYlGn_r',
                     labels={'avg_risk': 'Avg Risk', 'diag_1_category': 'Diagnosis'})
        fig.update_layout(height=350, margin=dict(t=10, b=10), font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Resource Utilization ──
    st.markdown("### 🏥 Resource Utilization Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Length of Stay Distribution")
        fig = px.histogram(df, x='time_in_hospital', nbins=14,
                           color_discrete_sequence=['#2d4a7a'],
                           labels={'time_in_hospital': 'Days in Hospital'})
        fig.update_layout(height=300, margin=dict(t=10, b=10), font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Gender Risk Distribution")
        gender_risk = df.groupby(['gender', 'risk_level'], observed=True).size().reset_index(name='count')
        fig = px.bar(gender_risk, x='gender', y='count', color='risk_level',
                     barmode='group',
                     color_discrete_map={'Low': '#48bb78', 'Medium': '#ecc94b', 'High': '#f56565'})
        fig.update_layout(height=300, margin=dict(t=10, b=10), font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Department Analysis ──
    st.markdown("### Department Risk Analysis")
    dept_risk = df.groupby('diag_1_category', observed=True).agg(
        patient_count=('patient_id', 'count'),
        avg_risk=('risk_score', 'mean'),
        avg_stay=('time_in_hospital', 'mean'),
        avg_meds=('num_medications', 'mean'),
    ).reset_index().rename(columns={'diag_1_category': 'Department'})
    st.dataframe(dept_risk.sort_values('avg_risk', ascending=False),
                 use_container_width=True)
