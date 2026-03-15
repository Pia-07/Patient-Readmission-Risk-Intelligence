"""
KenexAI — AI-Powered Patient Readmission Risk Intelligence Platform
Main Streamlit Application — Entry Point & Home Page
"""

import streamlit as st
import os
import sys

# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="KenexAI — AI Healthcare Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏥 KenexAI")
        st.markdown("**AI Healthcare Analytics Platform**")
        st.markdown("---")
        st.markdown("""
        **📑 Navigation:**
        - 📡 **Real-Time Monitoring**
        - 🔮 **Risk Prediction**
        - 🔬 **Data Quality & EDA**
        - 👥 **Persona Dashboards**
        - 🧠 **AI Explanation**
        - 🤖 **GenAI Assistant**
        """)
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; opacity:0.7; font-size:0.8rem; margin-top:40px;">
            KenexAI v2.0.0<br>
            AI-Powered Healthcare Analytics<br>
            Data & AI Challenge Platform
        </div>
        """, unsafe_allow_html=True)

    # ── Hero Section ──
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <div style="font-size: 4rem;">🏥</div>
        <h1 style="margin: 0; font-size: 2.5rem;">Patient Readmission Risk Intelligence</h1>
        <p style="opacity: 0.7; font-size: 1.1rem; margin-top: 10px;">
            AI-powered predictions to reduce hospital readmissions and improve patient outcomes
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI Row ──
    import pandas as pd
    import numpy as np

    kpi = _get_kpi_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏥 Total Patients", f"{kpi['total_patients']:,}")
    c2.metric("🚨 High Risk", f"{kpi['high_risk']:,}")
    c3.metric("📊 Avg Risk Score", f"{kpi['avg_risk']:.3f}")
    c4.metric("🛏️ Avg Length of Stay", f"{kpi['avg_los']} days")

    st.markdown("---")

    # ── Quick Access Cards ──
    st.markdown("### ⚡ Quick Access")
    cols = st.columns(3)

    cards = [
        ("📡", "Real-Time Monitoring", "pages/1_Real_Time_Monitoring.py", "Live patient analytics"),
        ("🔮", "Risk Prediction", "pages/2_Risk_Prediction.py", "Predict readmission risk"),
        ("🔬", "Data Quality & EDA", "pages/3_Data_Quality_EDA.py", "Explore dataset quality"),
        ("👥", "Persona Dashboards", "pages/4_Persona_Dashboards.py", "Doctor & Admin analytics"),
        ("🧠", "AI Explanation", "pages/5_AI_Explanation.py", "Model explainability"),
        ("🤖", "GenAI Assistant", "pages/6_AI_Assistant.py", "Ask AI about data"),
    ]

    for i, (icon, title, page_path, desc) in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {icon} {title}")
                st.write(desc)
                if st.button(f"Open {title}", key=f"btn_{i}", use_container_width=True):
                    st.switch_page(page_path)

    # Footer
    st.markdown("""
    <div style="text-align:center; opacity:0.5; margin-top:40px; padding:20px; font-size:0.8rem;">
        KenexAI v2.0.0 — AI-Powered Patient Readmission Risk Intelligence Platform<br>
        Built for Data & AI Challenge 2026
    </div>
    """, unsafe_allow_html=True)


def _get_kpi_data() -> dict:
    try:
        import duckdb
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb')
        if os.path.exists(db_path):
            con = duckdb.connect(db_path, read_only=True)
            # Try Gold Layer first
            try:
                gold_kpi = con.execute("SELECT * FROM gold_hospital_kpis").fetchone()
                if gold_kpi:
                    con.close()
                    return {
                        'total_patients': gold_kpi[0],
                        'high_risk': gold_kpi[1],
                        'avg_risk': gold_kpi[3],
                        'avg_los': gold_kpi[4]
                    }
            except Exception:
                pass

            # Fallback to raw tables
            total = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            high = con.execute("SELECT COUNT(*) FROM model_predictions WHERE risk_level='High'").fetchone()[0]
            avg_risk = con.execute("SELECT AVG(risk_score) FROM model_predictions").fetchone()[0] or 0
            avg_los = con.execute("SELECT AVG(time_in_hospital) FROM patients").fetchone()[0] or 0
            con.close()
            return {'total_patients': total, 'high_risk': high,
                    'avg_risk': round(avg_risk, 3), 'avg_los': round(avg_los, 1)}
    except Exception:
        pass
    return {'total_patients': 5000, 'high_risk': 823, 'avg_risk': 0.42, 'avg_los': 4.4}


if __name__ == '__main__':
    main()
