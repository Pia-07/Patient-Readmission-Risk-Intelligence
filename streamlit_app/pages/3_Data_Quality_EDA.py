"""
Page 3 — Data Quality & EDA Dashboard
Missing values, distributions, correlations, and outlier detection.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Quality & EDA", page_icon="🔬", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       'data', 'kenexai.duckdb')


@st.cache_data(ttl=60)
def load_data(layer='silver_patient_visits'):
    import duckdb
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        # Load sample for performance
        df = con.execute(f"SELECT * FROM {layer} LIMIT 10000").fetchdf()
        con.close()
        return df
    except Exception:
        return None


st.markdown("# 🔬 Data Quality & EDA Dashboard")
st.markdown("> Explore the dataset, check data quality, and discover patterns")
st.markdown("---")

# Layer selection
layer_col = st.sidebar.radio("📁 Select Data Layer", 
                             ["Silver (Cleaned)", "Bronze (Raw)"],
                             index=0)
layer_table = "silver_patient_visits" if "Silver" in layer_col else "bronze_patient_visits"

df = load_data(layer_table)

if df is None:
    st.error("No data found. Please run the pipeline first: `python3 pipeline/run_pipeline.py`")
    st.stop()

# ── Tab Layout ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview", "🔍 Missing Values", "📊 Distributions", "🔗 Correlations"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Numeric Cols", f"{len(df.select_dtypes(include=[np.number]).columns)}")
    c4.metric("Categorical Cols", f"{len(df.select_dtypes(include=['object']).columns)}")

    st.markdown("### Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Data Types")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['Type', 'Count']
    st.dataframe(dtype_counts, use_container_width=True)

with tab2:
    st.markdown("### Missing Values Heatmap")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Column': missing.index, 'Missing': missing.values,
                               'Percentage': missing_pct.values})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False)

    if len(missing_df) > 0:
        fig = px.bar(missing_df.head(20), x='Column', y='Percentage',
                     color='Percentage',
                     color_continuous_scale='Reds',
                     title='Top 20 Columns with Missing Values')
        fig.update_layout(height=400, margin=dict(t=40, b=10),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values in the dataset!")

    # Missing value matrix visualization
    st.markdown("### Missing Values Matrix (sample)")
    sample = df.sample(min(200, len(df)), random_state=42)
    numeric_missing = sample.isnull().astype(int)
    cols_with_missing = numeric_missing.columns[numeric_missing.sum() > 0]
    if len(cols_with_missing) > 0:
        fig = px.imshow(numeric_missing[cols_with_missing].T,
                        color_continuous_scale='RdYlGn_r',
                        labels=dict(x="Sample Row", y="Feature", color="Missing"),
                        aspect="auto")
        fig.update_layout(height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values to visualize.")

with tab3:
    st.markdown("### Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected = st.multiselect("Select features to plot",
                                  numeric_cols,
                                  default=numeric_cols[:4])
        if selected:
            cols = st.columns(min(len(selected), 2))
            for i, col_name in enumerate(selected):
                with cols[i % 2]:
                    fig = px.histogram(df, x=col_name, nbins=40,
                                       color_discrete_sequence=['#2d4a7a'],
                                       title=col_name)
                    fig.update_layout(height=300, margin=dict(t=40, b=10),
                                      font=dict(family="Inter"))
                    st.plotly_chart(fig, use_container_width=True)

    # Outlier detection
    st.markdown("### 🚨 Outlier Detection (Box Plots)")
    outlier_cols = st.multiselect("Select features for outlier detection",
                                  numeric_cols,
                                  default=numeric_cols[:6],
                                  key="outlier_select")
    if outlier_cols:
        fig = go.Figure()
        for col_name in outlier_cols:
            fig.add_trace(go.Box(y=df[col_name].dropna(), name=col_name))
        fig.update_layout(height=450, margin=dict(t=10, b=10),
                          font=dict(family="Inter"),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        fig = px.imshow(corr,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect="auto",
                        title="Pearson Correlation Matrix")
        fig.update_layout(height=600, margin=dict(t=40, b=10),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations
        st.markdown("### Top Correlations")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_unstacked = corr.where(~mask).unstack().dropna()
        corr_unstacked = corr_unstacked[corr_unstacked.abs() > 0.3]
        corr_unstacked = corr_unstacked.sort_values(ascending=False)
        if len(corr_unstacked) > 0:
            top_corr = pd.DataFrame({
                'Feature 1': [idx[0] for idx in corr_unstacked.index],
                'Feature 2': [idx[1] for idx in corr_unstacked.index],
                'Correlation': corr_unstacked.values
            })
            st.dataframe(top_corr.head(20), use_container_width=True)
        else:
            st.info("No strong correlations found (|r| > 0.3)")
