"""
Page 6 — AI Data Copilot (Production)
Text-to-SQL assistant powered by Gemini 2.5 Flash.
"""

import streamlit as st
import os
import sys
import pandas as pd

st.set_page_config(page_title="AI Data Copilot", page_icon="🤖", layout="wide")

# Add the PROJECT ROOT to sys.path so 'chatbot' package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chatbot.text_to_sql_agent import ask

# ── Page Header ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0;">
    <h1 style="margin:0;">🤖 AI Data Copilot</h1>
    <p style="opacity:0.7; margin-top:5px;">
        Ask questions in plain English — I'll query the hospital data warehouse for you
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Try These Questions")
    example_questions = [
        "How many high risk patients are there?",
        "Show average risk score by gender",
        "Top 10 highest risk patients",
        "What is the average length of stay for high risk patients?",
        "How many patients are there in each age group?",
        "What percentage of patients are high risk?",
        "Show risk distribution by primary diagnosis",
        "Which gender has higher average risk?",
        "Count patients by race",
        "Average number of medications for high risk vs low risk patients",
    ]
    for q in example_questions:
        if st.button(f"📝 {q}", key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.markdown("""
    <div style="opacity:0.6; font-size:0.8rem;">
        <b>Powered by:</b><br>
        • Google Gemini 2.5 Flash<br>
        • LangChain Text-to-SQL<br>
        • DuckDB Data Warehouse
    </div>
    """, unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your **AI Data Copilot**. Ask me anything about our hospital patient data and I'll query the database for you.\n\nTry: *\"How many high risk patients are there?\"*",
            "sql": None,
            "results_df": None
        }
    ]

# Render chat history
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "assistant" else "👨‍⚕️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("🔍 Generated SQL Query", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("results_df") is not None and len(msg["results_df"]) > 0:
            with st.expander(f"📊 Query Results ({len(msg['results_df'])} rows)", expanded=False):
                st.dataframe(msg["results_df"], use_container_width=True)

# ── Handle pending question from sidebar ──────────────────────
pending = st.session_state.pop("pending_question", None)

# ── Chat Input ────────────────────────────────────────────────
user_input = st.chat_input("Ask about patients, risk levels, demographics, or trends...")

# Use pending question if no direct input
question = user_input or pending

if question:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question, "sql": None, "results_df": None})
    with st.chat_message("user", avatar="👨‍⚕️"):
        st.markdown(question)

    # Process with the agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Analyzing your question..."):
            response = ask(question)

        if response["error"]:
            error_msg = f"⚠️ {response['error']}"
            st.error(error_msg)
            if response["sql"]:
                with st.expander("🔍 Generated SQL (failed)", expanded=True):
                    st.code(response["sql"], language="sql")
            st.session_state.chat_history.append({
                "role": "assistant", "content": error_msg,
                "sql": response["sql"], "results_df": None
            })
        else:
            # Show explanation
            st.markdown(response["explanation"])

            # Show SQL
            with st.expander("🔍 Generated SQL Query", expanded=False):
                st.code(response["sql"], language="sql")

            # Show results table
            results_df = pd.DataFrame(response["results"])
            if len(results_df) > 0:
                with st.expander(f"📊 Query Results ({len(results_df)} rows)", expanded=True):
                    st.dataframe(results_df, use_container_width=True)

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["explanation"],
                "sql": response["sql"],
                "results_df": results_df
            })
