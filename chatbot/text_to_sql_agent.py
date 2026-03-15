"""
Text-to-SQL Agent — Production Ready
Uses Google Gemini Flash 2.5 to convert natural language → SQL → DuckDB → explanation.
"""

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBnhIQWWK45sfLWWQKUTmxFiM8AD9BfCC0"
MODEL_NAME = "gemini-2.5-flash"

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb')
DB_URI = f"duckdb:///{DB_PATH}"

# ── Schema context for the LLM ────────────────────────────────
SCHEMA_CONTEXT = """
You have access to a DuckDB healthcare data warehouse with these tables:

TABLE: patients
Columns: patient_id (INT PK), age (INT), age_group (VARCHAR), gender (VARCHAR), race (VARCHAR),
         num_medications (INT), num_lab_procedures (INT), num_procedures (INT),
         number_diagnoses (INT), time_in_hospital (INT),
         diag_1_category (VARCHAR), diag_2_category (VARCHAR), diag_3_category (VARCHAR),
         diabetes_med (VARCHAR), insulin (VARCHAR), a1c_result (VARCHAR)

TABLE: patient_visits
Columns: visit_id (INT PK), patient_id (INT FK), admission_type_id (INT),
         discharge_disposition_id (INT), admission_source_id (INT),
         number_outpatient (INT), number_emergency (INT), number_inpatient (INT),
         total_visits (INT), medication_change (BOOL), high_lab_procedures (BOOL)

TABLE: model_predictions
Columns: prediction_id (INT PK), patient_id (INT FK), risk_score (DOUBLE),
         risk_percentage (DOUBLE), risk_level (VARCHAR), actual_readmitted (INT),
         top_factors (VARCHAR)

TABLE: fact_patient_visits
Columns: id (INT), patient_id (INT), visit_id (INT), timestamp (VARCHAR),
         risk_score (DOUBLE), risk_level (VARCHAR)

TABLE: dim_patient
Columns: patient_id (INT PK), age (INT), gender (VARCHAR)

TABLE: dim_visit_metrics
Columns: visit_id (INT PK), num_medications (INT), num_lab_procedures (INT),
         number_inpatient (INT), time_in_hospital (INT)

TABLE: gold_hospital_kpis
Columns: total_patients (BIGINT), high_risk_patients (BIGINT), high_risk_percentage (DOUBLE),
         avg_risk_score (DOUBLE), avg_time_in_hospital (DOUBLE)

TABLE: gold_risk_distribution
Columns: risk_level (VARCHAR), patient_count (BIGINT), avg_risk_score (DOUBLE)

TABLE: gold_patient_risk_summary
Columns: patient_id (INTEGER), age (INTEGER), gender (VARCHAR),
         risk_score (DOUBLE), risk_level (VARCHAR), last_visit_timestamp (VARCHAR)

IMPORTANT VALUES:
- risk_level can be: 'Low', 'Medium', 'High'
- gender can be: 'Male', 'Female'
- age_group can be: '[0-10)', '[10-20)', '[20-30)', ... '[90-100)'
- diag categories include: 'Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Other'

TIPS:
- For general hospital stats or overview questions, query gold_hospital_kpis.
- For patient risk lists or summaries, query gold_patient_risk_summary.
- For count of patients by risk level, query gold_risk_distribution.
"""

SQL_GENERATION_PROMPT = """You are a DuckDB SQL expert for a hospital readmission risk analytics platform.

{schema}

RULES:
1. Generate ONLY valid DuckDB SQL. Output ONLY the raw SQL query, nothing else.
2. NEVER use schema prefixes — query tables directly (e.g. FROM patients, not FROM main.patients).
3. ONLY use columns that exist in the schema above. Never hallucinate columns.
4. For analytics questions, use aggregations (COUNT, AVG, SUM, GROUP BY).
5. Always add LIMIT 25 for open-ended queries to prevent large outputs.
6. ONLY generate SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
7. Use JOIN when combining tables — join patients and model_predictions on patient_id.
8. For "high risk" questions, filter WHERE risk_level = 'High'.
9. Round decimal results to 2 places using ROUND().
10. If the question is unclear or unrelated to healthcare data, return: SELECT 'I can only answer questions about patient data.' AS message;

Question: {question}

SQL:"""

EXPLANATION_PROMPT = """You are a helpful healthcare analytics assistant for doctors and hospital administrators.

A user asked: "{question}"

This SQL query was executed:
```sql
{sql}
```

Results:
{results}

Provide a clear, concise natural language answer. Use markdown formatting:
- Bold important numbers
- Use bullet points for lists
- Keep it under 150 words
- Do NOT explain the SQL syntax
- If results are empty, say no matching data was found
- Add a brief clinical insight if relevant"""


def _get_llm(temperature: float = 0):
    """Create a Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


def _clean_sql(raw: str) -> str:
    """Extract clean SQL from LLM output (strips markdown fences, extra text)."""
    # Remove markdown code blocks
    if "```sql" in raw:
        raw = raw.split("```sql")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    sql = raw.strip().rstrip(";") + ";"

    # Safety: reject anything that isn't a SELECT
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word not in ("SELECT", "WITH"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {first_word}")

    return sql


def generate_sql(question: str) -> str:
    """Convert a natural language question into a DuckDB SQL query using Gemini."""
    llm = _get_llm(temperature=0)

    prompt = SQL_GENERATION_PROMPT.format(schema=SCHEMA_CONTEXT, question=question)
    response = llm.invoke(prompt)
    raw_sql = response.content

    logger.info(f"Raw LLM output: {raw_sql}")
    return _clean_sql(raw_sql)


def execute_sql(query: str) -> tuple:
    """Execute a SQL query against DuckDB. Returns (results_list, column_names)."""
    import duckdb

    # Double-check safety
    lower_q = query.lower().strip()
    for forbidden in ['insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'truncate ']:
        if forbidden in lower_q:
            raise ValueError(f"Destructive SQL operation blocked: {forbidden.strip()}")

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = con.execute(query)
        df = result.fetchdf()

        # Cap at 100 rows
        if len(df) > 100:
            df = df.head(100)

        columns = df.columns.tolist()
        records = df.to_dict('records')
        return records, columns
    finally:
        con.close()


def generate_explanation(question: str, sql: str, results: list) -> str:
    """Generate a natural language explanation of the query results."""
    llm = _get_llm(temperature=0.5)

    # Truncate results for the prompt to avoid token limits
    display_results = results[:20] if len(results) > 20 else results

    prompt = EXPLANATION_PROMPT.format(
        question=question,
        sql=sql,
        results=display_results
    )

    response = llm.invoke(prompt)
    return response.content


def ask(question: str) -> dict:
    """
    Full pipeline: question → SQL → execute → explain.
    Returns dict with keys: sql, results, columns, explanation, error
    """
    try:
        sql = generate_sql(question)
    except Exception as e:
        return {"sql": None, "results": [], "columns": [], "explanation": None,
                "error": f"Failed to generate SQL: {e}"}

    try:
        results, columns = execute_sql(sql)
    except Exception as e:
        return {"sql": sql, "results": [], "columns": [],
                "explanation": None, "error": f"Query execution failed: {e}"}

    try:
        explanation = generate_explanation(question, sql, results)
    except Exception as e:
        explanation = f"Results returned successfully but explanation generation failed: {e}"

    return {
        "sql": sql,
        "results": results,
        "columns": columns,
        "explanation": explanation,
        "error": None
    }
