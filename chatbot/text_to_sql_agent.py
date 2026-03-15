"""
Text-to-SQL Agent — Production Ready
Uses Google Gemini Flash 2.5 to convert natural language → SQL → DuckDB → explanation.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import re
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'kenexai.duckdb')
DB_URI = f"duckdb:///{DB_PATH}"

# ── Schema context for the LLM ────────────────────────────────
SCHEMA_CONTEXT = """
You have access to a DuckDB healthcare data warehouse with these tables:

TABLE: gold_patient_risk_summary (71,518 rows — best table for risk + demographics together)
Columns: patient_id (INT PK), age (INT), age_group (VARCHAR), gender (VARCHAR), race (VARCHAR),
         diag_1_category (VARCHAR), num_medications (INT), num_lab_procedures (INT),
         time_in_hospital (INT), total_visits (INT), number_inpatient (INT),
         insulin (VARCHAR), diabetes_med (VARCHAR), a1c_result (VARCHAR),
         risk_score (DOUBLE), risk_percentage (DOUBLE), risk_level (VARCHAR)

TABLE: gold_hospital_kpis (1 row — overall hospital statistics)
Columns: total_patients (BIGINT), high_risk_patients (BIGINT), high_risk_rate (DOUBLE),
         avg_risk_score (DOUBLE), avg_length_of_stay (DOUBLE), avg_medications (DOUBLE)

TABLE: gold_risk_distribution (3 rows — one per risk level)
Columns: risk_level (VARCHAR), patient_count (BIGINT), percentage (DOUBLE)

TABLE: fact_patient_visits (71,518 rows — star schema fact table)
Columns: id (INT), patient_id (INT), visit_id (INT), timestamp (VARCHAR),
         risk_score (DOUBLE), risk_level (VARCHAR), readmitted_binary (INT)

TABLE: dim_patient (71,518 rows — demographic dimension)
Columns: patient_id (INT PK), age (INT), age_group (VARCHAR), gender (VARCHAR), race (VARCHAR)

TABLE: dim_visit_metrics (71,518 rows — clinical metrics dimension)
Columns: visit_id (INT PK), time_in_hospital (INT), num_lab_procedures (INT), 
         num_medications (INT), total_visits (INT), number_inpatient (INT),
         diag_1_category (VARCHAR), insulin (VARCHAR), diabetes_med (VARCHAR), a1c_result (VARCHAR)

IMPORTANT VALUES:
- risk_level: 'Low', 'Medium', 'High'
- gender: 'Male', 'Female', 'Unknown/Invalid'
- age_group: '0-20', '21-40', '41-60', '61-80', '81-100'
- race: 'Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'
- diag_1_category: 'Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
                   'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'
- diabetes_med: 'Yes', 'No'
- insulin: 'No', 'Up', 'Down', 'Steady'
- a1c_result: 'None', 'Norm', '>7', '>8'

TIPS:
- For a quick hospital overview, query gold_hospital_kpis (single row).
- For risk + demographics together, use gold_patient_risk_summary.
- For risk level breakdown (Low/Medium/High counts), use gold_risk_distribution.
"""

TIPS:
- For a quick hospital overview, query gold_hospital_kpis (single row).
- For risk + demographics together, use gold_patient_risk_summary.
- For risk level breakdown (Low/Medium/High counts), use gold_risk_distribution.
- For detailed patient info, JOIN patients with model_predictions on patient_id.
- For visit details, JOIN patient_visits with patients on patient_id.
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


FASTAPI_URL = "http://localhost:8000"


def execute_sql(query: str) -> tuple:
    """Execute a SQL query via the FastAPI /query endpoint. Returns (records, columns)."""
    import requests

    lower_q = query.lower().strip()
    for forbidden in ['insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'truncate ']:
        if forbidden in lower_q:
            raise ValueError(f"Destructive SQL operation blocked: {forbidden.strip()}")

    response = requests.post(
        f"{FASTAPI_URL}/query",
        json={"sql": query},
        timeout=30
    )

    if response.status_code != 200:
        raise ValueError(f"Query failed: {response.json().get('detail', response.text)}")

    data = response.json()
    return data["records"], data["columns"]


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
