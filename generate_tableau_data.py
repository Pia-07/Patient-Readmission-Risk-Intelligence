import pandas as pd
import duckdb
import os

print("Generating Master Tableau Dataset...")

# Connect to the local DuckDB database which has all the merged patients, visits, and predictions
conn = duckdb.connect('data/kenexai.duckdb')

# Query to join all 3 tables perfectly for Tableau
query = """
SELECT * EXCLUDE (v.patient_id, m.patient_id)
FROM patients p
JOIN patient_visits v ON p.patient_id = v.patient_id
LEFT JOIN model_predictions m ON p.patient_id = m.patient_id
"""

# Export directly to a single CSV for easy Tableau import
df = conn.execute(query).fetchdf()
os.makedirs('tableau_export', exist_ok=True)
export_path = 'tableau_export/Master_Patient_Risk_Data.csv'
df.to_csv(export_path, index=False)

print(f"✅ Created Tableau Master Dataset at: {export_path}")
print(f"Total Rows: {len(df)}")
print("You can simply drag and drop this file into Tableau Desktop to start building your dashboard!")
