# AI-Powered Patient Readmission Risk Intelligence Platform

A complete end-to-end healthcare AI analytics platform that predicts 30-day hospital readmission risk and provides actionable insights for doctors and hospital administrators.

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Kaggle      │    │  Data        │    │  DuckDB     │    │  Power BI    │
│  Dataset     │───▶│  Pipeline    │───▶│  Database   │───▶│  Dashboard   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                         │                    │
                         ▼                    │
                   ┌──────────────┐           │
                   │  Kaggle      │           │
                   │  ML Notebook │           │
                   │  (model.pkl) │           │
                   └──────┬───────┘           │
                          │                   │
                          ▼                   ▼
                   ┌──────────────┐    ┌──────────────┐
                   │  FastAPI     │◀───│  Streamlit   │
                   │  Backend     │───▶│  Web UI      │
                   └──────────────┘    └──────────────┘
```

## 📁 Project Structure

```
KenexAI/
├── data/raw/              # Original dataset CSV
├── data/processed/        # Cleaned & feature-engineered data
├── notebooks/             # Kaggle ML training notebook
├── pipeline/              # Data processing pipeline
├── backend/               # FastAPI prediction API
├── database/              # DuckDB schema & seeding
├── models/                # Trained model (.pkl)
├── streamlit_app/         # Doctor-facing web UI
└── powerbi/               # Dashboard design & sample data
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the **Diabetes 130-US Hospitals** dataset from Kaggle and place `diabetic_data.csv` in `data/raw/`.

### 3. Run Data Pipeline
```bash
python -m pipeline.run_pipeline
```

### 4. Train Model (on Kaggle)
Upload `notebooks/kaggle_training.py` to a Kaggle notebook. Run all cells. Download `model.pkl` to `models/`.

### 5. Seed Database
```bash
python -m database.seed
```

### 6. Start API Server
```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Start Streamlit UI
```bash
streamlit run streamlit_app/app.py --server.port 8501
```

## 🧠 Models

| Model                | Purpose           |
|---------------------|--------------------|
| Logistic Regression | Baseline           |
| Random Forest       | Ensemble (trees)   |
| XGBoost             | Best performance   |

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score, ROC AUC

## 🛠️ Tech Stack

- **ML**: scikit-learn, XGBoost, SHAP
- **Backend**: FastAPI, Uvicorn
- **Database**: DuckDB
- **Frontend**: Streamlit, Plotly
- **Dashboard**: Power BI
- **Data**: Pandas, NumPy

## 📋 Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**
- 100,000+ patient encounters
- 50+ features including demographics, diagnoses, medications, lab results
- Target: readmission within 30 days
