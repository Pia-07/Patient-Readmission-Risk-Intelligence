"""
Kaggle Training Notebook — AI-Powered Patient Readmission Risk Prediction
===========================================================================
Upload this script to a Kaggle notebook along with the 'Diabetes 130-US Hospitals' dataset.
Run all cells to train models, evaluate, and export results.

Output files:
  - model.pkl            → Best trained model
  - predictions.csv      → Per-patient predictions
  - patient_risk_summary.csv → Aggregated risk data for Power BI
  - model_comparison.csv → Model performance metrics
"""

# %% [markdown]
# # 🏥 Patient Readmission Risk Prediction
# Predicting 30-day hospital readmission using the Diabetes 130-US Hospitals dataset.
# %% [markdown]
# ## 1. Load Dataset (Kaggle Compatible)

# %%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

print("Searching for diabetic_data.csv inside /kaggle/input ...")

dataset_path = None

# Automatically find the dataset file
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.lower() == 'diabetic_data.csv':
            dataset_path = os.path.join(dirname, filename)

# If found → load dataset
if dataset_path:
    print(f"Dataset found at: {dataset_path}")
    df = pd.read_csv(dataset_path, na_values='?')
else:
    # fallback for local machine
    print("Dataset not found in Kaggle input folder. Trying local file...")
    df = pd.read_csv('diabetic_data.csv', na_values='?')

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nTarget distribution:\n{df['readmitted'].value_counts()}")

df.head()
# %% [markdown]
# ## 2. Data Preprocessing

# %%
# --- 2a. Remove duplicates (keep first encounter per patient) ---
print(f"Before dedup: {len(df)} rows")
df = df.drop_duplicates(subset='patient_nbr', keep='first')
print(f"After dedup: {len(df)} rows")

# --- 2b. Create binary target ---
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
print(f"\nBinary target distribution:\n{df['readmitted_binary'].value_counts()}")
print(f"Readmission rate: {df['readmitted_binary'].mean():.2%}")

# --- 2c. Drop high-missing / non-useful columns ---
drop_cols = [
    'encounter_id', 'patient_nbr', 'readmitted',
    'weight', 'payer_code', 'medical_specialty',
    'examide', 'citoglipton'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --- 2d. Map diagnosis codes ---
def map_diagnosis(code):
    if pd.isna(code):
        return 'Other'
    code_str = str(code)
    if code_str.startswith('E'): return 'External'
    if code_str.startswith('V'): return 'Supplementary'
    try:
        num = float(code_str)
    except ValueError:
        return 'Other'
    if 390 <= num <= 459 or num == 785: return 'Circulatory'
    elif 460 <= num <= 519 or num == 786: return 'Respiratory'
    elif 520 <= num <= 579 or num == 787: return 'Digestive'
    elif 250 <= num < 251: return 'Diabetes'
    elif 800 <= num <= 999: return 'Injury'
    elif 710 <= num <= 739: return 'Musculoskeletal'
    elif 580 <= num <= 629 or num == 788: return 'Genitourinary'
    elif 140 <= num <= 239: return 'Neoplasms'
    else: return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    if col in df.columns:
        df[f'{col}_category'] = df[col].apply(map_diagnosis)
        df = df.drop(columns=[col])

print(f"\nShape after preprocessing: {df.shape}")

# %% [markdown]
# ## 3. Feature Engineering

# %%
# --- Derived features ---
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# Total visits
df['total_visits'] = df[['number_inpatient', 'number_outpatient', 'number_emergency']].sum(axis=1)

# Medication change flag
df['medication_change'] = (df['change'] == 'Ch').astype(int)

# High lab procedures indicator (above 75th percentile)
threshold = df['num_lab_procedures'].quantile(0.75)
df['high_lab_procedures'] = (df['num_lab_procedures'] > threshold).astype(int)

# Age to numeric
df['age_numeric'] = df['age'].map(AGE_MAP).fillna(55)

# Medication count bins
df['num_medications_bin'] = pd.cut(
    df['num_medications'], bins=[0, 5, 10, 20, 50, 100],
    labels=['very_low', 'low', 'medium', 'high', 'very_high']
).astype(str)

# Encode medication columns
med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

for col in med_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1)

existing_med_cols = [c for c in med_cols if c in df.columns]
df['total_medications_active'] = df[existing_med_cols].sum(axis=1)

print(f"Engineered features added. Shape: {df.shape}")
print(f"\nNew features: total_visits, medication_change, high_lab_procedures, "
      f"age_numeric, num_medications_bin, total_medications_active")

# %% [markdown]
# ## 4. Encode Categorical Variables

# %%
# Label-encode categoricals
label_enc_cols = [
    'race', 'gender', 'age', 'max_glu_serum', 'A1Cresult',
    'change', 'diabetesMed', 'num_medications_bin',
    'diag_1_category', 'diag_2_category', 'diag_3_category'
]

encoders = {}
for col in label_enc_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Drop remaining object columns
remaining_obj = df.select_dtypes(include=['object']).columns.tolist()
if remaining_obj:
    print(f"Dropping remaining object columns: {remaining_obj}")
    df = df.drop(columns=remaining_obj)

# Fill any remaining NaN
df = df.fillna(0)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Feature columns: {len(df.columns) - 1}")
print(f"Target: readmitted_binary")

# %% [markdown]
# ## 5. Train-Test Split

# %%
TARGET = 'readmitted_binary'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTrain target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nTest target distribution:\n{y_test.value_counts(normalize=True)}")

# Save feature names for later
feature_names = list(X.columns)

# %% [markdown]
# ## 6. Model Training

# %%
# --- 6a. Logistic Regression (Baseline) ---
print("=" * 50)
print("Training Logistic Regression...")
print("=" * 50)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='saga',
    n_jobs=-1
)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, lr_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, lr_proba):.4f}")

# %%
# --- 6b. Random Forest ---
print("=" * 50)
print("Training Random Forest...")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba):.4f}")

# %%
# --- 6c. XGBoost ---
print("=" * 50)
print("Training XGBoost...")
print("=" * 50)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, xgb_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

# %% [markdown]
# ## 7. Model Comparison & Evaluation

# %%
models = {
    'Logistic Regression': (lr_model, lr_pred, lr_proba),
    'Random Forest': (rf_model, rf_pred, rf_proba),
    'XGBoost': (xgb_model, xgb_pred, xgb_proba)
}

results = []
for name, (model, pred, proba) in models.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1 Score': f1_score(y_test, pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_test, proba)
    })

results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv('model_comparison.csv', index=False)
print("\n✅ Saved model_comparison.csv")

# %%
# --- Visualization: Model Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
results_df.set_index('Model')[metrics_to_plot].plot(
    kind='bar', ax=axes[0], rot=15, colormap='viridis'
)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
axes[0].legend(loc='lower right')

# ROC Curves
for name, (model, pred, proba) in models.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved model_comparison.png")

# %% [markdown]
# ## 8. Select Best Model & Save

# %%
# Select best model by ROC AUC
best_idx = results_df['ROC AUC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = models[best_model_name][0]

print(f"🏆 Best Model: {best_model_name}")
print(f"   ROC AUC: {results_df.loc[best_idx, 'ROC AUC']:.4f}")

# Save best model
joblib.dump(best_model, 'model.pkl')
print("✅ Saved model.pkl")

# Save feature names
joblib.dump(feature_names, 'feature_names.pkl')
print("✅ Saved feature_names.pkl")

# --- Confusion Matrix for best model ---
best_pred = models[best_model_name][1]
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Readmitted', 'Readmitted'],
            yticklabels=['Not Readmitted', 'Readmitted'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, best_pred, target_names=['Not Readmitted', 'Readmitted']))

# %% [markdown]
# ===============================
# 9. SHAP Explainability (FAST)
# ===============================




# ===============================
# 9. SHAP Explainability
# ===============================

print("Generating SHAP explanations...")

# Use small sample for speed
sample_data = X_test.iloc[:200]

if best_model_name in ["Random Forest", "XGBoost"]:

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(sample_data)

    # RandomForest returns list[class0,class1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # select readmitted class
        base_val = explainer.expected_value[1]
    else:
        shap_vals = shap_values
        base_val = explainer.expected_value

else:
    explainer = shap.LinearExplainer(best_model, X_train)
    shap_vals = explainer.shap_values(sample_data)
    base_val = explainer.expected_value


# ---------- SUMMARY PLOT ----------
plt.figure(figsize=(10,8))
shap.summary_plot(shap_vals, sample_data, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

print("Saved shap_summary.png")


# ---------- BAR PLOT ----------
plt.figure(figsize=(10,6))
shap.summary_plot(shap_vals, sample_data, plot_type="bar", show=False)
plt.title("Mean SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150)
plt.show()

print("Saved shap_bar.png")


# ---------- WATERFALL PLOT ----------
single_vals = shap_vals[0]

# if still (features,2) pick class 1
if len(single_vals.shape) == 2:
    single_vals = single_vals[:,1]

# ensure base value is scalar
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[1] if len(base_val) > 1 else base_val[0]

sample_explanation = shap.Explanation(
    values=single_vals,
    base_values=float(base_val),
    data=sample_data.iloc[0],
    feature_names=sample_data.columns
)

plt.figure(figsize=(10,6))
shap.plots.waterfall(sample_explanation, max_display=15, show=False)
plt.title("SHAP Waterfall — Example Patient")
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150)
plt.show()

print("Saved shap_waterfall.png")




# %% [markdown]
# ## 10. Export Predictions for Power BI

# %%
# --- Per-patient predictions ---
best_proba = models[best_model_name][2]

predictions_df = X_test.copy()
predictions_df['actual_readmitted'] = y_test.values
predictions_df['risk_score'] = best_proba
predictions_df['risk_percentage'] = (best_proba * 100).round(1)
predictions_df['risk_level'] = pd.cut(
    best_proba,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

predictions_df.to_csv('predictions.csv', index=False)
print(f"✅ Saved predictions.csv ({len(predictions_df)} rows)")

# %%
# --- Aggregated risk summary for Power BI ---
summary = predictions_df.copy()

# Add age group labels back
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
summary['age_group'] = pd.cut(summary['age_numeric'], bins=age_bins, labels=age_labels)

# Risk by age group
risk_by_age = summary.groupby('age_group', observed=True).agg(
    avg_risk_score=('risk_score', 'mean'),
    patient_count=('risk_score', 'count'),
    high_risk_count=('risk_level', lambda x: (x == 'High').sum())
).reset_index()

risk_by_age.to_csv('risk_by_demographics.csv', index=False)
print("✅ Saved risk_by_demographics.csv")

# Risk level distribution
risk_dist = summary['risk_level'].value_counts().reset_index()
risk_dist.columns = ['risk_level', 'patient_count']
risk_dist.to_csv('risk_distribution.csv', index=False)
print("✅ Saved risk_distribution.csv")

# Patient risk summary (top columns for Power BI)
patient_summary = summary[[
    'age_numeric', 'total_visits', 'num_medications',
    'num_lab_procedures', 'number_inpatient',
    'risk_score', 'risk_percentage', 'risk_level', 'actual_readmitted'
]].copy()
patient_summary = patient_summary.rename(columns={'age_numeric': 'age'})
patient_summary.to_csv('patient_risk_summary.csv', index=False)
print(f"✅ Saved patient_risk_summary.csv ({len(patient_summary)} rows)")

# %%
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"ROC AUC: {results_df.loc[best_idx, 'ROC AUC']:.4f}")
print(f"\nFiles saved:")
print("  📦 model.pkl — Trained model")
print("  📦 feature_names.pkl — Feature names list")
print("  📊 predictions.csv — Per-patient predictions")
print("  📊 patient_risk_summary.csv — Summary for Power BI")
print("  📊 risk_by_demographics.csv — Risk by age groups")
print("  📊 risk_distribution.csv — Risk level distribution")
print("  📊 model_comparison.csv — Model metrics comparison")
print("  🖼️ model_comparison.png — Performance charts")
print("  🖼️ confusion_matrix.png — Confusion matrix")
print("  🖼️ shap_summary.png — SHAP feature importance")
print("  🖼️ shap_bar.png — SHAP bar chart")
print("  🖼️ shap_waterfall.png — SHAP waterfall (example)")