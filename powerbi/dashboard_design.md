# Power BI Dashboard Design Specification

## KenexAI — Patient Readmission Risk Intelligence Dashboard

### Color Palette
| Color | Hex | Usage |
|-------|-----|-------|
| Navy Blue | #1a365d | Headers, primary text |
| Medical Blue | #2d4a7a | Charts, accent |
| Teal | #38b2ac | Positive indicators |
| Soft Green | #48bb78 | Low risk |
| Amber | #ecc94b | Medium risk |
| Coral Red | #f56565 | High risk |
| Light Gray | #f7fafc | Backgrounds |
| White | #ffffff | Cards |

### Typography
- **Titles**: Segoe UI Bold, 18pt
- **Subtitles**: Segoe UI Semibold, 12pt
- **Body**: Segoe UI, 10pt

---

## Page 1: Hospital Overview

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  🏥 Hospital Readmission Risk Overview          [Date Filter]│
├──────────┬──────────┬──────────┬─────────────────────────────┤
│ Total    │ Readmit  │ Avg LOS  │ High Risk                   │
│ Patients │ Rate     │          │ Patients                    │
│ [KPI]    │ [KPI]    │ [KPI]    │ [KPI]                       │
├──────────┴──────────┴──────────┴─────────────────────────────┤
│                     │                                        │
│ Risk Distribution   │  Readmission Trend Over Time           │
│ [Donut Chart]       │  [Line Chart]                          │
│                     │                                        │
├─────────────────────┴────────────────────────────────────────┤
│                                                              │
│  Readmissions by Department       [Stacked Bar Chart]        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### KPI Cards
1. **Total Patients**: `COUNT(patient_id)` from `patient_risk_summary.csv`
2. **Readmission Rate**: `DIVIDE(COUNTROWS(FILTER(data, [actual_readmitted]=1)), COUNTROWS(data)) * 100`
3. **Avg Length of Stay**: `AVERAGE(time_in_hospital)`
4. **High Risk Patients**: `COUNTROWS(FILTER(data, [risk_level]="High"))`

### Charts
- **Donut Chart**: risk_level → count; colors: Low=#48bb78, Medium=#ecc94b, High=#f56565
- **Line Chart**: month → readmission_rate from `readmission_trends.csv`
- **Stacked Bar**: diag_1_category → patient_count, color by risk_level

---

## Page 2: High Risk Patients

### Layout
```
┌──────────────────────────────────────────────────────────────┐
│  🚨 High Risk Patients                                      │
├──────────┬──────────┬────────────────────────────────────────┤
│ Age      │ Diagnosis│  Gender                                │
│ [Slicer] │ [Slicer] │  [Slicer]                              │
├──────────┴──────────┴────────────────────────────────────────┤
│                                                              │
│  Patient ID | Age | Diagnosis | Visits | Risk% | Risk Level  │
│  [Interactive Table with conditional formatting]             │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Risk Score Distribution           │ Top Risk Patients       │
│  [Histogram]                       │ [Horizontal Bar]        │
└──────────────────────────────────────────────────────────────┘
```

### Table Conditional Formatting
- Risk Score cells: Data bars (green → red gradient)
- Risk Level column: Background color (Low=green, Medium=amber, High=red)

### DAX Measures (in Power BI)
```dax
High Risk Count = COUNTROWS(FILTER(patient_risk_summary, [risk_level] = "High"))

Avg Risk Score = AVERAGE(patient_risk_summary[risk_percentage])

Risk Category Color = 
SWITCH(
    SELECTEDVALUE(patient_risk_summary[risk_level]),
    "High", "#f56565",
    "Medium", "#ecc94b",
    "Low", "#48bb78",
    "#a0aec0"
)
```

---

## Page 3: Risk Analytics

### Layout
```
┌──────────────────────────────────────────────────────────────┐
│  📊 Risk Analytics                                           │
├─────────────────────────────┬────────────────────────────────┤
│ Risk by Age Group           │ Risk by Diagnosis              │
│ [Grouped Bar Chart]         │ [Horizontal Bar Chart]         │
│                             │                                │
├─────────────────────────────┼────────────────────────────────┤
│ Risk by Gender              │ Risk Correlation Matrix        │
│ [Clustered Column]          │ [Matrix Visual / Heatmap]      │
│                             │                                │
├─────────────────────────────┴────────────────────────────────┤
│                                                              │
│  Risk Factor Comparison         [Scatter Plot]               │
│  X: num_medications  Y: risk_percentage  Size: total_visits  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Data Source
Use `risk_by_demographics.csv` and `patient_risk_summary.csv`.

---

## Page 4: Patient Profile (Drill-Through)

### Layout
```
┌──────────────────────────────────────────────────────────────┐
│  👤 Patient Profile          [← Back to High Risk Patients]  │
├─────────────────────────────┬────────────────────────────────┤
│ Patient Demographics        │ Risk Score Gauge               │
│ ┌─────────────────────────┐ │ ┌────────────────────────────┐ │
│ │ ID:     12345           │ │ │       Risk: 82%            │ │
│ │ Age:    72              │ │ │    [Gauge Visual]          │ │
│ │ Gender: Male            │ │ │    ● HIGH RISK             │ │
│ │ Race:   Caucasian       │ │ └────────────────────────────┘ │
│ └─────────────────────────┘ │                                │
├─────────────────────────────┼────────────────────────────────┤
│ Visit History               │ Top Contributing Factors       │
│ [Card Visuals]              │ [Horizontal Bar Chart]         │
│ Inpatient: 3                │ Feature → Importance           │
│ Emergency: 2                │                                │
│ Outpatient: 1               │                                │
│ Medications: 18             │                                │
└─────────────────────────────┴────────────────────────────────┘
```

### Drill-Through Setup
1. Add `patient_id` as the drill-through field.
2. Right-click any patient row on Page 2 → "Drill through" → Patient Profile.

### Gauge Visual
- **Value**: risk_percentage
- **Min**: 0, **Max**: 100
- **Target**: 30 (low threshold)
- **Colors**: 0–30 green, 30–60 amber, 60–100 red

---

## Data Import Instructions

### Step 1: Open Power BI Desktop
1. Click **Home** → **Get Data** → **Text/CSV**
2. Import these files from `powerbi/sample_data/`:
   - `patient_risk_summary.csv`
   - `risk_by_demographics.csv`
   - `readmission_trends.csv`

### Step 2: Data Transformation
1. Open **Power Query Editor**
2. Verify column types (numbers, text, dates)
3. Ensure `risk_level` is categorized (Low, Medium, High)
4. Close & Apply

### Step 3: Create Relationships
- No relationships needed — single-table analysis
- For multi-table: link `patient_id` across tables

### Step 4: Build Pages
Follow the layout specs above for each page. Use:
- **KPI**: Card visual
- **Charts**: Pie, Line, Bar, Gauge from Visualizations pane
- **Tables**: Table visual with conditional formatting
- **Filters**: Slicers for interactive filtering

### Step 5: Theme
1. Go to **View** → **Themes** → **Customize current theme**
2. Apply the color palette from this document
3. Set font to **Segoe UI**
