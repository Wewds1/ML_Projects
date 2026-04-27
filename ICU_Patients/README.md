# ICU Patient Deterioration Early Warning — End-to-End ML Project

**Target variable:** `deterioration_12h` (binary: 1 = clinical deterioration within 12 hours)  
**Model type:** Binary Classification  
**Domain:** Critical care medicine / clinical decision support  
**Dataset:** 8,500 synthetic ICU patient observations, deliberately dirty  
**Deterioration rate:** ~18% — higher than fraud, lower than typical churn  
**Stakes:** This is the highest-consequence classification problem in this portfolio. In real deployment, a false negative means a patient deteriorates without warning. A false positive means unnecessary intervention and alarm fatigue. Every modeling decision has a direct human cost.

---

## What makes this problem different from every other classification project

The fraud detection project introduced class imbalance and the cost asymmetry of false negatives vs false positives. This project takes both to their logical extreme in a healthcare context and adds three new challenges that don't appear anywhere else in this portfolio:

**Clinical validity constraint.** Your model's coefficients must be clinically interpretable. If logistic regression assigns a negative coefficient to `lactate` (suggesting high lactate is protective against deterioration), that is a medically impossible result and means something is wrong with your preprocessing — not just a number to ignore. Every feature importance ranking and every coefficient you publish needs to be checked against clinical knowledge.

**Missing data is MNAR in a dangerous way.** In the previous projects, missing values were inconvenient. Here, `gcs_score` is missing most often in sedated, mechanically ventilated patients — the sickest patients in the dataset. If you impute with the median and treat missingness as random, you will systematically underestimate risk in the highest-risk subgroup. The missingness pattern is the signal.

**Temporal leakage risk.** This dataset represents a single observation window per patient. In real ICU systems, you would have rolling hourly observations and you must never let future observations leak into features for past timepoints. Even in this static version, features like `sofa_score` and `news2_score` are computed from other columns in the dataset — you need to understand what they encode so you don't accidentally double-count their components.

**The baseline you must beat is a real clinical score.** NEWS2 and SOFA are validated, internationally deployed scoring systems used in ICUs today. Your model needs to outperform them — not just beat a naive majority-class baseline. If it doesn't, there's no case for deploying a black-box ML model over a transparent clinical score.

---

## What SOFA and NEWS2 are — read this before modeling

Both scores are already in the dataset as columns. Understanding them is essential.

**SOFA (Sequential Organ Failure Assessment, 0–24):** Measures dysfunction across six organ systems — respiratory (P/F ratio), coagulation (platelets), liver (bilirubin), cardiovascular (MAP + vasopressors), neurological (GCS), and renal (creatinine). Score ≥ 2 from baseline = sepsis definition. Score > 10 = ~50% ICU mortality. It's your strongest individual predictor baseline.

**NEWS2 (National Early Warning Score 2, 0–20):** Combines respiratory rate, SpO2, supplemental oxygen, temperature, systolic BP, heart rate, and consciousness level. Score ≥ 7 triggers urgent clinical review in UK hospitals. Widely used in general wards; less specific in ICU where all patients are already high-acuity.

Your job is to beat the AUROC of SOFA alone (typically ~0.72–0.78 for deterioration prediction) using the full feature set. If your XGBoost model reaches 0.85+, you can argue it captures signal that neither score encodes — which is the scientific contribution of the project.

---

## Dataset columns

### Demographics and admission

| Column | Type | Notes |
|---|---|---|
| `patient_id` | string | Unique ID — drop before modeling |
| `patient_age` | int | 18–95 |
| `gender` | categorical | Male / Female |
| `weight_kg` | float | 38–180 kg |
| `height_cm` | float | 145–205 cm |
| `bmi` | float | Derived — check for impossible values vs height/weight |
| `icu_unit` | categorical | **dirty** — 5 ICU types |
| `admission_type` | categorical | **dirty** — Emergency / Elective / Urgent / Transfer |
| `icu_los_days_at_obs` | float | Days in ICU at time of observation |

### Comorbidities

| Column | Type | Notes |
|---|---|---|
| `has_diabetes` | binary | |
| `has_hypertension` | binary | |
| `has_heart_failure` | binary | |
| `has_copd` | binary | |
| `has_ckd` | binary | |
| `has_immunosuppression` | binary | |
| `charlson_index` | int | Composite comorbidity score (0–15) — includes all above |

### Vital signs (6-hour observation window aggregates)

| Column | Type | Notes |
|---|---|---|
| `hr_mean` / `hr_min` / `hr_max` / `hr_std` | float | Heart rate (bpm) |
| `sbp_mean` / `sbp_min` / `sbp_max` | float | Systolic blood pressure (mmHg) |
| `map_mean` / `map_min` | float | Mean arterial pressure (mmHg) — MAP < 65 = shock threshold |
| `rr_mean` / `rr_max` | float | Respiratory rate (breaths/min) |
| `temp_mean` / `temp_max` | float | Temperature (°C) |
| `spo2_mean` / `spo2_min` | float | Oxygen saturation (%) |
| `gcs_score` | float | **~8% missing** — Glasgow Coma Scale (3–15), MNAR |

### Lab values

| Column | Type | Notes |
|---|---|---|
| `lactate` | float | **~9% missing** — MNAR, not ordered for elective admissions |
| `wbc` | float | White blood cell count (×10³/µL) |
| `creatinine` | float | Serum creatinine (mg/dL) — renal function marker |
| `bilirubin` | float | **~7% missing** — liver function, not always in panel |
| `platelet_count` | int | Platelets (×10³/µL) |
| `sodium` | float | Serum sodium (mEq/L) |
| `potassium` | float | Serum potassium (mEq/L) |
| `hemoglobin` | float | Hemoglobin (g/dL) |
| `pao2_fio2_ratio` | float | **~11% missing** — P/F ratio, requires ABG blood draw |

### Interventions and scores

| Column | Type | Notes |
|---|---|---|
| `on_vasopressor` | binary | Active vasopressor (norepinephrine, dopamine etc.) |
| `on_mechanical_vent` | binary | Intubated and mechanically ventilated |
| `on_rrt` | binary | Renal replacement therapy (dialysis) |
| `fluid_balance_24h` | float | **~5% missing** — net fluid in mL (negative = net loss) |
| `sofa_score` | int | SOFA score computed from labs + vitals (0–24) |
| `news2_score` | int | NEWS2 score computed from vitals (0–20) |

### Target

| Column | Type | Notes |
|---|---|---|
| `deterioration_12h` | binary | **Target** — 1 if patient deteriorated within 12 hours |

---

## Data issues to fix

### Duplicates
~30 duplicate rows from data export overlap. Drop before any analysis.

### Missing values — five columns, all clinically meaningful

**`gcs_score`** (~8%) — MNAR. Missing most often in sedated, mechanically ventilated patients. These are typically the sickest patients and their true GCS is unknown (not measurable under sedation), not simply unrecorded. Two approaches:
1. Create `gcs_missing` flag = 1, then impute with 3 (minimum — worst-case assumption). This is clinically defensible: an unassessable patient is assumed to have severe neurological compromise.
2. Create `gcs_missing` flag = 1, then impute with median. Less conservative but more statistically neutral.
Try both and compare model performance. The choice affects your recall on the highest-risk patients.

**`lactate`** (~9%) — MNAR. Lactate is ordered when clinicians suspect sepsis or poor perfusion. Not ordering it in an elective surgical patient is actually low-risk. But when lactate IS missing in an emergency admission, that absence of data can itself signal clinical uncertainty. Create `lactate_missing` flag before imputing. Impute with median grouped by `admission_type`.

**`pao2_fio2_ratio`** (~11%) — MNAR. Requires an arterial blood gas draw, which is more invasive than routine labs. Not done in stable patients. Create `pao2_fio2_missing` flag. Impute with 400 (normal value) — the missingness flag carries the MNAR signal.

**`bilirubin`** (~7%) — Closer to MAR. Not always included in the initial lab panel. Create flag, impute with median grouped by `has_heart_failure` and `has_ckd` (both affect liver perfusion).

**`fluid_balance_24h`** (~5%) — Documentation gaps in nursing records. Impute with 0 (neutral fluid balance) after creating a `fluid_balance_missing` flag.

**Critical rule for all five:** Create the missingness flag BEFORE imputing. The flag is often more predictive than the imputed value, especially for `gcs_score` and `lactate`.

### Categorical inconsistency

**`icu_unit`**: "medical icu", "MEDICAL ICU", "Micu", "surgical icu", "SURGICAL ICU", "cardiac icu", "neuro icu", "trauma icu" — strip, title-case, then apply a lookup dict: `{'Micu': 'Medical ICU', 'Medical Icu': 'Medical ICU'}` etc. Final canonical values: Medical ICU, Surgical ICU, Cardiac ICU, Neuro ICU, Trauma ICU.

**`admission_type`**: "emergency", "EMERGENCY", "elective", "ELECTIVE", "urgent", "URGENT", "transfer", "TRANSFER" — strip and title-case handles all of these.

### Clinical validity checks — unique to healthcare data

Run these before modeling. They catch impossible values that suggest data entry errors:

```python
# BMI derived from height and weight — recompute and compare
df['bmi_check'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
bmi_mismatch = (df['bmi'] - df['bmi_check']).abs() > 2
print(f"BMI inconsistencies: {bmi_mismatch.sum()}")

# SpO2 min can never exceed SpO2 mean
spo2_invalid = df['spo2_min'] > df['spo2_mean']
print(f"SpO2 min > mean: {spo2_invalid.sum()}")

# HR min can never exceed HR mean
hr_invalid = df['hr_min'] > df['hr_mean']
print(f"HR min > mean: {hr_invalid.sum()}")

# MAP must be between DBP and SBP — rough check
map_invalid = (df['map_mean'] > df['sbp_mean']) | (df['map_mean'] < df['sbp_min'] * 0.5)
print(f"MAP out of plausible range: {map_invalid.sum()}")

# Potassium outside survivable range
k_invalid = (df['potassium'] < 2.0) | (df['potassium'] > 7.0)
print(f"Potassium physiologically implausible: {k_invalid.sum()}")
```

Fix or null any rows that fail these checks. In a real clinical ML project, data quality validation is a formal step with documented rules.

---

## EDA checklist

### Always check clinical score distributions first

- Histogram of `sofa_score` — should be right-skewed, most patients 0–6, fewer above 10.
- Histogram of `news2_score` — similarly right-skewed.
- Box plot: `sofa_score` by `deterioration_12h` — the separation between the boxes tells you how strong SOFA is as a standalone predictor. This is your baseline to beat.
- Box plot: `news2_score` by `deterioration_12h` — same analysis.

### ROC-AUC of SOFA and NEWS2 alone — do this before any ML

```python
from sklearn.metrics import roc_auc_score
print("SOFA AUROC:", roc_auc_score(df['deterioration_12h'], df['sofa_score']))
print("NEWS2 AUROC:", roc_auc_score(df['deterioration_12h'], df['news2_score']))
```

Write down these numbers. They are your clinical baseline. Every model you build must beat both of them to justify ML over a simple scoring system. If your Random Forest AUROC is 0.74 and SOFA alone scores 0.76, your model adds no value and should not be deployed.

### Vital sign distributions by deterioration status

For each vital sign, plot overlapping histograms for `deterioration_12h = 0` vs `deterioration_12h = 1`:
- `hr_max` — elevated max heart rate is a strong deterioration signal
- `spo2_min` — low minimum SpO2 is critical
- `map_min` — MAP below 65 is a physiological shock threshold
- `rr_mean` — tachypnea (RR > 25) is an early respiratory distress marker
- `lactate` — even mild elevation (> 2 mmol/L) matters

The plots should show meaningful separation. If they don't, verify your data generation.

### Lab value distributions

- Scatter: `lactate` vs `sofa_score`, colored by `deterioration_12h` — you should see that high lactate + high SOFA is almost exclusively in the deterioration group.
- Box plot: `creatinine` by `deterioration_12h` — renal function deteriorates early in sepsis.
- Bar chart: mean `deterioration_12h` rate by `charlson_index` bucket (0, 1–2, 3–5, 6+) — should show a clear step function.

### Intervention analysis

- Bar chart: deterioration rate by `on_vasopressor` (0 vs 1) — vasopressor use should strongly predict deterioration.
- Bar chart: deterioration rate by `on_mechanical_vent` (0 vs 1) — same.
- These are partially circular (vasopressors are given because the patient is already deteriorating) — note this as a potential data leakage issue in your notebook commentary.

### Missing value heatmap

`missingno.matrix()` colored by `deterioration_12h`. This is the single most important diagnostic plot for this dataset. If you can visually see that `gcs_score` and `pao2_fio2_ratio` missingness is concentrated in the deterioration = 1 rows, that confirms the MNAR assumption and validates your strategy of creating missingness flags.

---

## Feature engineering checklist

### Missingness flags — always first

```python
for col in ['gcs_score', 'lactate', 'pao2_fio2_ratio', 'bilirubin', 'fluid_balance_24h']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### Clinically meaningful thresholds — binary flags

These mirror the actual thresholds used in clinical practice:

```python
df['map_below_65']      = (df['map_min'] < 65).astype(int)       # shock threshold
df['spo2_below_90']     = (df['spo2_min'] < 90).astype(int)      # severe hypoxia
df['rr_above_25']       = (df['rr_mean'] > 25).astype(int)       # tachypnea
df['hr_above_130']      = (df['hr_max'] > 130).astype(int)       # severe tachycardia
df['lactate_elevated']  = (df['lactate'] > 2.0).astype(float)    # hyperlactatemia
df['lactate_critical']  = (df['lactate'] > 4.0).astype(float)    # critical lactatemia
df['temp_fever']        = (df['temp_max'] > 38.3).astype(int)    # fever threshold
df['temp_hypothermia']  = (df['temp_mean'] < 36.0).astype(int)   # hypothermia (bad sign)
df['gcs_severe']        = (df['gcs_score'] < 9).astype(float)    # severe neuro compromise
df['pf_ratio_ards']     = (df['pao2_fio2_ratio'] < 200).astype(float)  # ARDS threshold
df['creatinine_aki']    = (df['creatinine'] > 2.0).astype(int)   # AKI threshold
df['platelets_low']     = (df['platelet_count'] < 100).astype(int)     # thrombocytopenia
```

### Variability features — trend is more informative than absolute value

```python
df['hr_pulse_pressure'] = df['hr_max'] - df['hr_min']    # HR variability
df['sbp_range']         = df['sbp_max'] - df['sbp_min']  # BP variability
df['temp_range']        = df['temp_max'] - df['temp_mean']
```

High variability in vitals is often more alarming than a single abnormal reading. A patient whose BP swings 60 points in 6 hours is in worse shape than one with a stable low BP.

### Composite risk features

```python
# Shock index: HR / SBP — values > 1.0 indicate hemodynamic instability
df['shock_index'] = df['hr_mean'] / df['sbp_mean'].clip(1)

# Fluid responsiveness proxy: large positive balance in already-ventilated patient
df['fluid_overload_flag'] = (
    (df['fluid_balance_24h'] > 2000) & (df['on_mechanical_vent'] == 1)
).astype(int)

# Cumulative organ failure count
df['organ_failure_count'] = (
    (df['map_min'] < 65).astype(int)
    + (df['pao2_fio2_ratio'] < 200).astype(float)
    + (df['creatinine'] > 2.0).astype(int)
    + (df['bilirubin'] > 2.0).astype(float)
    + (df['platelet_count'] < 100).astype(int)
    + (df['gcs_score'] < 9).astype(float)
).fillna(0)

# Interaction: vasopressor + AKI (renal + cardiovascular failure together)
df['vasopress_x_aki'] = df['on_vasopressor'] * df['creatinine_aki']
```

### Log transforms for skewed lab values

```python
df['log_lactate']     = np.log1p(df['lactate'].fillna(0))
df['log_creatinine']  = np.log1p(df['creatinine'])
df['log_bilirubin']   = np.log1p(df['bilirubin'].fillna(0))
df['log_wbc']         = np.log1p(df['wbc'])
```

### Encoding strategy

- `icu_unit`, `admission_type`, `gender`: One-hot encode.
- Drop `patient_id`.
- Keep `sofa_score` and `news2_score` as features — they will likely rank highly in feature importance, which is the correct answer. Your model should use and extend them, not ignore them.

---

## Handling class imbalance

18% positive rate is less extreme than fraud detection (2.8%) but still requires deliberate handling. Use the same toolkit as the fraud project:

```python
# Option 1: class_weight='balanced' in model constructor
RandomForestClassifier(class_weight='balanced')
LogisticRegression(class_weight='balanced')

# Option 2: scale_pos_weight in XGBoost
scale = (df['deterioration_12h'] == 0).sum() / (df['deterioration_12h'] == 1).sum()
XGBClassifier(scale_pos_weight=scale)

# Option 3: SMOTE via imblearn pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
```

At 18% imbalance, `class_weight='balanced'` is usually sufficient. SMOTE adds complexity without proportional gain at this imbalance ratio. Try both and compare recall on the positive class — that is your primary concern.

---

## Modeling checklist

### Baselines to beat (not optional)

Before any ML model:

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Clinical score baselines
print("SOFA AUROC:", roc_auc_score(y_test, sofa_test))
print("NEWS2 AUROC:", roc_auc_score(y_test, news2_test))
print("SOFA PR-AUC:", average_precision_score(y_test, sofa_test))
print("NEWS2 PR-AUC:", average_precision_score(y_test, news2_test))

# Majority class baseline
print("Majority class accuracy:", 1 - y_test.mean())
```

Write these down. Your model's success is measured against SOFA's AUROC first.

### Models to train in order

**1. Logistic Regression with clinical features only**
Start with the features a clinician would recognise: `sofa_score`, `news2_score`, `lactate`, `map_min`, `spo2_min`, `gcs_score`, `on_vasopressor`, `creatinine`, `patient_age`, `charlson_index`. This is your "clinician-informed" baseline. Check that all coefficients have the correct sign. Positive coefficient on `sofa_score` = clinically valid. Negative coefficient = data problem.

**2. Logistic Regression with full feature set + regularization**
Add all engineered features. Use `LogisticRegressionCV` with L2 (Ridge) penalty. Compare AUROC to the clinical-features-only version. If full features don't improve over clinical-only, your feature engineering didn't add signal.

**3. Random Forest Classifier**
Non-linear benchmark. Set `class_weight='balanced'`. Extract feature importances — `sofa_score` and `news2_score` should appear in the top 5. If they don't, something is wrong with the preprocessing pipeline. Plot top 20 feature importances.

**4. XGBoost with SHAP explanations**
Best expected performance. After training, compute SHAP values:

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot — shows which features push predictions highest
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for a single high-risk patient
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

The SHAP force plot for a single patient — showing exactly which features pushed the deterioration probability from 18% base to 84% for this specific patient — is a compelling portfolio visual. It's also exactly how clinical decision support systems present model output to clinicians.

**5. Threshold calibration — critical for clinical deployment**
The default 0.5 threshold optimizes accuracy, not clinical utility. In ICU early warning, you want high recall (catch as many deteriorations as possible) at the cost of some false positives. Plot the threshold sensitivity table:

```python
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
for t in thresholds:
    preds = (proba >= t).astype(int)
    sens  = recall_score(y_test, preds)         # sensitivity (recall)
    spec  = recall_score(y_test, preds, pos_label=0)  # specificity
    ppv   = precision_score(y_test, preds)      # positive predictive value
    print(f"Threshold {t:.2f} | Sensitivity: {sens:.3f} | Specificity: {spec:.3f} | PPV: {ppv:.3f}")
```

For an ICU early warning system, a sensitivity of 0.80+ at a specificity of 0.70+ is a clinically reasonable operating point. Below 0.70 sensitivity, you're missing too many deteriorations to justify deployment.

### Evaluation metrics

- **AUROC** — primary comparison metric against SOFA and NEWS2 baselines.
- **PR-AUC** — better than AUROC for imbalanced data. At 18% positive rate, a random classifier scores 0.18.
- **Sensitivity (Recall)** — proportion of deteriorations caught. Clinical teams care about this most.
- **Specificity** — proportion of non-deteriorations correctly cleared. Alarm fatigue is a real problem; too many false positives cause nurses to ignore alerts.
- **Confusion matrix at chosen threshold** — report actual TP, FP, TN, FN counts, not just rates. "The model missed 47 deteriorations and generated 312 false alarms per 1,000 patients" is how a clinical committee evaluates this.
- **Calibration curve** — plot predicted probability vs actual deterioration rate in decile bins. A well-calibrated model at 0.6 probability should have ~60% of patients in that bin actually deteriorate. Poor calibration undermines clinical trust even when AUROC is high.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Calibrate if needed
calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

fraction_pos, mean_pred = calibration_curve(y_test, calibrated.predict_proba(X_test)[:,1], n_bins=10)
plt.plot(mean_pred, fraction_pos, marker='o', label='Calibrated model')
plt.plot([0,1],[0,1], linestyle='--', label='Perfect calibration')
```

---

## Deployment — clinical ML has unique requirements

### This is not a standard software deployment

Clinical ML systems fall under medical device regulation in most jurisdictions. In the US, the FDA regulates AI/ML-based Software as a Medical Device (SaMD). In the EU, it falls under the Medical Device Regulation (MDR). A real ICU early warning system would require:

- **Clinical validation study** — prospective evaluation in real patients before deployment.
- **Institutional review** — hospital ethics committee approval.
- **Integration with EHR** — HL7 FHIR or vendor-specific APIs (Epic, Cerner, etc.) for real-time data ingestion.
- **Alert fatigue management** — governance around alert thresholds, escalation protocols, and clinician override rates.

For your portfolio, focus on the technical deployment architecture and acknowledge the regulatory layer explicitly. That awareness distinguishes serious ML engineers from those who haven't thought through the clinical context.

### Step 1 — Serialize the full pipeline

```python
import joblib

joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(xgb_model, 'models/deterioration_model.pkl')
joblib.dump(calibrated, 'models/calibrated_model.pkl')

import json
json.dump({
    'alert_threshold': 0.25,
    'model_version': 'v1.0',
    'training_date': '2024-01-01',
    'sofa_auroc_baseline': 0.74,
    'model_auroc': 0.86
}, open('models/model_card.json', 'w'))
```

Saving a `model_card.json` is good practice — it records what the model was trained on, what baseline it was compared to, and what threshold is being used in production. This is the clinical equivalent of the `config.json` from the fraud project, but with more accountability context.

### Step 2 — Inference API

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, numpy as np, pandas as pd

app         = FastAPI()
preprocessor = joblib.load('models/preprocessor.pkl')
model        = joblib.load('models/calibrated_model.pkl')
card         = json.load(open('models/model_card.json'))
THRESHOLD    = card['alert_threshold']

class PatientObservation(BaseModel):
    patient_age: int
    sofa_score: int
    news2_score: int
    hr_mean: float
    hr_max: float
    sbp_mean: float
    map_min: float
    spo2_min: float
    rr_mean: float
    temp_max: float
    gcs_score: float | None
    lactate: float | None
    creatinine: float
    pao2_fio2_ratio: float | None
    on_vasopressor: int
    on_mechanical_vent: int
    charlson_index: int
    # ... all other features

@app.post('/predict')
def predict(obs: PatientObservation):
    df = pd.DataFrame([obs.dict()])
    X  = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    alert = prob >= THRESHOLD
    return {
        'deterioration_probability': round(prob, 4),
        'alert_triggered': alert,
        'alert_threshold': THRESHOLD,
        'risk_level': 'HIGH' if prob >= 0.5 else 'ELEVATED' if prob >= THRESHOLD else 'NORMAL',
        'model_version': card['model_version'],
    }

@app.get('/model_info')
def model_info():
    return card
```

The `/model_info` endpoint exposes the model card — what version is running, what threshold, what the training AUROC was. In a hospital setting, this metadata is required for audit trails.

### Step 3 — Containerize and deploy

Same Dockerfile pattern as previous projects. One difference: add a health check endpoint that verifies the model can produce a prediction, not just that the server is running.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

For clinical systems, deployment is almost always on-premise (within the hospital network) or on a HIPAA-compliant cloud (AWS GovCloud, Azure Government, or GCP Healthcare API) — not on Railway or Render. For your portfolio project, deploy on Railway and note that a real deployment would require HIPAA Business Associate Agreements with the cloud provider.

### Step 4 — Monitoring in a clinical context

**Alert fatigue tracking:** Log every alert fired and whether it was acknowledged, overridden, or acted upon by clinical staff. High override rates (> 40%) indicate your threshold is too low and is generating too many false positives.

**Sensitivity monitoring:** As ground truth labels become available (patients who deteriorated), compute rolling sensitivity on the last 30 days of predictions. If sensitivity drops below 0.70, retrain or escalate to clinical review.

**Population shift:** ICU patient mix changes with seasons (flu season, COVID waves), hospital capacity, and referral patterns. Track distributions of `sofa_score`, `admission_type`, and `charlson_index` monthly against training baselines. A significant shift in patient acuity means the model is operating on a different population than it was trained on.

**Model decay timeline:** Clinical models typically need retraining every 12–18 months as treatment protocols evolve. A model trained before a major antibiotic protocol change may have systemically biased lactate-to-outcome relationships.

---

## Project structure

```
icu_deterioration_project/
│
├── data/
│   └── raw/
│       └── icu_patients.csv          # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Clinical score baselines, vital distributions
│   ├── 02_feature_engineering.ipynb  # Missingness flags, thresholds, composites
│   ├── 03_modeling.ipynb             # LR → RF → XGBoost, threshold calibration
│   └── 04_explainability.ipynb       # SHAP values, force plots, calibration curves
│
├── src/
│   ├── validate_clinical.py          # Clinical validity checks (BMI, SpO2 bounds etc.)
│   └── pipeline.py                   # Full preprocessing + feature engineering
│
├── models/
│   ├── preprocessor.pkl
│   ├── deterioration_model.pkl
│   ├── calibrated_model.pkl
│   └── model_card.json               # Version, threshold, baseline AUROC
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
missingno>=0.5
jupyterlab>=4.0
fastapi>=0.110
uvicorn>=0.29
joblib>=1.3
pydantic>=2.0
evidently>=0.4
```

---

## How this project differs from the fraud detection project

| Topic | Fraud Detection | ICU Deterioration |
|---|---|---|
| Positive rate | 2.8% | 18% |
| Baseline to beat | Naive majority class | Validated clinical scores (SOFA, NEWS2) |
| Missing data mechanism | Mostly MAR | Heavily MNAR — missingness is clinical signal |
| Coefficient check | Feature direction matters | Feature direction is medically verifiable |
| Threshold choice | Business cost ratio | Clinical sensitivity/specificity tradeoff |
| Explainability | Feature importances | SHAP force plots per patient |
| Deployment context | Payments API | Clinical decision support, regulatory oversight |
| Monitoring priority | Score drift, input drift | Alert fatigue rate, sensitivity over time |
| Retraining trigger | Fraud pattern shift | Treatment protocol changes, population shift |

---

*Dataset is fully synthetic. No real patient data was used. Clinical thresholds are based on published scoring systems (SOFA, NEWS2) but this dataset is not validated for clinical use.*
