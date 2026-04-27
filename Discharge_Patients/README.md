# 30-Day Hospital Readmission Risk Scoring — End-to-End ML Project

**Targets:** `readmission_risk_score` (continuous 0–100, regression) + `readmitted_30d` (binary, classification)  
**Model focus:** Regression + Ranking — predict a continuous risk score, then rank patients by that score for care management prioritization  
**Domain:** Hospital discharge planning / care transitions  
**Dataset:** 9,000 synthetic discharge records, deliberately dirty  
**Readmission rate:** ~19.5%  
**Project emphasis:** Feature engineering depth, preprocessing rigor, and visualization storytelling — this README is heavier on those three pillars than any previous project

---

## Why this problem matters operationally

The Centers for Medicare & Medicaid Services (CMS) runs the Hospital Readmissions Reduction Program (HRRP), which financially penalizes hospitals with excess 30-day readmission rates for heart failure, COPD, pneumonia, hip fracture, stroke, and CABG. In 2023 alone, CMS reduced payments to over 2,500 hospitals. A hospital that can accurately identify high-risk patients at discharge — and intervene with follow-up calls, home nursing visits, or medication reconciliation — directly avoids those penalties.

The output of this model is not a binary yes/no. It's a **ranked list of patients scored 0–100**, sorted from highest to lowest readmission risk, delivered to the care management team each morning. The team has capacity to follow up with the top 40–60 patients per day. Your model determines who those patients are.

This framing — regression as a ranking tool — is what separates this project from the previous classification projects. Precision at the top of the ranked list matters more than aggregate AUROC.

---

## Two targets, two uses

The dataset has both a continuous score (`readmission_risk_score`) and a binary label (`readmitted_30d`). Use both deliberately:

- **Regression on `readmission_risk_score`** — train your model to predict the continuous score. Evaluate with RMSE and MAE. Use predictions to rank patients.
- **Classification on `readmitted_30d`** — use as a secondary evaluation. Convert your regression predictions to probabilities and compute AUROC, PR-AUC, and the confusion matrix at your chosen operating threshold.
- **Ranking evaluation** — compute Precision@K: of the top K patients your model flags as highest risk, what fraction actually readmitted? This is the metric that maps directly to care team capacity.

---

## Dataset columns

### Demographics and social context

| Column | Type | Notes |
|---|---|---|
| `patient_id` | string | Drop before modeling |
| `age` | int | 18–95 |
| `gender` | categorical | Male / Female |
| `race_ethnicity` | categorical | White / Black / Hispanic / Asian / Other |
| `insurance_type` | categorical | **dirty** — Medicare / Medicaid / Private / Uninsured / VA |
| `zip_income_decile` | float | **~5% missing** — neighborhood income rank (1=poorest, 10=richest) |
| `lives_alone` | binary | No household support |
| `has_caregiver` | binary | Designated caregiver at home |

### Admission and discharge

| Column | Type | Notes |
|---|---|---|
| `primary_diagnosis` | categorical | **dirty** — 8 diagnosis categories + abbreviation variants |
| `admission_source` | categorical | Emergency Dept / Direct Admit / Transfer / Observation |
| `admission_month` | int | 1–12 (seasonality signal) |
| `admission_year` | int | 2021–2023 |
| `los_days` | float | Length of stay in days (right-skewed) |
| `discharge_disposition` | categorical | **dirty** — Home / Home with Services / SNF / Rehab / Hospice / AMA |
| `num_procedures` | int | Procedures performed during admission |

### Comorbidities

| Column | Type | Notes |
|---|---|---|
| `has_chf` / `has_copd` / `has_diabetes` | binary | Primary diagnosis flags |
| `has_ckd` / `has_depression` / `has_dementia` | binary | Secondary comorbidities |
| `has_cancer` / `has_afib` | binary | |
| `num_comorbidities` | int | Total count (0–12) |
| `elixhauser_score` | float | Composite comorbidity severity score |

### Discharge labs

| Column | Type | Notes |
|---|---|---|
| `discharge_sodium` | float | Hyponatremia (< 135) = readmission risk |
| `discharge_bun` | float | Blood urea nitrogen — renal/hydration marker |
| `discharge_creatinine` | float | Renal function |
| `discharge_hemoglobin` | float | Anemia marker |
| `discharge_wbc` | float | Infection/inflammation marker |
| `discharge_albumin` | float | **~10% missing** — nutritional/inflammation marker, MNAR |
| `discharge_bnp` | float | **~12% missing** — cardiac stress marker, MNAR for non-cardiac |

### Discharge vitals

| Column | Type | Notes |
|---|---|---|
| `dc_heart_rate` | float | At time of discharge |
| `dc_systolic_bp` | float | |
| `dc_resp_rate` | float | |
| `dc_spo2` | float | SpO2 < 93% at discharge = high risk |
| `dc_temperature` | float | |

### Medications and follow-up

| Column | Type | Notes |
|---|---|---|
| `num_discharge_meds` | int | Total medications at discharge |
| `med_changes_at_discharge` | int | Number of medication changes made at discharge |
| `followup_appt_days` | float | **~10% missing** — days until scheduled follow-up |
| `prior_admissions_12m` | float | **~4% missing** — admissions in prior 12 months, MNAR for transfers |
| `prior_ed_visits_6m` | int | ED visits in prior 6 months |

### Social determinants of health (SDOH)

| Column | Type | Notes |
|---|---|---|
| `substance_use_flag` | binary | Active substance use disorder |
| `housing_instability` | binary | Unstable housing or homelessness |
| `food_insecurity` | binary | Documented food insecurity |
| `transportation_barrier` | binary | No reliable transportation |
| `limited_english` | binary | Limited English proficiency |

### Targets

| Column | Type | Notes |
|---|---|---|
| `readmission_risk_score` | float | **Primary target** — continuous 0–100 risk score |
| `readmitted_30d` | binary | **Secondary target** — binary outcome for AUROC evaluation |

---

## Data issues to fix

### Duplicates
~32 duplicate rows from EHR export overlap. Drop before any analysis.

### Missing values — five columns, each needs its own strategy

**`discharge_albumin`** (~10%) — Albumin is not always ordered unless malnutrition or liver disease is suspected. This is MNAR: sicker and malnourished patients are more likely to have it ordered AND more likely to have low values — but the completely missing cases are often patients where no one thought to check. Create flag `albumin_missing = 1`, then impute with median grouped by `primary_diagnosis`. Malnutrition risk varies substantially by diagnosis (sepsis vs elective hip fracture).

**`discharge_bnp`** (~12%) — BNP is a cardiac stress biomarker ordered almost exclusively for heart failure and related conditions. Non-cardiac patients are missing it systematically. Do not impute BNP with the population median — that would assign a spurious cardiac signal to every stroke and diabetes patient. Correct approach: create `bnp_missing = 1` flag, impute with 0 for non-cardiac diagnoses, and impute with median within Heart Failure patients for their small missing fraction.

**`followup_appt_days`** (~10%) — Missing because no appointment was scheduled at discharge. This is itself a high-risk signal. Do not impute. Create `no_followup_scheduled = 1` where null, and fill the numeric column with 999 (a sentinel value meaning "no appointment"). Then cap the column at 30 for modeling (anything > 30 days = effectively no near-term follow-up for a 30-day outcome window).

**`prior_admissions_12m`** (~4%) — Mostly missing for Transfer patients who came from outside the hospital system and have no prior record here. Create `prior_admissions_unknown = 1`, impute with the median for Transfer patients specifically. Do not impute transfers with the full population median — they are a structurally different group.

**`zip_income_decile`** (~5%) — Missing address data. Create flag, impute with median (5 — middle decile). The flag preserves information about data completeness, which correlates with healthcare access.

### Categorical inconsistency — three columns

**`primary_diagnosis`**: The most complex cleaning task. Map all variants:
```python
diag_map = {
    'heart failure': 'Heart Failure', 'chf': 'Heart Failure',
    'copd': 'COPD', 'pneumonia': 'Pneumonia',
    'sepsis': 'Sepsis', 'sepsis ': 'Sepsis',
    'diabetes': 'Diabetes', 'aki ': 'AKI', 'aki': 'AKI',
    'stroke': 'Stroke',
}
df['primary_diagnosis'] = (df['primary_diagnosis']
    .str.strip().str.lower()
    .map(lambda x: diag_map.get(x, x))
    .str.title()
)
# Special cases that title-case breaks:
df['primary_diagnosis'] = df['primary_diagnosis'].replace({
    'Copd': 'COPD', 'Aki': 'AKI', 'Chf': 'Heart Failure'
})
```

**`discharge_disposition`**: "home", "HOME", "home with services", "snf", "SNF", "Snf", "rehab", "REHAB", "ama", "AMA" — strip, title-case, then fix: "Snf" → "SNF", "Ama" → "AMA".

**`insurance_type`**: "medicare", "MEDICARE", "medicaid", "MEDICAID", "va", "VA" — strip, title-case handles most. Fix "Va" → "VA".

### Clinical validity checks

```python
# SpO2 at discharge should not be below 80% in a discharged patient
spo2_invalid = df['dc_spo2'] < 82
print(f"Implausible SpO2 at discharge: {spo2_invalid.sum()}")

# LOS of 0 days is physiologically impossible
los_zero = df['los_days'] < 0.5
print(f"LOS < 0.5 days: {los_zero.sum()}")

# BUN/Creatinine ratio — very high ratios may indicate transcription error
df['bun_cr_ratio'] = df['discharge_bun'] / df['discharge_creatinine'].clip(0.1)
bun_implausible = df['bun_cr_ratio'] > 100
print(f"Implausible BUN/Creatinine ratio: {bun_implausible.sum()}")

# Age consistency: dementia in a 22-year-old warrants review
young_dementia = (df['age'] < 40) & (df['has_dementia'] == 1)
print(f"Dementia in patient under 40: {young_dementia.sum()}")
```

---

## Feature engineering — the deep focus section

This is the heart of the project. Go beyond obvious ratio features. Each block below has a clinical rationale.

### Block 1 — Missingness flags (always first)

```python
miss_cols = ['discharge_albumin', 'discharge_bnp', 'followup_appt_days',
             'prior_admissions_12m', 'zip_income_decile']
for col in miss_cols:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### Block 2 — Clinical threshold flags

These mirror evidence-based cutoffs used in clinical risk tools:

```python
# Hyponatremia at discharge — strong readmission predictor for CHF and cirrhosis
df['hyponatremia'] = (df['discharge_sodium'] < 135).astype(int)

# Hypoalbuminemia — malnutrition / inflammation / poor wound healing
df['low_albumin'] = (df['discharge_albumin'] < 3.0).astype(float)  # float preserves nan

# Anemia at discharge
df['anemia_at_discharge'] = (df['discharge_hemoglobin'] < 10.0).astype(int)

# Residual infection / inflammation
df['elevated_wbc'] = (df['discharge_wbc'] > 11.0).astype(int)

# Renal impairment — Stage 3+ CKD threshold
df['creatinine_elevated'] = (df['discharge_creatinine'] > 2.0).astype(int)

# Low SpO2 at discharge — patient not ready to go home
df['spo2_below_93'] = (df['dc_spo2'] < 93).astype(int)

# Tachycardia at discharge
df['tachycardia_at_dc'] = (df['dc_heart_rate'] > 100).astype(int)

# Fever at discharge — active infection not resolved
df['fever_at_dc'] = (df['dc_temperature'] > 37.8).astype(int)

# High BNP — ongoing cardiac stress (use filled version)
df['bnp_high'] = (df['discharge_bnp'] > 500).astype(float)

# Polypharmacy — > 10 medications is a care coordination risk
df['polypharmacy'] = (df['num_discharge_meds'] > 10).astype(int)
```

### Block 3 — Utilization history features

Prior utilization is one of the strongest readmission predictors across all diagnosis groups:

```python
# High prior utilizer — the single strongest predictor in most readmission models
df['prior_high_utilizer'] = (df['prior_admissions_12m'] >= 2).astype(float)

# Any prior ED visit (binary)
df['any_prior_ed'] = (df['prior_ed_visits_6m'] > 0).astype(int)

# Combined utilization burden
df['utilization_burden'] = (
    df['prior_admissions_12m'].fillna(0) * 2
    + df['prior_ed_visits_6m']
)

# Ratio of ED visits to total visits — patterns of crisis-driven care
df['ed_to_total_ratio'] = (
    df['prior_ed_visits_6m'] /
    (df['prior_admissions_12m'].fillna(0) + df['prior_ed_visits_6m'] + 1)
).round(3)
```

### Block 4 — Social determinants composite

Individual SDOH flags are weak individually. A composite score has more predictive power:

```python
df['sdoh_burden'] = (
    df['lives_alone']
    + df['housing_instability']
    + df['food_insecurity']
    + df['transportation_barrier']
    + df['limited_english']
    + df['substance_use_flag']
    + (df['has_caregiver'] == 0).astype(int)
    + (df['zip_income_decile'].fillna(5) < 4).astype(int)
)

# Low social support: lives alone, no caregiver, AND at least one other SDOH
df['low_social_support'] = (
    (df['lives_alone'] == 1) &
    (df['has_caregiver'] == 0)
).astype(int)

# Economic vulnerability composite
df['economic_vulnerability'] = (
    (df['insurance_type'].isin(['Medicaid', 'Uninsured'])).astype(int)
    + (df['zip_income_decile'].fillna(5) <= 3).astype(int)
    + df['food_insecurity']
    + df['housing_instability']
)
```

### Block 5 — Discharge complexity and care transition risk

```python
# AMA (Against Medical Advice) is the highest-risk single discharge disposition
df['ama_discharge'] = (df['discharge_disposition'] == 'AMA').astype(int)

# No structured follow-up within 7 days — evidence-based transition of care gap
df['no_early_followup'] = (
    (df['followup_appt_days'].fillna(999) > 7)
).astype(int)

# Follow-up within 3 days — strong protective factor
df['early_followup_3d'] = (df['followup_appt_days'] <= 3).astype(float)

# Medication burden: many changes at discharge = reconciliation complexity
df['high_med_complexity'] = (
    (df['num_discharge_meds'] > 10) &
    (df['med_changes_at_discharge'] >= 3)
).astype(int)

# LOS bucket: very short stays often mean premature discharge
df['los_under_2d'] = (df['los_days'] < 2).astype(int)
df['los_over_10d'] = (df['los_days'] > 10).astype(int)

# Discharge to SNF/Rehab is protective vs home (supervised recovery)
df['supervised_discharge'] = df['discharge_disposition'].isin(
    ['SNF', 'Rehab']
).astype(int)
```

### Block 6 — Interaction features

This is where the project gets analytically interesting. These combinations are non-obvious and require domain knowledge to construct:

```python
# Cardiac patient with high BNP AND no early follow-up — the textbook readmission profile
df['chf_no_followup'] = (
    (df['has_chf'] == 1) & (df['no_early_followup'] == 1)
).astype(int)

# High utilizer who lives alone — no safety net to catch deterioration
df['high_util_isolated'] = (
    (df['prior_high_utilizer'] == 1) & (df['low_social_support'] == 1)
).astype(float)

# Elderly + multiple comorbidities + polypharmacy = very high complexity
df['geriatric_complexity'] = (
    (df['age'] >= 75).astype(int)
    + (df['num_comorbidities'] >= 4).astype(int)
    + df['polypharmacy']
    + df['has_dementia']
).clip(0, 4)

# Discharged home without services AND has SDOH barriers
df['home_unmanaged'] = (
    (df['discharge_disposition'] == 'Home') &
    (df['sdoh_burden'] >= 2)
).astype(int)

# Substance use + housing instability = care continuity breakdown
df['social_crisis_risk'] = (
    (df['substance_use_flag'] == 1) & (df['housing_instability'] == 1)
).astype(int)
```

### Block 7 — Log and power transformations

```python
import numpy as np

df['log_los']              = np.log1p(df['los_days'])
df['log_bnp']              = np.log1p(df['discharge_bnp'].fillna(0))
df['log_bun']              = np.log1p(df['discharge_bun'])
df['log_prior_admissions'] = np.log1p(df['prior_admissions_12m'].fillna(0))
df['sqrt_num_meds']        = np.sqrt(df['num_discharge_meds'])

# BUN-to-creatinine ratio: dehydration / prerenal failure marker
df['bun_cr_ratio']         = (
    df['discharge_bun'] / df['discharge_creatinine'].clip(0.1)
).round(2)
```

### Block 8 — Temporal and seasonal features

```python
# Winter months = higher pneumonia/COPD readmission seasonality
df['is_winter'] = df['admission_month'].isin([12, 1, 2]).astype(int)
df['is_flu_season'] = df['admission_month'].isin([10, 11, 12, 1, 2, 3]).astype(int)

# Calendar quarter
df['admission_quarter'] = pd.cut(
    df['admission_month'], bins=[0,3,6,9,12],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
```

### Block 9 — Encoding strategy

- **`discharge_disposition`**: Ordinal encode by readmission risk order — AMA=5, Home=4, Home with Services=3, SNF=2, Rehab=1, Hospice=0. AMA is highest risk, Hospice patients are rarely readmitted (alternative pathway). This ordering is clinically defensible.
- **`primary_diagnosis`**, **`insurance_type`**, **`race_ethnicity`**, **`admission_source`**: One-hot encode.
- **`admission_quarter`**: One-hot encode (no natural numerical order).
- Drop `patient_id`, `admission_month` (captured by quarter/season flags).

---

## Visualization guide — the main emphasis of this project

Every visualization below has a purpose. Don't make charts to check a box — each one should lead to a modeling or feature engineering decision.

### Notebook 01 — Data quality and distributions

**Plot 1 — Missing value heatmap with outcome overlay**
```python
import missingno as msno
import matplotlib.pyplot as plt

# Sort rows by readmitted_30d to see if missingness clusters in one outcome group
df_sorted = df.sort_values('readmitted_30d')
msno.matrix(df_sorted[miss_cols + ['readmitted_30d']], figsize=(12, 6),
            color=(0.27, 0.52, 0.70))
plt.title('Missingness pattern sorted by 30-day readmission outcome')
```
What to look for: if you see horizontal bands of missing data that align with the readmitted rows, missingness is MNAR. That confirms your flag-first imputation strategy.

**Plot 2 — Target distribution (both targets side by side)**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['readmission_risk_score'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Continuous Risk Score Distribution')
axes[0].set_xlabel('Readmission Risk Score (0–100)')
axes[0].axvline(df['readmission_risk_score'].median(), color='red',
                linestyle='--', label=f"Median: {df['readmission_risk_score'].median():.1f}")
axes[0].legend()

axes[1].bar(['Not Readmitted (0)', 'Readmitted (1)'],
            df['readmitted_30d'].value_counts().sort_index().values,
            color=['steelblue', 'tomato'])
axes[1].set_title(f"30-Day Readmission (Rate: {df['readmitted_30d'].mean():.1%})")
```
Note the right-skew in the risk score. This motivates the ranking approach — the model needs to separate the top 10–20% from the rest, not predict exact scores.

**Plot 3 — LOS distribution by diagnosis**
```python
import seaborn as sns

plt.figure(figsize=(13, 5))
order = df.groupby('primary_diagnosis')['los_days'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='primary_diagnosis', y='los_days', order=order,
            palette='Blues_d', fliersize=2)
plt.yscale('log')
plt.xticks(rotation=30, ha='right')
plt.title('Length of Stay by Primary Diagnosis (log scale)')
```
Sepsis and stroke should have the longest and most variable stays. Hip fracture and COPD should be more consistent.

---

### Notebook 02 — EDA and target relationships

**Plot 4 — Readmission rate by primary diagnosis (the most important single plot)**
```python
diag_rates = (df.groupby('primary_diagnosis')['readmitted_30d']
                .agg(['mean', 'count'])
                .sort_values('mean', ascending=False)
                .reset_index())

fig, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(diag_rates['primary_diagnosis'], diag_rates['mean'],
               color='tomato', alpha=0.8)
ax1.axhline(df['readmitted_30d'].mean(), color='black', linestyle='--',
            label=f"Overall rate: {df['readmitted_30d'].mean():.1%}")
ax1.set_ylabel('30-Day Readmission Rate')
ax1.set_ylim(0, 0.45)

ax2 = ax1.twinx()
ax2.plot(diag_rates['primary_diagnosis'], diag_rates['count'],
         'o--', color='steelblue', label='Volume')
ax2.set_ylabel('Patient Volume')
plt.title('Readmission Rate and Volume by Primary Diagnosis')
fig.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
```
This dual-axis chart is critical for the hospital. A high readmission rate in a low-volume diagnosis (e.g., hip fracture) may still matter less operationally than a moderate rate in a high-volume diagnosis (e.g., heart failure).

**Plot 5 — Readmission rate by discharge disposition (heatmap-style)**
```python
disp_diag = (df.groupby(['discharge_disposition', 'primary_diagnosis'])
               ['readmitted_30d'].mean()
               .unstack()
               .round(3))

plt.figure(figsize=(13, 6))
sns.heatmap(disp_diag, annot=True, fmt='.2f', cmap='RdYlGn_r',
            linewidths=0.5, vmin=0, vmax=0.4)
plt.title('Readmission Rate: Discharge Disposition × Primary Diagnosis')
plt.xlabel('Primary Diagnosis')
plt.ylabel('Discharge Disposition')
```
The AMA × any diagnosis cell should be the darkest red. SNF × Heart Failure is often surprisingly high (sickest patients get sent to SNF, still readmit). This heatmap reveals which disposition-diagnosis combinations need targeted intervention programs.

**Plot 6 — Social determinants: individual vs composite**
```python
sdoh_cols = ['lives_alone', 'housing_instability', 'food_insecurity',
             'transportation_barrier', 'substance_use_flag', 'limited_english']

sdoh_rates = {col: df.groupby(col)['readmitted_30d'].mean() for col in sdoh_cols}
rate_diffs  = {col: rates[1] - rates[0] for col, rates in sdoh_rates.items()}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: individual SDOH factor readmission rate lift
axes[0].barh(list(rate_diffs.keys()), list(rate_diffs.values()),
             color=['tomato' if v > 0 else 'steelblue' for v in rate_diffs.values()])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Readmission Rate Lift vs Baseline')
axes[0].set_title('Readmission Rate Lift per SDOH Factor')

# Right: SDOH burden score vs readmission rate
burden_rates = df.groupby('sdoh_burden')['readmitted_30d'].agg(['mean', 'count'])
axes[1].bar(burden_rates.index, burden_rates['mean'], color='tomato', alpha=0.8)
axes[1].set_xlabel('SDOH Burden Score (0–8)')
axes[1].set_ylabel('Readmission Rate')
axes[1].set_title('Readmission Rate by SDOH Burden Composite')
```
This side-by-side shows the value of the composite: individual SDOH factors have modest lift, but the composite score rises steeply. That justifies creating the composite feature.

**Plot 7 — Prior utilization: the strongest predictor**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prior admissions
util_rates = df.groupby('prior_admissions_12m')['readmitted_30d'].mean()
axes[0].bar(util_rates.index, util_rates.values, color='steelblue')
axes[0].axhline(df['readmitted_30d'].mean(), color='red', linestyle='--')
axes[0].set_xlabel('Prior Admissions in Last 12 Months')
axes[0].set_ylabel('30-Day Readmission Rate')
axes[0].set_title('Readmission Rate by Prior Admission History')

# Heatmap: prior admissions × prior ED visits
pivot = df.groupby(['prior_admissions_12m', 'prior_ed_visits_6m'])['readmitted_30d'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
            ax=axes[1], linewidths=0.5, vmin=0.05, vmax=0.55)
axes[1].set_title('Readmission Rate: Prior Admissions × Prior ED Visits')
```
The heatmap is the visualization that most clearly justifies the `utilization_burden` composite feature. The top-right corner (3+ admissions, 3+ ED visits) should be deeply red.

**Plot 8 — Lab values at discharge: violin plots by outcome**
```python
lab_cols = ['discharge_albumin', 'discharge_sodium', 'discharge_creatinine',
            'discharge_hemoglobin', 'discharge_bun']

fig, axes = plt.subplots(1, len(lab_cols), figsize=(18, 5))
for ax, col in zip(axes, lab_cols):
    df_plot = df[[col, 'readmitted_30d']].dropna()
    sns.violinplot(data=df_plot, x='readmitted_30d', y=col,
                   palette=['steelblue', 'tomato'], ax=ax, inner='box')
    ax.set_title(col.replace('discharge_', '').replace('_', ' ').title())
    ax.set_xlabel('')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['No\nReadmit', 'Readmit'])
plt.suptitle('Discharge Lab Values by 30-Day Readmission Outcome', y=1.02)
plt.tight_layout()
```
Albumin should show the widest separation — lower albumin in readmitted patients. BUN separation reveals dehydration/renal dysfunction patterns. Save this plot for the final report.

**Plot 9 — Correlation heatmap (engineered features only)**

After feature engineering, plot a correlation heatmap of your engineered features vs `readmitted_30d`. Use a diverging colormap centered at zero. Features should show both positive (sdoh_burden, high_utilizer, ama_discharge) and negative (early_followup_3d, supervised_discharge, has_caregiver) correlations.

```python
eng_features = ['sdoh_burden', 'utilization_burden', 'geriatric_complexity',
                'high_med_complexity', 'low_social_support', 'chf_no_followup',
                'no_early_followup', 'polypharmacy', 'ama_discharge',
                'early_followup_3d', 'supervised_discharge',
                'low_albumin', 'creatinine_elevated', 'hyponatremia']

corr = df[eng_features + ['readmitted_30d']].corr()['readmitted_30d'].drop('readmitted_30d')
corr_sorted = corr.sort_values()

colors = ['tomato' if v > 0 else 'steelblue' for v in corr_sorted]
plt.figure(figsize=(10, 8))
corr_sorted.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Engineered Feature Correlation with 30-Day Readmission')
plt.xlabel('Pearson Correlation with readmitted_30d')
```

---

### Notebook 03 — Modeling and Ranking evaluation

**Plot 10 — Precision@K curve (the ranking evaluation)**
```python
def precision_at_k(y_true, y_scores, k_values):
    sorted_idx = np.argsort(y_scores)[::-1]
    results = []
    for k in k_values:
        top_k = sorted_idx[:k]
        results.append(y_true.iloc[top_k].mean())
    return results

k_values = list(range(10, 500, 10))

# Compute for each model and for random baseline
p_at_k_xgb    = precision_at_k(y_test, xgb_scores, k_values)
p_at_k_ridge  = precision_at_k(y_test, ridge_scores, k_values)
p_at_k_random = [y_test.mean()] * len(k_values)

plt.figure(figsize=(11, 6))
plt.plot(k_values, p_at_k_xgb, label='XGBoost', color='tomato', linewidth=2)
plt.plot(k_values, p_at_k_ridge, label='Ridge', color='steelblue', linewidth=2)
plt.axhline(y_test.mean(), color='gray', linestyle='--', label='Random baseline')
plt.xlabel('K (Top-K patients flagged)')
plt.ylabel('Precision@K (fraction who actually readmit)')
plt.title('Precision@K — Care Team Can Follow Up With Top-K Patients Per Day')
plt.axvline(50, color='black', linestyle=':', alpha=0.6, label='Typical daily capacity = 50')
plt.legend()
plt.grid(alpha=0.3)
```
This is the most operationally meaningful plot in the project. A care team with daily capacity of 50 patients wants to know: if they follow up with the top 50 patients our model flags, what fraction actually were at risk? Compare your model's Precision@50 to the random baseline (the overall readmission rate). A 2–3× lift is a strong operational result.

**Plot 11 — SHAP summary (for XGBoost)**
```python
import shap
explainer    = shap.TreeExplainer(xgb_model)
shap_values  = explainer.shap_values(X_test)

plt.figure(figsize=(10, 9))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Feature Importance — XGBoost Readmission Model')
plt.tight_layout()
```
SHAP values here do double duty: they show feature importance AND directionality. `prior_high_utilizer` and `sdoh_burden` should be the top features. `early_followup_3d` should appear as negative SHAP (reduces risk). This plot is what you show to a hospital CMO.

**Plot 12 — Risk score calibration**
```python
from sklearn.calibration import calibration_curve

fraction_pos, mean_pred = calibration_curve(
    y_test,
    (predicted_scores / 100),   # rescale to probability
    n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot(mean_pred, fraction_pos, 'o-', color='tomato', label='Model')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
plt.fill_between(mean_pred, fraction_pos,
                 [m for m in mean_pred],
                 alpha=0.2, color='tomato')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction Actually Readmitted')
plt.title('Calibration Curve — Is the Risk Score Trustworthy?')
plt.legend()
```
A well-calibrated model at score 70 should have ~70% of those patients actually readmit. Miscalibration undermines clinical trust even if AUROC is high. Use `CalibratedClassifierCV` if calibration is poor.

---

## Modeling checklist

### The right evaluation hierarchy for this problem

In order of operational importance:
1. **Precision@K** (K = 40–60 = typical daily care team capacity)
2. **PR-AUC** — area under precision-recall curve
3. **AUROC** — general discrimination ability
4. **RMSE on `readmission_risk_score`** — for regression evaluation
5. Calibration curve — trust and interpretability

### Models to train

**1. Ridge Regression on `readmission_risk_score`**
Baseline continuous predictor. Use cross-validated `RidgeCV`. Evaluate RMSE, then convert predictions to 0–100 range and compute Precision@K. Check that coefficients have the right direction: `prior_admissions_12m` positive, `early_followup_3d` negative, `has_caregiver` negative.

**2. Logistic Regression on `readmitted_30d`**
Binary classification baseline. Use `class_weight='balanced'`. AUROC and PR-AUC. Compute Precision@K from `predict_proba`.

**3. Random Forest (both targets)**
Feature importances. Prior utilization, SDOH composite, and no-followup flags should dominate.

**4. XGBoost + SHAP**
Best expected performance. Use `scale_pos_weight`. Add SHAP analysis. Compare Precision@50 to all other models.

**5. Ranking-specific model: LambdaMART (optional, advanced)**
```python
from lightgbm import LGBMRanker

ranker = LGBMRanker(objective='lambdarank', metric='ndcg')
# Requires group structure — patients grouped by admission cohort
```
LambdaMART directly optimizes NDCG (ranking quality) rather than classification accuracy. If your AUROC is already strong but Precision@K is weaker than expected, this is worth trying.

---

## Deployment

### Step 1 — Serialize

```python
import joblib, json
joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(xgb_model,    'models/readmission_model.pkl')
json.dump({
    'model_version': 'v1.0',
    'alert_threshold_score': 65,
    'daily_capacity_k': 50,
    'training_readmission_rate': 0.195,
    'precision_at_50': 0.41,
    'auroc': 0.82,
}, open('models/model_card.json', 'w'))
```

### Step 2 — Batch scoring API (morning report pattern)

This model runs as a daily batch job, not a real-time API. Each morning, it scores all patients discharged in the prior 48 hours and outputs a ranked worklist for the care management team.

```python
# score_daily.py
import pandas as pd, numpy as np, joblib, json, sqlite3
from datetime import datetime

preprocessor = joblib.load('models/preprocessor.pkl')
model        = joblib.load('models/readmission_model.pkl')
card         = json.load(open('models/model_card.json'))

def run_daily_scoring(df_new: pd.DataFrame) -> pd.DataFrame:
    X        = preprocessor.transform(df_new)
    scores   = model.predict(X)
    df_new['readmission_risk_score'] = np.clip(scores, 0, 100).round(1)
    df_new['risk_tier'] = pd.cut(
        df_new['readmission_risk_score'],
        bins=[0, 40, 60, 75, 100],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )
    df_new['scored_at'] = datetime.now().isoformat()
    df_ranked = df_new.sort_values('readmission_risk_score', ascending=False)

    # Write to SQLite for dashboard consumption
    conn = sqlite3.connect('outputs/daily_worklist.db')
    df_ranked.head(card['daily_capacity_k']).to_sql(
        'daily_worklist', conn, if_exists='replace', index=False
    )
    conn.close()
    return df_ranked

# Schedule via cron or Airflow at 6am daily
```

### Step 3 — FastAPI worklist endpoint

```python
@app.get('/worklist/today')
def get_today_worklist(limit: int = 50, min_score: float = 60.0):
    conn = sqlite3.connect('outputs/daily_worklist.db')
    df   = pd.read_sql(
        f"SELECT patient_id, age, primary_diagnosis, discharge_disposition, "
        f"readmission_risk_score, risk_tier, scored_at "
        f"FROM daily_worklist WHERE readmission_risk_score >= {min_score} "
        f"ORDER BY readmission_risk_score DESC LIMIT {limit}",
        conn
    )
    return {
        'date': datetime.now().date().isoformat(),
        'total_flagged': len(df),
        'patients': df.to_dict(orient='records')
    }
```

### Step 4 — Deploy and monitor

Same cloud deployment pattern as prior projects. Key monitoring additions specific to readmission:

**CMS compliance monitoring:** Track your hospital's actual 30-day readmission rate by diagnosis monthly. If the model's risk-stratified intervention is working, the readmission rate in your high-risk tier should trend down over 6–12 months after implementation.

**Intervention effectiveness tracking:** Log which flagged patients actually received follow-up calls or home visits. Compare 30-day readmission rates between intervened and non-intervened high-risk patients. This is the A/B test that proves model value to hospital administration.

**Seasonal drift:** Readmission risk patterns shift with flu season, holiday staffing changes, and post-COVID policy changes. Retrain the model at least annually, more often if seasonal drift is detected.

---

## Project structure

```
readmission_project/
│
├── data/
│   └── raw/
│       └── discharge_patients.csv    # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_data_quality.ipynb         # Missing value analysis, validity checks, distributions
│   ├── 02_eda_and_visualization.ipynb # 12 plots detailed above — the storytelling layer
│   ├── 03_feature_engineering.ipynb  # All 9 feature blocks with clinical rationale
│   └── 04_modeling_and_ranking.ipynb # Ridge → LR → RF → XGBoost, Precision@K, SHAP
│
├── src/
│   ├── validate.py                   # Clinical validity checks
│   ├── features.py                   # All feature engineering functions
│   └── score_daily.py                # Batch scoring job
│
├── models/
│   ├── preprocessor.pkl
│   ├── readmission_model.pkl
│   └── model_card.json
│
├── outputs/
│   └── daily_worklist.db             # SQLite scored worklist
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
lightgbm>=4.0
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

## How this project differs from the ICU deterioration project

| Topic | ICU Deterioration | 30-Day Readmission |
|---|---|---|
| Prediction window | 12 hours ahead | 30 days ahead |
| Output | Binary alert | Continuous risk score → ranked list |
| Primary metric | Sensitivity / AUROC | Precision@K |
| Feature emphasis | Vitals variability, clinical thresholds | SDOH composites, utilization history, discharge complexity |
| Visualization emphasis | Clinical score baselines, SHAP force plots | Heatmaps, dual-axis diagnosis charts, P@K curves |
| Deployment pattern | Real-time alert per observation | Daily batch worklist |
| Operational output | "Alert: Patient X is deteriorating now" | "Top 50 patients to call today, ranked by risk" |
| Regulatory context | FDA SaMD, immediate clinical response | CMS HRRP compliance, care management program |

---

*Dataset is fully synthetic. No real patient data was used. Clinical thresholds reflect published evidence but this dataset is not validated for clinical use.*
