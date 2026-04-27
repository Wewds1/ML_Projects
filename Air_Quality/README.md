# Air Quality Health Risk Classification — End-to-End ML Project

**Target:** 5 simultaneous binary labels (multi-label classification)  
**Model type:** Multi-label Classification  
**Domain:** Environmental monitoring / public health  
**Dataset:** 10,000 air quality sensor readings across 6 stations (2021–2023), deliberately dirty  
**What's completely new:** Multi-label classification — every reading can trigger *multiple* health risk alerts at the same time

---

## The new concept: multi-label classification

Every project in this portfolio so far has had one target. Fraud or not fraud. Readmitted or not. Risk score of X. This project has **five targets simultaneously**, and a single sensor reading can be positive on all five at once.

That's multi-label classification. It appears constantly in real-world ML:
- A news article tagged as both "politics" and "economy"
- A medical image flagged for both "pneumonia" and "pleural effusion"
- A transaction flagged for both "unusual location" and "velocity spike"
- A patient coded with multiple ICD diagnoses

The challenge is that labels are not independent. If `respiratory_risk` fires, `vulnerable_alert` almost certainly fires too. If `industrial_event` fires, `cardiovascular_risk` often follows. Your model needs to learn these correlations — not treat each label as a completely separate binary problem.

### The five labels in this dataset

| Label | Meaning | Approximate rate |
|---|---|---|
| `label_respiratory_risk` | Elevated PM2.5, PM10, or SO2 — direct lung irritation | ~28.5% |
| `label_cardiovascular_risk` | High PM2.5, CO, or NO2 — systemic cardiovascular stress | ~18.1% |
| `label_vulnerable_alert` | Lower-threshold alert for children, elderly, asthmatic patients | ~73.3% |
| `label_outdoor_warning` | Advise against outdoor physical activity | ~31.2% |
| `label_industrial_event` | SO2 + benzene spike suggesting industrial emission event | ~35.5% |

Average labels per reading: **1.87** — most readings trigger nearly two alerts simultaneously.

---

## Dataset columns

### Temporal features

| Column | Type | Notes |
|---|---|---|
| `reading_id` | string | Drop before modeling |
| `timestamp` | datetime string | Parse before use |
| `year` | int | 2021–2023 |
| `month` | int | 1–12 |
| `hour` | int | 0–23 |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `is_weekend` | binary | |
| `is_rush_hour` | binary | 7–9am and 4–7pm |
| `season` | categorical | **dirty** — Winter/Spring/Summer/Fall + "Autumn" variants |

### Station metadata

| Column | Type | Notes |
|---|---|---|
| `station_id` | categorical | 6 named stations |
| `station_type` | categorical | **dirty** — Urban / Suburban / Rural / Industrial |
| `elevation_m` | float | Station elevation in metres |
| `near_highway` | binary | Within 500m of major highway |
| `near_industry` | binary | Within 1km of industrial facility |

### Meteorological conditions

| Column | Type | Notes |
|---|---|---|
| `temp_c` | float | Temperature in Celsius (seasonal variation built in) |
| `humidity_pct` | float | Relative humidity 10–100% |
| `wind_speed_ms` | float | Wind speed m/s — high wind disperses pollutants |
| `wind_dir_deg` | float | **~5% missing** — anemometer failures |
| `pressure_hpa` | float | **~3% missing** — barometer calibration |
| `precipitation_mm` | float | Rainfall washes out PM particles |
| `visibility_km` | float | **~6% missing** — sensor fog/maintenance |
| `temp_inversion` | binary | Atmospheric inversion trapping pollutants near surface |

### Pollutant concentrations

| Column | Units | WHO / EPA threshold | Notes |
|---|---|---|---|
| `pm25` | µg/m³ | 15 µg/m³ (annual WHO 2021) | Fine particulate — deepest lung penetration |
| `pm10` | µg/m³ | 45 µg/m³ (annual WHO 2021) | Coarse particulate |
| `no2` | µg/m³ | 40 µg/m³ (annual WHO) | Traffic + combustion — respiratory irritant |
| `o3` | µg/m³ | 100 µg/m³ (8-hr WHO) | Ground-level ozone — peaks summer afternoons |
| `so2` | µg/m³ | 500 µg/m³ (10-min WHO) | Industrial / coal burning |
| `co` | mg/m³ | 4 mg/m³ (24-hr WHO) | Carbon monoxide — blocks oxygen transport |
| `benzene` | µg/m³ | **~8% missing** — 1.7 µg/m³ (annual EU) | Carcinogen, requires GC-MS analysis |
| `aqi` | 0–500 | 100 = "Unhealthy for Sensitive Groups" | US EPA composite index |

### Multi-label targets (5 columns)

`label_respiratory_risk`, `label_cardiovascular_risk`, `label_vulnerable_alert`, `label_outdoor_warning`, `label_industrial_event` — each is 0 or 1. A reading can be 1 on all five simultaneously.

---

## Data issues to fix

### Duplicates
~28 duplicate rows from sensor batch overlap. Drop on all columns except `reading_id`.

### Missing values

**`benzene`** (~8%) — Benzene measurement requires gas chromatography-mass spectrometry, an expensive slow analysis not run at every station continuously. This is MNAR: industrial stations are more likely to have it measured (higher suspicion of benzene sources) AND more likely to have high values. Create `benzene_missing = 1` before imputing. Impute with median grouped by `station_type`. Industrial station median ≠ rural station median.

**`visibility_km`** (~6%) — Sensor unavailability due to fog or maintenance. When visibility is missing during fog events, it is likely very low — MNAR. Create `visibility_missing = 1`, impute with 1.0 km (conservative low-visibility assumption) rather than the median. A missing visibility during a high-humidity winter reading is almost certainly < 2 km.

**`wind_dir_deg`** (~5%) — Anemometer failures. Wind direction is important for identifying pollution source direction (industrial station upwind vs downwind). Create `wind_dir_missing = 1`. For the numeric column, impute with 180 (neutral direction). Then engineer circular wind features (see Feature Engineering) — missing wind direction should map to zero sine/cosine components.

**`pressure_hpa`** (~3%) — Barometer calibration gaps. Closer to MAR. Create flag, impute with median grouped by `month` (pressure has seasonal variation).

### Categorical inconsistency

**`season`**: "winter", "WINTER", "spring", "SPRING", "summer", "SUMMER", "fall", "FALL", "Autumn", "autumn" — all need to collapse. Note "Autumn" and "autumn" must map to "Fall" (US convention used in the rest of the data). Strip, lower-case, then apply:
```python
season_map = {'autumn': 'Fall', 'fall': 'Fall', 'winter': 'Winter',
              'spring': 'Spring', 'summer': 'Summer'}
df['season'] = df['season'].str.strip().str.lower().map(season_map)
```

**`station_type`**: "urban", "URBAN", "suburban", "SUBURBAN", "rural", "RURAL", "industrial", "INDUSTRIAL" — strip and title-case handles all variants.

### Physical validity checks — unique to sensor data

Sensor data has instrument error patterns you won't see in administrative data:

```python
# PM10 must always be >= PM2.5 (fine particles are a subset of coarse)
pm_invalid = df['pm10'] < df['pm25']
print(f"PM10 < PM2.5 (physically impossible): {pm_invalid.sum()}")
df.loc[pm_invalid, 'pm10'] = df.loc[pm_invalid, 'pm25'] * 1.5  # fix with ratio

# O3 and NO2 have an inverse photochemical relationship
# Extreme high values of both simultaneously are suspicious
both_extreme = (df['o3'] > 200) & (df['no2'] > 200)
print(f"Suspicious O3 + NO2 both extreme: {both_extreme.sum()}")

# AQI should match PM2.5 (it's derived from PM2.5 in this dataset)
# Recompute and flag large discrepancies
def aqi_from_pm25(c):
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300)]
    result = np.zeros(len(c))
    for lo,hi,alo,ahi in bp:
        mask = (c >= lo) & (c <= hi)
        result[mask] = ((ahi-alo)/(hi-lo))*(c[mask]-lo) + alo
    return result.round().astype(int)

aqi_recomputed = aqi_from_pm25(df['pm25'].values)
aqi_discrepancy = np.abs(df['aqi'] - aqi_recomputed) > 30
print(f"AQI inconsistent with PM2.5: {aqi_discrepancy.sum()}")

# Wind speed of 0 exactly is often a stuck sensor, not true calm
stuck_anemometer = df['wind_speed_ms'] == 0.0
print(f"Possible stuck anemometer (wind = 0.0): {stuck_anemometer.sum()}")
```

---

## Feature engineering checklist

### Block 1 — Missingness flags (always first)

```python
for col in ['benzene', 'visibility_km', 'wind_dir_deg', 'pressure_hpa']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### Block 2 — Circular encoding for wind direction

Wind direction is circular — 359° and 1° are nearly the same direction, but treated as far apart by any linear model. Standard encoding breaks this. The fix is sine/cosine decomposition:

```python
wind_rad = np.deg2rad(df['wind_dir_deg'].fillna(180))
df['wind_sin'] = np.sin(wind_rad).round(4)
df['wind_cos'] = np.cos(wind_rad).round(4)
# Drop original wind_dir_deg after creating sin/cos
```

This is a technique that appears in any time-series or geospatial project with angular features — compass bearings, clock hours (encode hour as sin/cos too), calendar month. Get comfortable with it here.

Apply the same treatment to `hour` and `month`:
```python
df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24).round(4)
df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24).round(4)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).round(4)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).round(4)
```

This preserves the continuity that January (month=1) is close to December (month=12) — something `OrdinalEncoder` on month would destroy.

### Block 3 — Pollution ratio and composite features

```python
# PM2.5/PM10 ratio — high ratio means finer particles dominating (worse for health)
df['pm_fine_ratio'] = (df['pm25'] / df['pm10'].clip(0.1)).round(3)

# Oxidant load: combined ozone + NO2 — total oxidative stress
df['oxidant_load'] = (df['o3'] + df['no2']).round(1)

# Combustion index: NO2 + CO — combined traffic/combustion indicator
df['combustion_index'] = (df['no2'] * 0.6 + df['co'] * 10).round(2)

# Industrial signature: SO2 + benzene correlation
df['industrial_signature'] = (
    df['so2'] * 0.5 + df['benzene'].fillna(0) * 3
).round(2)

# AQI exceedance: how much above the "Unhealthy" threshold of 150
df['aqi_excess_150'] = (df['aqi'] - 150).clip(lower=0)
df['aqi_excess_100'] = (df['aqi'] - 100).clip(lower=0)

# Log transforms for right-skewed pollutants
for col in ['pm25', 'pm10', 'no2', 'so2', 'co', 'benzene', 'aqi']:
    df[f'log_{col}'] = np.log1p(df[col].fillna(0))
```

### Block 4 — Meteorological interaction features

```python
# Dispersion index: high wind + low inversion = pollutants disperse
# Low wind + inversion = pollutants trap. This is the most important met feature.
df['dispersion_index'] = (
    df['wind_speed_ms'] / (1 + df['temp_inversion'] * 3)
).round(3)

# Wet deposition: rain washes out particulates
df['precipitation_flag'] = (df['precipitation_mm'] > 0).astype(int)
df['heavy_rain']         = (df['precipitation_mm'] > 5).astype(int)

# Heat + ozone: high temperature accelerates ozone formation
df['heat_ozone_risk'] = (
    (df['temp_c'] > 25) & (df['o3'] > 100)
).astype(int)

# Inversion severity score
df['inversion_severity'] = (
    df['temp_inversion']
    * (1 / df['wind_speed_ms'].clip(0.5))
    * (df['humidity_pct'] / 100)
).round(3)

# Visibility as a pollution proxy (haze reduces visibility)
df['haze_flag'] = (df['visibility_km'].fillna(1.0) < 5).astype(int)
```

### Block 5 — Station context features

```python
# Urban heat island effect modifier
df['urban_heat'] = ((df['station_type'] == 'Urban') & (df['temp_c'] > 28)).astype(int)

# Industrial proximity × wind direction toward station
# If wind blows FROM industry (225–315° = southwest/westerly toward urban)
df['industrial_downwind'] = (
    (df['near_industry'] == 1) &
    (df['wind_dir_deg'].fillna(180).between(180, 315))
).astype(int)

# Rush hour at urban station: peak traffic pollution window
df['urban_rush'] = (
    (df['station_type'] == 'Urban') & (df['is_rush_hour'] == 1)
).astype(int)
```

### Block 6 — Rolling/lag features (simulate temporal awareness)

Since readings are ordered by timestamp, compute lag and rolling features within each station:

```python
df_sorted = df.sort_values(['station_id', 'timestamp']).copy()

for col in ['pm25', 'no2', 'aqi']:
    # 3-reading rolling mean (approx 3-hour window given hourly data)
    df_sorted[f'{col}_roll3_mean'] = (
        df_sorted.groupby('station_id')[col]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(2)
    )
    # Lag-1: previous reading
    df_sorted[f'{col}_lag1'] = (
        df_sorted.groupby('station_id')[col]
        .transform(lambda x: x.shift(1))
    )
    # Trend: current vs rolling mean (is pollution rising or falling?)
    df_sorted[f'{col}_trend'] = (
        df_sorted[col] - df_sorted[f'{col}_roll3_mean']
    ).round(2)

# Rising PM2.5 trend flag
df_sorted['pm25_rising'] = (df_sorted['pm25_trend'] > 5).astype(int)
```

Rolling features are the bridge between static tabular ML and time series ML. They're especially important here because a single high reading is less alarming than a sustained rising trend.

### Block 7 — Label correlation features (multi-label specific)

Since labels are correlated, the predicted probability of one label is useful as a feature for predicting others. This is the core idea behind **Classifier Chains** (covered in Modeling). For feature engineering, encode domain knowledge about label dependencies:

```python
# Proxy for cardiovascular_risk being active
# (high PM2.5 is both respiratory AND cardiovascular)
df['pm25_cardio_zone'] = (df['pm25'] > 55.4).astype(int)

# Proxy for all labels active simultaneously: extreme pollution event
df['multi_label_event'] = (df['aqi'] > 150).astype(int)

# Count of pollutants above WHO annual guidelines
df['pollutants_above_who'] = (
    (df['pm25'] > 15).astype(int)
    + (df['pm10'] > 45).astype(int)
    + (df['no2'] > 40).astype(int)
    + (df['o3'] > 100).astype(int)
    + (df['so2'] > 40).astype(int)
).clip(0, 5)
```

### Encoding strategy

- `season`, `station_type`, `station_id`: One-hot encode.
- `admission_source`: One-hot encode.
- Drop `reading_id`, `timestamp` (after extracting all temporal features), raw `hour`, `month`, `wind_dir_deg` (replaced by sin/cos).
- Keep both raw pollutant values AND log-transformed versions — let the model decide which scale works better for each label.

---

## Multi-label classification — the modeling section

### Step 1 — Understand your label matrix

Before any modeling, analyze label structure:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

label_cols = ['label_respiratory_risk', 'label_cardiovascular_risk',
              'label_vulnerable_alert', 'label_outdoor_warning',
              'label_industrial_event']
Y = df[label_cols]

# Label frequency
print(Y.mean().sort_values(ascending=False))

# Label co-occurrence matrix — which labels fire together?
co_occ = Y.T.dot(Y)
co_occ_norm = co_occ / len(Y)

plt.figure(figsize=(8, 6))
sns.heatmap(co_occ_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[c.replace('label_','') for c in label_cols],
            yticklabels=[c.replace('label_','') for c in label_cols])
plt.title('Label Co-occurrence Matrix (fraction of readings)')
```

This heatmap reveals which labels are nearly always active together (strong co-occurrence) vs which are relatively independent. The co-occurrence structure determines which multi-label strategy works best.

```python
# Label cardinality: distribution of how many labels are active per row
label_counts = Y.sum(axis=1)
print(label_counts.value_counts().sort_index())
print(f"Average labels per reading: {label_counts.mean():.2f}")
print(f"Readings with 0 labels: {(label_counts==0).sum()}")
print(f"Readings with all 5 labels: {(label_counts==5).sum()}")
```

### Step 2 — Three strategies for multi-label classification

**Strategy 1: Binary Relevance (simplest)**
Train one independent binary classifier per label. Ignores label correlation entirely.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Wraps any binary classifier to handle multiple outputs
br_model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    n_jobs=-1
)
br_model.fit(X_train, Y_train)
Y_pred = br_model.predict(X_test)
```

Pros: Simple, fast, parallelizable. Each label can use a different base model.
Cons: Ignores that "if respiratory_risk fires, cardiovascular_risk is more likely."

**Strategy 2: Classifier Chains (captures label correlation)**
Train labels in a sequence. Each classifier receives the original features PLUS the predictions of all previous classifiers as additional features. This propagates label correlations forward.

```python
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# Order matters — put correlated labels adjacent
# vulnerable_alert → respiratory_risk → cardiovascular_risk → outdoor_warning → industrial_event
chain = ClassifierChain(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    order=[2, 0, 1, 3, 4],   # indices into label_cols
    random_state=42
)
chain.fit(X_train, Y_train)
Y_pred_chain = chain.predict(X_test)
```

Pros: Models label dependencies — the single most important advantage over Binary Relevance.
Cons: Errors in early labels propagate to later ones. Order of labels matters.

For robustness, train an ensemble of Classifier Chains with different orderings and average their predicted probabilities:

```python
from sklearn.multioutput import ClassifierChain
import numpy as np

n_chains = 10
chains = [
    ClassifierChain(
        LogisticRegression(class_weight='balanced', max_iter=500),
        order='random',
        random_state=i
    )
    for i in range(n_chains)
]

for chain in chains:
    chain.fit(X_train, Y_train)

# Average predicted probabilities across chains
Y_prob_avg = np.mean([c.predict_proba(X_test) for c in chains], axis=0)
Y_pred_ensemble = (Y_prob_avg >= 0.5).astype(int)
```

**Strategy 3: Label Powerset (treats combinations as single classes)**
Converts the multi-label problem to a single multiclass problem by treating each unique label combination as one class. With 5 binary labels, you have up to 2⁵ = 32 possible combinations.

```python
# Create a combined label column
Y_train['label_combo'] = Y_train.apply(
    lambda row: '_'.join(row.astype(str)), axis=1
)
# Train a multiclass classifier
from sklearn.ensemble import GradientBoostingClassifier
lp_model = GradientBoostingClassifier(random_state=42)
lp_model.fit(X_train, Y_train['label_combo'])
```

Pros: Captures all label interactions simultaneously.
Cons: Exponential number of classes, rare combinations get very few training examples, doesn't generalize to unseen combinations.

### Step 3 — Evaluation metrics for multi-label classification

This is the most important new section. Standard accuracy doesn't apply.

```python
from sklearn.metrics import (
    hamming_loss, jaccard_score,
    classification_report, f1_score
)

# ── Hamming Loss ──────────────────────────────────────────────────────────────
# Fraction of labels that are incorrectly predicted
# Lower is better. 0 = perfect, 1 = worst
hl = hamming_loss(Y_test, Y_pred)
print(f"Hamming Loss: {hl:.4f}")
# Interpretation: if HL = 0.08, then 8% of all (sample, label) pairs are wrong

# ── Subset Accuracy (Exact Match Ratio) ───────────────────────────────────────
# Strictest metric: fraction of samples where ALL 5 labels are correct
exact_match = (Y_pred == Y_test.values).all(axis=1).mean()
print(f"Exact Match Ratio: {exact_match:.4f}")
# This will be low — getting all 5 labels right simultaneously is hard

# ── Micro-averaged F1 ──────────────────────────────────────────────────────────
# Treats each (sample, label) pair as one binary prediction
# Dominated by frequent labels (vulnerable_alert at 73% will dominate)
f1_micro = f1_score(Y_test, Y_pred, average='micro')
print(f"F1 Micro: {f1_micro:.4f}")

# ── Macro-averaged F1 ──────────────────────────────────────────────────────────
# Averages F1 across labels equally — rare labels have equal weight
f1_macro = f1_score(Y_test, Y_pred, average='macro')
print(f"F1 Macro: {f1_macro:.4f}")

# ── Per-label report ───────────────────────────────────────────────────────────
# The most informative evaluation — see each label's precision, recall, F1
label_names = [c.replace('label_','') for c in label_cols]
print(classification_report(Y_test, Y_pred, target_names=label_names))

# ── Jaccard Similarity Score ───────────────────────────────────────────────────
# |predicted ∩ true| / |predicted ∪ true| per sample, then averaged
# More lenient than exact match, more strict than Hamming
jaccard = jaccard_score(Y_test, Y_pred, average='samples')
print(f"Jaccard (samples): {jaccard:.4f}")
```

**Which metric to report?**
- Report **Hamming Loss** as your primary headline metric — it's the standard for multi-label.
- Report **per-label F1** in a table — this is what a public health team actually cares about: is the cardiovascular risk label accurate?
- Report **Exact Match Ratio** to show how often all five labels are simultaneously correct.
- Compare Micro vs Macro F1 — if they diverge significantly, your model is better at frequent labels than rare ones.

### Step 4 — Threshold tuning per label

Unlike binary classification where one threshold applies to everything, multi-label lets you tune a different threshold per label based on its cost structure:

```python
# Get per-label probabilities from Binary Relevance model
Y_proba = np.column_stack([
    est.predict_proba(X_test)[:, 1]
    for est in br_model.estimators_
])

from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.9, 0.05)
best_thresholds = []

for i, label in enumerate(label_cols):
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (Y_proba[:, i] >= t).astype(int)
        f1 = f1_score(Y_test.iloc[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
    print(f"{label}: best threshold = {best_t:.2f}, F1 = {best_f1:.3f}")

# Apply per-label thresholds
Y_pred_tuned = (Y_proba >= best_thresholds).astype(int)
```

For `vulnerable_alert` (73% positive rate), you might lower the threshold to 0.35 to maintain high recall — missing a vulnerable population alert is more costly than a false alarm. For `industrial_event` (35% rate), a higher threshold reduces false alarms to industrial operators who must respond to each one.

### Step 5 — Models to compare

| Model | Strategy | Expected Hamming Loss |
|---|---|---|
| Logistic Regression | Binary Relevance | ~0.14 |
| Random Forest | Binary Relevance | ~0.11 |
| XGBoost | Binary Relevance | ~0.10 |
| Logistic Regression | Classifier Chain | ~0.12 |
| XGBoost | Classifier Chain ensemble | ~0.09 |

Build a comparison table showing Hamming Loss, Micro F1, Macro F1, and Exact Match Ratio for each combination. The Classifier Chain XGBoost ensemble should win on Macro F1 because it captures label correlations.

---

## Visualization guide

### Plot 1 — Label co-occurrence heatmap
Already covered in the modeling setup above. This should be your first plot in Notebook 01. It tells you everything about label structure before you write a single line of modeling code.

### Plot 2 — Pollutant distributions by station type

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
pollutants = ['pm25', 'no2', 'o3', 'so2', 'co', 'benzene']
station_order = ['Rural', 'Suburban', 'Urban', 'Industrial']
palette = {'Rural':'#4CAF50','Suburban':'#2196F3','Urban':'#FF9800','Industrial':'#F44336'}

for ax, pol in zip(axes.flat, pollutants):
    df_plot = df[['station_type', pol]].dropna()
    sns.boxplot(data=df_plot, x='station_type', y=pol,
                order=station_order, palette=palette, ax=ax, fliersize=1)
    ax.set_title(pol.upper())
    ax.set_yscale('log')
    ax.set_xlabel('')
plt.suptitle('Pollutant Distributions by Station Type (log scale)', fontsize=14)
plt.tight_layout()
```

Industrial > Urban > Suburban > Rural should hold for every combustion-related pollutant. O3 may be highest in rural areas (less NO2 to destroy it via photochemistry — the "ozone titration" effect). That reversal is worth noting.

### Plot 3 — Temporal heatmap: pollution by hour × season

```python
pivot = df.groupby(['season', 'hour'])['pm25'].mean().unstack()
season_order = ['Winter', 'Spring', 'Summer', 'Fall']

plt.figure(figsize=(14, 5))
sns.heatmap(pivot.loc[season_order], cmap='YlOrRd',
            annot=False, linewidths=0,
            xticklabels=range(24))
plt.title('Mean PM2.5 by Season × Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Season')
```

You should see rush hour peaks (8am, 6pm) consistently, winter readings darker than summer, and summer afternoons lighter (more UV = faster PM oxidation). This plot tells a complete story about pollution dynamics without any model.

### Plot 4 — Wind rose colored by AQI

A wind rose shows which wind directions are associated with high pollution. It requires the `windrose` library:

```python
# pip install windrose
from windrose import WindroseAxes

fig = plt.figure(figsize=(8, 8))
ax = WindroseAxes.from_ax(fig=fig)

clean_mask = df['wind_dir_deg'].notna() & df['aqi'].notna()
ax.bar(df.loc[clean_mask, 'wind_dir_deg'],
       df.loc[clean_mask, 'aqi'],
       normed=True, opening=0.8, edgecolor='white',
       bins=[0, 50, 100, 150, 200, 300],
       cmap=plt.cm.YlOrRd)
ax.set_legend(title='AQI Range')
plt.title('Wind Rose — AQI by Wind Direction')
```

For the industrial station, the wind rose will show high AQI concentrating from the direction of the industrial source. This is the visualization that would justify a pollution source investigation.

### Plot 5 — Temperature inversion events: before and after PM2.5

```python
# Compare PM2.5 distribution on inversion vs non-inversion days
fig, ax = plt.subplots(figsize=(10, 5))

df[df['temp_inversion']==0]['pm25'].plot.kde(ax=ax, color='steelblue',
                                               label='No inversion', linewidth=2)
df[df['temp_inversion']==1]['pm25'].plot.kde(ax=ax, color='tomato',
                                               label='Temperature inversion', linewidth=2)
ax.axvline(35.4, color='black', linestyle='--', label='PM2.5 "Unhealthy" threshold')
ax.set_xlabel('PM2.5 (µg/m³)')
ax.set_xlim(0, 200)
ax.set_title('PM2.5 Distribution: Inversion vs No Inversion Events')
ax.legend()
```

The inversion distribution should shift significantly rightward — inversions trap pollutants. This directly justifies the `temp_inversion` and `inversion_severity` features.

### Plot 6 — Multi-label evaluation: per-label F1 comparison across models

```python
models = {'Binary Relevance RF': br_pred,
          'Classifier Chain XGB': chain_pred,
          'Binary Relevance LR': lr_pred}

f1_results = {}
for model_name, preds in models.items():
    f1_results[model_name] = f1_score(Y_test, preds, average=None)

f1_df = pd.DataFrame(f1_results, index=label_names)

f1_df.plot(kind='bar', figsize=(12, 5), colormap='tab10', edgecolor='white')
plt.title('Per-Label F1 Score by Model')
plt.ylabel('F1 Score')
plt.xticks(rotation=20, ha='right')
plt.axhline(0.7, color='gray', linestyle='--', alpha=0.5, label='F1=0.7 reference')
plt.legend(loc='lower right')
plt.tight_layout()
```

This bar chart is the deliverable for a public health audience. It shows not just aggregate performance but whether the model is reliable on the specific labels that matter most — cardiovascular risk and vulnerable population alert.

### Plot 7 — Label prediction heatmap for sample patients

Visualize the model's output across a sample of 50 patients — rows are patients, columns are labels, color is predicted probability:

```python
sample_proba = Y_proba[:50]  # predicted probabilities for 50 test patients

plt.figure(figsize=(10, 12))
sns.heatmap(sample_proba,
            xticklabels=[c.replace('label_','') for c in label_cols],
            yticklabels=False,
            cmap='RdYlGn_r', vmin=0, vmax=1,
            linewidths=0.3, linecolor='white')
plt.title('Predicted Risk Probabilities per Patient (sample of 50)')
plt.xlabel('Risk Label')
plt.ylabel('Patient Index')
```

This is the operational output — the grid a public health dashboard would show. Rows with multiple dark cells are multi-label high-risk readings requiring coordinated response.

---

## Deployment

### Batch alert system

Like the readmission project, this is better as a batch job than a real-time API. Air quality alerts are typically issued every hour based on the previous hour's readings.

```python
# score_hourly.py
import joblib, json, pandas as pd, numpy as np
from datetime import datetime

preprocessor = joblib.load('models/preprocessor.pkl')
chain_model  = joblib.load('models/classifier_chain.pkl')
thresholds   = json.load(open('models/label_thresholds.json'))

def score_readings(df_new: pd.DataFrame) -> pd.DataFrame:
    X       = preprocessor.transform(df_new)
    Y_proba = chain_model.predict_proba(X)

    label_cols = ['respiratory_risk', 'cardiovascular_risk',
                  'vulnerable_alert', 'outdoor_warning', 'industrial_event']

    for i, label in enumerate(label_cols):
        df_new[f'prob_{label}']  = Y_proba[:, i].round(4)
        df_new[f'alert_{label}'] = (Y_proba[:, i] >= thresholds[label]).astype(int)

    df_new['total_alerts']  = df_new[[f'alert_{l}' for l in label_cols]].sum(axis=1)
    df_new['max_risk_prob'] = Y_proba.max(axis=1).round(4)
    df_new['scored_at']     = datetime.now().isoformat()
    return df_new
```

### FastAPI endpoint

```python
@app.get('/current_alerts')
def current_alerts():
    """Returns active alerts across all stations in the last hour."""
    conn = sqlite3.connect('outputs/air_quality_alerts.db')
    df   = pd.read_sql(
        "SELECT station_id, station_type, aqi, total_alerts, "
        "alert_respiratory_risk, alert_cardiovascular_risk, "
        "alert_vulnerable_alert, alert_outdoor_warning, alert_industrial_event, "
        "max_risk_prob, scored_at "
        "FROM alerts WHERE scored_at >= datetime('now', '-1 hour') "
        "ORDER BY max_risk_prob DESC",
        conn
    )
    return {
        'stations_with_alerts': int((df['total_alerts'] > 0).sum()),
        'highest_risk_station': df.iloc[0]['station_id'] if len(df) else None,
        'readings': df.to_dict(orient='records')
    }
```

### What to monitor

**Per-label drift** — track the daily rate of each label separately. If `industrial_event` alerts spike on weekdays 2–4pm, that's a real signal. If `vulnerable_alert` drops to near zero in January (normally it should be elevated in winter), the model or the sensors may have a problem.

**Label correlation drift** — the co-occurrence structure of your labels in production should match training. If `respiratory_risk` and `cardiovascular_risk` start firing independently when they should almost always fire together, input feature distributions have shifted.

**Sensor drift** — air quality sensors degrade over time. PM2.5 sensors in particular drift upward as the optical chamber collects dust. Track raw feature distributions per station monthly. A gradual upward trend in `pm25` at one station while others are stable = sensor calibration issue, not real pollution.

---

## Project structure

```
airquality_project/
│
├── data/
│   └── raw/
│       └── air_quality_readings.csv  # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_data_quality.ipynb         # Sensor validity checks, missing patterns
│   ├── 02_eda_visualization.ipynb    # 7 plots above — wind rose, temporal heatmaps
│   ├── 03_feature_engineering.ipynb  # Circular encoding, rolling features, composites
│   └── 04_multilabel_modeling.ipynb  # Binary Relevance, Chains, evaluation metrics
│
├── src/
│   ├── validate_sensors.py           # Physical validity checks
│   ├── features.py                   # Feature engineering functions
│   └── score_hourly.py               # Hourly batch scoring job
│
├── models/
│   ├── preprocessor.pkl
│   ├── classifier_chain.pkl
│   └── label_thresholds.json         # Per-label operating thresholds
│
├── outputs/
│   └── air_quality_alerts.db
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
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
missingno>=0.5
windrose>=1.9
jupyterlab>=4.0
fastapi>=0.110
uvicorn>=0.29
joblib>=1.3
pydantic>=2.0
evidently>=0.4
```

---

## What's completely new in this project vs every previous one

| Concept | First introduced here |
|---|---|
| Multi-label targets (5 simultaneous) | ✓ |
| Label co-occurrence matrix | ✓ |
| Binary Relevance (`MultiOutputClassifier`) | ✓ |
| Classifier Chains (`ClassifierChain`) | ✓ |
| Classifier Chain ensemble (random orderings) | ✓ |
| Hamming Loss | ✓ |
| Exact Match Ratio | ✓ |
| Jaccard Similarity Score | ✓ |
| Per-label threshold tuning | ✓ |
| Micro vs Macro F1 distinction for multi-label | ✓ |
| Circular encoding (sin/cos for angles, hours, months) | ✓ |
| Rolling / lag features within station groups | ✓ |
| Sensor drift monitoring | ✓ |
| Wind rose visualization | ✓ |
| Physical sensor validity checks | ✓ |

---

*Dataset is fully synthetic. Pollutant thresholds are based on WHO 2021 Air Quality Guidelines and US EPA AQI breakpoints but this dataset is not validated for real environmental monitoring.*
