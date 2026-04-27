# Philippine Soil-Based Crop Suitability Ranking — End-to-End ML Project

**Target:** Rank 25 Philippine crops by suitability score (0–100) for a given soil sample  
**Model type:** Regression + Ranking (Learning to Rank)  
**Domain:** Agriculture / soil science — Philippines-specific  
**Dataset:** 8,000 synthetic soil samples across all 16 Philippine regions, deliberately dirty  
**Unique angle:** Every soil sample has a per-crop suitability score for all 25 crops simultaneously — modeling both regression (predict score) and ranking (order crops by fit)

---

## The real-world problem

A farmer in Bukidnon sends a soil sample to the Bureau of Soils and Water Management (BSWM). The lab returns NPK values, pH, organic matter, and texture class. The question is: **given everything we know about this soil, this location, this season, and this farm history — which crops should this farmer plant, and in what order of suitability?**

This is not a simple classification problem. It's a ranking problem. The output isn't "plant rice" — it's "here are your top 5 crops ranked by how well your soil fits their requirements, with scores so you can see how close each one is." A score of 85 for banana and 82 for cacao means both are viable; a score of 85 vs 40 means one is clearly dominant.

This kind of system is exactly what the DA-BSWM and PhilRice use in their soil fertility management tools.

---

## Philippine context baked into the data

### Regions (all 16)
Every sample is tagged to one of the 16 Philippine administrative regions, each with region-appropriate soil series names from the Bureau of Soils classification, matching PAGASA climate types, and realistic NPK/pH ranges for that region's dominant agricultural soils.

| Region | Key agricultural context |
|---|---|
| Ilocos Region | Dry Type I climate; garlic, onion, tobacco country; Sandy Loam dominant |
| Cagayan Valley | Fertile alluvial valley; rice and corn granary of North Luzon |
| Central Luzon | The rice bowl; Pampanga Clay and Bulacan Clay Loam; Type I climate |
| CAR (Cordillera) | Highland 800–2,600m; Benguet Sandy Loam; cabbage, strawberry, coffee |
| Western Visayas | Sugarcane country; Iloilo and Bacolod Clay Loam; Type I |
| Davao Region | Banana, durian, cacao, coffee; volcanic Clay Loam; Type IV |
| SOCCSKSARGEN | Pineapple, corn, asparagus; South Cotabato Loam; Type IV |
| Northern Mindanao | Pineapple, cacao, rice; Bukidnon Clay Loam |

### PAGASA climate types
- **Type I** — Distinct wet (Jun–Oct) and dry (Nov–Apr) seasons. West coast: Ilocos, Central Luzon, Western Visayas
- **Type II** — No dry season; heavy rain Nov–Jan. East coast: Bicol, Eastern Visayas, Caraga
- **Type III** — Short dry season (Mar–May). CAR, most Visayas, parts of Mindanao
- **Type IV** — Rainfall distributed all year. Mindanao interior, BARMM

### PAGASA seasons in the dataset
Three seasons are represented: **Wet Season**, **Dry Season**, and **Transition**. Season interacts with crop suitability scores — onion and garlic require Dry Season only; rice scores highest in Wet Season; coconut and malunggay are year-round.

### Philippine soil series names (Bureau of Soils)
The `province_soil_name` column uses actual Bureau of Soils series names — Pampanga Clay, Benguet Sandy Loam, Davao Clay Loam, Bacolod Clay Loam, Tuguegarao Clay Loam, etc. The `soil_texture_class` column is the simplified texture derived from the full name.

---

## Dataset columns

### Location and context

| Column | Type | Notes |
|---|---|---|
| `sample_id` | string | Drop before modeling |
| `region` | categorical | 16 Philippine administrative regions |
| `province_soil_name` | categorical | Bureau of Soils series name (47 unique) |
| `soil_texture_class` | categorical | Clay / Clay Loam / Sandy Loam / Loam / Sandy Clay / Muck |
| `climate_type` | categorical | PAGASA Type I–IV |
| `season` | categorical | **dirty** — Wet Season / Dry Season / Transition |
| `elevation_m` | float | 0–2,600m (CAR has highest values) |
| `temp_c` | float | Mean monthly temperature at site (°C) |
| `rainfall_mm` | float | Mean monthly rainfall (mm) |
| `slope_pct` | float | Land slope percentage |
| `near_water_body` | binary | Proximity to river/lake/irrigation |
| `farm_history` | categorical | Fallow / Monocrop / Rotation / Virgin Land / Orchard |
| `previous_crop` | categorical | Last crop grown on this plot |

### Soil NPK and chemistry

| Column | Units | Interpretation |
|---|---|---|
| `nitrogen_pct` | % total N | Low < 0.10, Medium 0.10–0.20, High > 0.20 |
| `phosphorus_ppm` | ppm (Olsen method) | **~7% missing** — Low < 10, Medium 10–25, High > 25 |
| `potassium_cmol` | cmol/kg (exchangeable K) | Low < 0.2, Medium 0.2–0.5, High > 0.5 |
| `soil_ph` | pH units | Philippine target: 5.5–6.5 for most crops |
| `organic_matter_pct` | % | **~6% missing** — Low < 1.5, Medium 1.5–3.0, High > 3.0 |
| `moisture_pct` | % | Varies strongly by season |
| `cation_exchange_capacity` | cmol/kg | **~9% missing** — expensive test |
| `base_saturation_pct` | % | **~8% missing** — requires CEC |
| `soil_depth_cm` | cm | **~4% missing** — effective rooting depth |
| `bulk_density_gcc` | g/cm³ | > 1.5 = compacted (limits root growth) |
| `drainage_class` | categorical | **dirty** — 5 classes from Poorly to Excessively Drained |

### Targets

| Column | Notes |
|---|---|
| `score_<CropName>` | 25 columns — suitability score 0–100 per crop |
| `best_crop` | Crop with highest score for this sample |
| `top3_crops` | Pipe-separated top 3 crops |
| `suitability_spread` | Difference between best and worst score (range) |

---

## The 25 Philippine crops

### Staple crops
- **Palay (Rice)** — Needs wet/flooded conditions; Clay and Clay Loam; pH 5.0–7.0; Wet Season
- **Mais (Corn)** — Adaptable; Loam preferred; pH 5.5–7.5; all seasons
- **Cassava** — Tolerates poor soils and drought; Sandy Loam; Dry Season; pH 4.5–7.5
- **Camote (Sweet Potato)** — Sandy Loam; Dry Season; pH 5.0–7.0; moderate K demand

### Cash crops
- **Saging/Pisan (Banana)** — High K demand; Loam-Clay Loam; pH 5.5–7.0; year-round
- **Tubo (Sugarcane)** — High N+K; Clay Loam; pH 5.5–7.5; Western Visayas and SOCCSKSARGEN
- **Niyog (Coconut)** — Sandy Loam; tolerates wide pH (5.0–8.0); coastal and lowland
- **Kape (Coffee)** — Loam; pH 5.0–6.5; temp 18–28°C; CAR and Benguet highlands
- **Kakaw (Cacao)** — Clay Loam; pH 5.0–7.5; Davao and Northern Mindanao
- **Abaka (Abaca)** — Loam; wet conditions; Bicol, Eastern Visayas
- **Luya (Ginger)** — Loam-Sandy Loam; pH 5.5–7.0; Wet Season-Transition
- **Langka (Jackfruit)** — Adaptable; Sandy Loam to Clay Loam; year-round

### Vegetables and herbs
- **Pechay** — Fast-growing; Loam; pH 5.5–7.5; Dry Season; low temp tolerance
- **Sitaw (String Beans)** — Sandy Loam; pH 6.0–7.5; Dry Season
- **Ampalaya (Bitter Gourd)** — Sandy Loam; pH 6.0–7.0; Dry Season
- **Malunggay (Moringa)** — Extremely adaptable; tolerates drought and alkaline soils; pH 5.0–9.0
- **Talong (Eggplant)** — Sandy Loam-Loam; pH 5.5–7.5; Dry Season
- **Kamatis (Tomato)** — Requires good drainage; pH 6.0–7.5; cool Dry Season
- **Sibuyas (Onion)** — Sandy Loam only; pH 6.0–7.5; strict Dry Season; Ilocos
- **Bawang (Garlic)** — Sandy Loam; pH 6.0–7.5; strict Dry Season; Ilocos Region
- **Repolyo (Cabbage)** — Loam; pH 6.0–7.5; temp 14–22°C; CAR highlands only
- **Strawberry** — Sandy Loam; pH 5.5–6.5; temp 14–22°C; Benguet, CAR

### Fruits and trees
- **Pinya (Pineapple)** — Sandy Loam; pH 4.5–6.5 (acid-tolerant); SOCCSKSARGEN
- **Mangga (Mango)** — Sandy Loam-Loam; pH 5.0–7.5; Dry Season flowering; widespread
- **Durian** — Clay Loam; pH 5.0–6.5; Davao; high rainfall; Wet Season-Transition

---

## How suitability scores are computed (know this before modeling)

Each score column was generated by penalizing deviations from agronomic requirement ranges for 8 factors: N, P, K, pH, rainfall, temperature, season match, soil texture match, and drainage class match. Organic matter contributes a bonus/penalty. Random noise (±2 points) is added to simulate real-world variability.

This means:
- **Score columns are not independent** — if a soil has the right pH for rice, it likely also scores well for sugarcane (both prefer pH 5.5–7.0)
- **Score columns are correlated in clusters** — highland vegetables (cabbage, strawberry, coffee) all score high together in CAR; lowland cash crops (banana, coconut, cacao) cluster together in Mindanao
- **The `best_crop` label is derived directly from scores** — it's the argmax of the 25 score columns

Understanding this structure is essential for choosing the right modeling approach.

---

## Data issues to fix

### Duplicates
~30 duplicate rows from laboratory batch re-export. Drop on all columns except `sample_id`.

### Missing values — five soil lab columns

**`phosphorus_ppm`** (~7%) — The Olsen phosphorus test requires separate sample preparation and is not run at all BSWM labs. Missing mostly in remote areas (MIMAROPA, BARMM, Caraga) where lab capacity is limited. Create `phosphorus_missing = 1` flag, impute with median grouped by `soil_texture_class`. Sandy Loam soils have characteristically lower P than Clay Loam.

**`organic_matter_pct`** (~6%) — The Walkley-Black method is the Philippine standard but is not universally applied. OM correlates with `nitrogen_pct` (OM ≈ N × 10 as a rough rule). If OM is missing, you can use `nitrogen_pct * 10` as an imputed estimate — this is a domain-knowledge imputation that outperforms median fill here.

```python
om_imputed = df['nitrogen_pct'] * 10
df['organic_matter_pct'] = df['organic_matter_pct'].fillna(om_imputed).clip(0.5, 8.0)
```

**`cation_exchange_capacity`** (~9%) — CEC is the most expensive standard soil test. Missing mostly in small farm samples and barangay-level surveys. Create flag, impute with median grouped by `soil_texture_class`. Clay soils have CEC 20–35+; Sandy Loam 5–12.

**`base_saturation_pct`** (~8%) — Base saturation requires CEC to compute, so it's missing wherever CEC is missing (and sometimes additional rows). Create flag, impute with median grouped by `soil_ph` bucket (acidic < 5.5, neutral 5.5–7.0, alkaline > 7.0).

**`soil_depth_cm`** (~4%) — Not measured in rapid reconnaissance surveys. Impute with mode (75 cm) — the most common shallow-to-medium depth in Philippine agricultural soils.

### Categorical cleaning

**`season`**: "wet season", "WET SEASON", "wet", "DRY SEASON", "dry season", "dry", "transition", "TRANSITION", "rainy", "Rainy Season" — all need to collapse. "rainy" and "Rainy Season" map to "Wet Season".

```python
def clean_season(s):
    s = str(s).strip().lower()
    if any(w in s for w in ['wet', 'rainy']): return 'Wet Season'
    if 'dry' in s: return 'Dry Season'
    return 'Transition'
df['season'] = df['season'].apply(clean_season)
```

**`drainage_class`**: "poorly drained", "POORLY DRAINED", "well drained", "WELL DRAINED", "moderately well drained", "excessively drained" — strip, lower, then title-case with canonical mapping:

```python
drain_map = {
    'poorly drained': 'Poorly Drained',
    'somewhat poorly drained': 'Somewhat Poorly Drained',
    'moderately well drained': 'Moderately Well Drained',
    'well drained': 'Well Drained',
    'excessively drained': 'Excessively Drained',
}
df['drainage_class'] = df['drainage_class'].str.strip().str.lower().map(drain_map)
```

### Soil chemistry validity checks

```python
# pH must be within measurable range
ph_invalid = (df['soil_ph'] < 3.5) | (df['soil_ph'] > 9.5)
print(f"pH out of range: {ph_invalid.sum()}")

# Organic matter and nitrogen should correlate (OM ≈ N × ~10)
# Flag samples where ratio is extreme (likely transcription error)
df['om_n_ratio'] = df['organic_matter_pct'] / df['nitrogen_pct'].clip(0.01)
ratio_flag = (df['om_n_ratio'] < 4) | (df['om_n_ratio'] > 25)
print(f"Suspicious OM/N ratio: {ratio_flag.sum()}")

# Bulk density above 1.6 g/cm³ indicates severely compacted soil
# These soils support almost no crops — verify, don't silently keep
high_bd = df['bulk_density_gcc'] > 1.6
print(f"Severely compacted (BD > 1.6): {high_bd.sum()}")

# Slope > 30% is above the BSWM threshold for row crop cultivation
steep_slope = df['slope_pct'] > 30
print(f"Slope > 30% (marginal for most crops): {steep_slope.sum()}")
```

---

## Feature engineering checklist

### Block 1 — Missingness flags first

```python
for col in ['phosphorus_ppm', 'organic_matter_pct',
            'cation_exchange_capacity', 'base_saturation_pct', 'soil_depth_cm']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
```

### Block 2 — NPK interpretation features

These translate raw numbers into agronomic interpretation classes used by the BSWM:

```python
# BSWM nitrogen rating
df['n_rating'] = pd.cut(df['nitrogen_pct'],
    bins=[0, 0.10, 0.20, 1.0],
    labels=['Low', 'Medium', 'High'])

# BSWM phosphorus rating (Olsen method)
df['p_rating'] = pd.cut(df['phosphorus_ppm'].fillna(15),
    bins=[0, 10, 25, 200],
    labels=['Low', 'Medium', 'High'])

# BSWM potassium rating
df['k_rating'] = pd.cut(df['potassium_cmol'],
    bins=[0, 0.2, 0.5, 10],
    labels=['Low', 'Medium', 'High'])

# pH interpretation
df['ph_class'] = pd.cut(df['soil_ph'],
    bins=[0, 4.5, 5.5, 6.5, 7.5, 14],
    labels=['Strongly Acidic', 'Moderately Acidic', 'Slightly Acidic', 'Near Neutral', 'Alkaline'])

# Overall fertility index (composite)
df['fertility_index'] = (
    (df['nitrogen_pct'] / 0.20).clip(0, 1) * 0.30
    + (df['phosphorus_ppm'].fillna(15) / 25).clip(0, 1) * 0.25
    + (df['potassium_cmol'] / 0.5).clip(0, 1) * 0.20
    + (df['organic_matter_pct'].fillna(2) / 3.0).clip(0, 1) * 0.25
).round(3)
```

### Block 3 — Soil health and structural features

```python
# pH distance from ideal for Philippine crops (6.0–6.5 is optimal for most)
df['ph_deviation_from_ideal'] = (df['soil_ph'] - 6.2).abs().round(2)

# Compaction flag
df['is_compacted'] = (df['bulk_density_gcc'] > 1.4).astype(int)

# Rooting depth adequacy
df['shallow_soil'] = (df['soil_depth_cm'].fillna(75) < 45).astype(int)
df['deep_soil']    = (df['soil_depth_cm'].fillna(75) > 90).astype(int)

# Drainage suitability score (ordinal: Poorly=1, Somewhat Poorly=2, Moderately=3, Well=4, Excessive=5)
drain_ord = {'Poorly Drained':1,'Somewhat Poorly Drained':2,
             'Moderately Well Drained':3,'Well Drained':4,'Excessively Drained':5}
df['drainage_ordinal'] = df['drainage_class'].map(drain_ord).fillna(3)

# Organic matter adequacy
df['om_adequate'] = (df['organic_matter_pct'].fillna(2) >= 2.0).astype(int)
```

### Block 4 — Climate and seasonal interactions

```python
# Agroclimatic zone: combines climate type + elevation
df['agroclimatic_zone'] = df['climate_type'] + '_' + pd.cut(
    df['elevation_m'],
    bins=[-1, 200, 600, 1000, 3000],
    labels=['Lowland', 'Midland', 'Highland', 'Upland']
).astype(str)

# Wet season in Type I climate = peak growing period
df['peak_growing_period'] = (
    (df['climate_type'] == 'Type I') & (df['season'] == 'Wet Season')
).astype(int)

# Highland crop zone: CAR + elevation > 800m + cool temp
df['highland_zone'] = (
    (df['elevation_m'] > 800) & (df['temp_c'] < 24)
).astype(int)

# Lowland irrigated potential: flat + near water + good drainage
df['irrigated_potential'] = (
    (df['slope_pct'] < 5) &
    (df['near_water_body'] == 1) &
    df['drainage_class'].isin(['Moderately Well Drained', 'Well Drained'])
).astype(int)

# Temperature suitability by zone
df['lowland_heat'] = (df['temp_c'] > 27).astype(int)  # suits tropical crops
df['highland_cool'] = (df['temp_c'] < 22).astype(int)  # suits temperate vegetables
```

### Block 5 — Farm history features

```python
# Soil depletion risk: monocrop without rotation
df['soil_depletion_risk'] = (df['farm_history'] == 'Monocrop').astype(int)

# N boost from legume rotation
df['prior_legume'] = df['previous_crop'].isin(['Sitaw', 'Munggo', 'Vegetable']).astype(int)

# Virgin land: likely high OM, no pest pressure
df['virgin_land_bonus'] = (df['farm_history'] == 'Virgin Land').astype(int)

# Encode farm_history ordinally by fertility preservation score
fh_ord = {'Virgin Land':5,'Rotation':4,'Fallow':3,'Orchard':2,'Monocrop':1}
df['farm_history_score'] = df['farm_history'].map(fh_ord).fillna(3)
```

### Block 6 — Log transforms for skewed soil variables

```python
df['log_phosphorus']    = np.log1p(df['phosphorus_ppm'].fillna(15))
df['log_rainfall']      = np.log1p(df['rainfall_mm'])
df['log_elevation']     = np.log1p(df['elevation_m'])
df['log_slope']         = np.log1p(df['slope_pct'])
df['log_bnp_discharge'] = np.log1p(df['discharge_bnp'] if 'discharge_bnp' in df.columns else 0)
```

### Block 7 — Encoding strategy

- `region`, `soil_texture_class`, `climate_type`, `season`, `farm_history`, `previous_crop`: One-hot encode
- `drainage_class`: Use the `drainage_ordinal` already created
- `ph_class`, `n_rating`, `p_rating`, `k_rating`: Ordinal encode (they have natural order)
- `province_soil_name`: Too many categories (47) — use `soil_texture_class` instead, or target-encode using mean `suitability_spread`
- Drop `sample_id`, `best_crop`, `top3_crops` when training the score regressors (they're derived from scores)

---

## Modeling — regression + ranking

### What you're modeling

You have **three levels of targets** and you should work through all three:

**Level 1 — Regression on individual crop scores**
Train one model per crop to predict its suitability score. Use `score_Palay (Rice)` as your regression target, then separately `score_Mais (Corn)`, etc. Evaluate RMSE per crop. Some crops will be easier to predict than others — malunggay (adaptable to almost anything) will have low variance; strawberry and cabbage (strict requirements) will show clear feature splits.

**Level 2 — Multi-output regression**
Predict all 25 scores simultaneously:

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Train one model that predicts all 25 crop scores at once
multi_reg = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    n_jobs=-1
)
score_cols = [c for c in df.columns if c.startswith('score_')]
multi_reg.fit(X_train, y_train[score_cols])
```

**Level 3 — Ranking: which crop should be #1?**
After predicting 25 scores, rank the crops for each sample by predicted score. Evaluate with:
- **NDCG@K** (Normalized Discounted Cumulative Gain at K) — the gold standard ranking metric
- **Precision@1** — is the top-ranked predicted crop the actual best crop?
- **Precision@3** — is the actual best crop in your top 3?

```python
from sklearn.metrics import ndcg_score

# y_true and y_score must be (n_samples, n_crops) arrays
ndcg = ndcg_score(y_true_scores, y_pred_scores, k=3)
print(f"NDCG@3: {ndcg:.4f}")

# Precision@1: fraction where predicted rank-1 == actual rank-1
p_at_1 = (predicted_best_crop == actual_best_crop).mean()
print(f"Precision@1: {p_at_1:.4f}")

# Precision@3: fraction where actual best crop is in predicted top 3
p_at_3 = np.mean([
    actual_best[i] in predicted_top3[i]
    for i in range(len(actual_best))
])
print(f"Precision@3: {p_at_3:.4f}")
```

### Models to train

**1. Random Forest Regressor (per crop, one model per crop)**
Start here. For each of the 25 crops, train a separate RF. Extract feature importances. `soil_ph`, `nitrogen_pct`, `season`, `elevation_m`, and `temp_c` should dominate for most crops. Crops with strict requirements (onion, garlic, strawberry, cabbage) will have higher feature importance on `temp_c` and `season` than adaptable crops (malunggay, cassava, coconut).

**2. XGBoost MultiOutput Regressor**
Best performance. Use `MultiOutputRegressor(XGBRegressor(...))`. Compare NDCG@3 to the RF baseline.

**3. LightGBM Ranker (Learning to Rank — new concept)**
This is the proper ranking model. Instead of predicting individual scores, it directly optimizes ranking quality using LambdaRank:

```python
import lightgbm as lgb

# For LambdaRank, we need group structure: each soil sample = one query
# All 25 crops for that sample = the documents to rank
# Reshape data: each row is (sample × crop), target = suitability score

# Melt the dataset
score_cols = [c for c in df.columns if c.startswith('score_')]
df_long = df.melt(id_vars=feature_cols + ['sample_id'],
                   value_vars=score_cols,
                   var_name='crop', value_name='suitability_score')
df_long['crop'] = df_long['crop'].str.replace('score_','')

# Add crop-specific features (NPK requirements — one-hot or embedding)
# Group sizes: 25 crops per sample
group_sizes = df_long.groupby('sample_id').size().values

ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg',
                         ndcg_eval_at=[1, 3, 5])
ranker.fit(X_train_long, y_train_long, group=group_train)
```

This is the model that most closely matches what a real precision agriculture recommendation system would use.

### Crop-specific feature importance comparison

One of the most insightful analyses in this project: for each crop, what are the top 3 features driving its score?

```python
importance_df = pd.DataFrame({
    crop.replace('score_', ''): model.feature_importances_
    for crop, model in zip(score_cols, multi_reg.estimators_)
}, index=feature_names)

# Heatmap: crops on columns, features on rows
plt.figure(figsize=(20, 12))
sns.heatmap(importance_df.T, cmap='YlOrRd', yticklabels=True, xticklabels=True)
plt.title('Feature Importance per Crop — What drives each crop suitability score?')
```

---

## Visualization guide

### Plot 1 — NPK triangle plot by region

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Ternary plot of N:P:K ratio colored by region
# Use the 'python-ternary' library
# pip install python-ternary
import ternary

fig, ax = ternary.figure(scale=100)
# Normalize NPK to percentages
total = df['nitrogen_pct'] + df['phosphorus_ppm']/100 + df['potassium_cmol']/5
points = list(zip(
    (df['nitrogen_pct']/total*100).clip(0,100),
    (df['phosphorus_ppm'].fillna(15)/100/total*100).clip(0,100),
    (df['potassium_cmol']/5/total*100).clip(0,100)
))
ax.scatter(points, s=3, alpha=0.4)
ax.set_title('N:P:K Balance across Philippine Soil Samples')
```

This triangle plot is unique to soil science — you won't see it in finance or healthcare projects. It shows whether samples are nitrogen-dominated, phosphorus-limited, or potassium-rich at a glance.

### Plot 2 — pH distribution by region (violin plot)

```python
fig, ax = plt.subplots(figsize=(16, 6))
region_order = df.groupby('region')['soil_ph'].median().sort_values().index
sns.violinplot(data=df, x='region', y='soil_ph', order=region_order,
               palette='viridis', ax=ax, inner='box', scale='width')
ax.axhline(5.5, color='orange', linestyle='--', label='Lower ideal pH (5.5)')
ax.axhline(6.5, color='green',  linestyle='--', label='Upper ideal pH (6.5)')
ax.axhline(7.0, color='red',    linestyle='--', label='Alkalinity threshold (7.0)')
plt.xticks(rotation=45, ha='right')
plt.title('Soil pH Distribution by Philippine Region')
plt.legend()
```

Central Luzon and Cagayan Valley should cluster near the ideal 5.5–6.5 band. CAR should show lower pH (acidic upland soils). This is the chart a BSWM extension worker would show to explain regional soil amendment needs.

### Plot 3 — Crop suitability heatmap by region

```python
score_cols  = [c for c in df.columns if c.startswith('score_')]
crop_labels = [c.replace('score_','') for c in score_cols]
region_crop_mean = df.groupby('region')[score_cols].mean()
region_crop_mean.columns = crop_labels

plt.figure(figsize=(22, 9))
sns.heatmap(region_crop_mean, cmap='RdYlGn', annot=False,
            linewidths=0.3, vmin=40, vmax=85)
plt.title('Mean Crop Suitability Score by Philippine Region')
plt.xlabel('Crop')
plt.ylabel('Region')
plt.xticks(rotation=45, ha='right')
```

This is the master visualization of the project. It shows at a glance: CAR is green for coffee, strawberry, and cabbage; Davao is green for durian, cacao, and banana; Western Visayas is green for sugarcane; Central Luzon is green for rice. The heatmap validates that the synthetic data aligns with real Philippine agricultural geography.

### Plot 4 — Season × crop suitability matrix

```python
season_crop = df.groupby('season')[score_cols].mean()
season_crop.columns = crop_labels

fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)
for ax, seas in zip(axes, ['Wet Season','Dry Season','Transition']):
    if seas in season_crop.index:
        vals = season_crop.loc[seas].sort_values(ascending=False)
        ax.barh(vals.index, vals.values,
                color=['green' if v >= 65 else 'orange' if v >= 50 else 'tomato' for v in vals.values])
        ax.axvline(65, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{seas}\nMean Suitability Score')
        ax.set_xlim(0, 100)
plt.suptitle('Crop Suitability by PAGASA Season', fontsize=14)
plt.tight_layout()
```

Dry Season: onion, garlic, tomato, camote at top. Wet Season: rice, abaca, durian at top. This directly informs planting calendar recommendations.

### Plot 5 — NPK deficiency radar chart per soil texture class

```python
import numpy as np
import matplotlib.pyplot as plt

categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter', 'pH Balance']
textures   = df['soil_texture_class'].unique()
angles     = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles    += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                          subplot_kw=dict(polar=True))
for ax, texture in zip(axes.flat, textures):
    sub = df[df['soil_texture_class'] == texture]
    values = [
        sub['nitrogen_pct'].mean() / 0.20,                        # normalized to medium
        sub['phosphorus_ppm'].fillna(15).mean() / 25,
        sub['potassium_cmol'].mean() / 0.5,
        sub['organic_matter_pct'].fillna(2).mean() / 3.0,
        1 - abs(sub['soil_ph'].mean() - 6.2) / 2.0,               # closeness to ideal pH
    ]
    values = [min(v, 1.5) for v in values]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.2, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1.5)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax.set_title(texture, size=10, pad=10)
plt.suptitle('Soil Fertility Radar by Texture Class', fontsize=13)
plt.tight_layout()
```

Radar/spider charts are the visualization language of soil science extension work. Benguet Sandy Loam will show low P and low K relative to Clay Loam soils. This is the chart a farmer's field school would use.

### Plot 6 — Ranking quality: Precision@K curves

```python
k_values   = range(1, 10)
p_at_k_rf  = []
p_at_k_xgb = []

for k in k_values:
    # For each sample, check if actual best crop is in predicted top-k
    p_rf  = np.mean([actual[i] in predicted_rf_topk[i][:k]  for i in range(len(actual))])
    p_xgb = np.mean([actual[i] in predicted_xgb_topk[i][:k] for i in range(len(actual))])
    p_at_k_rf.append(p_rf)
    p_at_k_xgb.append(p_xgb)

plt.figure(figsize=(9, 5))
plt.plot(k_values, p_at_k_rf,  'o-', label='Random Forest', color='steelblue')
plt.plot(k_values, p_at_k_xgb, 's-', label='XGBoost',       color='tomato')
plt.axhline(1/25, color='gray', linestyle='--', label='Random baseline (1/25)')
plt.xlabel('K (Top-K recommended crops)')
plt.ylabel("Precision@K (actual best crop in top-K)")
plt.title('Ranking Quality: Is the Best Crop in the Top-K Recommendations?')
plt.legend()
plt.grid(alpha=0.3)
```

At K=1, expect ~55–70% for a good model. At K=3, expect ~80%+. This tells a farmer: "if we give you 3 crop recommendations, the truly best crop for your soil will be among them 80% of the time."

### Plot 7 — SHAP waterfall for a single soil sample

```python
import shap

explainer   = shap.TreeExplainer(xgb_model_rice)  # rice model
shap_values = explainer.shap_values(X_test)

# Single sample explanation
sample_idx = 42
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx],
        feature_names=feature_names
    )
)
```

This tells a farmer exactly why their soil scored 73 for rice: "your pH is ideal (+8 points), your drainage is slightly poor (+3 points), but your nitrogen is low (−12 points) and it's currently Dry Season (−15 points)." That's actionable: amend N, wait for wet season, or switch crops.

---

## Deployment — agricultural recommendation API

### Batch recommendation endpoint

```python
@app.post('/recommend_crops')
def recommend_crops(soil: SoilInput, top_k: int = 5):
    df_in = pd.DataFrame([soil.dict()])
    X     = preprocessor.transform(df_in)
    scores = multi_model.predict(X)[0]  # shape: (25,)

    crop_names   = [c.replace('score_','') for c in score_cols]
    ranked_idx   = np.argsort(scores)[::-1]
    recommendations = [
        {
            'rank': int(r+1),
            'crop': crop_names[i],
            'suitability_score': round(float(scores[i]), 1),
            'suitability_class': 'Highly Suitable' if scores[i]>=75 else
                                  'Moderately Suitable' if scores[i]>=55 else
                                  'Marginally Suitable' if scores[i]>=40 else 'Not Suitable'
        }
        for r, i in enumerate(ranked_idx[:top_k])
    ]
    return {
        'region': soil.region,
        'season': soil.season,
        'soil_texture': soil.soil_texture_class,
        'recommendations': recommendations
    }
```

The `suitability_class` labels follow the FAO Land Suitability Classification (S1 = Highly Suitable, S2 = Moderately Suitable, S3 = Marginally Suitable, N = Not Suitable) — the international standard for agricultural land evaluation.

---

## Project structure

```
ph_crop_ranking_project/
│
├── data/
│   └── raw/
│       └── ph_soil_crop.csv          # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_eda_visualization.ipynb    # NPK triangle, pH by region, crop heatmap
│   ├── 02_feature_engineering.ipynb  # NPK ratings, fertility index, agroclimatic zones
│   ├── 03_regression_per_crop.ipynb  # One model per crop, feature importance comparison
│   └── 04_ranking_models.ipynb       # MultiOutput, LightGBM Ranker, NDCG, Precision@K
│
├── src/
│   ├── validate_soil.py              # pH range checks, OM/N ratio checks
│   └── pipeline.py                   # Full preprocessing + feature engineering
│
├── models/
│   ├── preprocessor.pkl
│   ├── multi_crop_regressor.pkl
│   └── model_card.json
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
lightgbm>=4.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
python-ternary>=1.0
missingno>=0.5
jupyterlab>=4.0
fastapi>=0.110
uvicorn>=0.29
joblib>=1.3
pydantic>=2.0
```

---

## What's new vs every previous project

| Concept | First introduced here |
|---|---|
| 25 simultaneous regression targets (one per crop) | ✓ |
| `MultiOutputRegressor` for multi-target regression | ✓ |
| Learning to Rank — LightGBM `LGBMRanker` with LambdaRank | ✓ |
| NDCG@K (Normalized Discounted Cumulative Gain) | ✓ |
| Precision@K for ranking evaluation | ✓ |
| Ternary/triangle plot (N:P:K balance) | ✓ |
| Radar/spider chart (soil fertility by texture class) | ✓ |
| Domain-knowledge imputation (OM = N × 10) | ✓ |
| FAO Land Suitability Classification in output | ✓ |
| Philippine Bureau of Soils series names | ✓ |
| PAGASA climate type encoding | ✓ |
| Crop-specific feature importance comparison heatmap | ✓ |

---

*Dataset is fully synthetic. Crop agronomic requirements are based on DA-BAR, PhilRice, and FAO crop production guidelines for the Philippines, but this dataset is not validated for actual farm advisory use.*
