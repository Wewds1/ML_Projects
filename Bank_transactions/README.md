# Bank Transaction Anomaly Detection — End-to-End ML Project

**Target variable:** None — this is unsupervised learning  
**Model type:** Clustering (K-Means, DBSCAN, Hierarchical) + Anomaly Scoring  
**Domain:** AML (Anti-Money Laundering) / transaction surveillance  
**Dataset:** 8,000 synthetic bank transactions, deliberately dirty  
**Core challenge:** No labels. You don't know which transactions are suspicious — you have to find them.

---

## Why this project is structurally different from everything before it

The previous three projects all had a target column. You cleaned data, engineered features, fit a model, and measured error against ground truth. That loop doesn't exist here.

Clustering is unsupervised. There's no `is_fraud` column, no RMSE to minimize, no accuracy to report. Instead you're asking: *what natural groupings exist in this data, and which groups look behaviorally unusual compared to the rest?*

This changes almost everything:

- **Evaluation is interpretive, not numeric.** You can't compute precision and recall. You assess cluster quality using silhouette score, Davies-Bouldin index, and most importantly — whether the clusters make business sense when you profile them.
- **Preprocessing decisions matter more.** K-Means uses Euclidean distance. If `txn_amount` ranges from $1 to $250,000 and `kyc_risk_score` ranges from 1 to 5, amount will dominate every distance calculation before any learning happens. Scaling isn't optional.
- **Feature selection is a modeling decision.** You choose which features to cluster on. Including irrelevant features adds noise and degrades cluster quality — this is called the "curse of dimensionality."
- **The output is a cluster label + a profile, not a prediction.** The deliverable is a table that says "Cluster 3 contains 87 accounts — here's their behavioral fingerprint" and a judgment call about whether Cluster 3 warrants investigation.

---

## Dataset columns

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | string | Unique ID — drop before modeling |
| `timestamp` | datetime string | Parse before use |
| `hour_of_day` | int | 0–23 |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `is_weekend` | binary | Derived from day_of_week |
| `month` | int | 1–12 |
| `account_type` | categorical | **dirty** — Checking / Savings / Business / Student / Premium |
| `account_age_months` | int | Months since account opened |
| `account_balance` | float | Current balance USD |
| `customer_age` | int | 18–78 |
| `customer_segment` | categorical | **dirty** — Retail / SME / Corporate / Student / Wealth |
| `txn_type` | categorical | **dirty** — Wire Transfer / ACH / POS Purchase / ATM / etc. |
| `txn_direction` | categorical | Debit / Credit |
| `txn_amount` | float | USD, heavily right-skewed |
| `counterparty_country` | categorical | **~4% missing** — internal/system gaps |
| `is_high_risk_country` | binary | 1 if NG, RU, KY, AE |
| `is_cross_border` | binary | 1 if not US |
| `txn_count_7d` | int | Transactions last 7 days |
| `txn_count_30d` | int | Transactions last 30 days |
| `avg_txn_amount_30d` | float | **~5% missing** — new accounts |
| `max_txn_amount_30d` | float | Largest single transaction in 30 days |
| `unique_counterparties_30d` | int | Distinct counterparties in 30 days |
| `cross_border_ratio_30d` | float | Fraction of 30d transactions that were cross-border |
| `structuring_flag` | binary | 1 if amount between $8,500–$9,999 (below CTR threshold) |
| `prior_sar_count` | int | Prior Suspicious Activity Reports filed |
| `kyc_risk_score` | float | **~3% missing** — pending KYC review (1=low, 5=high) |

### Three anomalous behavioral patterns injected (no label column — clustering should surface these)

**Pattern A — Structuring + high-risk wire transfers:** ~90 transactions. Large round-number wire transfers to Nigeria, Russia, or Cayman Islands between 1–4am, amounts just below the $10,000 CTR reporting threshold. Classic money laundering structuring pattern.

**Pattern B — Account takeover velocity spike:** ~75 transactions. Sudden spike in transaction count (25–60 in 7 days vs normal of 6), large number of new unique counterparties (40–80), small individual amounts. Behavioral signature of a compromised account being used to disperse funds.

**Pattern C — Layering:** ~60 transactions. Single very large transaction with average 30-day amount being a fraction of the current transaction. High cross-border ratio. Inbound/outbound mismatch. Classic layering — placing funds in, then moving them through multiple hops.

---

## Data issues to fix

### Duplicates
~30 duplicate rows. Drop on all columns except `transaction_id`.

### Missing values

- **`counterparty_country`** (~4%) — Missing mostly for internal transfers and system gaps, not randomly. Create a binary flag `counterparty_country_missing = 1` before imputing. Impute with "US" (domestic default). The flag captures the missingness signal; K-Means can't handle NaN directly so you must fill.

- **`avg_txn_amount_30d`** (~5%) — Missing for new accounts with fewer than 30 days of history (`account_age_months <= 1`). Impute with the median grouped by `account_type`. A new Business account and a new Student account have very different typical transaction sizes. Do not use global median.

- **`kyc_risk_score`** (~3%) — Missing because KYC review is pending. This is MNAR — accounts pending review are often higher risk than cleared accounts. Create a flag `kyc_pending = 1` before imputing. Impute with the median (3 = midpoint). The flag is more important than the fill value.

### Categorical inconsistency

- **`txn_type`**: "wire transfer", "WIRE TRANSFER", "Wire transfer", "ACH ", "ach", "POS purchase", "pos purchase", "ATM withdrawal", "atm withdrawal" — strip whitespace, apply `.str.strip().str.title()`, then manually map: "Ach" → "ACH", "Atm Withdrawal" → "ATM Withdrawal", "Pos Purchase" → "POS Purchase".
- **`account_type`**: "checking", "CHECKING", "savings", "SAVINGS", "business ", "student " — strip and title-case.
- **`customer_segment`**: "retail", "RETAIL", "sme", "SME " — strip, title-case, then map "Sme" → "SME".

### Outlier treatment — different logic for clustering vs regression

In regression, you clip outliers to prevent them from distorting model coefficients. In clustering, outliers are often the *point* — they may be the anomalous transactions you're looking for. **Do not clip `txn_amount` or `max_txn_amount_30d` before clustering.**

Instead:
- Log-transform skewed features (`txn_amount`, `account_balance`, `max_txn_amount_30d`, `avg_txn_amount_30d`) to compress the range without removing extremes.
- After clustering, check if outliers naturally fall into their own small cluster. If they do, that cluster is a candidate anomaly group.

---

## EDA checklist

EDA for clustering has a different goal than EDA for supervised learning. You're not looking for correlations with a target — you're looking for the natural structure and distribution of the data itself. You're also looking for features that separate meaningfully so you can decide which ones to cluster on.

### Understand the distributions first
- Histogram of `txn_amount` (raw and log-transformed) — note the extreme right skew and any suspicious clustering near $9,000–$9,999.
- Histogram of `txn_count_7d` — most accounts will cluster around 3–8. Values above 20 are notable.
- Histogram of `cross_border_ratio_30d` — most accounts should be near 0. Any account above 0.5 deserves attention.
- Bar chart of `txn_type` and `account_type` counts — just to understand the data mix.

### Identify potential anomaly signals before modeling
These plots tell you what the three injected patterns look like before you've run any algorithm:

- Scatter: `txn_amount` vs `txn_count_7d`, colored by `is_high_risk_country` — Pattern A and B both show up here.
- Scatter: `avg_txn_amount_30d` vs `txn_amount` — the ratio between these two is the core Pattern C signal. Transactions where `txn_amount` >> `avg_txn_amount_30d` should be visible as a separate cloud in the upper-left of this plot.
- Histogram: `txn_amount` filtered to `structuring_flag == 1` — should show a tight cluster just below $10,000.
- Bar chart: `structuring_flag` rate by `counterparty_country` — NG, RU, KY should have disproportionately high structuring rates.
- Box plot: `unique_counterparties_30d` overall vs for rows where `txn_count_7d > 20` — Pattern B shows up here.

### Time patterns
- Heatmap: transaction count by `hour_of_day` × `day_of_week` — most legitimate transactions happen 8am–9pm on weekdays. Late-night weekend transactions are worth noting.
- Bar chart: mean `txn_amount` by `hour_of_day` — the 1–4am band should have a higher mean than daytime if Pattern A is visible.

### Feature correlation for clustering
Compute a correlation matrix on all numeric features. Features that are highly correlated (>0.85) add redundancy without new information to clustering. You'll need to decide whether to drop one of a correlated pair or use dimensionality reduction (PCA) before clustering. Specifically check: `txn_count_7d` vs `txn_count_30d`, and `max_txn_amount_30d` vs `avg_txn_amount_30d`.

---

## Feature engineering checklist

### The most important ratio features for AML clustering

| Feature | Formula | AML rationale |
|---|---|---|
| `amount_vs_avg_ratio` | txn_amount / avg_txn_amount_30d.clip(1) | How unusual is this transaction vs the account's normal behavior |
| `amount_to_balance_ratio` | txn_amount / account_balance.clip(1) | Relative size — large transaction vs a small balance is suspicious |
| `velocity_spike_ratio` | txn_count_7d / (txn_count_30d / 4).clip(0.1) | Is transaction frequency accelerating in the last 7 days |
| `counterparty_diversity` | unique_counterparties_30d / txn_count_30d.clip(1) | High ratio = many new counterparties = account takeover signal |
| `max_to_avg_ratio` | max_txn_amount_30d / avg_txn_amount_30d.clip(1) | High ratio = one large transaction in otherwise small-amount account |

### Log transformations — required before K-Means and hierarchical clustering

```python
df['log_txn_amount']       = np.log1p(df['txn_amount'])
df['log_account_balance']  = np.log1p(df['account_balance'])
df['log_avg_30d_spend']    = np.log1p(df['avg_txn_amount_30d'])
df['log_max_30d_spend']    = np.log1p(df['max_txn_amount_30d'])
```

Without this, `txn_amount` will dominate every distance calculation and effectively be the only feature K-Means uses.

### Time-based flags

```python
df['is_late_night']     = df['hour_of_day'].between(1, 4).astype(int)
df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
```

### Missing value flags — create before imputing

```python
df['counterparty_missing'] = df['counterparty_country'].isnull().astype(int)
df['kyc_pending']          = df['kyc_risk_score'].isnull().astype(int)
df['is_new_account']       = (df['account_age_months'] <= 1).astype(int)
```

### Encoding for clustering

Clustering algorithms work on numeric vectors. One-hot encode all categoricals.

- `txn_type`, `account_type`, `customer_segment`, `txn_direction`: One-hot encode.
- `counterparty_country`: Group rare countries into "Other" (keep US, UK, CA, DE, SG, NG, RU, KY, AE as explicit — rest → "Other") before OHE. This prevents the country dimension from ballooning.

Drop `transaction_id` and `timestamp` after feature extraction.

### Feature selection for clustering — critical step
More features is not better in clustering. After engineering, you'll have ~40+ columns. Run PCA first to understand how much variance each component captures. If 5–6 components explain 80%+ of variance, cluster on those PCA components rather than raw features. This reduces noise and the curse of dimensionality.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.85, random_state=42)  # keep components explaining 85% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Components retained: {pca.n_components_}")
```

Try clustering on both the raw scaled features and the PCA-reduced version. Compare silhouette scores. Use whichever produces better-separated clusters.

---

## Modeling checklist

### Always scale before clustering
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
This is not optional. K-Means minimizes within-cluster variance measured in Euclidean distance. Unscaled features with large ranges will dominate.

---

### Model 1 — K-Means

**Step 1: Find the right K using the elbow method**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias, silhouettes = [], []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Plot both curves — the elbow in inertia and the peak in silhouette should roughly agree
```

The elbow in the inertia curve is where adding another cluster gives diminishing returns. The silhouette score peak tells you which K produces the most internally cohesive and externally separated clusters. If they disagree, try both K values and profile the resulting clusters.

**Step 2: Fit the final model**
```python
km = KMeans(n_clusters=K_best, random_state=42, n_init=20, max_iter=500)
df['cluster_kmeans'] = km.fit_predict(X_scaled)
```

**Step 3: Profile each cluster — this is the actual deliverable**
```python
profile = df.groupby('cluster_kmeans').agg({
    'txn_amount': ['mean', 'median', 'max'],
    'txn_count_7d': 'mean',
    'is_high_risk_country': 'mean',
    'cross_border_ratio_30d': 'mean',
    'structuring_flag': 'mean',
    'kyc_risk_score': 'mean',
    'unique_counterparties_30d': 'mean',
    'amount_vs_avg_ratio': 'mean',
}).round(3)
print(profile)
```

Sort by `structuring_flag` mean and `is_high_risk_country` mean descending. The clusters with the highest values on those two columns are your candidates for manual review. Assign each cluster a human-readable label: "Normal Retail," "High-Value Business," "Suspicious Wire Activity," etc.

---

### Model 2 — DBSCAN

DBSCAN doesn't require you to specify K. It finds clusters of arbitrary shape and, critically, **explicitly labels outliers as noise** (label = -1). In anomaly detection, the noise points are often the most interesting.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.5, min_samples=10, metric='euclidean')
df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)

n_clusters = len(set(df['cluster_dbscan'])) - (1 if -1 in df['cluster_dbscan'].values else 0)
n_noise    = (df['cluster_dbscan'] == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points (anomaly candidates): {n_noise} ({n_noise/len(df)*100:.1f}%)")
```

**Tuning `eps` and `min_samples`:**
- Use the k-distance plot to find a good `eps`. For each point, compute the distance to its k-th nearest neighbor. Sort and plot. The "elbow" in that plot is a good starting `eps`.
- `min_samples` should be at least `n_features + 1`. For ~15 features, try 15–20.
- A good result has 1–4% noise points. If you get 30%+ noise, `eps` is too small. If you get 0 noise, `eps` is too large.

```python
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=15).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.title('K-Distance Plot — find the elbow for eps')
```

**Profile the noise points:**
```python
noise_txns = df[df['cluster_dbscan'] == -1]
print(noise_txns[['txn_amount', 'is_high_risk_country', 'structuring_flag',
                   'txn_count_7d', 'unique_counterparties_30d']].describe())
```

Compare mean values of noise points vs the full dataset. If noise points have significantly higher `structuring_flag`, `is_high_risk_country`, and `amount_vs_avg_ratio`, DBSCAN is surfacing real anomalies.

---

### Model 3 — Agglomerative Hierarchical Clustering

Hierarchical clustering builds a tree (dendrogram) that shows how transactions merge into groups at different similarity thresholds. Useful here because you don't need to commit to a K upfront — you cut the tree at whichever level makes business sense.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Plot dendrogram on a sample (full dataset is too large for a clear plot)
sample_idx = np.random.choice(len(X_scaled), 300, replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(16, 6))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (sample of 300)')
plt.xlabel('Transaction index')
plt.ylabel('Ward linkage distance')
plt.axhline(y=8, color='red', linestyle='--', label='Cut here for ~5 clusters')
```

The long vertical lines in the dendrogram show where clusters merge at high distance — those are the "natural" cut points. A horizontal red line at that height gives you a sensible K.

```python
# Fit on full data with chosen K
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['cluster_hierarchical'] = agg.fit_predict(X_scaled)
```

Ward linkage minimizes within-cluster variance — analogous to K-Means but without the centroid constraint. Use ward unless you have strong reason to use average or complete linkage.

---

### Model 4 — Isolation Forest (anomaly scoring complement)

After clustering, use Isolation Forest as a second lens. It assigns an anomaly score to every transaction independently of cluster membership. High anomaly score + membership in a suspicious cluster = strong candidate for review.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
df['anomaly_score'] = iso.fit_predict(X_scaled)           # -1 = anomaly, 1 = normal
df['anomaly_raw_score'] = iso.decision_function(X_scaled) # lower = more anomalous
```

The `contamination` parameter is your prior on what fraction of transactions are anomalous. Set it to ~2–4% — consistent with the patterns injected in this dataset.

**Cross-reference with DBSCAN noise:**
```python
# Transactions flagged by both DBSCAN and Isolation Forest
both_flagged = df[(df['cluster_dbscan'] == -1) & (df['anomaly_score'] == -1)]
print(f"Transactions flagged by both methods: {len(both_flagged)}")
```

Consensus between two independent methods is much stronger evidence than either alone.

---

### Cluster evaluation metrics

These replace the RMSE / R² / PR-AUC from supervised projects:

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print("Silhouette Score (higher = better, range -1 to 1):",
      silhouette_score(X_scaled, df['cluster_kmeans']))

print("Davies-Bouldin Index (lower = better):",
      davies_bouldin_score(X_scaled, df['cluster_kmeans']))

print("Calinski-Harabasz Index (higher = better):",
      calinski_harabasz_score(X_scaled, df['cluster_kmeans']))
```

- **Silhouette score** measures how similar each point is to its own cluster vs the nearest other cluster. Above 0.4 is decent; above 0.6 is strong. Below 0.2 means your clusters are barely separated.
- **Davies-Bouldin** penalizes clusters that are close together or have large intra-cluster spread. Lower is better.
- **Calinski-Harabasz** rewards compact, well-separated clusters. Higher is better.

Compare these across K-Means, DBSCAN (ignoring noise points), and Hierarchical. No single metric is definitive — use them together alongside business interpretability of the cluster profiles.

---

### Final output — the anomaly report

The deliverable is a ranked table of suspicious transactions:

```python
# Score each transaction: higher = more suspicious
df['risk_score'] = (
    (df['cluster_dbscan'] == -1).astype(int) * 3
  + (df['anomaly_score'] == -1).astype(int) * 2
  + df['structuring_flag']
  + df['is_high_risk_country']
  + df['prior_sar_count']
  + (df['kyc_risk_score'] >= 4).astype(int)
  + (df['amount_vs_avg_ratio'] > 5).astype(int)
)

top_suspicious = (df.sort_values('risk_score', ascending=False)
                    .head(50)[['transaction_id', 'txn_amount', 'txn_type',
                                'counterparty_country', 'structuring_flag',
                                'txn_count_7d', 'unique_counterparties_30d',
                                'risk_score']])
```

This is what a compliance analyst actually receives. The model doesn't decide — it ranks. Humans investigate.

---

## Deployment

### How clustering deploys differently from classification

A fraud classifier gets deployed as a real-time inference API — one transaction comes in, one prediction goes out. Clustering is different. You don't deploy K-Means to score a single transaction in isolation; a transaction's cluster membership depends on the entire dataset it was trained on.

In practice, AML clustering systems work in one of two ways:

**Batch scoring (most common):** Run the clustering pipeline nightly or weekly on a rolling window of transactions. Output a risk-ranked list to the compliance team each morning. The "deployment" is a scheduled job, not a live API.

**Profile-based scoring (scalable alternative):** Instead of re-clustering in production, extract the cluster centroids from training and score new transactions by their distance to the nearest centroid. This gives you a continuous anomaly signal in real time.

### Step 1 — Serialize the preprocessing and centroids

```python
import joblib

# Save the full preprocessing pipeline
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca.pkl')          # if using PCA reduction
joblib.dump(km, 'models/kmeans.pkl')
joblib.dump(iso, 'models/isolation_forest.pkl')

# Save cluster profiles for human-readable output
cluster_profiles.to_csv('models/cluster_profiles.csv')
```

### Step 2 — Batch scoring script

```python
# score_batch.py
import pandas as pd, numpy as np, joblib

scaler  = joblib.load('models/scaler.pkl')
pca     = joblib.load('models/pca.pkl')
km      = joblib.load('models/kmeans.pkl')
iso     = joblib.load('models/isolation_forest.pkl')

def score_transactions(df_new: pd.DataFrame) -> pd.DataFrame:
    # Apply same feature engineering as training
    X = preprocess(df_new)           # your feature engineering function
    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)

    df_new['cluster']       = km.predict(X_pca)
    df_new['iso_flag']      = iso.predict(X_scaled)
    df_new['iso_score']     = iso.decision_function(X_scaled)
    df_new['risk_score']    = compute_risk_score(df_new)
    return df_new.sort_values('risk_score', ascending=False)
```

### Step 3 — Schedule the batch job

**Locally / on a server:** Use `cron` or `Airflow` to run `score_batch.py` nightly.

```bash
# crontab entry — run at 2am every day
0 2 * * * python /app/score_batch.py >> /logs/scoring.log 2>&1
```

**On cloud:**
- **AWS:** EventBridge (scheduled) → Lambda or ECS task
- **GCP:** Cloud Scheduler → Cloud Run job
- **Azure:** Azure Data Factory pipeline with a timer trigger

### Step 4 — Output delivery

The job outputs a CSV or writes to a database table: top N flagged transactions, ranked by risk score. A compliance dashboard (Tableau, Power BI, or a simple FastAPI + React app) reads from that table and displays the queue to analysts.

For a portfolio project, write the output to a SQLite database and build a minimal FastAPI endpoint:

```python
@app.get("/flagged_transactions")
def get_flagged(limit: int = 50, min_risk_score: int = 3):
    conn = sqlite3.connect('outputs/flagged.db')
    df = pd.read_sql(
        f"SELECT * FROM flagged WHERE risk_score >= {min_risk_score} "
        f"ORDER BY risk_score DESC LIMIT {limit}", conn
    )
    return df.to_dict(orient='records')
```

### Step 5 — Monitoring in production

Clustering drift is subtle because there's no ground truth to compare against. Watch these signals instead:

**Cluster size drift:** If Cluster 3 (your suspicious cluster) suddenly grows from 90 to 400 transactions week-over-week, either fraud is spiking or normal behavior has shifted to look like fraud. Both require investigation.

**Silhouette score on new batches:** Re-compute silhouette on each week's scored data against the training centroids. Degradation signals the transaction mix has changed and you should retrain.

**Noise point rate (DBSCAN):** If the fraction of -1 noise points shifts significantly, the outlier structure of the data has changed.

**Analyst feedback loop:** The most valuable monitoring signal is whether the transactions flagged by the model are actually being confirmed as suspicious by analysts. Build a simple form where analysts mark each flagged transaction as TP or FP. Over time, this builds a labeled dataset you can use to retrain the risk scoring function — and eventually shift from clustering to supervised classification.

---

## Project structure

```
bank_clustering_project/
│
├── data/
│   └── raw/
│       └── bank_transactions.csv     # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Distribution analysis, anomaly signal plots
│   ├── 02_feature_engineering.ipynb  # Ratios, log transforms, encoding, PCA
│   ├── 03_clustering.ipynb           # K-Means, DBSCAN, Hierarchical + evaluation
│   └── 04_anomaly_report.ipynb       # Risk scoring, final ranked output
│
├── src/
│   ├── preprocess.py                 # Feature engineering pipeline
│   └── score_batch.py                # Nightly scoring job
│
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   ├── isolation_forest.pkl
│   └── cluster_profiles.csv          # Human-readable cluster summary
│
├── outputs/
│   └── flagged.db                    # SQLite output of scored transactions
│
├── app.py                            # FastAPI endpoint to serve flagged transactions
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
scipy>=1.11
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

## How this project differs from the previous three

| Topic | Regression / Classification | This project |
|---|---|---|
| Target column | Always present | Does not exist |
| Model evaluation | RMSE, R², PR-AUC | Silhouette, Davies-Bouldin, cluster profiling |
| Outlier treatment | Clip before modeling | Preserve — outliers are the signal |
| Scaling importance | Helpful | Mandatory — distance-based algorithms break without it |
| Deployment pattern | Real-time inference API | Nightly batch job → analyst queue |
| Output | Single prediction | Ranked list of candidates for human review |
| Ground truth | Available for training | Arrives weeks later via analyst feedback |
| What "done" looks like | Model beats baseline metric | Cluster profiles make business sense + anomaly groups surface the injected patterns |

---

*Dataset is fully synthetic. No real transaction data was used.*
