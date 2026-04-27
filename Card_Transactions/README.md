# Credit Card Fraud Detection — End-to-End ML Project

**Target variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)  
**Model type:** Binary Classification  
**Domain:** Payments / financial crime  
**Dataset:** 9,000 synthetic card transactions, deliberately dirty  
**Class imbalance:** ~2.8% fraud — the defining challenge of this project

---

## Why this problem is different from regression

The previous projects (loan rates, insurance premiums, salary benchmarking) all had continuous targets and roughly balanced data. Fraud detection breaks both assumptions. The target is binary, and 97.2% of your data is the same class. That changes almost every decision you make:

- Accuracy is a useless metric. A model that predicts "not fraud" for every transaction gets 97.2% accuracy and catches zero fraud.
- The cost of errors is asymmetric. Missing a fraud (false negative) costs the bank money. Flagging a legitimate transaction (false positive) costs a customer relationship. The threshold you pick between them is a business decision, not a model decision.
- Standard train/test splits can accidentally put all fraud cases in training or test. You need stratified splits.
- Most algorithms will learn to ignore the minority class unless you intervene with resampling or class weights.

These are the real-world challenges this project is designed to make you confront directly.

---

## Dataset columns

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | string | Unique ID — drop before modeling |
| `timestamp` | datetime string | Parse before use |
| `hour_of_day` | int | 0–23, already extracted |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `is_weekend` | binary | Derived from day_of_week |
| `month` | int | 1–12 |
| `cardholder_age` | int | 18–80 |
| `account_age_days` | int | Days since card was issued |
| `credit_limit` | float | Card credit limit USD |
| `merchant_category` | categorical | **dirty** — mixed case, trailing spaces |
| `merchant_country` | categorical | **~4% missing** — foreign terminals don't always transmit |
| `is_foreign` | binary | 1 if transaction outside US |
| `transaction_amount` | float | USD, right-skewed |
| `avg_7day_spend` | float | **~3% missing** — 7-day rolling average for cardholder |
| `num_txn_24h` | int | Transactions in last 24 hours (velocity) |
| `num_txn_7d` | int | Transactions in last 7 days |
| `days_since_last_txn` | float | **~5% missing** — null for new cards |
| `distinct_merchants_7d` | int | Unique merchants visited in 7 days |
| `declined_last_30d` | int | Declined transactions in last 30 days |
| `channel` | categorical | **dirty** — Chip / Swipe / Online / Contactless / ATM |
| `device_type` | categorical | **dirty** — Mobile / Desktop / POS / ATM / Unknown |
| `is_fraud` | binary | **Target** (0 = legit, 1 = fraud) |

---

## Data issues to fix

### Duplicates
~25 duplicate rows from a batch ingestion error. Drop on all columns except `transaction_id` before any analysis.

### Missing values — three columns, three different reasons

- **`merchant_country`** (~4%) — Foreign POS terminals frequently don't transmit country codes. This is MNAR: missingness correlates with the transaction being foreign, which correlates with fraud. Do not impute with "US." Create a binary flag `merchant_country_missing` before imputing, then impute with mode ("US"). The flag preserves the signal; the imputation fills the gap.

- **`days_since_last_txn`** (~5%) — Missing because the card is new and has no prior transaction. This is MAR conditional on `account_age_days < 60`. Impute with a high value (e.g., 999) to signal "no history," not with median. A new card with no prior transactions behaves differently from a card whose last transaction was 3 days ago.

- **`avg_7day_spend`** (~3%) — Missing for new cards or accounts with very few prior transactions. Impute with the median grouped by `merchant_category`. Grocery cardholders and Electronics cardholders have very different spend baselines.

### Categorical inconsistency

- **`channel`**: "chip", "CHIP", "online", "ONLINE", "contactless ", "swipe " — strip and title-case, then verify only 5 valid values remain.
- **`device_type`**: "mobile", "MOBILE", "pos", "POS " — same treatment.
- **`merchant_category`**: "grocery", "GROCERY", "online_retail", "Online retail", "ATM " — strip, title-case, map "Online retail" → "Online_Retail".

### Class imbalance — the core challenge
The dataset is 97.2% legitimate and 2.8% fraud. This is not a data quality issue — it's the real-world distribution. Your entire modeling section needs to account for it. The cleaning step is just to be aware: when you do train/test split, use `stratify=y` to preserve the fraud rate in both sets.

---

## EDA checklist

EDA on imbalanced data is different. You're not just exploring distributions — you're building a case for which features separate fraud from legitimate transactions.

### Check the imbalance first
Print `value_counts()` and `value_counts(normalize=True)` on `is_fraud`. Write down the fraud rate. Every plot you make after this should be in the context of that number.

### Time patterns
- Bar chart: fraud count by `hour_of_day` — expect fraud to spike between midnight and 5am, when cardholders are asleep and less likely to notice.
- Bar chart: fraud rate (not count) by `hour_of_day` — the distinction matters. High fraud count at 2pm just means lots of transactions happen at 2pm. High fraud *rate* at 2am means 2am transactions are disproportionately suspicious.
- Bar chart: fraud rate by `day_of_week` — weekends may show different patterns.

### Amount distribution
- Overlay histogram: `transaction_amount` for fraud vs legitimate transactions on the same plot (use alpha=0.5). Fraud transactions tend to cluster at specific amounts — round numbers, amounts just under card limits.
- Box plot: `transaction_amount` by `is_fraud` — use log scale on y-axis given the skew.

### Categorical feature analysis — the most important EDA section
For each categorical, compute fraud rate per category (not raw counts):

```python
df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
df.groupby('channel')['is_fraud'].mean().sort_values(ascending=False)
df.groupby('device_type')['is_fraud'].mean().sort_values(ascending=False)
```

These bar charts will show you which categories are highest risk. ATM and Online channels should have higher fraud rates than Chip. Luxury and Electronics merchants should outrank Grocery. If they don't, revisit the cleaning step.

### Velocity features
- Scatter: `num_txn_24h` vs `transaction_amount`, colored by `is_fraud` — fraud often shows up as high velocity + high amount.
- Box plot: `declined_last_30d` by `is_fraud` — prior declines are a strong fraud signal.

### Correlation with target
Compute point-biserial correlation between each numeric feature and `is_fraud`. Sort descending. This gives you a ranked list of features before you even build a model. `is_foreign`, `num_txn_24h`, and `transaction_amount / credit_limit` should be near the top.

### Missing value analysis
Use `missingno.matrix()`. Check whether `merchant_country` missingness overlaps with `is_fraud = 1`. If it does, that confirms the MNAR assumption and the need for the missing flag.

---

## Feature engineering checklist

### The single most important feature
```python
df['amount_to_limit_ratio'] = df['transaction_amount'] / df['credit_limit']
```
A $500 charge on a $600 limit is far more suspicious than a $500 charge on a $50,000 limit. Raw amount alone misses this.

### Velocity and behavioral ratios

| Feature | Formula | Rationale |
|---|---|---|
| `amount_to_limit_ratio` | transaction_amount / credit_limit | Relative spend pressure |
| `amount_vs_7day_avg` | transaction_amount / avg_7day_spend.clip(1) | How unusual is this amount vs normal behavior |
| `txn_velocity_ratio` | num_txn_24h / (num_txn_7d / 7).clip(0.1) | Are transactions accelerating in last 24h vs weekly average |
| `merchant_concentration` | num_txn_7d / distinct_merchants_7d.clip(1) | Transactions per merchant — high = shopping spree or testing stolen card |

### Time-based features
```python
df['is_late_night'] = df['hour_of_day'].between(0, 4).astype(int)
df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
```
Fraud clusters at late night. Legitimate transactions cluster during business hours. These binary flags are more useful than raw hour in a linear model; tree models can learn the split themselves.

### Account risk features
```python
df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
df['log_account_age'] = np.log1p(df['account_age_days'])
df['log_transaction_amount'] = np.log1p(df['transaction_amount'])
```
New accounts are higher fraud risk. Log-transforming amount reduces the influence of extreme values on distance-based models and logistic regression.

### Interaction flags
```python
df['foreign_online'] = ((df['is_foreign'] == 1) & (df['channel'] == 'Online')).astype(int)
df['new_account_high_amount'] = ((df['is_new_account'] == 1) &
                                  (df['amount_to_limit_ratio'] > 0.5)).astype(int)
df['late_night_foreign'] = ((df['is_late_night'] == 1) & (df['is_foreign'] == 1)).astype(int)
```
These compound flags capture combinations that are greater than the sum of their parts. A foreign online transaction at 3am from a new account is not just the sum of three risk factors.

### Missing value flags — do this before imputing
```python
df['merchant_country_missing'] = df['merchant_country'].isnull().astype(int)
df['days_since_last_missing'] = df['days_since_last_txn'].isnull().astype(int)
```
Create these before any imputation. Missingness is a feature, not just a gap to fill.

### Encoding strategy
- `merchant_category`, `channel`, `device_type`: One-hot encode. No ordinal relationship.
- `merchant_country`: Frequency encode or group rare countries into "Other" before one-hot encoding. There are 12 countries — full OHE is fine here.
- Drop `transaction_id` and `timestamp` after extracting all time features.

---

## Handling class imbalance — the core section

This is where fraud detection diverges completely from the regression projects. You have four main tools, and you should try at least two and compare them.

### Option 1: Class weights (simplest, try this first)
Most sklearn classifiers accept `class_weight='balanced'`. This internally up-weights the minority class during training without changing the data.

```python
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')
```

Pros: No data modification, fast, no risk of overfitting to synthetic samples.  
Cons: May not be aggressive enough for extreme imbalance.

### Option 2: SMOTE — Synthetic Minority Over-sampling Technique
Generates synthetic fraud samples by interpolating between real fraud cases in feature space.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('model', RandomForestClassifier()),
])
```

**Critical rule:** SMOTE must go inside the cross-validation fold, not before it. If you SMOTE the full dataset before splitting, synthetic fraud samples based on test set data leak into training. Use `imblearn.pipeline.Pipeline`, not `sklearn.pipeline.Pipeline` — it handles this correctly.

Pros: Balances training data, often improves recall on minority class.  
Cons: Synthetic samples can introduce noise; overfitting risk increases.

### Option 3: Threshold tuning
By default, classifiers predict fraud when `predict_proba > 0.5`. In fraud detection, you almost always want a lower threshold (say 0.2–0.3) to catch more fraud at the cost of more false positives. The threshold is a business decision based on the cost ratio of false negatives vs false positives.

```python
proba = model.predict_proba(X_test)[:, 1]
# Plot precision-recall curve and pick the threshold that fits business requirements
threshold = 0.25
predictions = (proba >= threshold).astype(int)
```

### Option 4: Under-sampling (use cautiously)
Randomly remove legitimate transactions to balance classes. Simple but throws away real data. Only use this if your dataset is very large and you can afford it. At 9,000 rows, don't use random under-sampling.

---

## Modeling checklist

### Never use accuracy as your metric
With 2.8% fraud, a model predicting all-zeros scores 97.2% accuracy. Use:

- **Precision** — of the transactions flagged as fraud, how many actually were? (Minimizes false positives / customer annoyance)
- **Recall** — of all actual fraud, how many did we catch? (Minimizes missed fraud / financial loss)
- **F1 Score** — harmonic mean of precision and recall. Use when you need one number but want to balance both.
- **ROC-AUC** — model's ability to rank fraud above legitimate across all thresholds. Good for comparing models.
- **PR-AUC (Precision-Recall AUC)** — better than ROC-AUC for imbalanced data. A random classifier gets PR-AUC = fraud_rate = 0.028, not 0.5. Use this as your primary comparison metric.
- **Confusion matrix** — always print it. Shows the actual counts of TP, FP, TN, FN.

### Models to train (in order)

**1. Logistic Regression (baseline)**  
Use `class_weight='balanced'`. Interpret coefficients: positive coefficients increase fraud probability, negative decrease it. `amount_to_limit_ratio`, `is_foreign`, `is_late_night` should have positive coefficients. If they don't, check your feature engineering.

```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
```

**2. Random Forest Classifier**  
Try both `class_weight='balanced'` and `class_weight='balanced_subsample'`. Extract `feature_importances_` and plot the top 15 features. They should broadly agree with the logistic regression coefficients. If they disagree significantly, Random Forest has found a non-linear interaction that linear models missed.

```python
RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
```

**3. XGBoost**  
Use `scale_pos_weight = n_negative / n_positive` to handle imbalance natively.

```python
import xgboost as xgb
scale = (df['is_fraud'] == 0).sum() / (df['is_fraud'] == 1).sum()
xgb.XGBClassifier(scale_pos_weight=scale, eval_metric='aucpr', random_state=42)
```

**4. Logistic Regression + SMOTE (compare to #1)**  
Same architecture as the baseline but with SMOTE in the pipeline. Compare PR-AUC, precision, and recall between the two. Sometimes SMOTE helps recall significantly; sometimes class_weight does the same job with less risk.

**5. (optional) Isolation Forest for anomaly detection**  
A completely different framing: treat fraud as anomalies without using labels during training. Useful to understand how an unsupervised model compares to supervised ones.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.028, random_state=42)
iso.fit(X_train)
# Returns -1 for anomalies, 1 for normal — convert to 0/1
preds = (iso.predict(X_test) == -1).astype(int)
```

### Pipeline structure (with imblearn)

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), categorical_cols),
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42)),
])
```

Use `StratifiedKFold(n_splits=5)` for cross-validation to preserve the fraud rate in each fold.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='average_precision')
```

### Evaluation deep-dive

After training your best model, run all of these:

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              RocCurveDisplay, PrecisionRecallDisplay)

# 1. Classification report
print(classification_report(y_test, y_pred))

# 2. Confusion matrix — annotate with dollar cost if possible
print(confusion_matrix(y_test, y_pred))

# 3. ROC curve
RocCurveDisplay.from_estimator(model, X_test, y_test)

# 4. PR curve — more informative than ROC for imbalanced data
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)

# 5. Threshold analysis — what happens as you lower the threshold?
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    preds = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    p = precision_score(y_test, preds)
    r = recall_score(y_test, preds)
    print(f"Threshold {threshold:.1f} → Precision: {p:.3f}, Recall: {r:.3f}")
```

The threshold table is something you can show in an interview and explain the business tradeoff directly. "At threshold 0.3, we catch 84% of fraud but generate 3x more false positives than at 0.5. The right threshold depends on the cost ratio of a missed fraud vs a blocked legitimate transaction."

---

## Deployment

### Step 1 — Serialize the pipeline

```python
import joblib
joblib.dump(pipeline, 'models/fraud_detector_v1.pkl')
# Also save the threshold separately — it's a business parameter, not a model parameter
import json
json.dump({'threshold': 0.3}, open('models/config.json', 'w'))
```

### Step 2 — Build the API

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, pandas as pd, numpy as np

app = FastAPI()
pipeline = joblib.load("models/fraud_detector_v1.pkl")
config = json.load(open("models/config.json"))
THRESHOLD = config['threshold']

class TransactionInput(BaseModel):
    cardholder_age: int
    account_age_days: int
    credit_limit: float
    merchant_category: str
    merchant_country: str | None   # nullable — mirrors real-world missing data
    is_foreign: int
    transaction_amount: float
    avg_7day_spend: float | None
    num_txn_24h: int
    num_txn_7d: int
    days_since_last_txn: float | None
    distinct_merchants_7d: int
    declined_last_30d: int
    channel: str
    device_type: str
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    month: int

@app.post("/predict")
def predict(data: TransactionInput):
    df = pd.DataFrame([data.dict()])
    fraud_proba = pipeline.predict_proba(df)[0][1]
    is_fraud = int(fraud_proba >= THRESHOLD)
    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "is_fraud": is_fraud,
        "threshold_used": THRESHOLD,
        "action": "BLOCK" if is_fraud else "APPROVE"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

The API returns both the probability and the binary decision. In a real payments system, the calling service decides whether to hard-block or soft-decline based on the probability — the model just provides the score.

### Step 3 — Containerize

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 4 — Deploy

| Platform | Best for |
|---|---|
| **Railway / Render** | Portfolio — free tier, public HTTPS URL in minutes |
| **Google Cloud Run** | Production — serverless, scales to zero, low cost |
| **AWS Lambda + API Gateway** | High-volume, per-request billing, sub-100ms latency targets |

Fraud detection APIs are latency-sensitive — a real payments system needs a response in under 200ms before the card terminal times out. Cloud Run cold starts can be ~2s. If latency matters, keep the container warm or use Lambda with provisioned concurrency.

### Step 5 — What to monitor in production

Fraud detection drift is especially tricky because fraudsters adapt. The patterns that worked six months ago may not work today.

**Data drift:** Track the distribution of `is_foreign`, `channel`, `merchant_category` in incoming transactions. If "Online" transactions spike (as they do around holidays), your fraud rate will spike too — that's real, not model degradation.

**Score drift:** Plot the distribution of `fraud_probability` weekly. If the distribution shifts toward higher scores across the board, either fraud is increasing or your feature distributions have drifted.

**Ground truth latency:** In fraud, you often don't get the true label (confirmed fraud via chargeback) for 30–90 days after the transaction. This means you're monitoring a model blind for weeks. Build a delayed label pipeline: log all predictions, then join chargebacks back to transactions when they arrive to compute real-world precision and recall.

**Threshold review cadence:** The optimal threshold should be reviewed quarterly. As fraud tactics change, the cost ratio of false negatives vs false positives may shift, and the threshold should move with it.

---

## Project structure

```
fraud_detection_project/
│
├── data/
│   └── raw/
│       └── card_transactions.csv     # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Imbalance analysis, fraud rate by segment
│   ├── 02_feature_engineering.ipynb  # Velocity ratios, interaction flags, encoding
│   └── 03_modeling.ipynb             # Logistic, RF, XGBoost, SMOTE comparison
│
├── src/
│   └── pipeline.py                   # imblearn Pipeline + ColumnTransformer
│
├── models/
│   ├── fraud_detector_v1.pkl
│   └── config.json                   # threshold stored separately
│
├── app.py                            # FastAPI scoring endpoint
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

## Key differences from the regression projects

| Topic | Regression projects | This project |
|---|---|---|
| Target | Continuous (salary, premium, rate) | Binary (fraud / not fraud) |
| Primary metric | RMSE, R² | PR-AUC, F1, Recall |
| Imbalance handling | Not needed | SMOTE, class weights, threshold tuning |
| Train/test split | Random | Stratified |
| Pipeline library | `sklearn.pipeline` | `imblearn.pipeline` |
| Deployment output | Single float | Probability + binary decision + action |
| Production monitoring | Prediction drift | Score drift + delayed ground truth labels |

---

*Dataset is fully synthetic. No real transaction data was used.*
