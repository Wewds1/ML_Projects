# Customer Lifetime Value Prediction with NLP Features — End-to-End ML Project

**Target variable:** `clv_12m` (continuous USD — 12-month forward customer value)  
**Model type:** Regression + NLP feature extraction (TF-IDF, sentiment, BERT embeddings)  
**Domain:** Fintech / SaaS — customer analytics  
**Dataset:** 7,000 synthetic customer records with free-text feedback, deliberately dirty  
**Core challenge:** Combining structured behavioral data with unstructured text to predict future customer value — and proving the text features actually improve the model

---

## Why this project is different from everything before it

Every previous project used structured tabular data. This one has a text column — `customer_feedback` — and the entire NLP section is about turning that raw text into numeric features that a regression model can use.

That's a two-stage problem. First you build a text processing pipeline (cleaning → vectorization → feature extraction). Then you combine those features with the structured columns and run the regression. The challenge is proving the text features add predictive power beyond what the structured data already captures. If NPS score and support ticket count already encode customer sentiment, does the raw text teach the model anything new? Answering that question honestly is the analytical core of the project.

The target — CLV — is also not a raw observed value. It's a forward projection: how much will this customer be worth over the next 12 months? That means even the target variable involves assumptions and modeling choices, which is true of every real CLV implementation.

---

## Dataset columns

| Column | Type | Notes |
|---|---|---|
| `customer_id` | string | Unique ID — drop before modeling |
| `customer_age` | int | 18–72 |
| `gender` | categorical | Male / Female / Non-Binary |
| `country` | categorical | 10 countries |
| `acquisition_channel` | categorical | **dirty** — mixed case |
| `product_tier` | categorical | **dirty** — Free / Basic / Pro / Enterprise |
| `tenure_months` | int | Months since signup |
| `monthly_spend` | float | **~5% missing** — billing lag for new accounts |
| `login_freq_monthly` | int | Average logins per month |
| `feature_adoption` | float | **~4% missing** — fraction of product features used (0–1) |
| `support_tickets_6m` | int | Support tickets opened in last 6 months |
| `nps_score` | float | **~8% missing** — Net Promoter Score (0–10), skipped surveys |
| `payment_failures_6m` | int | Failed payment attempts in last 6 months |
| `referrals_made` | int | Customers referred by this account |
| `days_since_login` | int | Days since last login |
| `customer_feedback` | text | **~8% missing** (matches nps_score nulls) — free-text survey response |
| `clv_12m` | float | **Target** — projected 12-month customer value in USD |

---

## What the text column looks like

`customer_feedback` contains free-text survey responses. They range from clearly positive to clearly negative, with noise injected to mimic real-world messiness: some are lowercased ("love this product..."), some have excessive punctuation ("Great product... but the app crashes..."), some are terse ("It's fine."), some are detailed paragraphs. About 8% are null — customers who skipped the survey entirely, matching the `nps_score` nulls.

The feedback correlates with `nps_score` and `support_tickets_6m` by construction — high NPS customers wrote positive feedback, high-ticket customers wrote negative feedback. But the text captures nuance the scores don't: a customer who scores NPS 7 (neutral) and wrote "the mobile app needs work but the desktop is excellent" is a different retention risk than one who wrote "thinking about cancelling." Both score 7. Only the text distinguishes them.

---

## Data issues to fix

### Duplicates
~28 duplicate rows. Drop on all columns except `customer_id`.

### Missing values — four columns, four strategies

- **`nps_score`** (~8%) — Customers who skipped the survey. Not random — low-engagement customers skip more often. `days_since_login` and `login_freq_monthly` correlate with this missingness. Create a flag `nps_missing = 1` before imputing. Impute with the median grouped by `product_tier`. A Free-tier skip is a different signal than an Enterprise-tier skip.

- **`customer_feedback`** (~8%, same rows as nps_score) — These nulls are the same customers who skipped the NPS survey. Do not impute with a string — the absence of text is meaningful. Keep nulls as-is through the text pipeline; handle them explicitly during feature extraction by assigning a neutral/zero vector. Create a binary flag `feedback_missing = 1`.

- **`monthly_spend`** (~5%) — Missing due to billing system lag for new accounts (`tenure_months <= 1`). Impute with 0 for Free-tier customers (they have no spend by definition) and with the median grouped by `product_tier` for paid tiers. A missing Pro-tier spend is very different from a missing Free-tier spend.

- **`feature_adoption`** (~4%) — No strong pattern to missingness. Impute with median grouped by `product_tier`. Enterprise customers use more features on average than Basic customers.

### Categorical inconsistency

- **`product_tier`**: "free", "FREE", "basic", "BASIC", "pro ", "PRO", "enterprise", "ENTERPRISE" — strip whitespace, `.str.strip().str.title()` handles most cases. Then verify only 4 valid values remain: Free, Basic, Pro, Enterprise.
- **`acquisition_channel`**: "organic search", "Organic search", "paid ads", "REFERRAL", "social media", "email campaign" — same treatment: strip, title-case. Final canonical values: Organic Search, Paid Ads, Referral, Social Media, Email Campaign, Direct.

### Target distribution
`clv_12m` is heavily right-skewed. Free-tier customers cluster at near-zero; Enterprise customers reach $50,000+. Log-transform the target before modeling:

```python
df['log_clv_12m'] = np.log1p(df['clv_12m'])
```

Remember to exponentiate (`np.expm1`) predictions back to USD before reporting RMSE. RMSE on log-CLV is not interpretable to a business audience.

---

## EDA checklist

### Understand the CLV distribution first

- Histogram of `clv_12m` (raw) — expect a spike near zero (Free-tier), a long right tail (Enterprise). This bimodal shape tells you the `product_tier` is likely the single most powerful feature.
- Histogram of `log_clv_12m` — should look approximately normal after transformation. If it doesn't, investigate the spike at zero separately (Free-tier customers may need to be modeled differently or excluded).
- Box plot: `clv_12m` by `product_tier` — the single most important plot before modeling. Enterprise >> Pro >> Basic >> Free, with large within-tier variance.

### Structured feature relationships

- Scatter: `monthly_spend` vs `clv_12m` — should be near-linear, with variance explained by `retention_prob` proxies.
- Scatter: `tenure_months` vs `clv_12m` — longer tenure = higher CLV, but with diminishing returns.
- Bar chart: mean `clv_12m` by `acquisition_channel` — Referral customers tend to have higher CLV than Paid Ads customers. Check whether this holds in your data.
- Scatter: `days_since_login` vs `clv_12m` — customers who haven't logged in recently churn sooner.
- Bar chart: mean `clv_12m` by `nps_score` bucket (0–6 detractor, 7–8 passive, 9–10 promoter) — promoters should have meaningfully higher CLV than detractors. If they don't, the NPS signal is weak for this dataset.
- Box plot: `clv_12m` by `payment_failures_6m` — even 1 payment failure should correlate with lower CLV.

### Text column exploration — before any NLP

- Print 10 random samples of `customer_feedback` — read them. Understand the vocabulary, length, noise patterns.
- Distribution of text length (word count) — plot a histogram. Very short texts ("It's fine.") carry less signal than longer ones.
- Word frequency analysis: compute top 30 words in positive feedback (NPS >= 9) vs negative feedback (NPS <= 4). The vocabulary should be visibly different. This is your sanity check that text carries signal before you invest in a full NLP pipeline.

```python
from collections import Counter
import re

def tokenize(text):
    return re.findall(r'\b[a-z]{3,}\b', str(text).lower())

positive_words = Counter()
negative_words = Counter()
for _, row in df[df['nps_score'] >= 9].iterrows():
    positive_words.update(tokenize(str(row['customer_feedback'])))
for _, row in df[df['nps_score'] <= 4].iterrows():
    negative_words.update(tokenize(str(row['customer_feedback'])))

print("Top positive words:", positive_words.most_common(15))
print("Top negative words:", negative_words.most_common(15))
```

---

## NLP pipeline — the new section

This is the part of the project that doesn't appear in any of the previous four datasets. Work through it in Notebook 02 before joining with structured features.

### Step 1 — Text cleaning

```python
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\.{2,}', '.', text)          # normalize ellipses
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['feedback_clean'] = df['customer_feedback'].apply(clean_text)
```

Note: keep punctuation at this stage — exclamation marks and question marks carry sentiment signal. Remove them only if your vectorizer doesn't handle them.

### Step 2 — Approach A: TF-IDF vectorization (baseline NLP)

TF-IDF (Term Frequency-Inverse Document Frequency) converts text to a sparse numeric matrix. Each column is a word; each value is how distinctive that word is to that document vs the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=500,        # keep top 500 most informative terms
    ngram_range=(1, 2),      # unigrams and bigrams ("not good" is different from "good")
    min_df=5,                # ignore terms appearing in fewer than 5 documents
    max_df=0.85,             # ignore terms appearing in more than 85% of documents
    stop_words='english',
    sublinear_tf=True,       # apply log normalization to term frequency
)

# Fill nulls with empty string before fitting
tfidf_matrix = tfidf.fit_transform(df['feedback_clean'].fillna(''))
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
)
```

500 TF-IDF features is a lot to add directly to the model. Use PCA to reduce to 20–30 components first:

```python
from sklearn.decomposition import TruncatedSVD  # use this for sparse matrices, not PCA

svd = TruncatedSVD(n_components=30, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)
tfidf_features = pd.DataFrame(tfidf_reduced, columns=[f'tfidf_svd_{i}' for i in range(30)])
print(f"Variance explained: {svd.explained_variance_ratio_.sum():.3f}")
```

This is called Latent Semantic Analysis (LSA). It's TF-IDF + dimensionality reduction, and it's the standard approach for feeding text into traditional ML models.

### Step 3 — Approach B: Sentiment scoring (interpretable NLP)

Extract scalar sentiment features that are directly interpretable. These are easier to explain to a business audience than TF-IDF components.

```python
from textblob import TextBlob

def get_sentiment(text):
    if not text or text == "":
        return pd.Series({'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0})
    blob = TextBlob(str(text))
    return pd.Series({
        'sentiment_polarity':     blob.sentiment.polarity,      # -1 to 1
        'sentiment_subjectivity': blob.sentiment.subjectivity,  # 0 to 1
    })

sentiment_features = df['feedback_clean'].apply(get_sentiment)
df = pd.concat([df, sentiment_features], axis=1)
```

Then engineer explicit signal features:

```python
# Negation: "not good", "never works", "can't recommend" — critical for sentiment
df['has_negation'] = df['feedback_clean'].str.contains(
    r'\b(not|never|no|can\'t|cannot|won\'t|wouldn\'t|don\'t)\b', regex=True
).astype(int)

# Churn intent language
df['churn_language'] = df['feedback_clean'].str.contains(
    r'\b(cancel|cancell|switching|alternative|competitor|leaving|downgrad|refund)\b',
    regex=True
).astype(int)

# Praise language
df['praise_language'] = df['feedback_clean'].str.contains(
    r'\b(love|excellent|amazing|best|recommend|worth|outstanding|perfect)\b',
    regex=True
).astype(int)

# Text length as a feature (longer = more engaged)
df['feedback_word_count'] = df['feedback_clean'].apply(
    lambda x: len(str(x).split()) if x else 0
)
```

### Step 4 — Approach C: BERT sentence embeddings (advanced)

BERT produces dense 768-dimensional embeddings that capture semantic meaning far better than TF-IDF. Two customers writing "great product" and "absolutely love using this" will get similar embeddings from BERT but very different TF-IDF vectors. Use the lighter `sentence-transformers` library — no GPU needed for inference on 7,000 rows.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB model, fast CPU inference

texts = df['feedback_clean'].fillna('').tolist()
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
# embeddings shape: (7000, 384)  — MiniLM produces 384-dim vectors

# Reduce to 20 components before joining with structured features
from sklearn.decomposition import PCA
pca = PCA(n_components=20, random_state=42)
bert_reduced = pca.fit_transform(embeddings)
bert_features = pd.DataFrame(bert_reduced, columns=[f'bert_{i}' for i in range(20)])
print(f"BERT PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
```

**Three approaches, three experiments.** Run your regression model with each text representation separately and compare R² and RMSE. Then run a fourth experiment with all structured features but no text. This ablation table is the analytical core of the project — it shows which NLP approach adds the most value and how much.

| Experiment | Features | R² | RMSE |
|---|---|---|---|
| Baseline | Structured only | ? | ? |
| + TF-IDF LSA | Structured + 30 SVD components | ? | ? |
| + Sentiment | Structured + polarity, subjectivity, flags | ? | ? |
| + BERT | Structured + 20 BERT-PCA components | ? | ? |
| + All NLP | Structured + all text features | ? | ? |

Fill this table in your notebook. The gap between "Structured only" and the best NLP approach is the value of the text data.

---

## Feature engineering checklist (structured features)

### Behavioral ratio features

| Feature | Formula | Rationale |
|---|---|---|
| `spend_per_login` | monthly_spend / login_freq_monthly.clip(1) | Revenue efficiency per engagement event |
| `ticket_rate` | support_tickets_6m / tenure_months.clip(1) | Annualized complaint rate — higher = more friction |
| `referral_value_est` | referrals_made × monthly_spend | Estimated referral revenue contribution |
| `engagement_score` | (login_freq_monthly / 30) × feature_adoption | Combined engagement index (0–1) |
| `churn_risk_proxy` | payment_failures_6m × 0.4 + (days_since_login > 30) × 0.3 + (nps_score < 5) × 0.3 | Composite risk score before modeling |

### Transformations

```python
df['log_monthly_spend']   = np.log1p(df['monthly_spend'])
df['log_tenure_months']   = np.log1p(df['tenure_months'])
df['log_days_since_login']= np.log1p(df['days_since_login'])
df['sqrt_login_freq']     = np.sqrt(df['login_freq_monthly'])
```

### NPS bucketing

```python
def nps_bucket(score):
    if pd.isna(score): return 'Unknown'
    if score <= 6:  return 'Detractor'
    if score <= 8:  return 'Passive'
    return 'Promoter'

df['nps_category'] = df['nps_score'].apply(nps_bucket)
```

The category is useful for tree models. The raw score is better for linear models. Include both.

### Encoding strategy

- `product_tier`: Ordinal encode — Free=0, Basic=1, Pro=2, Enterprise=3. There is a natural value order.
- `acquisition_channel`, `country`, `gender`, `nps_category`: One-hot encode.
- Drop `customer_id`, `customer_feedback` (after extracting NLP features), `feedback_clean`.

---

## Modeling checklist

### The right way to combine text and structured features

After extracting NLP features, combine them with structured features into a single DataFrame before the train/test split:

```python
# Join all feature sets
X = pd.concat([
    structured_features,   # cleaned + engineered structured columns
    sentiment_features,    # polarity, subjectivity, flag columns
    tfidf_features,        # 30 SVD components from TF-IDF
    # bert_features,       # 20 PCA components from BERT (optional)
], axis=1)

y = np.log1p(df['clv_12m'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Models to train

**1. Ridge Regression (baseline)**
With 30+ TF-IDF components added, you have more features than you might expect — regularization is necessary. Ridge is the right starting model. Interpret coefficients on the structured features after fitting. The sentiment polarity coefficient should be positive; the churn_language flag should be negative.

**2. Lasso Regression**
With many TF-IDF SVD components added, Lasso will zero out the least useful ones. Check how many NLP components survive. If Lasso drops all TF-IDF components but keeps sentiment polarity and `churn_language`, that tells you rule-based text features outperform bag-of-words here.

**3. Random Forest Regressor**
Extract `feature_importances_` and plot top 20. The interesting question is: do any NLP features appear in the top 10? If `sentiment_polarity` or `churn_language` rank higher than `payment_failures_6m`, the text is adding meaningful signal beyond what the structured fields already capture.

**4. Gradient Boosting / XGBoost**
Best overall performance. Use the same feature importance analysis. Run 5-fold cross-validation and report mean ± std of RMSE (in log-CLV) and back-transformed MAE (in USD).

**5. The ablation study — most important modeling step**
Train each model four times: once with structured features only, once adding TF-IDF, once adding sentiment flags, once adding BERT. Compare metrics. This is the analysis you present. It answers: "does the NLP investment pay off, and which approach is worth the added complexity?"

### Evaluation metrics

- **RMSE on log-CLV** — for comparing models during training (scale-normalized).
- **MAE in USD** — `np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_test)))` — interpretable for business audiences. "The model is off by $X on average."
- **R²** — overall variance explained.
- **Segment-level RMSE** — compute RMSE separately for Free, Basic, Pro, Enterprise tiers. Models almost always underperform on Enterprise because it's the smallest and most variable segment. Reporting this honestly is more useful than a single aggregate metric.
- **Residual plot** — predicted vs residuals on log-CLV scale. Look for any systematic pattern by product tier (plot residuals colored by tier). If Enterprise residuals are all negative (underprediction), the model needs tier-specific calibration.

---

## Deployment

### Two-stage pipeline to serialize

The text pipeline and the structured pipeline are separate transformers that need to be serialized together.

```python
import joblib

# Save everything the inference function needs
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(svd, 'models/tfidf_svd.pkl')
joblib.dump(scaler, 'models/feature_scaler.pkl')
joblib.dump(model, 'models/clv_model.pkl')

# Optionally save the BERT model name (re-load at inference time)
import json
json.dump({'sbert_model': 'all-MiniLM-L6-v2'}, open('models/config.json', 'w'))
```

### FastAPI inference endpoint

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
from textblob import TextBlob
import re

app    = FastAPI()
tfidf  = joblib.load('models/tfidf_vectorizer.pkl')
svd    = joblib.load('models/tfidf_svd.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
model  = joblib.load('models/clv_model.pkl')

class CustomerInput(BaseModel):
    customer_age: int
    product_tier: str
    tenure_months: int
    monthly_spend: float | None
    login_freq_monthly: int
    feature_adoption: float | None
    support_tickets_6m: int
    nps_score: float | None
    payment_failures_6m: int
    referrals_made: int
    days_since_login: int
    customer_feedback: str | None

def extract_text_features(text: str | None) -> dict:
    if not text:
        return {'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0,
                'churn_language': 0, 'praise_language': 0,
                'has_negation': 0, 'feedback_word_count': 0,
                'feedback_missing': 1}
    clean = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text.lower()).strip()
    blob  = TextBlob(clean)
    tfidf_vec   = tfidf.transform([clean])
    tfidf_svd   = svd.transform(tfidf_vec)[0]
    result = {
        'sentiment_polarity':     blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'churn_language': int(bool(re.search(r'\b(cancel|switching|leaving|competitor)\b', clean))),
        'praise_language': int(bool(re.search(r'\b(love|excellent|best|recommend)\b', clean))),
        'has_negation':    int(bool(re.search(r'\b(not|never|can\'t|won\'t)\b', clean))),
        'feedback_word_count': len(clean.split()),
        'feedback_missing': 0,
    }
    for i, val in enumerate(tfidf_svd):
        result[f'tfidf_svd_{i}'] = val
    return result

@app.post('/predict')
def predict(data: CustomerInput):
    structured = data.dict()
    text_features = extract_text_features(structured.pop('customer_feedback'))
    features = {**structured, **text_features}
    df_input = pd.DataFrame([features])
    X_scaled = scaler.transform(df_input)
    log_pred = model.predict(X_scaled)[0]
    clv_pred = float(np.expm1(log_pred))
    return {
        'clv_12m_estimate_usd': round(clv_pred, 2),
        'sentiment_polarity':   round(text_features['sentiment_polarity'], 3),
        'churn_risk_flag':      text_features['churn_language'],
    }
```

The response returns the CLV estimate plus two interpretable NLP-derived signals — sentiment polarity and churn language flag. A product team can act on those directly even without understanding the full model.

### Deploy to cloud

Same pattern as the previous projects — Docker → Railway/Render for portfolio, Cloud Run or ECS for production. One additional consideration: the BERT model (`all-MiniLM-L6-v2`) is ~80MB. If you're using BERT at inference time, include it in the Docker image or load it from a model registry. Cold start times will be slower than the previous projects.

For a portfolio project, stick with TF-IDF + sentiment at inference time. BERT adds complexity for modest gain on this dataset.

### What to monitor after deployment

**Text drift** — are the words customers use shifting over time? If "crash" and "slow" start appearing more frequently, product quality may be declining before it shows up in churn metrics. Track top TF-IDF term frequencies weekly.

**Sentiment drift** — plot mean `sentiment_polarity` of incoming feedback weekly. A declining trend predicts future CLV decline before the structured metrics catch it. This is the core value proposition of including text in the model.

**CLV prediction calibration** — as actual 12-month revenue data arrives for older cohorts, compare predictions to actuals. CLV models degrade as product pricing and market conditions change. Retrain at least annually.

**Segment performance** — track prediction error separately by product tier. Enterprise CLV is the most valuable to predict correctly and the hardest. Monitor it independently.

---

## Project structure

```
clv_nlp_project/
│
├── data/
│   └── raw/
│       └── customer_clv.csv          # Raw dirty dataset — never modify
│
├── notebooks/
│   ├── 01_eda.ipynb                  # CLV distribution, structured correlations, text exploration
│   ├── 02_nlp_pipeline.ipynb         # Text cleaning, TF-IDF, sentiment, BERT
│   ├── 03_feature_engineering.ipynb  # Structured features, joins, encoding
│   └── 04_modeling.ipynb             # Ablation study, model comparison, evaluation
│
├── src/
│   ├── text_features.py              # Text cleaning + feature extraction functions
│   └── pipeline.py                   # Full preprocessing pipeline
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_svd.pkl
│   ├── feature_scaler.pkl
│   ├── clv_model.pkl
│   └── config.json
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
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
missingno>=0.5
jupyterlab>=4.0
textblob>=0.17
sentence-transformers>=2.2
xgboost>=2.0
fastapi>=0.110
uvicorn>=0.29
joblib>=1.3
pydantic>=2.0
evidently>=0.4
```

---

## How this project differs from the previous four

| Topic | Previous projects | This project |
|---|---|---|
| Data types | Tabular only | Tabular + free text |
| Feature engineering | Numeric ratios, binning, encoding | All of the above + TF-IDF, sentiment extraction, BERT embeddings |
| Dimensionality | 20–40 features | 60–80 features after NLP (needs PCA/SVD to manage) |
| Key preprocessing step | Imputation, scaling | Text cleaning pipeline before any modeling |
| Core analytical question | Which features predict the target? | Does text add value beyond structured features? (ablation study) |
| Evaluation structure | Single model comparison table | Ablation table: structured only vs +TF-IDF vs +sentiment vs +BERT |
| Deployment complexity | Single sklearn pipeline | Two-stage: text pipeline + structured pipeline both need to be serialized |
| Monitoring signal | Prediction drift, feature drift | + Sentiment drift, vocabulary shift |

---

*Dataset is fully synthetic. No real customer data was used.*
