# Insurance Premium Prediction API

An end-to-end Machine Learning project that predicts the annual health insurance premium for policyholders based on demographic data, coverage choices, and actuarial health risks.

## Project Overview
Insurance pricing relies heavily on understanding compounding health risks. This project takes raw, messy synthetic policyholder data, cleans it, engineers actuarial feature interactions, and serves predictions via a production-ready REST API.

### Key Highlights:
* **Missing Data Imputation:** Handled specialized missingness (e.g., self-reported MNAR data for BMI, conditional MAR data for new enrollees).
* **Feature Engineering:** Captured non-linear risk compounding using interaction terms (e.g., `BMI × Smoker Status`, `Age × Chronic Conditions`). The target variable was log-transformed to handle the extreme right-skewness common in healthcare costs.
* **Modeling:** Trained and evaluated Linear Regression, Ridge, Lasso, and Random Forest baseline models.
* **Deployment:** The best performing model (`Ridge CV`) was serialized into an `sklearn.Pipeline` (preventing data leakage) and wrapped in a containerized `FastAPI` application.

## Model Performance
Evaluated on a 20% holdout test set:

| Metric | Baseline (Mean Prediction) | Ridge Regression (Final) | Improvement |
|--------|--------------------------|--------------------------|-------------|
| **RMSE** | $3,276.70                | **$1,069.81**            | **$2,206.89 reduction** |
| **MAE**  | $2,552.50                | **$822.99**              | **$1,729.51 reduction** |
| **R²**   | 0.0000                   | **0.8914**               | Explains ~89% of variance |

*Note: The model correctly identified smoking status and the `BMI × Smoker` interaction as the highest positive coefficients driving premium costs.*

## Project Structure
```text
insurance_claims/
├── data/
│   ├── raw/                      # Original synthetic dataset
│   └── processed/                # Cleaned and engineered features
├── models/
│   └── premium_predictor_v1.pkl  # Serialized scikit-learn Pipeline
├── notebooks/
│   ├── 01_eda.ipynb              # Data Cleaning & Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Interaction terms & Transformations
│   └── 03_modeling.ipynb         # Model training, evaluation, and selection
├── app.py                        # FastAPI application
├── Dockerfile                    # Containerization blueprint
└── requirements.txt              # Production dependencies
```

## How to Run Locally

You can easily run this API on your own machine using Docker.

**1. Clone the repository and navigate to the directory**
```bash
git clone <your-github-repo-url>
cd insurance_claims
```

**2. Build the Docker Image**
```bash
docker build -t premium-predictor .
```

**3. Run the Container**
```bash
docker run -p 8000:8000 premium-predictor
```

**4. Test the API**
Once running, navigate to `http://localhost:8000/docs` in your browser to access the interactive Swagger UI.

Alternatively, you can test it via `curl` in your terminal:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 45,
  "gender": "Male",
  "region": "Northeast",
  "bmi": 28.5,
  "smoker": "Yes",
  "alcohol_units_per_week": 5,
  "has_diabetes": 1,
  "has_hypertension": 0,
  "has_heart_disease": 0,
  "num_chronic_conditions": 1,
  "coverage_type": "Family",
  "plan_tier": "Gold",
  "deductible_amount": 1000,
  "num_dependents": 2,
  "employment_status": "Employed",
  "policy_start_year": 2023,
  "annual_income": 85000,
  "prior_claims_count": 2,
  "prior_claims_amount": 1500,
  "policy_tenure_years": 3.5,
  "region_cost_index": 1.15
}'
```
*Expected Output: `{"annual_premium_estimate": 21721.8}`*