from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Insurance Premium Predictor")

# Load the saved Ridge pipeline
pipeline = joblib.load("models/premium_predictor_v1.pkl")

# Define the expected input payload based on base features
class PolicyInput(BaseModel):
    age: int
    gender: str
    region: str
    bmi: float | None = None
    smoker: str
    alcohol_units_per_week: float
    has_diabetes: int
    has_hypertension: int
    has_heart_disease: int
    num_chronic_conditions: int
    coverage_type: str
    plan_tier: str
    deductible_amount: int
    num_dependents: int
    employment_status: str
    policy_start_year: int 
    annual_income: float | None = None
    prior_claims_count: int
    prior_claims_amount: float | None = None
    policy_tenure_years: float
    region_cost_index: float

def engineer_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Applies the same feature engineering used in 02_feature_engineering.ipynb"""
    df_feat['bmi_smoker_interaction'] = df_feat['bmi'] * (df_feat['smoker'] == 'Yes').astype(int)
    df_feat['age_chronic_interaction'] = df_feat['age'] * df_feat['num_chronic_conditions']
    
    df_feat['claims_per_year'] = df_feat['prior_claims_count'] / np.maximum(df_feat['policy_tenure_years'], 0.1)
    df_feat['avg_claim_amount'] = df_feat['prior_claims_amount'] / df_feat['prior_claims_count'].clip(lower=1)
    
    df_feat['is_high_risk'] = ((df_feat['smoker'] == 'Yes') & (df_feat['num_chronic_conditions'] >= 2)).astype(int)
    df_feat['is_new_enrollee'] = (df_feat['policy_tenure_years'] < 1).astype(int)
    df_feat['has_prior_claims'] = (df_feat['prior_claims_count'] > 0).astype(int)
    
    df_feat['age_group'] = pd.cut(df_feat['age'], bins=[17, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    df_feat['bmi_category'] = pd.cut(df_feat['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Needs a placeholder income bracket if income is null, pipeline imputer handles the rest
    df_feat['income_bracket'] = pd.cut(
        df_feat['annual_income'].fillna(50000), # fallback if null
        bins=[-1, 40000, 80000, float('inf')], 
        labels=['Low', 'Medium', 'High']
    )    
    # Log transforms for numeric columns
    df_feat['annual_income_log'] = np.log1p(df_feat['annual_income'].fillna(0))
    df_feat['prior_claims_amount_log'] = np.log1p(df_feat['prior_claims_amount'].fillna(0))
    
    return df_feat

@app.post("/predict")
def predict_premium(data: PolicyInput):
    # Convert input to DataFrame
    df_raw = pd.DataFrame([data.model_dump()])
    
    # Apply feature engineering
    df_processed = engineer_features(df_raw)
    
    # Pipeline handles the remaining imputation, scaling, and OHE
    prediction_log = pipeline.predict(df_processed)[0]
    
    # Convert back from log to dollars
    prediction_dollars = np.expm1(prediction_log)
    
    return {"annual_premium_estimate": round(prediction_dollars, 2)}