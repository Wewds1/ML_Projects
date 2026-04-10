

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings("ignore")

RAW_PATH = "data/raw/loan_applications.csv"
OUT_PATH = "data/processed/loans_clean.csv"

# Load

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]   Raw shape: {df.shape}")
    return df


# clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "borrower_id"])
    print(f"[dedup]  Removed {before - len(df)} duplicate rows → {len(df)} rows")
    return df.reset_index(drop=True)


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize case/whitespace inconsistencies in string columns."""
    cat_cols = ["employment_type", "loan_purpose"]
    mapping = {
        "employment_type": {
            "salaried": "Salaried",
            "saried": "Salaried",
            "self employed": "Self-Employed",
            "self_employed": "Self-Employed",
        },
        "loan_purpose": {
            "auto": "Auto",
            "debt consolidation": "Debt Consolidation",
            "business": "Business",
            "vacation ": "Vacation",
            "vacation": "Vacation",
        },
    }
    for col in cat_cols:
        df[col] = df[col].str.strip()
        # Title-case first, then apply explicit overrides
        df[col] = df[col].str.title()
        for raw_val, clean_val in mapping.get(col, {}).items():
            df[col] = df[col].replace(raw_val.title(), clean_val)
    print(f"[clean]  Standardized categoricals: {cat_cols}")
    return df


def clip_dti_outliers(df: pd.DataFrame, upper: float = 3.0) -> pd.DataFrame:
    """DTI ratios above 3.0 are almost always data errors for retail loans."""
    n = (df["dti_ratio"] > upper).sum()
    df["dti_ratio"] = df["dti_ratio"].clip(upper=upper)
    print(f"[clip]   Clipped {n} DTI outlier rows to {upper}")
    return df


# Impute

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy:
    - annual_income  → median by employment_type (group-aware imputation)
    - credit_score   → KNN imputation using age, employment_years, num_late_payments
    """
    # Group median imputation for income
    df["annual_income"] = df.groupby("employment_type")["annual_income"].transform(
        lambda x: x.fillna(x.median())
    )

    # KNN imputation for credit score
    knn_features = ["age", "employment_years", "num_late_payments", "credit_score"]
    knn_imputer = KNNImputer(n_neighbors=5)
    df[knn_features] = knn_imputer.fit_transform(df[knn_features])
    df["credit_score"] = df["credit_score"].round().astype(int).clip(300, 850)

    print(f"[impute] Remaining nulls: {df.isnull().sum().sum()}")
    return df


# Feature Engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ratio features
    df["loan_to_income_ratio"] = (df["loan_amount"] / df["annual_income"]).round(4)
    df["debt_to_loan_ratio"] = (df["existing_debt"] / df["loan_amount"]).round(4)
    df["monthly_payment_est"] = (df["loan_amount"] / df["loan_term_months"]).round(2)
    df["payment_to_income"] = (df["monthly_payment_est"] / (df["annual_income"] / 12)).round(4)

    # Credit risk buckets (ordinal)
    bins = [0, 580, 670, 740, 800, 851]
    labels = ["Very Poor", "Fair", "Good", "Very Good", "Exceptional"]
    df["credit_tier"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False)

    # Log transforms for skewed financial fields
    df["log_annual_income"] = np.log1p(df["annual_income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    df["log_existing_debt"] = np.log1p(df["existing_debt"])

    # Macroeconomic interaction
    df["rate_spread"] = (df["interest_rate_offered"] - df["fed_funds_rate"]).round(3)

    # Employment stability flag
    df["is_stable_employment"] = df["employment_type"].isin(["Salaried", "Retired"]).astype(int)

    # Late payment flag (binary vs count)
    df["has_late_payments"] = (df["num_late_payments"] > 0).astype(int)

    # Term bucket
    df["term_bucket"] = pd.cut(
        df["loan_term_months"],
        bins=[0, 24, 60, 121],
        labels=["Short", "Medium", "Long"],
    )

    print(f"[feat]   Engineered {df.shape[1]} total columns")
    return df



def build_sklearn_pipeline(df: pd.DataFrame):
    """
    Returns a fitted ColumnTransformer that scales numerics and
    one-hot encodes categoricals. Useful for model training.
    """
    num_cols = [
        "age", "employment_years", "annual_income", "credit_score",
        "loan_amount", "loan_term_months", "existing_debt",
        "num_open_accounts", "num_late_payments", "dti_ratio",
        "fed_funds_rate", "loan_to_income_ratio", "debt_to_loan_ratio",
        "monthly_payment_est", "payment_to_income",
        "log_annual_income", "log_loan_amount", "log_existing_debt",
        "is_stable_employment", "has_late_payments",
    ]
    cat_cols = ["employment_type", "loan_purpose", "origination_quarter",
                "credit_tier", "term_bucket"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, [c for c in num_cols if c in df.columns]),
        ("cat", cat_pipeline, [c for c in cat_cols if c in df.columns]),
    ], remainder="drop")

    return preprocessor



def main():
    df = load_raw()
    df = remove_duplicates(df)
    df = standardize_categoricals(df)
    df = clip_dti_outliers(df)
    df = impute_features(df)
    df = engineer_features(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[done]   Saved processed data → {OUT_PATH}")
    print(f"         Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    main()
