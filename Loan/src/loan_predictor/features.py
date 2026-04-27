from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_REQUIRED_COLUMNS = [
    "borrower_id",
    "age",
    "employment_type",
    "employment_years",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "loan_purpose",
    "existing_debt",
    "num_open_accounts",
    "num_late_payments",
    "dti_ratio",
    "origination_date",
    "fed_funds_rate",
]

TARGET_COLUMN = "interest_rate_offered"

NUMERIC_MODEL_COLUMNS = [
    "age",
    "employment_years",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "existing_debt",
    "num_open_accounts",
    "num_late_payments",
    "dti_ratio",
    "fed_funds_rate",
    "loan_to_income_ratio",
    "debt_to_loan_ratio",
    "monthly_payment_est",
    "payment_to_income",
    "log_annual_income",
    "log_loan_amount",
    "log_existing_debt",
    "is_stable_employment",
    "has_late_payments",
]

CATEGORICAL_MODEL_COLUMNS = ["employment_type", "loan_purpose", "origination_quarter"]


def load_raw(path: str | None = None) -> pd.DataFrame:
    dataset_path = path or "data/raw/loan_applications.csv"
    return pd.read_csv(dataset_path)


def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "origination_date" in output.columns:
        parsed = pd.to_datetime(output["origination_date"], errors="coerce")
        if "origination_year" not in output.columns or output["origination_year"].isna().any():
            output["origination_year"] = output.get("origination_year")
            output["origination_year"] = output["origination_year"].fillna(parsed.dt.year)
        if "origination_quarter" not in output.columns or output["origination_quarter"].isna().any():
            output["origination_quarter"] = output.get("origination_quarter")
            output["origination_quarter"] = output["origination_quarter"].fillna("Q" + parsed.dt.quarter.astype("Int64").astype(str))
    return output


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = [column for column in df.columns if column != "borrower_id"]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    mappings = {
        "employment_type": {
            "salaried": "Salaried",
            "saried": "Salaried",
            "self employed": "Self-Employed",
            "self_employed": "Self-Employed",
            "self-employed": "Self-Employed",
        },
        "loan_purpose": {
            "auto": "Auto",
            "debt consolidation": "Debt Consolidation",
            "business": "Business",
            "vacation": "Vacation",
        },
    }

    for column, replacements in mappings.items():
        if column not in output.columns:
            continue
        normalized = output[column].astype("string").str.strip().str.lower()
        output[column] = normalized.replace(replacements).str.title()
        output[column] = output[column].replace({"Self-Employed": "Self-Employed"})

    if "origination_quarter" in output.columns:
        output["origination_quarter"] = output["origination_quarter"].astype("string").str.strip().str.upper()

    return output


def clip_dti_outliers(df: pd.DataFrame, upper: float = 3.0) -> pd.DataFrame:
    output = df.copy()
    if "dti_ratio" in output.columns:
        output["dti_ratio"] = output["dti_ratio"].clip(upper=upper)
    return output


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    if {"annual_income", "employment_type"}.issubset(output.columns):
        output["annual_income"] = output.groupby("employment_type")["annual_income"].transform(
            lambda values: values.fillna(values.median())
        )
        output["annual_income"] = output["annual_income"].fillna(output["annual_income"].median())

    knn_features = ["age", "employment_years", "num_late_payments", "credit_score"]
    if set(knn_features).issubset(output.columns):
        knn_imputer = KNNImputer(n_neighbors=5)
        output[knn_features] = knn_imputer.fit_transform(output[knn_features])
        output["credit_score"] = output["credit_score"].round().clip(300, 850)

    return output


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["loan_to_income_ratio"] = (output["loan_amount"] / output["annual_income"]).replace([np.inf, -np.inf], np.nan)
    output["debt_to_loan_ratio"] = (output["existing_debt"] / output["loan_amount"]).replace([np.inf, -np.inf], np.nan)
    output["monthly_payment_est"] = (output["loan_amount"] / output["loan_term_months"]).replace([np.inf, -np.inf], np.nan)
    output["payment_to_income"] = (
        output["monthly_payment_est"] / (output["annual_income"] / 12)
    ).replace([np.inf, -np.inf], np.nan)

    bins = [0, 580, 670, 740, 800, 851]
    labels = ["Very Poor", "Fair", "Good", "Very Good", "Exceptional"]
    output["credit_tier"] = pd.cut(output["credit_score"], bins=bins, labels=labels, right=False)

    output["log_annual_income"] = np.log1p(output["annual_income"].clip(lower=0))
    output["log_loan_amount"] = np.log1p(output["loan_amount"].clip(lower=0))
    output["log_existing_debt"] = np.log1p(output["existing_debt"].clip(lower=0))

    if TARGET_COLUMN in output.columns:
        output["rate_spread"] = output[TARGET_COLUMN] - output["fed_funds_rate"]

    output["is_stable_employment"] = output["employment_type"].isin(["Salaried", "Retired"]).astype(int)
    output["has_late_payments"] = (output["num_late_payments"] > 0).astype(int)
    output["term_bucket"] = pd.cut(
        output["loan_term_months"],
        bins=[0, 24, 60, 121],
        labels=["Short", "Medium", "Long"],
    )

    return output


def prepare_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    required = set(RAW_REQUIRED_COLUMNS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output = ensure_time_columns(df)
    if training:
        output = remove_duplicates(output)
    output = standardize_categoricals(output)
    output = clip_dti_outliers(output)
    output = impute_features(output)
    output = engineer_features(output)
    return output


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in NUMERIC_MODEL_COLUMNS if column in df.columns]
    categorical_columns = [column for column in CATEGORICAL_MODEL_COLUMNS if column in df.columns]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def model_input_columns() -> Iterable[str]:
    return [*RAW_REQUIRED_COLUMNS, "origination_year", "origination_quarter"]
