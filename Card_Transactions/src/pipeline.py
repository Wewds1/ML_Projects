from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "data/raw/card_transactions.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_transactions.csv"
TARGET_COLUMN = "is_fraud"

VALID_CHANNELS = ["ATM", "Chip", "Contactless", "Online", "Swipe"]
VALID_DEVICE_TYPES = ["ATM", "Desktop", "Mobile", "POS", "Unknown"]
VALID_MERCHANT_CATEGORIES = [
    "ATM",
    "Electronics",
    "Entertainment",
    "Gas",
    "Grocery",
    "Healthcare",
    "Luxury",
    "Online_Retail",
    "Restaurant",
    "Travel",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_channel(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {
        "atm": "ATM",
        "chip": "Chip",
        "contactless": "Contactless",
        "online": "Online",
        "swipe": "Swipe",
    }
    return mapping.get(text, str(value).strip().title())


def normalize_device_type(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {
        "atm": "ATM",
        "desktop": "Desktop",
        "mobile": "Mobile",
        "pos": "POS",
        "unknown": "Unknown",
    }
    return mapping.get(text, str(value).strip().title())


def normalize_merchant_category(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    mapping = {
        "atm": "ATM",
        "electronics": "Electronics",
        "entertainment": "Entertainment",
        "gas": "Gas",
        "grocery": "Grocery",
        "healthcare": "Healthcare",
        "luxury": "Luxury",
        "online_retail": "Online_Retail",
        "restaurant": "Restaurant",
        "travel": "Travel",
    }
    return mapping.get(text, str(value).strip().title().replace(" ", "_"))


@dataclass
class FeatureGroups:
    numeric: list[str]
    categorical: list[str]


FEATURE_GROUPS = FeatureGroups(
    numeric=[
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "cardholder_age",
        "account_age_days",
        "credit_limit",
        "is_foreign",
        "transaction_amount",
        "avg_7day_spend",
        "num_txn_24h",
        "num_txn_7d",
        "days_since_last_txn",
        "distinct_merchants_7d",
        "declined_last_30d",
        "merchant_country_missing",
        "days_since_last_missing",
        "amount_to_limit_ratio",
        "amount_vs_7day_avg",
        "txn_velocity_ratio",
        "merchant_concentration",
        "is_late_night",
        "is_business_hours",
        "is_new_account",
        "log_account_age",
        "log_transaction_amount",
        "foreign_online",
        "new_account_high_amount",
        "late_night_foreign",
    ],
    categorical=["merchant_category", "merchant_country", "channel", "device_type"],
)


class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """Clean and engineer features from raw transaction records."""

    def __init__(self) -> None:
        self.country_mode_: str | None = None
        self.avg_7day_spend_medians_: dict[str, float] | None = None
        self.global_avg_7day_spend_median_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureBuilder":
        frame = self._prepare_frame(X.copy())
        self.country_mode_ = frame["merchant_country"].mode(dropna=True).iloc[0]
        self.avg_7day_spend_medians_ = (
            frame.groupby("merchant_category")["avg_7day_spend"].median().fillna(frame["avg_7day_spend"].median()).to_dict()
        )
        self.global_avg_7day_spend_median_ = float(frame["avg_7day_spend"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_frame(X.copy())
        frame["merchant_country_missing"] = frame["merchant_country"].isna().astype(int)
        frame["days_since_last_missing"] = frame["days_since_last_txn"].isna().astype(int)

        frame["merchant_country"] = frame["merchant_country"].fillna(self.country_mode_ or "US")
        grouped_fill = frame["merchant_category"].map(self.avg_7day_spend_medians_ or {})
        frame["avg_7day_spend"] = frame["avg_7day_spend"].fillna(grouped_fill)
        frame["avg_7day_spend"] = frame["avg_7day_spend"].fillna(self.global_avg_7day_spend_median_ or 0.0)
        frame["days_since_last_txn"] = frame["days_since_last_txn"].fillna(999.0)

        frame["amount_to_limit_ratio"] = frame["transaction_amount"] / frame["credit_limit"].clip(lower=1)
        frame["amount_vs_7day_avg"] = frame["transaction_amount"] / frame["avg_7day_spend"].clip(lower=1)
        frame["txn_velocity_ratio"] = frame["num_txn_24h"] / (frame["num_txn_7d"].div(7).clip(lower=0.1))
        frame["merchant_concentration"] = frame["num_txn_7d"] / frame["distinct_merchants_7d"].clip(lower=1)
        frame["is_late_night"] = frame["hour_of_day"].between(0, 4).astype(int)
        frame["is_business_hours"] = frame["hour_of_day"].between(9, 17).astype(int)
        frame["is_new_account"] = (frame["account_age_days"] < 90).astype(int)
        frame["log_account_age"] = np.log1p(frame["account_age_days"])
        frame["log_transaction_amount"] = np.log1p(frame["transaction_amount"])
        frame["foreign_online"] = ((frame["is_foreign"] == 1) & (frame["channel"] == "Online")).astype(int)
        frame["new_account_high_amount"] = (
            (frame["is_new_account"] == 1) & (frame["amount_to_limit_ratio"] > 0.5)
        ).astype(int)
        frame["late_night_foreign"] = ((frame["is_late_night"] == 1) & (frame["is_foreign"] == 1)).astype(int)

        numeric_frame = frame[FEATURE_GROUPS.numeric].apply(pd.to_numeric, errors="coerce")
        categorical_frame = frame[FEATURE_GROUPS.categorical].astype("string")
        return pd.concat([numeric_frame, categorical_frame], axis=1)

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "transaction_id" in frame.columns:
            frame = frame.drop(columns=["transaction_id"])

        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["hour_of_day"] = frame["hour_of_day"].fillna(frame["timestamp"].dt.hour)
            frame["day_of_week"] = frame["day_of_week"].fillna(frame["timestamp"].dt.dayofweek)
            frame["month"] = frame["month"].fillna(frame["timestamp"].dt.month)

        frame["merchant_category"] = frame["merchant_category"].map(normalize_merchant_category)
        frame["channel"] = frame["channel"].map(normalize_channel)
        frame["device_type"] = frame["device_type"].map(normalize_device_type)
        return frame


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    dedupe_subset = [column for column in working.columns if column != "transaction_id"]
    working = working.drop_duplicates(subset=dedupe_subset).reset_index(drop=True)

    builder = FraudFeatureBuilder().fit(working.drop(columns=[TARGET_COLUMN]), working[TARGET_COLUMN])
    features = builder.transform(working.drop(columns=[TARGET_COLUMN]))
    cleaned = pd.concat([features, working[[TARGET_COLUMN]].reset_index(drop=True)], axis=1)
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURE_GROUPS.numeric),
            ("cat", categorical_pipeline, FEATURE_GROUPS.categorical),
        ]
    )
