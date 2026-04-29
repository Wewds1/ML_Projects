from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import canonicalize_channel, canonicalize_title, nps_bucket
from .settings import MODELS_DIR, VALID_CHANNELS, VALID_GENDERS, VALID_PRODUCT_TIERS
from .text_features import clean_text, extract_sentiment_features


class InferenceEngine:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        self.svd = joblib.load(models_dir / "tfidf_svd.pkl")
        self.model = joblib.load(models_dir / "clv_model.pkl")
        self.metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = self.metadata["feature_columns"]
        self.imputers = self.metadata["imputers"]
        self.product_tier_map = self.metadata["product_tier_map"]
        self.category_levels = self.metadata["category_levels"]

    def prepare_payload(self, payload: dict[str, object]) -> tuple[pd.DataFrame, dict[str, float | int]]:
        frame = pd.DataFrame([payload]).copy()
        frame["product_tier"] = frame["product_tier"].map(canonicalize_title)
        frame["acquisition_channel"] = frame["acquisition_channel"].map(canonicalize_channel)
        frame["gender"] = frame["gender"].map(canonicalize_title)
        frame["country"] = frame["country"].astype(str).str.strip().str.upper()

        frame["product_tier"] = frame["product_tier"].where(
            frame["product_tier"].isin(VALID_PRODUCT_TIERS), "Basic"
        )
        frame["acquisition_channel"] = frame["acquisition_channel"].where(
            frame["acquisition_channel"].isin(VALID_CHANNELS), "Direct"
        )
        frame["gender"] = frame["gender"].where(frame["gender"].isin(VALID_GENDERS), "Non-Binary")

        frame["nps_missing"] = frame["nps_score"].isna().astype(int)
        frame["feedback_missing"] = frame["customer_feedback"].isna().astype(int)

        frame["nps_score"] = frame.apply(
            lambda row: self.imputers["tier_nps_median"].get(row["product_tier"], 7.0)
            if pd.isna(row["nps_score"])
            else row["nps_score"],
            axis=1,
        )
        frame["monthly_spend"] = frame.apply(
            lambda row: 0.0
            if pd.isna(row["monthly_spend"]) and row["product_tier"] == "Free"
            else (
                self.imputers["tier_spend_median"].get(row["product_tier"], 0.0)
                if pd.isna(row["monthly_spend"])
                else row["monthly_spend"]
            ),
            axis=1,
        )
        frame["feature_adoption"] = frame.apply(
            lambda row: self.imputers["tier_feature_adoption_median"].get(
                row["product_tier"], 0.0
            )
            if pd.isna(row["feature_adoption"])
            else row["feature_adoption"],
            axis=1,
        )

        text_features = extract_sentiment_features(frame.at[0, "customer_feedback"])
        cleaned_feedback = clean_text(frame.at[0, "customer_feedback"])
        tfidf_matrix = self.vectorizer.transform([cleaned_feedback])
        reduced = self.svd.transform(tfidf_matrix)[0]

        for key, value in text_features.items():
            frame[key] = value
        for idx, value in enumerate(reduced):
            frame[f"tfidf_svd_{idx}"] = float(value)

        frame["spend_per_login"] = frame["monthly_spend"] / frame["login_freq_monthly"].clip(lower=1)
        frame["ticket_rate"] = frame["support_tickets_6m"] / frame["tenure_months"].clip(lower=1)
        frame["referral_value_est"] = frame["referrals_made"] * frame["monthly_spend"]
        frame["engagement_score"] = (frame["login_freq_monthly"] / 30.0) * frame["feature_adoption"]
        frame["churn_risk_proxy"] = (
            frame["payment_failures_6m"] * 0.4
            + (frame["days_since_login"] > 30).astype(int) * 0.3
            + (frame["nps_score"] < 5).astype(int) * 0.3
        )
        frame["log_monthly_spend"] = np.log1p(frame["monthly_spend"])
        frame["log_tenure_months"] = np.log1p(frame["tenure_months"])
        frame["log_days_since_login"] = np.log1p(frame["days_since_login"])
        frame["sqrt_login_freq"] = np.sqrt(frame["login_freq_monthly"])
        frame["product_tier_ordinal"] = frame["product_tier"].map(self.product_tier_map)
        frame["nps_category"] = frame["nps_score"].map(nps_bucket)

        dummies = pd.get_dummies(
            frame[["country", "acquisition_channel", "gender", "nps_category"]],
            drop_first=False,
            dtype=float,
        )
        for prefix, categories in self.category_levels.items():
            if prefix == "countries":
                for country in categories:
                    column = f"country_{country}"
                    dummies[column] = dummies.get(column, 0.0)
            else:
                source = "acquisition_channel" if prefix == "acquisition_channel" else prefix
                for category in categories:
                    column = f"{source}_{category}"
                    dummies[column] = dummies.get(column, 0.0)

        numeric = frame[
            [
                "customer_age",
                "tenure_months",
                "monthly_spend",
                "login_freq_monthly",
                "feature_adoption",
                "support_tickets_6m",
                "nps_score",
                "payment_failures_6m",
                "referrals_made",
                "days_since_login",
                "nps_missing",
                "feedback_missing",
                "spend_per_login",
                "ticket_rate",
                "referral_value_est",
                "engagement_score",
                "churn_risk_proxy",
                "log_monthly_spend",
                "log_tenure_months",
                "log_days_since_login",
                "sqrt_login_freq",
                "product_tier_ordinal",
                "sentiment_polarity",
                "sentiment_subjectivity",
                "has_negation",
                "churn_language",
                "praise_language",
                "feedback_word_count",
            ]
            + [f"tfidf_svd_{idx}" for idx in range(len(reduced))]
        ].astype(float)
        features = pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        return features, text_features

    def predict(self, payload: dict[str, object]) -> dict[str, object]:
        features, text_features = self.prepare_payload(payload)
        log_prediction = float(self.model.predict(features)[0])
        clv_usd = float(np.expm1(log_prediction))
        return {
            "clv_12m_estimate_usd": round(clv_usd, 2),
            "log_clv_prediction": round(log_prediction, 4),
            "sentiment_polarity": round(float(text_features["sentiment_polarity"]), 3),
            "sentiment_subjectivity": round(float(text_features["sentiment_subjectivity"]), 3),
            "churn_risk_flag": int(text_features["churn_language"]),
            "praise_language_flag": int(text_features["praise_language"]),
        }
