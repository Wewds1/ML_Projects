from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .settings import (
    MODELS_DIR,
    PRODUCT_TIER_MAP,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEXT_COMPONENTS,
    VALID_CHANNELS,
    VALID_GENDERS,
    VALID_PRODUCT_TIERS,
)
from .text_features import clean_text, extract_sentiment_features


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    original_rows: int
    duplicates_removed: int


def canonicalize_title(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().title()


def canonicalize_channel(value: object) -> str:
    cleaned = canonicalize_title(value)
    aliases = {
        "Organic Search": "Organic Search",
        "Paid Ads": "Paid Ads",
        "Referral": "Referral",
        "Social Media": "Social Media",
        "Email Campaign": "Email Campaign",
        "Direct": "Direct",
    }
    return aliases.get(cleaned, cleaned)


def load_dataset(path: Path = RAW_DATA_PATH) -> DatasetBundle:
    df = pd.read_csv(path)
    original_rows = len(df)
    deduped = df.drop_duplicates(subset=[col for col in df.columns if col != "customer_id"]).copy()
    duplicates_removed = original_rows - len(deduped)
    return DatasetBundle(deduped.reset_index(drop=True), original_rows, duplicates_removed)


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    frame = df.copy()

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

    tier_nps_median = (
        frame.groupby("product_tier", dropna=False)["nps_score"].median().to_dict()
    )
    tier_spend_median = (
        frame.groupby("product_tier", dropna=False)["monthly_spend"].median().fillna(0).to_dict()
    )
    tier_feature_adoption_median = (
        frame.groupby("product_tier", dropna=False)["feature_adoption"].median().to_dict()
    )

    frame["nps_score"] = frame.apply(
        lambda row: tier_nps_median.get(row["product_tier"], frame["nps_score"].median())
        if pd.isna(row["nps_score"])
        else row["nps_score"],
        axis=1,
    )

    frame["monthly_spend"] = frame.apply(
        lambda row: 0.0
        if pd.isna(row["monthly_spend"]) and row["product_tier"] == "Free"
        else (
            tier_spend_median.get(row["product_tier"], frame["monthly_spend"].median())
            if pd.isna(row["monthly_spend"])
            else row["monthly_spend"]
        ),
        axis=1,
    )

    frame["feature_adoption"] = frame.apply(
        lambda row: tier_feature_adoption_median.get(
            row["product_tier"], frame["feature_adoption"].median()
        )
        if pd.isna(row["feature_adoption"])
        else row["feature_adoption"],
        axis=1,
    )

    frame["feedback_clean"] = frame["customer_feedback"].map(clean_text)
    sentiment = frame["customer_feedback"].apply(extract_sentiment_features).apply(pd.Series)
    frame = pd.concat([frame, sentiment], axis=1)

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
    frame["product_tier_ordinal"] = frame["product_tier"].map(PRODUCT_TIER_MAP)
    frame["log_clv_12m"] = np.log1p(frame["clv_12m"])
    frame["nps_category"] = frame["nps_score"].map(nps_bucket)

    imputers = {
        "tier_nps_median": tier_nps_median,
        "tier_spend_median": tier_spend_median,
        "tier_feature_adoption_median": tier_feature_adoption_median,
    }
    return frame, imputers


def nps_bucket(score: float) -> str:
    if score <= 6:
        return "Detractor"
    if score <= 8:
        return "Passive"
    return "Promoter"


def build_text_projection(text_series: pd.Series) -> tuple[TfidfVectorizer, TruncatedSVD, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(text_series.fillna(""))
    components = min(TEXT_COMPONENTS, max(2, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(tfidf_matrix)
    reduced_df = pd.DataFrame(reduced, columns=[f"tfidf_svd_{i}" for i in range(components)])
    return vectorizer, svd, reduced_df


def build_feature_views(
    frame: pd.DataFrame,
    tfidf_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical = pd.get_dummies(
        frame[["country", "acquisition_channel", "gender", "nps_category"]],
        drop_first=False,
        dtype=float,
    )
    structured_numeric = frame[
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
        ]
    ].astype(float)
    sentiment = frame[
        [
            "sentiment_polarity",
            "sentiment_subjectivity",
            "has_negation",
            "churn_language",
            "praise_language",
            "feedback_word_count",
        ]
    ].astype(float)
    structured = pd.concat([structured_numeric, categorical], axis=1)
    with_sentiment = pd.concat([structured, sentiment], axis=1)
    with_tfidf = pd.concat([structured, tfidf_features], axis=1)
    with_all = pd.concat([structured, sentiment, tfidf_features], axis=1)
    return structured, with_sentiment, with_tfidf, with_all


def evaluate_regression(
    X: pd.DataFrame,
    y_log: pd.Series,
    tiers: pd.Series,
) -> dict[str, object]:
    X_train, X_test, y_train, y_test, _tier_train, tier_test = train_test_split(
        X,
        y_log,
        tiers,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse_log = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    actual_usd = np.expm1(y_test)
    pred_usd = np.expm1(predictions)
    mae_usd = float(mean_absolute_error(actual_usd, pred_usd))

    segment_rmse: dict[str, float] = {}
    eval_frame = pd.DataFrame(
        {
            "tier": tier_test.to_numpy(),
            "actual": y_test.to_numpy(),
            "pred": predictions,
        }
    )
    for tier_name, group in eval_frame.groupby("tier"):
        segment_rmse[str(tier_name)] = float(
            np.sqrt(mean_squared_error(group["actual"], group["pred"]))
        )

    return {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "rmse_log": rmse_log,
        "r2": r2,
        "mae_usd": mae_usd,
        "segment_rmse_log": segment_rmse,
        "test_size": len(X_test),
    }


def train_project_artifacts() -> dict[str, object]:
    bundle = load_dataset()
    frame, imputers = preprocess_dataframe(bundle.data)
    vectorizer, svd, tfidf_features = build_text_projection(frame["feedback_clean"])
    structured, with_sentiment, with_tfidf, with_all = build_feature_views(frame, tfidf_features)
    target = frame["log_clv_12m"]
    tiers = frame["product_tier"]

    experiments = {
        "structured_only": evaluate_regression(structured, target, tiers),
        "structured_plus_sentiment": evaluate_regression(with_sentiment, target, tiers),
        "structured_plus_tfidf": evaluate_regression(with_tfidf, target, tiers),
        "all_nlp_features": evaluate_regression(with_all, target, tiers),
    }
    best_name = min(experiments.items(), key=lambda item: item[1]["rmse_log"])[0]
    best_run = experiments[best_name]

    metadata = {
        "dataset": {
            "raw_rows": bundle.original_rows,
            "modeled_rows": len(frame),
            "duplicates_removed": bundle.duplicates_removed,
        },
        "experiments": {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"model", "feature_columns"}
            }
            for name, result in experiments.items()
        },
        "best_experiment": best_name,
        "feature_columns": best_run["feature_columns"],
        "tfidf_components": [col for col in tfidf_features.columns],
        "vectorizer_vocabulary_size": len(vectorizer.get_feature_names_out()),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "category_levels": {
            "countries": sorted(frame["country"].dropna().unique().tolist()),
            "acquisition_channel": VALID_CHANNELS,
            "gender": VALID_GENDERS,
            "nps_category": ["Detractor", "Passive", "Promoter"],
        },
        "imputers": imputers,
        "product_tier_map": PRODUCT_TIER_MAP,
    }

    return {
        "frame": frame,
        "tfidf_features": tfidf_features,
        "vectorizer": vectorizer,
        "svd": svd,
        "best_model": best_run["model"],
        "metadata": metadata,
    }


def save_artifacts(artifacts: dict[str, object], models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["vectorizer"], models_dir / "tfidf_vectorizer.pkl")
    joblib.dump(artifacts["svd"], models_dir / "tfidf_svd.pkl")
    joblib.dump(artifacts["best_model"], models_dir / "clv_model.pkl")
    with (models_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts["metadata"], handle, indent=2)
