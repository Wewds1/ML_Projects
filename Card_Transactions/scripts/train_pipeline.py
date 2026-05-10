from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import pointbiserialr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    FEATURE_GROUPS,
    FraudFeatureBuilder,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    build_preprocessor,
    clean_transactions,
    load_raw_data,
)

sns.set_theme(style="whitegrid")

EDA_DIR = ROOT / "01_eda"
FIG_DIR = EDA_DIR / "figures"
MODELS_DIR = ROOT / "models"
PROCESSED_PATH = ROOT / PROCESSED_DATA_PATH


def ensure_dirs() -> None:
    for path in [EDA_DIR, FIG_DIR, MODELS_DIR, PROCESSED_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def series_to_markdown(series: pd.Series, value_name: str = "value") -> str:
    lines = [f"| index | {value_name} |", "|---|---:|"]
    for idx, value in series.items():
        if isinstance(value, (float, np.floating)):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| {idx} | {rendered} |")
    return "\n".join(lines)


def generate_eda(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict[str, object]:
    fraud_rate = raw_df[TARGET_COLUMN].mean()
    duplicates = raw_df.drop(columns=["transaction_id"]).duplicated().sum()
    missing_summary = raw_df.isna().sum().sort_values(ascending=False)

    hour_counts = raw_df.groupby("hour_of_day")[TARGET_COLUMN].sum()
    hour_rates = raw_df.groupby("hour_of_day")[TARGET_COLUMN].mean()
    day_rates = raw_df.groupby("day_of_week")[TARGET_COLUMN].mean()
    merchant_rates = clean_df.groupby("merchant_category")[TARGET_COLUMN].mean().sort_values(ascending=False)
    channel_rates = clean_df.groupby("channel")[TARGET_COLUMN].mean().sort_values(ascending=False)
    device_rates = clean_df.groupby("device_type")[TARGET_COLUMN].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, color="#d1495b")
    plt.title("Fraud Count by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Count")
    save_plot(FIG_DIR / "fraud_count_by_hour.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=hour_rates.index, y=hour_rates.values, color="#00798c")
    plt.title("Fraud Rate by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Rate")
    save_plot(FIG_DIR / "fraud_rate_by_hour.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(x=day_rates.index, y=day_rates.values, color="#edae49")
    plt.title("Fraud Rate by Day of Week")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Fraud Rate")
    save_plot(FIG_DIR / "fraud_rate_by_day.png")

    plt.figure(figsize=(10, 5))
    legit = raw_df.loc[raw_df[TARGET_COLUMN] == 0, "transaction_amount"]
    fraud = raw_df.loc[raw_df[TARGET_COLUMN] == 1, "transaction_amount"]
    plt.hist(legit, bins=40, alpha=0.55, label="Legit", color="#31708e")
    plt.hist(fraud, bins=40, alpha=0.55, label="Fraud", color="#d1495b")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Count")
    plt.legend()
    save_plot(FIG_DIR / "transaction_amount_hist.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=raw_df,
        x=TARGET_COLUMN,
        y="transaction_amount",
        hue=TARGET_COLUMN,
        palette=["#31708e", "#d1495b"],
        legend=False,
    )
    plt.yscale("log")
    plt.title("Transaction Amount by Fraud Label")
    plt.xlabel("Is Fraud")
    plt.ylabel("Transaction Amount (log scale)")
    save_plot(FIG_DIR / "transaction_amount_boxplot.png")

    for series, filename, title, color in [
        (merchant_rates, "merchant_category_fraud_rate.png", "Fraud Rate by Merchant Category", "#d1495b"),
        (channel_rates, "channel_fraud_rate.png", "Fraud Rate by Channel", "#00798c"),
        (device_rates, "device_type_fraud_rate.png", "Fraud Rate by Device Type", "#edae49"),
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=series.values, y=series.index, color=color)
        plt.title(title)
        plt.xlabel("Fraud Rate")
        plt.ylabel("")
        save_plot(FIG_DIR / filename)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=raw_df.sample(min(len(raw_df), 2000), random_state=42),
        x="num_txn_24h",
        y="transaction_amount",
        hue=TARGET_COLUMN,
        palette={0: "#31708e", 1: "#d1495b"},
        alpha=0.7,
    )
    plt.title("Velocity vs Amount")
    plt.xlabel("Transactions in Last 24h")
    plt.ylabel("Transaction Amount")
    save_plot(FIG_DIR / "velocity_scatter.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=raw_df,
        x=TARGET_COLUMN,
        y="declined_last_30d",
        hue=TARGET_COLUMN,
        palette=["#31708e", "#d1495b"],
        legend=False,
    )
    plt.title("Declined Transactions in Last 30 Days")
    plt.xlabel("Is Fraud")
    plt.ylabel("Declined Count")
    save_plot(FIG_DIR / "declined_boxplot.png")

    plt.figure(figsize=(12, 4))
    msno.matrix(raw_df[["merchant_country", "avg_7day_spend", "days_since_last_txn"]], sparkline=False, color=(0.19, 0.44, 0.56))
    plt.title("Missingness Snapshot")
    save_plot(FIG_DIR / "missingness_matrix.png")

    corrs = {}
    for column in clean_df.columns:
        if column == TARGET_COLUMN or clean_df[column].dtype == "string":
            continue
        if clean_df[column].nunique(dropna=False) <= 1:
            continue
        corrs[column] = pointbiserialr(clean_df[TARGET_COLUMN], clean_df[column]).statistic
    corr_series = pd.Series(corrs).sort_values(ascending=False)
    corr_series.to_csv(EDA_DIR / "point_biserial_correlations.csv", header=["correlation"])

    overview_md = f"""# EDA Overview

This dataset starts at **{len(raw_df):,} rows** and lands at **{len(clean_df):,} rows** after the duplicate cleanup, so yeah, the batch issue in the brief was real.

The fraud rate is **{fraud_rate:.2%}**. That number kinda rules the whole project, because accuracy would look fake-good here and still miss every bad transaction.

Quick notes from the first pass:

- Duplicate rows found when ignoring `transaction_id`: **{duplicates}**
- Missing `merchant_country`: **{int(raw_df['merchant_country'].isna().sum())}**
- Missing `avg_7day_spend`: **{int(raw_df['avg_7day_spend'].isna().sum())}**
- Missing `days_since_last_txn`: **{int(raw_df['days_since_last_txn'].isna().sum())}**

The fraud class is tiny, but it does not look random. A couple patterns show up pretty fast once the dirty categories get fixed.

## Files saved

- `figures/fraud_count_by_hour.png`
- `figures/fraud_rate_by_hour.png`
- `figures/fraud_rate_by_day.png`
- `figures/missingness_matrix.png`
- `point_biserial_correlations.csv`
"""

    patterns_md = f"""# Risk Patterns

This part is where the data gets more intresting.

## Time stuff

Fraud counts peak around **hour {int(hour_counts.idxmax())}**, but the fraud **rate** peaks around **hour {int(hour_rates.idxmax())}**. That split matters a lot, otherwise we'd just be charting volume and calling it insight.

## Categorical risk

Top merchant categories by fraud rate:

{series_to_markdown(merchant_rates.head(5), "fraud_rate")}

Top channels by fraud rate:

{series_to_markdown(channel_rates, "fraud_rate")}

Top device types by fraud rate:

{series_to_markdown(device_rates, "fraud_rate")}

The online-ish behavior and ATM-style behavior are defintely riskier here, which is pretty much what we hoped to confirm after the cleanup.
"""

    feature_md = f"""# Feature Notes

These are the engineered bits that probly matter most before modeling:

- `amount_to_limit_ratio`: catches pressure against the credit limit, not just raw spend.
- `amount_vs_7day_avg`: flags when a charge is way above the cardholder's usual rhythm.
- `txn_velocity_ratio`: tells us if the last 24 hours are moving way faster than the last week.
- `foreign_online`, `late_night_foreign`, `new_account_high_amount`: these combo flags are kinda where the fraud story gets more real.

## Correlation check

Top numeric features by point-biserial correlation with fraud:

{series_to_markdown(corr_series.head(10).round(4), "correlation")}

It's not a perfect ranking, but it gives a good sanity check before fitting the heavier models.
"""

    (EDA_DIR / "01_overview.md").write_text(overview_md, encoding="utf-8")
    (EDA_DIR / "02_risk_patterns.md").write_text(patterns_md, encoding="utf-8")
    (EDA_DIR / "03_feature_notes.md").write_text(feature_md, encoding="utf-8")

    return {
        "fraud_rate": float(fraud_rate),
        "duplicates_removed": int(duplicates),
        "missing_summary": {k: int(v) for k, v in missing_summary.items() if int(v) > 0},
        "top_correlations": corr_series.head(10).round(4).to_dict(),
    }


def threshold_search(y_true: pd.Series, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    thresholds = np.append(thresholds, 1.0)
    records = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
            }
        )
    threshold_df = pd.DataFrame(records).sort_values(["f1", "recall", "precision"], ascending=False)
    best = threshold_df.iloc[0].to_dict()
    return float(best["threshold"]), best


def make_model_pipelines(scale_pos_weight: float) -> dict[str, ImbPipeline]:
    preprocessor: ColumnTransformer = build_preprocessor()
    base_steps = [("features", FraudFeatureBuilder()), ("preprocessor", preprocessor)]
    return {
        "logistic_balanced": ImbPipeline(
            steps=base_steps
            + [
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1, random_state=42)),
            ]
        ),
        "random_forest_balanced": ImbPipeline(
            steps=base_steps
            + [
                ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        ),
        "xgboost_weighted": ImbPipeline(
            steps=base_steps
            + [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="aucpr",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logistic_smote": ImbPipeline(
            steps=base_steps
            + [
                ("smote", SMOTE(random_state=42)),
                ("model", LogisticRegression(max_iter=2000, C=0.1, random_state=42)),
            ]
        ),
    }


def train_and_evaluate(raw_df: pd.DataFrame) -> dict[str, object]:
    deduped = raw_df.drop_duplicates(subset=[col for col in raw_df.columns if col != "transaction_id"]).reset_index(drop=True)
    X = deduped.drop(columns=[TARGET_COLUMN])
    y = deduped[TARGET_COLUMN]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pipelines = make_model_pipelines(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    fitted_candidates = {}
    for name, pipeline in pipelines.items():
        cv_scores = cross_val_score(clone(pipeline), X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        pipeline.fit(X_train, y_train)
        val_scores = pipeline.predict_proba(X_val)[:, 1]
        threshold, threshold_metrics = threshold_search(y_val, val_scores)
        val_preds = (val_scores >= threshold).astype(int)
        result = {
            "model": name,
            "cv_pr_auc_mean": float(cv_scores.mean()),
            "cv_pr_auc_std": float(cv_scores.std()),
            "val_pr_auc": float(average_precision_score(y_val, val_scores)),
            "val_roc_auc": float(roc_auc_score(y_val, val_scores)),
            "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
            "val_f1": float(f1_score(y_val, val_preds, zero_division=0)),
            "threshold": float(threshold),
        }
        result.update({f"threshold_{k}": float(v) for k, v in threshold_metrics.items() if k != "threshold"})
        results.append(result)
        fitted_candidates[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        ["val_pr_auc", "val_recall", "val_precision"], ascending=False
    )
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_threshold = float(results_df.iloc[0]["threshold"])

    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_trainval, y_trainval)
    test_scores = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_scores >= best_threshold).astype(int)

    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, test_preds).tolist()

    pd.DataFrame(
        [
            {
                "threshold": threshold,
                "precision": precision_score(y_test, (test_scores >= threshold).astype(int), zero_division=0),
                "recall": recall_score(y_test, (test_scores >= threshold).astype(int), zero_division=0),
                "f1": f1_score(y_test, (test_scores >= threshold).astype(int), zero_division=0),
            }
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, best_threshold]
        ]
    ).drop_duplicates(subset=["threshold"]).to_csv(MODELS_DIR / "threshold_analysis.csv", index=False)

    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, test_scores)
    plt.title(f"Precision-Recall Curve ({best_name})")
    save_plot(FIG_DIR / "precision_recall_curve.png")

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, test_scores)
    plt.title(f"ROC Curve ({best_name})")
    save_plot(FIG_DIR / "roc_curve.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(confusion), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix @ {best_threshold:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(FIG_DIR / "confusion_matrix.png")

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importances = (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, color="#00798c")
        plt.title("Top Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("")
        save_plot(FIG_DIR / "feature_importances.png")
    elif hasattr(model, "coef_"):
        feature_names = preprocessor.get_feature_names_out()
        importances = (
            pd.Series(model.coef_[0], index=feature_names)
            .abs()
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, color="#00798c")
        plt.title("Top Absolute Coefficients")
        plt.xlabel("Absolute Coefficient")
        plt.ylabel("")
        save_plot(FIG_DIR / "feature_importances.png")

    joblib.dump(best_pipeline, MODELS_DIR / "fraud_detector_v1.pkl")
    config = {
        "threshold": best_threshold,
        "model_name": best_name,
        "port": 6000,
        "metrics": {
            "test_pr_auc": float(average_precision_score(y_test, test_scores)),
            "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
            "test_precision": float(precision_score(y_test, test_preds, zero_division=0)),
            "test_recall": float(recall_score(y_test, test_preds, zero_division=0)),
            "test_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        },
    }
    (MODELS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (MODELS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_md = f"""# Modeling Notes

The best validation model ended up being **{best_name}** with a chosen threshold of **{best_threshold:.2f}**.

This was the test-set readout after retraining on train+validation:

- PR-AUC: **{config['metrics']['test_pr_auc']:.4f}**
- ROC-AUC: **{config['metrics']['test_roc_auc']:.4f}**
- Precision: **{config['metrics']['test_precision']:.4f}**
- Recall: **{config['metrics']['test_recall']:.4f}**
- F1: **{config['metrics']['test_f1']:.4f}**

Honestly, the big thing here is not perfect precision. It's catching enough fraud without letting the false positives get totally silly. The threshold file in `models/config.json` keeps that decision explicit.

## Saved artifacts

- `models/fraud_detector_v1.pkl`
- `models/config.json`
- `models/model_comparison.csv`
- `models/threshold_analysis.csv`
- `figures/precision_recall_curve.png`
- `figures/confusion_matrix.png`
"""
    (EDA_DIR / "04_modeling_notes.md").write_text(model_md, encoding="utf-8")

    return {
        "best_model": best_name,
        "threshold": best_threshold,
        "test_metrics": config["metrics"],
        "confusion_matrix": confusion,
    }


def main() -> None:
    ensure_dirs()
    raw_df = load_raw_data()
    clean_df = clean_transactions(raw_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    eda_summary = generate_eda(raw_df, clean_df)
    model_summary = train_and_evaluate(raw_df)

    summary = {
        "eda": eda_summary,
        "modeling": model_summary,
        "row_counts": {"raw": int(len(raw_df)), "clean": int(len(clean_df))},
    }
    (MODELS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
