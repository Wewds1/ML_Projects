from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .loan_predictor.config import ROOT_DIR, settings
from .loan_predictor.features import TARGET_COLUMN, build_preprocessor, load_raw, prepare_features


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


def evaluate(model_name: str, fitted_model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = fitted_model.predict(x_test)
    return {
        "model": model_name,
        "rmse": rmse(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def build_candidate_pipelines(x_train: pd.DataFrame) -> tuple[list[tuple[str, Pipeline]], RandomizedSearchCV]:
    candidates = [
        ("linear_regression", Pipeline([("preprocessor", build_preprocessor(x_train)), ("model", LinearRegression())])),
        ("ridge", Pipeline([("preprocessor", build_preprocessor(x_train)), ("model", Ridge(alpha=1.0))])),
        (
            "random_forest",
            Pipeline(
                [
                    ("preprocessor", build_preprocessor(x_train)),
                    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
                ]
            ),
        ),
    ]

    search = RandomizedSearchCV(
        estimator=Pipeline(
            [
                ("preprocessor", build_preprocessor(x_train)),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
            ]
        ),
        param_distributions={
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 12, 18, 24],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    return candidates, search


def main() -> None:
    raw = load_raw(str(settings.raw_data_path))
    prepared = prepare_features(raw, training=True)

    x = prepared.drop(columns=[TARGET_COLUMN, "rate_spread"], errors="ignore")
    y = prepared[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    candidates, search = build_candidate_pipelines(x_train)

    metrics: list[dict[str, float | str]] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_score = float("inf")

    for name, pipeline in candidates:
        pipeline.fit(x_train, y_train)
        result = evaluate(name, pipeline, x_test, y_test)
        metrics.append(result)
        if result["rmse"] < best_score:
            best_name = name
            best_model = pipeline
            best_score = float(result["rmse"])

    search.fit(x_train, y_train)
    tuned_model = search.best_estimator_
    tuned_result = evaluate("tuned_random_forest", tuned_model, x_test, y_test)
    tuned_result["best_cv_rmse"] = abs(search.best_score_)
    metrics.append(tuned_result)

    if float(tuned_result["rmse"]) < best_score:
        best_name = "tuned_random_forest"
        best_model = tuned_model

    model_dir = Path(settings.model_path).parent
    reports_dir = ROOT_DIR / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, settings.model_path)
    pd.DataFrame(metrics).to_csv(reports_dir / "final_model_metrics.csv", index=False)

    print(f"[done] Best model: {best_name}")
    print(f"[done] Saved model to {settings.model_path}")
    print(f"[done] Saved metrics to {reports_dir / 'final_model_metrics.csv'}")


if __name__ == "__main__":
    main()
