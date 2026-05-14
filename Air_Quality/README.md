# Air Quality Production-Ready Build Plan

## Summary
Turn the repo from `dataset + README` into a complete, reproducible ML project with three layers:
1. analysis notebooks for understanding and validation,
2. reusable `src/` pipeline code for cleaning, features, training, and scoring,
3. deployment/ops pieces for batch prediction, API access, testing, and packaging.

Default assumption: build this as a strong production-style ML project first, with hourly batch scoring as the primary runtime and a small API for serving latest results.

## Implementation Changes

### Project structure
Create and use this structure consistently:
- `data/raw/air_quality_readings.csv` as immutable input
- `data/processed/` for cleaned training-ready datasets
- `notebooks/` for EDA and reporting only
- `src/` for all reusable Python logic
- `models/` for trained artifacts and thresholds
- `outputs/` for predictions, reports, and monitoring summaries
- `tests/` for unit and smoke tests
- root files: `app.py`, `requirements.txt`, `Dockerfile`, `.gitignore`, `README.md`

### Notebooks to create
Create 4 notebooks, each with a narrow purpose:
1. `01_data_quality.ipynb`
   - load raw CSV
   - inspect schema, target balance, duplicates, missingness
   - validate physical rules like `pm10 >= pm25`
   - document all cleaning rules before automation

2. `02_eda_visualization.ipynb`
   - label co-occurrence heatmap
   - pollutant distributions by station type
   - season x hour PM2.5 heatmap
   - inversion vs non-inversion PM2.5 comparison
   - per-label target frequency and multi-label count distribution

3. `03_feature_engineering.ipynb`
   - prototype missingness flags
   - normalize `season` and `station_type`
   - add circular time/wind features
   - add pollutant composite features
   - add lag/rolling station-level features
   - save a sample engineered dataset definition for parity with `src/`

4. `04_multilabel_modeling.ipynb`
   - baseline train/test split with time-aware logic
   - compare Binary Relevance vs Classifier Chains
   - report Hamming loss, micro/macro F1, exact match, per-label F1
   - tune per-label thresholds
   - summarize final model choice

Rule: notebooks explore and explain; they should not contain the only copy of business logic.

### Scripts/modules to create in `src/`
Build reusable modules instead of one large script:
- `src/config.py`
  - central paths, label column names, feature groups, random seed

- `src/data_loading.py`
  - read raw data
  - parse timestamps
  - basic schema checks

- `src/validation.py`
  - duplicate detection
  - physical sensor validity checks
  - data quality summary outputs

- `src/preprocessing.py`
  - category cleanup
  - missingness flag creation
  - imputation rules
  - raw-to-clean dataframe transformation

- `src/features.py`
  - circular encodings
  - composite pollution features
  - meteorological interactions
  - station-context features
  - rolling and lag features by station

- `src/train.py`
  - assemble features and labels
  - train baseline and candidate models
  - evaluate metrics
  - persist best model, preprocessor, and metadata

- `src/evaluate.py`
  - generate comparison tables, confusion-style per-label summaries, threshold metrics

- `src/score_batch.py`
  - load saved model artifacts
  - score new hourly readings
  - write predictions to `outputs/` or SQLite

- `src/monitoring.py`
  - daily label-rate drift
  - feature drift summary
  - simple model health report

### Entry-point scripts
Expose simple runnable scripts:
- `scripts/run_data_validation.py`
- `scripts/run_training.py`
- `scripts/run_batch_scoring.py`
- `scripts/run_monitoring_report.py`

Each should call `src/` modules and accept minimal config such as input/output paths.

### Model and deployment approach
Use this runtime design:
- training is offline and reproducible
- scoring is hourly batch-first
- API is read-only for latest alerts and health checks

Add:
- `app.py`
  - `/health`
  - `/current_alerts`
  - optional `/score` only if you want ad hoc scoring later
- SQLite or JSON output store for latest scored alerts
- `Dockerfile` for API + scoring environment
- `requirements.txt` pinned enough for reproducibility
- optional CI later for tests and linting

## Public Interfaces
These should be stable and documented:
- model artifacts:
  - `models/preprocessor.pkl`
  - `models/multilabel_model.pkl`
  - `models/label_thresholds.json`
  - `models/metrics_summary.json`
- batch output:
  - one row per reading with `prob_*`, `alert_*`, `total_alerts`, `scored_at`
- API response:
  - list of latest readings with station, AQI, per-label alerts, and top risk score

## Test Plan
Add tests for the parts most likely to break:
- preprocessing maps `Autumn` to `Fall` and standardizes station types
- missingness flags are created for the four dirty columns
- invalid `pm10 < pm25` rows are corrected or flagged
- engineered features are created with expected names and no silent null explosion
- rolling features respect station boundaries
- training pipeline fits and saves artifacts on a small sample
- batch scoring returns all `prob_*` and `alert_*` columns
- API `/health` and `/current_alerts` return valid responses

Also run one end-to-end smoke test:
- raw CSV -> cleaned dataset -> trained model -> saved artifacts -> scored outputs

## Assumptions
- Primary use case is portfolio-quality production readiness, not full enterprise MLOps.
- Hourly batch scoring is the main deployment path; real-time inference is secondary.
- The current repo has no implemented pipeline yet, so this is a greenfield build from the dataset.
- Start with `ClassifierChain` plus a strong tabular baseline; only add XGBoost if dependency setup stays manageable.
- Keep notebooks for communication and learning, but put all durable logic in `src/`.
