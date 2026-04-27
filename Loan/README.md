# Loan Interest Rate Predictor

Portfolio-ready machine learning project for predicting offered loan interest rates from borrower profile, loan structure, and macroeconomic context. The repository now supports both reproducible model development and deployable API inference.

## What Changed

This project was upgraded from a notebook-first repo into a deployable application:

- reusable feature engineering package under `src/loan_predictor`
- FastAPI inference service with `/health` and `/predict`
- reproducible training script in `src/train.py`
- cleanup utility in `scripts/clean.py`
- lean runtime dependencies and separate dev dependencies
- deployment assets: `Dockerfile`, `Procfile`, `runtime.txt`
- test scaffolding for feature engineering and API routes

## Project Structure

```text
Loan/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── scripts/
│   └── clean.py
├── src/
│   ├── pipeline.py
│   ├── train.py
│   └── loan_predictor/
│       ├── api.py
│       ├── config.py
│       ├── features.py
│       ├── model.py
│       └── schemas.py
├── tests/
├── Dockerfile
├── Procfile
├── requirements.txt
└── requirements-dev.txt
```

## Local Setup

```bash
python -m venv .venv
pip install -r requirements-dev.txt
```

Activate the virtual environment with the command appropriate for your shell before installing dependencies.

## Core Workflows

### 1. Rebuild processed data

```bash
python -m src.pipeline
```

### 2. Retrain the model artifact

```bash
python -m src.train
```

This saves:

- processed dataset to `data/processed/loans_clean.csv`
- model artifact to `models/best_random_forest_pipeline.joblib`
- evaluation metrics to `reports/final_model_metrics.csv`

### 3. Run the API locally

```bash
uvicorn src.loan_predictor.api:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Prediction Contract

`POST /predict`

Request body:

```json
{
  "records": [
    {
      "borrower_id": "BRW99999",
      "age": 36,
      "employment_type": "Salaried",
      "employment_years": 7.5,
      "annual_income": 40590.92,
      "credit_score": 690,
      "loan_amount": 14000,
      "loan_term_months": 12,
      "loan_purpose": "Auto",
      "existing_debt": 5382.22,
      "num_open_accounts": 5,
      "num_late_payments": 0,
      "dti_ratio": 0.1326,
      "origination_date": "2023-06-20",
      "origination_year": 2023,
      "origination_quarter": "Q2",
      "fed_funds_rate": 5.278
    }
  ]
}
```

Response body:

```json
{
  "predictions": [
    {
      "borrower_id": "BRW99999",
      "predicted_interest_rate": 4.3701
    }
  ]
}
```

## Deployment

### Docker

```bash
docker build -t loan-rate-api .
docker run -p 8000:8000 loan-rate-api
```

### Platform-as-a-service

The repo also includes:

- `Procfile` for Heroku-style process platforms
- `runtime.txt` for Python version pinning

If you deploy to Render, Railway, Fly.io, or Heroku-style platforms, point the web service command to:

```bash
uvicorn src.loan_predictor.api:app --host 0.0.0.0 --port $PORT
```

## Testing and Cleanup

Run tests:

```bash
pytest
```

Run cleanup:

```bash
python scripts/clean.py
```

## Portfolio Notes

This project now presents well in a portfolio because it shows:

- messy data handling instead of toy-clean datasets
- feature engineering separated from notebooks
- train vs inference separation
- a callable API instead of notebook-only output
- deployment assets that make the repo easy to review and run

## Limitations

- The saved model artifact must exist before the API can serve predictions.
- The dataset is synthetic and intended for educational and portfolio use.
- The current API is synchronous and optimized for simple portfolio deployments rather than high-throughput production traffic.
