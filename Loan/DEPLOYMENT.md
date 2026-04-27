# Deployment Guide

## Recommended Baseline

Use Python `3.11` and deploy the API process below:

```bash
uvicorn src.loan_predictor.api:app --host 0.0.0.0 --port $PORT
```

## Environment Variables

Optional variables:

- `PORT`
- `HOST`
- `MODEL_PATH`
- `RAW_DATA_PATH`
- `PROCESSED_DATA_PATH`

Defaults are defined in [src/loan_predictor/config.py](D:/Projects/Loan/src/loan_predictor/config.py).

## Docker Deployment

```bash
docker build -t loan-rate-api .
docker run -p 8000:8000 loan-rate-api
```

## Render / Railway / Similar

Use:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.loan_predictor.api:app --host 0.0.0.0 --port $PORT`

## Pre-Deployment Checklist

- confirm `models/best_random_forest_pipeline.joblib` exists
- confirm `requirements.txt` installs cleanly
- hit `/health` after deploy
- test one `/predict` request from Swagger UI or curl
