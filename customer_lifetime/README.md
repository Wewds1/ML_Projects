# Customer Lifetime Value Prediction with NLP

This repository packages the `scratch.md` brief into a working ML application: a trained customer lifetime value model, reusable Python pipeline code, a FastAPI inference service, a browser-based frontend, and Docker deployment assets.

## What is included

- `src/`: reusable preprocessing, feature engineering, training, and inference code.
- `models/`: generated model artifacts and experiment metadata.
- `frontend/`: static UI for manual scoring against the API.
- `notebooks/`: EDA, NLP, feature engineering, and modeling walkthroughs.
- `app.py`: FastAPI application with `/api/health`, `/api/metadata`, `/api/sample`, and `/api/predict`.
- `Dockerfile`: container build that installs dependencies, trains artifacts, and serves the app with Uvicorn.

## Local run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train or refresh artifacts:

```bash
python -m src.train
```

3. Start the application:

```bash
uvicorn app:app --reload
```

4. Open `http://localhost:8000`.

## Model summary

The current best run is `structured_plus_sentiment`, which slightly outperformed the TF-IDF variants on the included synthetic dataset. Experiment metrics are stored in `models/metadata.json` and surfaced in the frontend.

## Docker

```bash
docker build -t clv-nlp-app .
docker run -p 8000:8000 clv-nlp-app
```
