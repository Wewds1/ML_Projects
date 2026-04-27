from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import settings
from .model import load_model, predict_dataframe
from .schemas import HealthResponse, PredictionRequest, PredictionResponse, PredictionResult


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "message": "Loan interest rate predictor is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model artifact not found: {exc}") from exc

    return HealthResponse(
        status="ok",
        model_path=str(settings.model_path),
        app_version=settings.app_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    frame = pd.DataFrame([record.model_dump(mode="json") for record in payload.records])

    try:
        predictions = predict_dataframe(frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model artifact not found: {exc}") from exc

    results = [
        PredictionResult(
            borrower_id=row["borrower_id"],
            predicted_interest_rate=round(float(row["predicted_interest_rate"]), 4),
        )
        for _, row in predictions.iterrows()
    ]
    return PredictionResponse(predictions=results)
