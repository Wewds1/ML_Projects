from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="CardShield Fraud Detection API",
    description="Portfolio-ready fraud scoring service for synthetic credit card transactions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ROOT / "frontend" / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "frontend" / "templates"))

pipeline = joblib.load(MODELS_DIR / "fraud_detector_v1.pkl")
config = json.loads((MODELS_DIR / "config.json").read_text(encoding="utf-8"))
THRESHOLD = float(config["threshold"])


class TransactionInput(BaseModel):
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    month: int = Field(ge=1, le=12)
    cardholder_age: int = Field(ge=18, le=100)
    account_age_days: int = Field(ge=0)
    credit_limit: float = Field(gt=0)
    merchant_category: str
    merchant_country: str | None = None
    is_foreign: int = Field(ge=0, le=1)
    transaction_amount: float = Field(gt=0)
    avg_7day_spend: float | None = Field(default=None, ge=0)
    num_txn_24h: int = Field(ge=0)
    num_txn_7d: int = Field(ge=0)
    days_since_last_txn: float | None = Field(default=None, ge=0)
    distinct_merchants_7d: int = Field(ge=0)
    declined_last_30d: int = Field(ge=0)
    channel: str
    device_type: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"threshold": THRESHOLD, "model_name": config["model_name"]},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(data: TransactionInput) -> dict[str, object]:
    frame = pd.DataFrame([data.model_dump()])
    fraud_probability = float(pipeline.predict_proba(frame)[0][1])
    decision = int(fraud_probability >= THRESHOLD)
    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": decision,
        "threshold_used": THRESHOLD,
        "action": "BLOCK" if decision else "APPROVE",
        "model_name": config["model_name"],
    }
