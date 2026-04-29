from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference import InferenceEngine
from src.pipeline import save_artifacts, train_project_artifacts
from src.settings import FRONTEND_DIR, MODELS_DIR

engine: InferenceEngine | None = None


class CustomerInput(BaseModel):
    customer_age: int = Field(ge=18, le=100)
    gender: str
    country: str = Field(min_length=2, max_length=3)
    acquisition_channel: str
    product_tier: str
    tenure_months: int = Field(ge=0, le=240)
    monthly_spend: float | None = Field(default=None, ge=0)
    login_freq_monthly: int = Field(ge=0, le=500)
    feature_adoption: float | None = Field(default=None, ge=0, le=1)
    support_tickets_6m: int = Field(ge=0, le=200)
    nps_score: float | None = Field(default=None, ge=0, le=10)
    payment_failures_6m: int = Field(ge=0, le=50)
    referrals_made: int = Field(ge=0, le=100)
    days_since_login: int = Field(ge=0, le=3650)
    customer_feedback: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    required = [
        MODELS_DIR / "tfidf_vectorizer.pkl",
        MODELS_DIR / "tfidf_svd.pkl",
        MODELS_DIR / "clv_model.pkl",
        MODELS_DIR / "metadata.json",
    ]
    if not all(path.exists() for path in required):
        artifacts = train_project_artifacts()
        save_artifacts(artifacts, MODELS_DIR)
    engine = InferenceEngine(MODELS_DIR)
    yield


app = FastAPI(
    title="Customer Lifetime Value Predictor",
    version="1.0.0",
    description="Predict 12-month customer lifetime value using behavioral and NLP features.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> dict[str, object]:
    assert engine is not None
    return {
        "best_experiment": engine.metadata["best_experiment"],
        "experiments": engine.metadata["experiments"],
        "vectorizer_vocabulary_size": engine.metadata["vectorizer_vocabulary_size"],
        "svd_explained_variance": engine.metadata["svd_explained_variance"],
    }


@app.get("/api/sample")
def sample_payload() -> dict[str, object]:
    return {
        "customer_age": 34,
        "gender": "Female",
        "country": "US",
        "acquisition_channel": "Referral",
        "product_tier": "Pro",
        "tenure_months": 22,
        "monthly_spend": 189.0,
        "login_freq_monthly": 23,
        "feature_adoption": 0.71,
        "support_tickets_6m": 1,
        "nps_score": 9.0,
        "payment_failures_6m": 0,
        "referrals_made": 2,
        "days_since_login": 4,
        "customer_feedback": "Love the reporting workflow. The mobile app still needs work, but overall I would recommend it.",
    }


@app.post("/api/predict")
def predict(payload: CustomerInput) -> dict[str, object]:
    assert engine is not None
    return engine.predict(payload.model_dump())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(FRONTEND_DIR / "index.html"))

