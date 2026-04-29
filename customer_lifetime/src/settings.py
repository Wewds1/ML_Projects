from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_clv.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FRONTEND_DIR = ROOT_DIR / "frontend"

RANDOM_STATE = 42
TEXT_COMPONENTS = 30

VALID_PRODUCT_TIERS = ["Free", "Basic", "Pro", "Enterprise"]
PRODUCT_TIER_MAP = {name: idx for idx, name in enumerate(VALID_PRODUCT_TIERS)}
VALID_CHANNELS = [
    "Organic Search",
    "Paid Ads",
    "Referral",
    "Social Media",
    "Email Campaign",
    "Direct",
]
VALID_GENDERS = ["Male", "Female", "Non-Binary"]

