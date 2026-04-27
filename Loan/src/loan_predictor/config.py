from dataclasses import dataclass
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Loan Interest Rate Predictor"
    app_version: str = "1.0.0"
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "best_random_forest_pipeline.joblib"))
    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", ROOT_DIR / "data" / "raw" / "loan_applications.csv"))
    processed_data_path: Path = Path(
        os.getenv("PROCESSED_DATA_PATH", ROOT_DIR / "data" / "processed" / "loans_clean.csv")
    )
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
