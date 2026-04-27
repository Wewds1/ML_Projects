from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from .config import settings
from .features import prepare_features


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> Any:
    path = model_path or str(settings.model_path)
    return joblib.load(path)


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_features(df, training=False)
    model = load_model()
    predictions = model.predict(prepared)
    output = prepared.copy()
    output["predicted_interest_rate"] = predictions
    return output
