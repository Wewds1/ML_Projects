from pathlib import Path

from .loan_predictor.config import settings
from .loan_predictor.features import load_raw, prepare_features


def main() -> None:
    output_path = Path(settings.processed_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw(str(settings.raw_data_path))
    processed = prepare_features(raw, training=True)
    processed.to_csv(output_path, index=False)

    print(f"[done] Saved processed data to {output_path}")
    print(f"[done] Final shape: {processed.shape}")


if __name__ == "__main__":
    main()
