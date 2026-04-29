from __future__ import annotations

import argparse
import json

from .pipeline import save_artifacts, train_project_artifacts
from .settings import MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist CLV project artifacts.")
    parser.parse_args()

    artifacts = train_project_artifacts()
    save_artifacts(artifacts, MODELS_DIR)
    print(json.dumps(artifacts["metadata"]["experiments"], indent=2))
    print(f"Best experiment: {artifacts['metadata']['best_experiment']}")


if __name__ == "__main__":
    main()
