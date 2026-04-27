from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {"venv", ".venv", ".git"}


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"[clean] removed directory: {path}")
    elif path.is_file():
        path.unlink(missing_ok=True)
        print(f"[clean] removed file: {path}")


def main() -> None:
    targets: list[Path] = []

    for path in ROOT.rglob("*"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}:
            targets.append(path)

    for target in sorted(set(targets)):
        remove_path(target)


if __name__ == "__main__":
    main()
