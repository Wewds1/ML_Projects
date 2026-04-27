# Cleanup Notes

## Why cleanup was added

The original repo included a local `venv/` folder and generated cache files, which are not portable and should not be part of a deployment story.

## What to run

```bash
python scripts/clean.py
```

## What this removes

- Python cache folders
- pytest cache
- notebook checkpoint folders

## What stays in the repo

- raw and processed datasets
- saved model artifact
- reports and portfolio assets
