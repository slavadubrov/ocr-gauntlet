# OCR Gauntlet — Claude Code Instructions

Simple standalone demo repo. Notebooks + minimal Python modules. No CLI framework.

## Stack
- Python 3.11+, uv
- Notebooks: Jupyter
- Viz: matplotlib, seaborn
- Metrics: editdistance, jiwer

## Structure
- `src/ocr_gauntlet/` — 4 files (engines, metrics, utils, visualize)
- `scripts/download_samples.py` — downloads 5 samples from HuggingFace at runtime
- `notebooks/` — 3 Jupyter notebooks
- `data/samples/` — created at runtime, NOT checked into git

## Rules
- Keep it simple. No abstractions, no registries, no factories.
- Each engine = plain function: image in → (text, metadata) out.
- Notebooks degrade gracefully: skip unavailable engines, never crash.
- API keys from environment variables only.
- Type hints everywhere. Format with ruff.

## Test
    uv run pytest
    uv run ruff check .
