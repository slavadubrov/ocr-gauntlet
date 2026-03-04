# 🥊 OCR Gauntlet

**5 tiers of OCR compared side-by-side** — from Tesseract to Gemini 3 Flash.

Companion repo for ["The Definitive Guide to OCR in 2026: From Pipelines to VLMs"](https://slavadubrov.github.io/blog/2026/03/04/the-definitive-guide-to-ocr-in-2026-from-pipelines-to-vlms/).

## What's Inside

| Notebook | Shows |
|----------|-------|
| `01_gauntlet.ipynb` | 5 OCR engines on the same documents — CER/WER/speed/cost |
| `02_docling_deep_dive.ipynb` | Docling for structured conversion (tables→DataFrames, RAG-ready markdown) |
| `03_cost_calculator.ipynb` | Cost analysis at 1K–10M pages/month |

## Quick Start

    git clone https://github.com/slavadubrov/ocr-gauntlet
    cd ocr-gauntlet

    # Minimal (Tesseract only — any machine, no GPU, no API keys)
    uv sync --extra tesseract --extra notebooks
    uv run python scripts/download_samples.py   # downloads 5 samples from HuggingFace
    uv run jupyter notebook notebooks/01_gauntlet.ipynb

    # Full gauntlet (all 5 tiers)
    uv sync --extra all
    uv run python scripts/download_samples.py
    cp .env.example .env   # add your API keys
    uv run jupyter notebook notebooks/01_gauntlet.ipynb

The notebook auto-downloads samples on first run if you skip the manual step.
It also runs whatever engines are available and skips the rest.

## Engine Setup

Each tier has different requirements. The notebook skips any engine that isn't available.

| Tier | What you need |
|------|---------------|
| 1 — Tesseract | `brew install tesseract` (or `apt install tesseract-ocr`) + `uv sync --extra tesseract` |
| 2 — Docling | `uv sync --extra docling` (+ Tesseract system dep) |
| 3 — dots.ocr | GPU with ~4GB VRAM + Python <=3.13. Run the vLLM server separately: `uvx --python 3.13 --from vllm vllm serve rednote-hilab/dots.ocr --trust-remote-code`. Then `uv sync --extra dots-ocr` for the client. |
| 4 — Mistral OCR | `MISTRAL_API_KEY` in `.env` + `uv sync --extra mistral` |
| 5 — Gemini Flash | `GEMINI_API_KEY` in `.env` + `uv sync --extra gemini` |

## The 5 Tiers

| Tier | Engine | Type | GPU? | Cost | OCR Arena ELO |
|------|--------|------|------|------|---------------|
| 1 | Tesseract | Traditional | No | Free | — |
| 2 | Docling + Tesseract | Framework | No | Free | — |
| 3 | dots.ocr (1.7B) | Lightweight VLM | ~4GB | Free | 1382 |
| 4 | Mistral OCR v3 | Dedicated OCR API | No | $2/1K pages | 1460 |
| 5 | Gemini 3 Flash | Frontier VLM | No | ~$0.8/1K pages | 1770 |

## Related

- 📖 [The Definitive Guide to OCR in 2026: From Pipelines to VLMs](https://slavadubrov.github.io/blog/2026/03/04/the-definitive-guide-to-ocr-in-2026-from-pipelines-to-vlms/)
- 🏟️ [OCR Arena](https://ocrarena.ai) — independent crowd-sourced rankings
- 🔧 Built with [uv](https://docs.astral.sh/uv/)

## Data Sources

Sample documents are downloaded at runtime from public HuggingFace datasets:

| Sample | Source | License |
|--------|--------|---------|
| Printed form | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | Non-commercial research |
| Receipt | [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | CORD — research use |
| Handwriting | [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | Non-commercial research |
| Dense form | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | Non-commercial research |
| Noisy form | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | Non-commercial research |

No dataset content is included in this repository.

## License

Apache-2.0
