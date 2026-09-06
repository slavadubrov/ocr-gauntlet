# OCR Gauntlet

Compare OCR **pipelines and their output contracts** on the same reviewed inputs.
Tesseract, PaddleOCR, Docling with Tesseract/EasyOCR/Granite, dots.ocr,
Mistral OCR and Gemini are optional adapters. Failures remain in the results.

Companion to [OCR in 2026](https://slavadubrov.com/blog/2026/03/04/the-definitive-guide-to-ocr-in-2026-from-pipelines-to-vlms/).
This is an instructional evaluation project, not a validated model leaderboard.

## Start locally

Use Python 3.12 (selected by `.python-version`) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked --extra tesseract --extra notebooks
# macOS: brew install tesseract; Debian/Ubuntu: apt install tesseract-ocr
uv run python scripts/make_fixture.py data/demo
uv run ocr-gauntlet data/demo/manifest.json --engines tesseract --output results/demo.jsonl
uv run jupyter notebook notebooks/01_gauntlet.ipynb
```

The generated pages are original **synthetic wiring checks**. They do not measure
real-document OCR quality. Existing datasets/results are never overwritten; choose
a new output directory/file for a new run. API keys are not needed for this path.

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Real documents

The importer pins dataset commits and records sample IDs, split, preprocessing,
licenses and image/reference hashes. It does not run OCR or upload documents.

```sh
uv sync --locked --extra datasets --extra tesseract
uv run python scripts/download_samples.py --dataset cord --count 2 --output data/cord
# Reproduce the visually reviewed receipt region used during implementation:
uv run python scripts/prepare_reviewed_cord.py data/cord/manifest.json data/cord-region
uv run ocr-gauntlet data/cord-region/manifest.json --engines tesseract --output results/cord.jsonl
```

Use `--count` and `--start` for a larger deterministic slice. General imports are
**unreviewed** and remain unscored until their visible text, coverage and order have
been checked. CORD semantic annotations live in `fields`, never in text gold.
The prepared region is a one-region integration example, not the official CORD
protocol. [Evaluation and dataset choices](docs/evaluation.md) explains expansion,
IAM restrictions, OmniDocBench and reference review.

## Select a pipeline

| Engine argument | Extra / runtime | Output |
|---|---|---|
| `tesseract` | `tesseract` + system Tesseract/language files | Text |
| `paddle` | `paddle` + platform-specific PaddlePaddle | Text + native boxes/scores |
| `docling-tesseract` | `docling` + system Tesseract | Markdown + Docling JSON/tables |
| `docling-easyocr` | `docling` + EasyOCR runtime | Markdown + Docling JSON/tables |
| `docling-granite` | `docling`; pinned Granite-Docling weights | Markdown + Docling JSON/tables |
| `dots-ocr` | `dots-ocr`; separately deployed model server | Text-prompt experiment + raw response |
| `mistral` | `mistral`; `MISTRAL_API_KEY` | Markdown + raw page response |
| `gemini` | `gemini`; `GEMINI_API_KEY` | Text + usage/finish metadata |

Install only the extras you run, e.g. `uv sync --locked --extra tesseract --extra
paddle --extra docling`. For PaddlePaddle follow the [official installation
instructions](https://www.paddleocr.ai/latest/en/quick_start.html); its runtime is
platform-specific. The tested CPU runtime is PaddlePaddle 3.3.1 on macOS ARM64. Local
PaddleOCR/Docling runs can download weights on first use. The default Paddle
candidate is the explicitly selected **PP-OCRv5** baseline, not an implied latest
pipeline; change `ocr_version` only as a new experiment. Do not equate PaddleOCR
recognition with the separate PaddleOCR-VL document parsing pipeline.

```sh
uv sync --locked --extra gemini --extra mistral
cp .env.example .env  # fill keys locally
uv run --env-file .env ocr-gauntlet data/cord-region/manifest.json \
  --engines gemini mistral --allow-remote --output results/hosted.jsonl
```

`--allow-remote` explicitly enables document uploads and possible spend **only for
selected engines**. It is also required for dots.ocr, whose endpoint can be remote.
The notebooks default to local/offline operation. `.env` is loaded only with the
shown `uv run --env-file` command; restart the Jupyter process with that command
when changing its environment. Never commit keys or raw private documents.

Gemini defaults to **`gemini-3.8-flash`**, verified against [Google's model
card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) on 2026-09-06.
Mistral defaults to [`mistral-ocr-4-1`](https://docs.mistral.ai/models/ocr-4-1).
Neither is declared an OCR winner. Request/response IDs, returned model versions,
usage and dated estimated pricing remain attached to each result. Unknown usage
is not free. Thinking tokens count toward Gemini output cost.

Use a JSON settings file to vary real adapter options:

```json
{
  "tesseract": {"lang": "eng", "psm": 6},
  "gemini": {"model": "gemini-3.8-flash", "media_resolution": "high"}
}
```

Pass `--settings settings.json` with exactly those engines selected. No settings
are silently ignored. For two configurations of one engine, create two separate
runs and compare their identical manifest hashes. See [runtime details](docs/evaluation.md#runtime-and-model-identity)
for the dots.ocr deployment contract and historical model limitations.

## Read the results

Every planned pair gets a JSONL row: success/error/skipped, reason, hashes, raw
text/native output, configuration, timing, usage and known/unknown cost.
The CLI prints a summary from these records:

- **Completion and scoring coverage first.** Unavailable/truncated/failed pages stay visible.
- **Corpus CER/WER** apply only to compatible reviewed references. Case, numbers
  and punctuation are preserved; whitespace is collapsed and Unicode is NFC.
- **All-planned text quality** gives failed/skipped/unscored pages zero contribution.
  Compare it only on identical inputs and the same output contract.
- **Fields, table cells and reading order** are separate tasks. Unsupported outputs
  are explicit. No Markdown stripping or field guessing is used to manufacture a score.
  Docling can explicitly project native blocks and table cells with
  `{"docling-tesseract": {"output_format": "text"}}`; raw Markdown remains saved.
- **Cost per success** includes known failed spend. Any unknown attempted spend
  makes the overall cost per success unknown. Local zero means API fee only.

| Notebook | Purpose |
|---|---|
| `01_gauntlet.ipynb` | Shared runner, completion, conditional heatmap, raw outputs |
| `02_docling_deep_dive.ipynb` | Explicit pipeline selection, cold/warm timing, tables and coordinates |
| `03_cost_calculator.ipynb` | Hypothetical capacity/retries/review costs; critical-field/order counterexamples |

The previous notebooks contained invalid labels/rankings. Their original content
remains in Git at `deb94ebd747f0e7cc14b523ac3d0ad19032f855a`; current notebooks have
no historical outputs relabeled as new measurements.

Code license: Apache-2.0. Dataset/model licenses are separate; no downloaded dataset
content is distributed in the repository.
