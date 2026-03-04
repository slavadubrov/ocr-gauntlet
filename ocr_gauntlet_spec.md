# OCR Gauntlet: Implementation Spec

## Overview

A simple, standalone demo repo that runs the same documents through 5 tiers of OCR and shows results side-by-side. No framework, no CLI, no complex abstractions — just notebooks and minimal Python modules. The repo is the hands-on companion to the blog article "The Definitive Guide to OCR in 2025–2026."

**Repo name:** `ocr-gauntlet`

**Goal:** Reader clones the repo, runs `uv sync`, opens a notebook, and sees exactly where traditional OCR breaks down and VLMs take over — with real numbers on quality, speed, and cost.

---

## Repo Structure

```
ocr-gauntlet/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example              # API key template
├── .gitignore
├── CLAUDE.md                 # Claude Code instructions
│
├── data/
│   └── samples/              # Downloaded at runtime — NOT checked into git
│       └── .gitkeep
│
├── scripts/
│   └── download_samples.py   # Downloads 5 sample images from HuggingFace
│
├── notebooks/
│   ├── 01_gauntlet.ipynb               # Main demo: 5 tiers compared
│   ├── 02_docling_deep_dive.ipynb      # Docling for RAG pipelines
│   └── 03_cost_calculator.ipynb        # Cost analysis at different scales
│
└── src/
    └── ocr_gauntlet/
        ├── __init__.py
        ├── engines.py          # All engine wrappers in ONE file
        ├── metrics.py          # CER, WER, ANLS — simple functions
        ├── utils.py            # Image loading, display helpers
        └── visualize.py        # Charts and comparison tables
```

That's it. No CLI, no configs directory, no dataset downloaders, no registries, no factories. Just modules that the notebooks import.

---

## pyproject.toml

```toml
[project]
name = "ocr-gauntlet"
version = "0.1.0"
description = "5 tiers of OCR compared: from Tesseract to Gemini 3 Flash"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "Slava Dubrov" }]

dependencies = [
    "Pillow>=10.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "tqdm>=4.65",
    "rich>=13.0",
    "editdistance>=0.8",
    "jiwer>=3.0",           # WER computation
    "datasets>=2.14",       # HuggingFace datasets for sample downloads
]

[project.optional-dependencies]
tesseract = ["pytesseract>=0.3.10"]
docling = ["docling>=2.70"]
gemini = ["google-generativeai>=0.9.0"]
mistral = ["mistralai>=1.0.0"]
dots-ocr = ["openai>=1.0"]    # dots.ocr via vLLM OpenAI-compatible API
notebooks = ["jupyter>=1.0", "ipywidgets>=8.0"]
all = ["ocr-gauntlet[tesseract,docling,gemini,mistral,dots-ocr,notebooks]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ocr_gauntlet"]

[dependency-groups]
dev = ["pytest>=7.0", "ruff>=0.4"]
```

---

## src/ocr_gauntlet/engines.py

All engines in one flat file. Each engine is a simple function: image in, (text, metadata) out.

```python
"""
OCR engine wrappers. Each function takes a PIL Image and returns:
    (extracted_text: str, metadata: dict)

metadata contains: latency_ms, engine_name, and engine-specific fields
(tokens, cost_estimate, etc.)
"""
from __future__ import annotations
import base64
import io
import os
import time
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────
# Tier 1: Tesseract (Traditional, CPU, Free)
# ─────────────────────────────────────────────

def run_tesseract(image: Image.Image, lang: str = "eng") -> tuple[str, dict]:
    """Run Tesseract OCR on a PIL Image.
    
    Requires: pip install pytesseract
    System dep: tesseract-ocr (apt install tesseract-ocr / brew install tesseract)
    """
    import pytesseract
    
    start = time.perf_counter()
    text = pytesseract.image_to_string(image, lang=lang)
    elapsed = (time.perf_counter() - start) * 1000
    
    return text.strip(), {
        "engine": "Tesseract",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
    }


# ─────────────────────────────────────────────
# Tier 2: Docling + Tesseract (Framework, CPU, Free)
# ─────────────────────────────────────────────

def run_docling(
    image_path: str | Path,
    ocr_backend: str = "tesseract",
    use_vlm: bool = False,
) -> tuple[str, dict]:
    """Run Docling document converter on an image or PDF.
    
    Requires: pip install docling
    
    Returns markdown text (Docling's native structured output).
    Docling adds layout understanding, reading order, and table structure
    on top of the raw OCR engine.
    
    Args:
        image_path: Path to image or PDF file.
        ocr_backend: "tesseract" or "easyocr".
        use_vlm: If True, use Granite-Docling-258M VLM mode (~8GB VRAM).
    """
    from docling.document_converter import DocumentConverter, ImageFormatOption
    
    start = time.perf_counter()
    
    if use_vlm:
        converter = DocumentConverter()
    else:
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        image_options = ImageFormatOption(
            ocr_options=ocr_options,
            do_table_structure=True,
        )
        converter = DocumentConverter(format_options={"image": image_options})
    
    result = converter.convert(str(image_path))
    markdown_text = result.document.export_to_markdown()
    elapsed = (time.perf_counter() - start) * 1000
    
    tables = []
    for table in result.document.tables:
        try:
            tables.append(table.export_to_dataframe())
        except Exception:
            pass
    
    return markdown_text.strip(), {
        "engine": "Docling (VLM)" if use_vlm else f"Docling + {ocr_backend}",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
        "tables_found": len(tables),
        "tables": tables,
        "output_format": "markdown",
    }


# ─────────────────────────────────────────────
# Tier 3: dots.ocr (Lightweight VLM, 1.7B, Free)
# ─────────────────────────────────────────────

def run_dots_ocr(
    image: Image.Image,
    base_url: str = "http://localhost:8000/v1",
    model: str = "rednote-hilab/dots.ocr",
    api_key: str = "token-abc123",
) -> tuple[str, dict]:
    """Run dots.ocr via a vLLM OpenAI-compatible server.
    
    Requires:
      - A running vLLM server: vllm serve rednote-hilab/dots.ocr
      - pip install openai
    
    dots.ocr is a 1.7B VLM achieving SOTA on OmniDocBench.
    Supports 100+ languages.
    """
    from openai import OpenAI
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": (
                    "Extract all text from this document image. "
                    "Return only the extracted text, preserving the original "
                    "layout and reading order."
                )},
            ],
        }],
        max_tokens=4096,
        temperature=0.0,
    )
    elapsed = (time.perf_counter() - start) * 1000
    text = response.choices[0].message.content or ""
    
    usage = response.usage
    return text.strip(), {
        "engine": "dots.ocr (1.7B)",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 0.0,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


# ─────────────────────────────────────────────
# Tier 4: Mistral OCR v3 (Dedicated OCR API)
# ─────────────────────────────────────────────

def run_mistral_ocr(
    image: Image.Image,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Run Mistral OCR v3 via the dedicated /v1/ocr endpoint.
    
    Requires: pip install mistralai, env MISTRAL_API_KEY
    
    Mistral OCR is a specialized OCR model (not a general VLM).
    Excels at forms, handwriting (88.9%), and tables (96.6%).
    Cost: $2/1K pages ($1/1K with batch API).
    """
    from mistralai import Mistral
    
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
    
    client = Mistral(api_key=api_key)
    
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    start = time.perf_counter()
    response = client.ocr.process(
        model="mistral-ocr-2512",
        document={"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    text_parts = [page.markdown for page in response.pages]
    full_text = "\n".join(text_parts)
    
    return full_text.strip(), {
        "engine": "Mistral OCR v3",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": 2.0,
        "output_format": "markdown",
    }


# ─────────────────────────────────────────────
# Tier 5: Gemini 3 Flash (Frontier VLM API)
# ─────────────────────────────────────────────

def run_gemini(
    image: Image.Image,
    api_key: str | None = None,
    model: str = "gemini-3-flash",
    media_resolution: str = "medium",
) -> tuple[str, dict]:
    """Run Gemini 3 Flash via Google Generative AI SDK.
    
    Requires: pip install google-generativeai, env GEMINI_API_KEY
    
    #1 on OCR Arena (ELO 1770, 77.2% win rate).
    Cost: $0.50/M input, $3.00/M output tokens.
    
    media_resolution: Google says OCR quality saturates at "medium".
    """
    import google.generativeai as genai
    
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    
    prompt = (
        "Extract all text from this document image. "
        "Return only the extracted text, preserving the original "
        "layout and reading order. "
        "Do not add any explanations or formatting beyond what is "
        "in the original document."
    )
    
    start = time.perf_counter()
    response = gen_model.generate_content(
        [prompt, image],
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    text = response.text or ""
    
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0
    cost = (input_tokens * 0.50 + output_tokens * 3.00) / 1_000_000
    
    return text.strip(), {
        "engine": "Gemini 3 Flash",
        "latency_ms": round(elapsed, 1),
        "cost_per_1k_pages": round(cost * 1000, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_this_page": round(cost, 6),
    }


# ─────────────────────────────────────────────
# Availability checker
# ─────────────────────────────────────────────

def check_available_engines() -> dict[str, bool]:
    """Check which engines can run on this system."""
    available = {}
    
    # Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        available["tesseract"] = True
    except Exception:
        available["tesseract"] = False
    
    # Docling
    try:
        import docling  # noqa: F401
        available["docling"] = True
    except ImportError:
        available["docling"] = False
    
    # dots.ocr (check if vLLM server is reachable)
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
        client.models.list()
        available["dots_ocr"] = True
    except Exception:
        available["dots_ocr"] = False
    
    # Mistral OCR
    available["mistral_ocr"] = bool(os.environ.get("MISTRAL_API_KEY"))
    
    # Gemini
    available["gemini"] = bool(os.environ.get("GEMINI_API_KEY"))
    
    return available
```

---

## src/ocr_gauntlet/metrics.py

```python
"""
Simple metric functions. Each: (prediction, ground_truth) -> float.
"""
import editdistance


def cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate. Lower is better. Range: [0, inf)."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else float("inf")
    return editdistance.eval(prediction, ground_truth) / len(ground_truth)


def wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate. Lower is better."""
    from jiwer import wer as jiwer_wer
    if not ground_truth.strip():
        return 0.0 if not prediction.strip() else float("inf")
    return jiwer_wer(ground_truth, prediction)


def anls(prediction: str, ground_truth: str, tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity. Higher is better. Range: [0, 1]."""
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 1.0
    max_len = max(len(prediction), len(ground_truth))
    if max_len == 0:
        return 1.0
    nls = 1.0 - editdistance.eval(prediction, ground_truth) / max_len
    return nls if nls >= tau else 0.0


def normalized_edit_distance(prediction: str, ground_truth: str) -> float:
    """Normalized Edit Distance. Lower is better. Range: [0, 1]."""
    max_len = max(len(prediction), len(ground_truth))
    if max_len == 0:
        return 0.0
    return editdistance.eval(prediction, ground_truth) / max_len
```

---

## src/ocr_gauntlet/utils.py

```python
"""Image loading, display, and text normalization helpers."""
from pathlib import Path
from PIL import Image
import re

SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"


def ensure_samples() -> Path:
    """Download samples if not already present. Returns samples dir."""
    manifest = SAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        print("📥 Samples not found — downloading from HuggingFace...")
        # Import here to avoid circular deps at module level
        import subprocess, sys
        script = Path(__file__).parent.parent.parent / "scripts" / "download_samples.py"
        subprocess.run([sys.executable, str(script)], check=True)
    return SAMPLES_DIR


def load_sample(name: str) -> tuple[Image.Image, str, Path]:
    """Load a sample image and its ground truth.
    Auto-downloads samples on first call if missing.
    Returns: (image, ground_truth_text, image_path)
    """
    ensure_samples()
    img_path = SAMPLES_DIR / f"{name}.png"
    gt_path = SAMPLES_DIR / f"{name}.txt"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Sample not found: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    
    return Image.open(img_path), gt_path.read_text(encoding="utf-8").strip(), img_path


def list_samples() -> list[str]:
    """List available sample names (excludes samples without ground truth)."""
    ensure_samples()
    samples = []
    for p in sorted(SAMPLES_DIR.glob("*.png")):
        gt = p.with_suffix(".txt")
        if gt.exists():
            text = gt.read_text(encoding="utf-8").strip()
            if not text.startswith("[NO_GROUND_TRUTH"):
                samples.append(p.stem)
    return samples


def list_all_samples() -> list[str]:
    """List ALL sample names including visual-only ones."""
    ensure_samples()
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.png"))


def normalize_text(text: str) -> str:
    """Normalize for fair metric comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
```

---

## src/ocr_gauntlet/visualize.py

```python
"""Visualization helpers for comparing OCR results."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TIER_COLORS = {
    "Tesseract": "#6B7280",
    "Docling + tesseract": "#3B82F6",
    "dots.ocr (1.7B)": "#8B5CF6",
    "Mistral OCR v3": "#F59E0B",
    "Gemini 3 Flash": "#10B981",
}


def results_heatmap(df: pd.DataFrame, metric: str = "cer"):
    """Heatmap: tiers (rows) x documents (columns), colored by metric."""
    pivot = df.pivot(index="engine", columns="document", values=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = "RdYlGn_r" if metric in ("cer", "wer") else "RdYlGn"
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5, ax=ax)
    ax.set_title(f"OCR {metric.upper()} by Engine x Document")
    plt.tight_layout()
    return fig


def quality_cliff_chart(df: pd.DataFrame, metric: str = "cer"):
    """Line chart showing where traditional OCR falls apart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        edf = df[df["engine"] == engine]
        color = TIER_COLORS.get(engine, "#999")
        ax.plot(edf["document"], edf[metric], marker="o", label=engine, color=color, linewidth=2)
    ax.set_xlabel("Document (easy → hard)")
    ax.set_ylabel(metric.upper())
    ax.set_title("The Quality Cliff: Where Traditional OCR Breaks Down")
    ax.legend(loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def speed_vs_quality(df: pd.DataFrame, metric: str = "cer"):
    """Scatter: latency (x, log) vs quality (y). Points sized by cost."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        edf = df[df["engine"] == engine]
        color = TIER_COLORS.get(engine, "#999")
        cost = edf["cost_per_1k_pages"].iloc[0]
        size = max(50, cost * 100) if cost > 0 else 50
        ax.scatter(edf["latency_ms"], edf[metric], label=engine, color=color, s=size, alpha=0.7, edgecolors="white")
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel(metric.upper())
    ax.set_title("Speed vs Quality Tradeoff")
    ax.legend()
    plt.tight_layout()
    return fig


def cost_table(pages_per_month: int, avg_metrics: dict[str, float]) -> pd.DataFrame:
    """Cost comparison table for a given monthly volume."""
    tiers = [
        ("Tesseract", 0.0, "CPU only"),
        ("Docling + tesseract", 0.0, "CPU only"),
        ("dots.ocr (1.7B)", 0.0, "~$0.50/hr GPU"),
        ("Mistral OCR v3", 2.0, "$2/1K pages"),
        ("Gemini 3 Flash", 0.8, "~$0.50/M input tokens"),
    ]
    rows = []
    for name, cost_1k, note in tiers:
        monthly = (pages_per_month / 1000) * cost_1k
        avg_cer = avg_metrics.get(name, float("nan"))
        rows.append({
            "Engine": name,
            "Avg CER": f"{avg_cer:.3f}" if avg_cer == avg_cer else "N/A",
            "Monthly Cost": f"${monthly:,.0f}" if monthly > 0 else "Free",
            "Pricing": note,
        })
    return pd.DataFrame(rows)
```

---

## Notebook 1: notebooks/01_gauntlet.ipynb

The main demo. Implement as an actual .ipynb file with these cells:

### Cell 1 — Setup
```python
# 🥊 The OCR Gauntlet: 5 Tiers of OCR Compared
# Companion to "The Definitive Guide to OCR in 2025-2026"

import warnings; warnings.filterwarnings("ignore")

from ocr_gauntlet.engines import check_available_engines
from ocr_gauntlet.utils import load_sample, list_samples, list_all_samples, normalize_text, ensure_samples
from ocr_gauntlet.metrics import cer, wer, anls
from ocr_gauntlet.visualize import results_heatmap, quality_cliff_chart, speed_vs_quality, cost_table

# Download samples from HuggingFace if not already present
ensure_samples()

# Check what engines can run
engines = check_available_engines()
print("\nAvailable engines:")
for name, ok in engines.items():
    print(f"  {name}: {'✅' if ok else '❌ (skipped)'}")
```

### Cell 2 — Load samples
```python
import matplotlib.pyplot as plt

# list_samples() returns only samples WITH ground truth (for metrics)
# list_all_samples() includes visual-only samples too
samples = list_samples()
all_samples = list_all_samples()

print(f"Samples with ground truth (for metrics): {len(samples)}")
print(f"All samples (including visual-only):     {len(all_samples)}\n")

fig, axes = plt.subplots(1, len(all_samples), figsize=(4 * len(all_samples), 5))
if len(all_samples) == 1: axes = [axes]

for ax, name in zip(axes, all_samples):
    img_path = f"data/samples/{name}.png"
    from PIL import Image as PILImage
    img = PILImage.open(img_path)
    ax.imshow(img)
    label = name.replace("_", " ").title()
    if name not in samples:
        label += "\n(visual only)"
    ax.set_title(label, fontsize=10)
    ax.axis("off")

plt.suptitle("The OCR Gauntlet: Test Documents", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()
```

### Cell 3 — Run the gauntlet
```python
import pandas as pd
from tqdm.notebook import tqdm
from ocr_gauntlet.engines import run_tesseract, run_docling, run_dots_ocr, run_mistral_ocr, run_gemini

TIERS = [
    ("Tesseract",           run_tesseract,   {},                   "tesseract"),
    ("Docling + tesseract", run_docling,     {"needs_path": True}, "docling"),
    ("dots.ocr (1.7B)",     run_dots_ocr,    {},                   "dots_ocr"),
    ("Mistral OCR v3",      run_mistral_ocr, {},                   "mistral_ocr"),
    ("Gemini 3 Flash",      run_gemini,      {},                   "gemini"),
]

results = []
raw_outputs = {}

for tier_name, runner, kwargs, requires in tqdm(TIERS, desc="Tiers"):
    if not engines.get(requires, False):
        print(f"⏭️  Skipping {tier_name}"); continue
    
    for sample_name in tqdm(samples, desc=tier_name, leave=False):
        img, gt, img_path = load_sample(sample_name)
        try:
            if kwargs.get("needs_path"):
                text, meta = runner(image_path=img_path)
            else:
                text, meta = runner(img)
            
            pred_norm = normalize_text(text)
            gt_norm = normalize_text(gt)
            
            results.append({
                "engine": tier_name, "document": sample_name,
                "cer": cer(pred_norm, gt_norm), "wer": wer(pred_norm, gt_norm),
                "anls": anls(pred_norm, gt_norm), "latency_ms": meta["latency_ms"],
                "cost_per_1k_pages": meta.get("cost_per_1k_pages", 0),
            })
            raw_outputs[(tier_name, sample_name)] = text
        except Exception as e:
            print(f"   ⚠️ {tier_name} x {sample_name}: {e}")

df = pd.DataFrame(results)
print(f"\n✅ {len(df)} results from {df['engine'].nunique()} engines")
```

### Cell 4 — Heatmap
```python
fig = results_heatmap(df); plt.show()
```

### Cell 5 — Quality cliff
```python
fig = quality_cliff_chart(df); plt.show()
```

### Cell 6 — Speed vs quality
```python
fig = speed_vs_quality(df); plt.show()
```

### Cell 7 — Side-by-side text for hardest document
```python
hardest = df.groupby("document")["cer"].mean().idxmax()
print(f"Hardest document: {hardest}\n{'='*80}")
for tier_name, *_ in TIERS:
    key = (tier_name, hardest)
    if key in raw_outputs:
        print(f"\n--- {tier_name} ---")
        print(raw_outputs[key][:500])
```

### Cell 8 — Cost calculator
```python
avg_cers = df.groupby("engine")["cer"].mean().to_dict()
for vol in [1_000, 10_000, 100_000, 1_000_000]:
    print(f"\n{'='*60}\n  📊 {vol:,} pages/month\n{'='*60}")
    display(cost_table(vol, avg_cers))
```

### Cell 9 — Summary
```python
best = df.loc[df.groupby("document")["cer"].idxmin()][["document", "engine", "cer"]]
print("🏆 Best engine per document type:")
display(best)
print("\n📖 Full guide: [blog link]")
print("🏟️ Live rankings: https://ocrarena.ai")
```

---

## Notebook 2: notebooks/02_docling_deep_dive.ipynb

### Cell 1 — Intro
```python
# Docling Deep Dive: Document Intelligence for RAG Pipelines
#
# Docling is NOT an OCR engine — it's a document CONVERTER.
# It uses OCR internally but adds:
#   - Layout detection (columns, headers, footers, figures)
#   - Reading order (which text block comes first?)
#   - Table structure (rows, columns, merged cells → DataFrame)
#   - Multi-format support (PDF, DOCX, PPTX, images, HTML)
#   - Export to Markdown, JSON, HTML for downstream LLM/RAG use

from ocr_gauntlet.utils import ensure_samples
ensure_samples()  # download if needed
```

### Cell 2 — Basic: 3 lines of code
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("data/samples/04_document_page.png")

print(result.document.export_to_markdown()[:1000])
```

### Cell 3 — Explore the structure
```python
import json

doc = result.document
print(f"Tables: {len(doc.tables)}, Figures: {len(doc.pictures)}")
print(json.dumps(doc.export_to_dict(), indent=2)[:2000])
```

### Cell 4 — Tables → DataFrames
```python
for i, table in enumerate(result.document.tables):
    print(f"\n📋 Table {i+1}:")
    display(table.export_to_dataframe())
```

### Cell 5 — OCR on scanned images
```python
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.pipeline_options import TesseractCliOcrOptions

ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
image_options = ImageFormatOption(ocr_options=ocr_options, do_table_structure=True)
converter = DocumentConverter(format_options={"image": image_options})

result = converter.convert("data/samples/02_receipt.png")
print(result.document.export_to_markdown()[:500])
```

### Cell 6 — Raw Tesseract vs Docling+Tesseract
```python
import pytesseract
from PIL import Image

img = Image.open("data/samples/02_receipt.png")

raw = pytesseract.image_to_string(img)
structured = result.document.export_to_markdown()

print("=== Raw Tesseract ===")
print(raw[:500])
print("\n=== Docling + Tesseract ===")
print(structured[:500])

# Key: Docling preserves structure and reading order.
# For RAG pipelines, Docling output is dramatically more useful.
```

### Cell 7 — When to use what
```python
# ┌────────────────────────────────┬─────────────────────────────┐
# │ Use Docling when...            │ Use raw VLM API when...     │
# ├────────────────────────────────┼─────────────────────────────┤
# │ You need markdown/JSON for RAG │ You need max OCR accuracy   │
# │ Mixed formats (PDF+DOCX+PPTX) │ Single document type        │
# │ Tables must become DataFrames  │ Plain text extraction only  │
# │ You need reading order         │ Speed matters most          │
# │ On-premise / air-gapped        │ Cloud APIs are fine         │
# │ Budget is zero (CPU-only)      │ Budget allows API costs     │
# └────────────────────────────────┴─────────────────────────────┘
#
# Best combo: Docling for structure + Gemini 3 Flash for hard pages.
print("Docling is infrastructure for RAG. VLMs are accuracy engines.")
```

---

## Notebook 3: notebooks/03_cost_calculator.ipynb

Short notebook: call `cost_table()` at 1K / 10K / 100K / 1M / 10M pages. Add a break-even chart (matplotlib) showing where self-hosting crosses API pricing.

---

## scripts/download_samples.py

Downloads 5 curated sample images + ground truth from public HuggingFace datasets at runtime. Nothing is checked into the repo — the user runs this once (or the notebook calls it automatically on first run).

```python
"""
Download 5 sample documents from HuggingFace for the OCR Gauntlet demo.

Usage:
    python scripts/download_samples.py
    
Or called automatically by the notebook on first run.

Datasets used (all publicly available for research):
    - nielsr/funsd          — scanned forms (from RVL-CDIP / tobacco docs)
    - Teklia/IAM-line       — handwritten English text lines
    - darentang/sroie       — scanned receipts (ICDAR 2019)
    - ds4sd/DocLayNet       — diverse document layouts (IBM, Apache 2.0)

These datasets are downloaded by the user at runtime. We do NOT redistribute
any dataset content in this repository. Users are responsible for complying
with each dataset's license terms.
"""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


def download_samples(force: bool = False) -> Path:
    """Download 5 sample images + ground truth to data/samples/.
    
    Returns the samples directory path.
    Skips download if samples already exist (unless force=True).
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest_path = SAMPLES_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"✅ Samples already downloaded at {SAMPLES_DIR}")
        return SAMPLES_DIR
    
    manifest = {}
    
    # ── Sample 1: Clean form (FUNSD — clean, printed, structured) ──
    print("📥 Downloading FUNSD form...")
    funsd = load_dataset("nielsr/funsd", split="test")
    # Pick a sample with moderate text density
    sample = funsd[0]
    img = sample["image"]
    # Extract ground truth: concatenate all word-level text annotations
    words = sample["words"]
    gt_text = " ".join(words)
    
    img.save(SAMPLES_DIR / "01_printed_form.png")
    (SAMPLES_DIR / "01_printed_form.txt").write_text(gt_text, encoding="utf-8")
    manifest["01_printed_form"] = {
        "source": "nielsr/funsd (test split, index 0)",
        "license": "Non-commercial research only",
        "description": "Scanned printed form — clean, structured layout",
    }
    
    # ── Sample 2: Receipt (SROIE — noisy, real-world) ──
    print("📥 Downloading SROIE receipt...")
    sroie = load_dataset("darentang/sroie", split="test")
    # Pick a receipt with visible noise/distortion
    sample = sroie[5]
    img = sample["image"]
    gt_text = sample.get("ground_truth", "") or sample.get("text", "")
    # If ground truth is JSON (key fields), convert to plain text
    if gt_text.startswith("{"):
        fields = json.loads(gt_text)
        gt_text = "\n".join(f"{k}: {v}" for k, v in fields.items())
    
    img.save(SAMPLES_DIR / "02_receipt.png")
    (SAMPLES_DIR / "02_receipt.txt").write_text(gt_text, encoding="utf-8")
    manifest["02_receipt"] = {
        "source": "darentang/sroie (test split, index 5)",
        "license": "ICDAR 2019 competition — research use",
        "description": "Scanned receipt — noisy, thermal print artifacts",
    }
    
    # ── Sample 3: Handwritten text (IAM) ──
    print("📥 Downloading IAM handwriting...")
    iam = load_dataset("Teklia/IAM-line", split="test")
    # Pick a line with moderate handwriting complexity
    sample = iam[10]
    img = sample["image"]
    gt_text = sample["text"]
    
    img.save(SAMPLES_DIR / "03_handwritten.png")
    (SAMPLES_DIR / "03_handwritten.txt").write_text(gt_text, encoding="utf-8")
    manifest["03_handwritten"] = {
        "source": "Teklia/IAM-line (test split, index 10)",
        "license": "Non-commercial research only (IAM Handwriting Database)",
        "description": "Handwritten English text line",
    }
    
    # ── Sample 4: Dense document page (DocLayNet — Apache 2.0!) ──
    print("📥 Downloading DocLayNet page...")
    doclay = load_dataset("ds4sd/DocLayNet", split="test", streaming=True)
    # Stream to avoid downloading the full 30GB dataset
    # Pick a page with mixed content (text + table)
    for i, sample in enumerate(doclay):
        img = sample["image"]
        # DocLayNet doesn't have OCR ground truth, but the images are
        # real documents. We use this for visual comparison only;
        # metrics are computed on the other 4 samples.
        if img.size[0] > 500 and img.size[1] > 500:  # skip tiny thumbnails
            break
        if i > 50:  # safety limit
            break
    
    img.save(SAMPLES_DIR / "04_document_page.png")
    # For DocLayNet we don't have text GT — mark as visual-only
    (SAMPLES_DIR / "04_document_page.txt").write_text(
        "[NO_GROUND_TRUTH — used for visual comparison only]",
        encoding="utf-8",
    )
    manifest["04_document_page"] = {
        "source": "ds4sd/DocLayNet (test split, streamed)",
        "license": "Apache 2.0 (IBM)",
        "description": "Dense document page with mixed layout — visual comparison only",
    }
    
    # ── Sample 5: Second handwritten sample or different FUNSD form ──
    print("📥 Downloading second form sample...")
    sample = funsd[10]  # different form
    img = sample["image"]
    words = sample["words"]
    gt_text = " ".join(words)
    
    img.save(SAMPLES_DIR / "05_noisy_form.png")
    (SAMPLES_DIR / "05_noisy_form.txt").write_text(gt_text, encoding="utf-8")
    manifest["05_noisy_form"] = {
        "source": "nielsr/funsd (test split, index 10)",
        "license": "Non-commercial research only",
        "description": "Noisier scanned form — degraded scan quality",
    }
    
    # ── Save manifest ──
    (SAMPLES_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    
    print(f"\n✅ Downloaded {len(manifest)} samples to {SAMPLES_DIR}")
    print("📋 Sources and licenses saved to manifest.json")
    return SAMPLES_DIR


if __name__ == "__main__":
    download_samples()
```

**Key design decisions:**

- **DocLayNet is Apache 2.0** (IBM) — the only dataset with a permissive license. Great for the "document page" sample. However it lacks OCR ground truth text, so it's used for visual output comparison only, not for CER/WER metrics.
- **FUNSD, IAM, SROIE** are research-only — but since the user downloads them at runtime (not us redistributing), this is standard practice. Every OCR paper on HuggingFace does this.
- **Streaming for DocLayNet** — the full dataset is ~30GB. We stream and grab just one image to avoid downloading everything.
- **manifest.json** tracks exactly which sample came from where, with license info. Good for reproducibility and attribution.
- **Idempotent** — skips download if `manifest.json` already exists (unless `force=True`).

**HuggingFace dataset specifics to verify during implementation:**

The exact field names vary per dataset. The script above uses approximate field names. During implementation, Claude Code should:
1. Run `load_dataset(name, split="test")` and inspect `dataset[0].keys()` to find the correct field names for image and ground truth text
2. FUNSD stores annotations as word-level bounding boxes + text — need to concatenate words
3. SROIE may store ground truth as JSON with key fields (company, date, address, total) — flatten to text
4. IAM-line has `image` and `text` fields directly
5. DocLayNet has `image` but no OCR text — it's a layout dataset

**If any dataset is unavailable or changes:** The notebook should catch download errors and print a helpful message telling the user which dataset failed, rather than crashing. The gauntlet can run with whatever samples are available.

---

## .env.example

```bash
# Copy to .env and fill in your keys.
# Only needed for Tiers 4 & 5. Tiers 1-3 work without any keys.
GEMINI_API_KEY=your-gemini-api-key-here
MISTRAL_API_KEY=your-mistral-api-key-here
```

---

## .gitignore (important entries)

```gitignore
# Downloaded samples — never commit dataset content
data/samples/*.png
data/samples/*.txt
data/samples/manifest.json

# Keep the directory itself
!data/samples/.gitkeep

# Standard Python
__pycache__/
*.egg-info/
.venv/
dist/
.env
```

---

## README.md

```markdown
# 🥊 OCR Gauntlet

**5 tiers of OCR compared side-by-side** — from Tesseract to Gemini 3 Flash.

Companion repo for ["The Definitive Guide to OCR in 2025–2026"](link-to-blog).

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
    python scripts/download_samples.py          # downloads 5 samples from HuggingFace
    uv run jupyter notebook notebooks/01_gauntlet.ipynb

    # Full gauntlet (all 5 tiers)
    uv sync --extra all
    python scripts/download_samples.py
    cp .env.example .env   # add your API keys
    uv run jupyter notebook notebooks/01_gauntlet.ipynb

The notebook auto-downloads samples on first run if you skip the manual step.
It also runs whatever engines are available and skips the rest.

## The 5 Tiers

| Tier | Engine | Type | GPU? | Cost | OCR Arena ELO |
|------|--------|------|------|------|---------------|
| 1 | Tesseract | Traditional | No | Free | — |
| 2 | Docling + Tesseract | Framework | No | Free | — |
| 3 | dots.ocr (1.7B) | Lightweight VLM | ~4GB | Free | 1382 |
| 4 | Mistral OCR v3 | Dedicated OCR API | No | $2/1K pages | 1460 |
| 5 | Gemini 3 Flash | Frontier VLM | No | ~$0.8/1K pages | 1770 |

## Related

- 📖 [The Definitive Guide to OCR in 2025–2026](link-to-blog)
- 🏟️ [OCR Arena](https://ocrarena.ai) — independent crowd-sourced rankings
- 🔧 Built with [uv](https://docs.astral.sh/uv/)

## Data Sources

Sample documents are downloaded at runtime from public HuggingFace datasets:

| Sample | Source | License |
|--------|--------|---------|
| Printed form | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | Non-commercial research |
| Receipt | [darentang/sroie](https://huggingface.co/datasets/darentang/sroie) | ICDAR 2019 — research use |
| Handwriting | [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | Non-commercial research |
| Document page | [ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet) | Apache 2.0 |
| Noisy form | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | Non-commercial research |

No dataset content is included in this repository.

## License

Apache-2.0
```

---

## CLAUDE.md

```markdown
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
```

---

## Implementation Order

1. `pyproject.toml` + project skeleton (dirs, `__init__.py`, `.gitignore` with `data/samples/`)
2. `src/ocr_gauntlet/metrics.py` — simplest, no external deps
3. `scripts/download_samples.py` — HuggingFace download script
4. `src/ocr_gauntlet/utils.py` — image loading + auto-download trigger
5. `src/ocr_gauntlet/engines.py` — all 5 engine functions
6. `src/ocr_gauntlet/visualize.py` — charts
7. `notebooks/01_gauntlet.ipynb` — main demo
8. `notebooks/02_docling_deep_dive.ipynb` — Docling showcase
9. `notebooks/03_cost_calculator.ipynb` — cost analysis
10. `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`
11. Basic tests for metrics and utils
