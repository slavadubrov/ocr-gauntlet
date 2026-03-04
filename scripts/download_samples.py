"""
Download 5 sample documents from HuggingFace for the OCR Gauntlet demo.

Usage:
    python scripts/download_samples.py

Or called automatically by the notebook on first run.

Datasets used (all publicly available for research):
    - nielsr/funsd          — scanned forms (from RVL-CDIP / tobacco docs)
    - Teklia/IAM-line       — handwritten English text lines
    - naver-clova-ix/cord-v2 — scanned receipts (CORD dataset)
    - nielsr/funsd          — additional form samples for layout variety

These datasets are downloaded by the user at runtime. We do NOT redistribute
any dataset content in this repository. Users are responsible for complying
with each dataset's license terms.
"""

from __future__ import annotations

import json
from pathlib import Path
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

    # ── Sample 2: Receipt (CORD — scanned receipts) ──
    print("📥 Downloading CORD receipt...")
    cord = load_dataset("naver-clova-ix/cord-v2", split="test")
    # Pick a receipt with visible content
    sample = cord[5]
    img = sample["image"]
    gt_json = json.loads(sample["ground_truth"])
    # Flatten the parsed fields into plain text
    gt_parts = []
    for menu_item in gt_json.get("gt_parse", {}).get("menu", [{}]):
        if isinstance(menu_item, dict):
            for k, v in menu_item.items():
                if v:
                    gt_parts.append(f"{k}: {v}")
    for section in ("sub_total", "total"):
        sec_data = gt_json.get("gt_parse", {}).get(section, {})
        if isinstance(sec_data, dict):
            for k, v in sec_data.items():
                if v:
                    gt_parts.append(f"{k}: {v}")
    gt_text = "\n".join(gt_parts) if gt_parts else str(gt_json.get("gt_parse", ""))

    img.save(SAMPLES_DIR / "02_receipt.png")
    (SAMPLES_DIR / "02_receipt.txt").write_text(gt_text, encoding="utf-8")
    manifest["02_receipt"] = {
        "source": "naver-clova-ix/cord-v2 (test split, index 5)",
        "license": "CORD dataset — research use",
        "description": "Scanned receipt — real-world OCR challenge",
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

    # ── Sample 4: Dense form (FUNSD — different form with more content) ──
    print("📥 Downloading dense form sample...")
    sample = funsd[5]  # pick a denser form
    img = sample["image"]
    words = sample["words"]
    gt_text = " ".join(words)

    img.save(SAMPLES_DIR / "04_dense_form.png")
    (SAMPLES_DIR / "04_dense_form.txt").write_text(gt_text, encoding="utf-8")
    manifest["04_dense_form"] = {
        "source": "nielsr/funsd (test split, index 5)",
        "license": "Non-commercial research only",
        "description": "Dense scanned form — more text content and layout complexity",
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
