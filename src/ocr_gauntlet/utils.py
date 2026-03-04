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
        import subprocess
        import sys

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
