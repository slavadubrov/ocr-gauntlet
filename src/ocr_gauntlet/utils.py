"""Explicit, hash-verified evaluation manifests; no implicit downloads."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from ocr_gauntlet.metrics import normalize_text  # noqa: F401


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    json.dumps(data, allow_nan=False)
    if data.get("schema_version") != 1 or not data.get("samples"):
        raise ValueError(
            "Expected a nonempty schema_version=1 manifest; legacy samples must be re-imported"
        )
    seen = set()
    samples = []
    for source in data["samples"]:
        sample = dict(source)
        if not isinstance(sample["id"], str) or sample["id"] in seen:
            raise ValueError("Sample IDs must be unique strings")
        seen.add(sample["id"])
        for key in ("image", "reference"):
            target = (path.parent / sample[key]).resolve()
            if not target.is_relative_to(path.parent):
                raise ValueError(f"{key} must remain inside manifest directory")
            if sha256(target) != sample[f"{key}_sha256"]:
                raise ValueError(f"Hash mismatch: {target}")
            sample[key] = str(target)
        if not isinstance(sample.get("reviewed"), bool) or not sample.get("protocol"):
            raise ValueError(
                "Each sample needs reviewed boolean and reference protocol"
            )
        reference = json.loads(Path(sample["reference"]).read_text())
        if not isinstance(reference.get("text"), str):
            raise ValueError("Reference text must be a string")
        sample["target"] = reference
        samples.append(sample)
    return samples


def load_sample(sample: dict) -> Image.Image:
    with Image.open(sample["image"]) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Benchmark inputs must be single-page images")
        return image.convert("RGB")
