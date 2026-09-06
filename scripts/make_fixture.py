"""Generate original offline fixtures; these are wiring checks, not benchmark data."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ocr_gauntlet.utils import sha256


def make_fixture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    font = ImageFont.load_default(size=28)
    for name, lines in {
        "receipt": [
            "OCR DEMO",
            "Item       Qty    Price",
            "Tea        2      4.00",
            "Total             8.00",
        ],
        "letter": [
            "Dear reader,",
            "OCR compares visible text.",
            "Preserve numbers: 12345.",
        ],
    }.items():
        image = Image.new("RGB", (700, 300), "white")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((25, 25 + index * 55), line, fill="black", font=font)
        image_path, reference_path = output / f"{name}.png", output / f"{name}.json"
        image.save(image_path)
        ref = {"text": "\n".join(lines), "output_format": "text"}
        if name == "receipt":
            ref.update(
                fields={"total": "8.00", "tax": None},
                table=[["Item", "Qty", "Price"], ["Tea", "2", "4.00"]],
                reading_order=["header", "table", "total"],
            )
        reference_path.write_text(json.dumps(ref, indent=2))
        samples.append(
            {
                "id": name,
                "image": image_path.name,
                "reference": reference_path.name,
                "image_sha256": sha256(image_path),
                "reference_sha256": sha256(reference_path),
                "dataset": "original synthetic fixture",
                "license": "Apache-2.0",
                "split": "smoke",
                "protocol": "generated lines in top-to-bottom order; not model-quality evidence",
                "reviewed": True,
                "preprocessing": "Pillow default font; saved hashes identify exact rendering",
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": samples}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(make_fixture(parser.parse_args().output))
