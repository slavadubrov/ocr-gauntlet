"""Reproduce the manually reviewed CORD test-1 text region used by this demo.

The source page has blurred headers/footer. Evaluate only the visible five-line
item/total region, with row-major gold checked against the image on 2026-09-06.
This is a one-region integration check, not a representative benchmark.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from ocr_gauntlet.utils import load_manifest, sha256


def prepare(source: Path, output: Path) -> Path:
    samples = load_manifest(source)
    sample = next(s for s in samples if s["id"] == "cord-test-00001")
    if (
        sample["image_sha256"]
        != "b9d90c30eb7ac134ffe9c45e87c650c34de4c3f8dba94ed4ac9c5ec503f547d1"
    ):
        raise ValueError("Review applies only to the visually inspected source image")
    if sample.get("revision") != "7f0115a4b758a71d6473b8d085751692da2fef98":
        raise ValueError("Review applies only to the pinned CORD revision")
    output.mkdir(parents=True, exist_ok=False)
    box = (100, 640, 860, 935)
    with Image.open(sample["image"]) as image:
        if image.size != (960, 1280):
            raise ValueError("Unexpected source image dimensions")
        image.crop(box).save(output / "receipt-region.png")
    ref = {
        "text": "J.STB PROMO 17500\nY.B.BAT 46000\nY.BASO PROM 27500\nTOTAL 91000\nCASH 91000",
        "output_format": "text",
        "fields": {"total": "91000", "cash": "91000", "tax": None},
        "table": [
            ["J.STB PROMO", "17500"],
            ["Y.B.BAT", "46000"],
            ["Y.BASO PROM", "27500"],
        ],
    }
    (output / "reference.json").write_text(json.dumps(ref, indent=2))
    sample.pop("target")
    sample.update(
        id="cord-test-00001-region",
        image="receipt-region.png",
        reference="reference.json",
        source_image_sha256=sample["image_sha256"],
        crop_xyxy=list(box),
        image_sha256=sha256(output / "receipt-region.png"),
        reference_sha256=sha256(output / "reference.json"),
        width=760,
        height=295,
        reviewed=True,
        protocol="manual visible region transcription, row-major; excludes blurred page regions",
        review="Visual review 2026-09-06; scoped integration example, not official CORD evaluation",
        preprocessing="RGB source cropped at recorded xyxy; no rescale",
    )
    path = output / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "samples": [sample]}, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.source, args.output))
