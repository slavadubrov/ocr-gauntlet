"""Explicit public dataset import. No models or hosted OCR calls.

CORD references use valid_line words, never gt_parse keys. Imported gold is
unreviewed: check annotation coverage/reading order against each image before
setting reviewed=true in the manifest. IAM's source terms still apply even if
its mirror advertises a different license.
"""

import argparse
import json
from pathlib import Path

from ocr_gauntlet.utils import sha256

DATASETS = {
    "cord": {
        "repo": "naver-clova-ix/cord-v2",
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98",
        "license": "CC-BY-4.0 (mirror metadata); see source terms",
        "license_url": "https://github.com/clovaai/cord",
        "protocol": "CORD valid_line annotation order; may omit unannotated visible text",
    },
    "iam": {
        "repo": "Teklia/IAM-line",
        "revision": "fbdad97500ce54635c0d1ba306bf535cb40656cf",
        "license": "IAM original research terms; mirror MIT does not relicense underlying images",
        "license_url": "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
        "protocol": "IAM line transcription, test split; not full-page OCR",
    },
}


def reference_for(kind: str, row: dict) -> dict:
    if kind == "iam":
        return {"text": row["text"], "output_format": "text"}
    gt = json.loads(row["ground_truth"])
    lines = gt.get("valid_line")
    if not lines:
        raise ValueError(
            "CORD sample has no visible-text annotations; semantic fields are not transcription gold"
        )
    return {
        "text": "\n".join(" ".join(w["text"] for w in line["words"]) for line in lines),
        "output_format": "text",
        "fields": gt.get("gt_parse", {}),
        "annotated_lines": lines,
    }


def download_samples(kind: str, output: Path, count: int = 5, start: int = 0) -> Path:
    from datasets import load_dataset

    if count < 1 or start < 0:
        raise ValueError("count must be positive and start nonnegative")
    config = DATASETS[kind]
    # Exclusive directory avoids silently overwriting reviewed references or partial imports.
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(
        config["repo"], revision=config["revision"], split="test", streaming=True
    )
    samples = []
    for index, row in enumerate(dataset.skip(start).take(count), start):
        name = f"{kind}-test-{index:05}"
        image_path, ref_path = output / f"{name}.png", output / f"{name}.json"
        image = row["image"].convert("RGB")
        image.save(image_path)
        ref_path.write_text(
            json.dumps(reference_for(kind, row), ensure_ascii=False, indent=2)
        )
        samples.append(
            {
                "id": name,
                "image": image_path.name,
                "reference": ref_path.name,
                "image_sha256": sha256(image_path),
                "reference_sha256": sha256(ref_path),
                "dataset": config["repo"],
                "revision": config["revision"],
                "split": "test",
                "index": index,
                "source_id": str(row.get("id", index)),
                "license": config["license"],
                "license_url": config["license_url"],
                "protocol": config["protocol"],
                "reviewed": False,
                "preprocessing": "decode source image, convert RGB, lossless PNG; no resize",
                "width": image.width,
                "height": image.height,
            }
        )
    if len(samples) != count:
        raise ValueError(
            f"Requested {count} records, found {len(samples)}; partial directory retained without manifest"
        )
    path = output / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "samples": samples}, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cord")
    parser.add_argument("--output", type=Path, default=Path("data/cord"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    print(download_samples(args.dataset, args.output, args.count, args.start))
