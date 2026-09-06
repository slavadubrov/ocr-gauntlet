"""Sequential, append-as-you-go evaluation. Run with python -m ocr_gauntlet.benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ocr_gauntlet import engines
from ocr_gauntlet.metrics import (
    NORMALIZATION,
    field_metrics,
    reading_order_accuracy,
    table_metrics,
    text_metrics,
)
from ocr_gauntlet.utils import load_manifest, load_sample, sha256

ENGINE_NAMES = (
    "tesseract",
    "paddle",
    "docling-tesseract",
    "docling-easyocr",
    "docling-granite",
    "dots-ocr",
    "mistral",
    "gemini",
)
REMOTE = {"gemini", "mistral", "dots-ocr"}


def _run(name: str, sample: dict, settings: dict) -> tuple[str, dict]:
    if name.startswith("docling-"):
        return engines.run_docling(
            sample["image"],
            ocr_backend="easyocr" if name.endswith("easyocr") else "tesseract",
            use_vlm=name.endswith("granite"),
            **settings,
        )
    image = load_sample(sample)
    try:
        if name == "tesseract":
            return engines.run_tesseract(image, **settings)
        if name == "paddle":
            return engines.run_paddle(image, **settings)
        if name == "gemini":
            return engines.run_gemini(image, **settings)
        if name == "mistral":
            return engines.run_mistral_ocr(image, **settings)
        if name == "dots-ocr":
            return engines.run_dots_ocr(image, **settings)
        raise ValueError(f"Unknown engine: {name}")
    finally:
        image.close()


def score_output(text: str, meta: dict, sample: dict) -> dict:
    target = sample["target"]
    out = {"metrics": {}, "evaluation_status": "not_scored", "evaluation_reason": None}
    if meta["status"] != "success":
        out["evaluation_reason"] = "extraction did not complete"
        return out
    if not sample["reviewed"]:
        out["evaluation_reason"] = "reference not reviewed"
        return out
    if meta.get("output_format") == target.get("output_format", "text"):
        out["metrics"].update(text_metrics(text, target["text"]))
        out["evaluation_status"] = "scored"
    else:
        out["evaluation_reason"] = (
            "output/reference formats differ; raw output retained for structural evaluation"
        )
    # Structured predictions are explicit, never inferred from gold or string containment.
    for key, scorer in (("fields", field_metrics), ("table", table_metrics)):
        if key in target:
            if key in meta:
                out["metrics"].update(scorer(meta[key], target[key]))
            else:
                out[f"{key}_status"] = "unsupported"
    if "reading_order" in target:
        if "reading_order" in meta:
            out["metrics"]["reading_order_accuracy"] = reading_order_accuracy(
                meta["reading_order"], target["reading_order"]
            )
        else:
            out["reading_order_status"] = "unsupported"
    return out


def run_benchmark(
    manifest: str | Path,
    names: list[str],
    output: str | Path,
    *,
    allow_remote: bool = False,
    settings: dict | None = None,
    runner=None,
) -> list[dict]:
    if not names or len(set(names)) != len(names) or set(names) - set(ENGINE_NAMES):
        raise ValueError(f"Select unique engines from {ENGINE_NAMES}")
    settings = settings or {}
    if set(settings) - set(names):
        raise ValueError("Settings provided for an unselected engine")
    # Secrets belong in env; do not serialize caller-supplied credentials.
    if any(
        "key" in key.lower() or "token" in key.lower()
        for config in settings.values()
        for key in config
    ):
        raise ValueError(
            "Pass credentials through environment variables, not benchmark settings"
        )
    samples = load_manifest(manifest)
    implementation_hash = hashlib.sha256(
        b"".join(
            p.name.encode() + p.read_bytes()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        )
    ).hexdigest()
    model_defaults = {
        "gemini": engines.GEMINI_MODEL,
        "mistral": engines.MISTRAL_MODEL,
        "paddle": "PP-OCRv5",
        "dots-ocr": "rednote-hilab/dots.ocr",
        "docling-granite": "ibm-granite/granite-docling-258M",
    }
    run_id = str(uuid.uuid4())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head, dirty = None, None
    records = []
    # ponytail: sequential pages keep timing interpretable; add concurrency only for a separate throughput experiment.
    with output.open("x", encoding="utf-8") as stream:
        for name in names:
            for sample in samples:
                row = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "engine": name,
                    "model_requested": settings.get(name, {}).get(
                        "model", model_defaults.get(name)
                    ),
                    "document": sample["id"],
                    "input_sha256": sample["image_sha256"],
                    "reference_sha256": sample["reference_sha256"],
                    "manifest_sha256": sha256(Path(manifest)),
                    "provenance": {
                        k: v
                        for k, v in sample.items()
                        if k not in {"target", "image", "reference"}
                    },
                    "normalization": NORMALIZATION,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "git_head": git_head,
                    "git_dirty": dirty,
                    "implementation_sha256": implementation_hash,
                    "requested_settings": settings.get(name, {}),
                    "status": "skipped",
                    "reason": "remote execution not enabled",
                    "text": "",
                    "cost_usd": None,
                    "metrics": {},
                    "evaluation_status": "not_scored",
                    "attempted": False,
                }
                start = time.perf_counter()
                if name not in REMOTE or allow_remote:
                    row["attempted"] = True
                    try:
                        text, meta = (runner or _run)(
                            name, sample, settings.get(name, {})
                        )
                        if meta.get("status") not in {"success", "error"}:
                            raise ValueError("Adapter must declare success/error")
                        row["text"] = text
                        cost = meta.get("cost_usd")
                        if (
                            isinstance(cost, (int, float))
                            and math.isfinite(cost)
                            and cost >= 0
                        ):
                            row["cost_usd"] = cost
                        # Validate before merging: malformed metadata cannot abort later planned pairs.
                        json.dumps(meta, allow_nan=False)
                        if meta["status"] == "success" and not text.strip():
                            meta = {**meta, "status": "error", "reason": "empty output"}
                        row.update(meta, text=text)
                        row["engine"] = name
                        row["document"] = sample["id"]
                        row.update(score_output(text, meta, sample))
                    except (ImportError, FileNotFoundError) as exc:
                        row.update(
                            status="error",
                            reason=f"dependency/input unavailable: {type(exc).__name__}",
                        )
                    except Exception as exc:
                        # Do not persist exception messages: SDK errors can echo documents or credentials.
                        row.update(
                            status="error",
                            reason=f"{type(exc).__name__}; inspect locally for details",
                        )
                        code = getattr(exc, "status_code", getattr(exc, "code", None))
                        if isinstance(code, int):
                            row["http_status"] = code
                row["wall_ms"] = (time.perf_counter() - start) * 1000
                row["review_required"] = (
                    row["status"] != "success"
                    or row["evaluation_status"] != "scored"
                    or not row["metrics"].get("text_exact", False)
                    or any(
                        row.get(f"{key}_status") == "unsupported"
                        for key in ("fields", "table", "reading_order")
                    )
                    or row["metrics"].get("all_fields_exact") is False
                    or row["metrics"].get("table_exact") is False
                    or row["metrics"].get("reading_order_accuracy", 1) < 1
                )
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                records.append(row)
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[row["engine"]].append(row)
    summaries = []
    for name, rows in groups.items():
        successful = [r for r in rows if r["status"] == "success"]
        scored = [r for r in rows if r.get("evaluation_status") == "scored"]
        known = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        unknown = sum(
            r.get("attempted", True) and r.get("cost_usd") is None for r in rows
        )
        chars = sum(r["metrics"].get("reference_chars", 0) for r in scored)
        words = sum(r["metrics"].get("reference_words", 0) for r in scored)
        times = [r["latency_ms"] for r in successful if r.get("latency_ms") is not None]
        # All planned denominator: failed/skipped/unscorable rows contribute zero.
        summaries.append(
            {
                "engine": name,
                "planned": len(rows),
                "attempted": sum(r.get("attempted", True) for r in rows),
                "successes": len(successful),
                "completion_rate": len(successful) / len(rows),
                "scored": len(scored),
                "score_coverage": len(scored) / len(rows),
                "all_planned_text_quality": sum(
                    r["metrics"].get("text_quality", 0) for r in scored
                )
                / len(rows),
                "conditional_corpus_cer": sum(
                    r["metrics"].get("char_edits", 0) for r in scored
                )
                / chars
                if chars
                else None,
                "conditional_corpus_wer": sum(
                    r["metrics"].get("word_edits", 0) for r in scored
                )
                / words
                if words
                else None,
                "known_cost_usd": sum(known),
                "unknown_cost_attempts": unknown,
                "usd_per_success": sum(known) / len(successful)
                if successful and not unknown
                else None,
                "warm_p50_ms": statistics.median(times) if times else None,
                "warm_p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1]
                if times
                else None,
                "review_fraction": sum(r["review_required"] for r in rows) / len(rows),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINE_NAMES, default=["tesseract"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSONL file; existing files are never overwritten",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow document uploads and possible API spend for selected engines",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="JSON mapping engine names to adapter kwargs; no credentials",
    )
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text()) if args.settings else None
    rows = run_benchmark(
        args.manifest,
        args.engines,
        args.output,
        allow_remote=args.allow_remote,
        settings=settings,
    )
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
