"""Task-specific metrics. Scores never imply a common OCR leaderboard."""

import math
import re
import unicodedata
from collections.abc import Sequence

import editdistance

NORMALIZATION = "NFC-case-sensitive-whitespace-v1"


def normalize_text(text: str) -> str:
    """Preserve case, punctuation and digits; NFC and collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def cer(prediction: str, ground_truth: str) -> float:
    """Character edits/reference characters; nonempty vs empty gold is infinity."""
    return (
        editdistance.eval(prediction, ground_truth) / len(ground_truth)
        if ground_truth
        else (math.inf if prediction else 0.0)
    )


def wer(prediction: str, ground_truth: str) -> float:
    """Whitespace-token edits/reference tokens (same empty-reference rule as CER)."""
    p, g = prediction.split(), ground_truth.split()
    return editdistance.eval(p, g) / len(g) if g else (math.inf if p else 0.0)


def normalized_edit_distance(prediction: str, ground_truth: str) -> float:
    return editdistance.eval(prediction, ground_truth) / max(
        len(prediction), len(ground_truth), 1
    )


def anls(prediction: str, ground_truth: str | Sequence[str], tau: float = 0.5) -> float:
    """One VQA answer: best reference, lowercase/space normalization, distance < tau.

    Biten et al., ICCV 2019 Eq. 2. Average over questions outside this function.
    Empty/empty is explicitly defined as 1; no accepted references is an error.
    This function is NOT used for whole-page transcription ranking.
    """
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    if not refs:
        raise ValueError("At least one accepted answer is required")
    distances = [
        normalized_edit_distance(
            normalize_text(prediction).lower(), normalize_text(g).lower()
        )
        for g in refs
    ]
    return max(1 - d if d < tau else 0.0 for d in distances)


def text_metrics(prediction: str, reference: str) -> dict:
    p, g = normalize_text(prediction), normalize_text(reference)
    c, w = cer(p, g), wer(p, g)
    return {
        "cer": c if math.isfinite(c) else None,
        "wer": w if math.isfinite(w) else None,
        "char_edits": editdistance.eval(p, g),
        "reference_chars": len(g),
        "word_edits": editdistance.eval(p.split(), g.split()),
        "reference_words": len(g.split()),
        "text_exact": p == g,
        "text_quality": 1 - normalized_edit_distance(p, g),
        "empty_reference_insertion": not g and bool(p),
    }


def field_metrics(prediction: dict, reference: dict) -> dict:
    """Exact key/value pairs; explicit null differs from missing. Extra keys count."""
    correct = sum(k in prediction and prediction[k] == v for k, v in reference.items())
    precision = correct / len(prediction) if prediction else float(not reference)
    recall = correct / len(reference) if reference else float(not prediction)
    return {
        "field_exact_match": correct / len(reference)
        if reference
        else float(not prediction),
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "all_fields_exact": prediction == reference,
    }


def table_metrics(prediction: list, reference: list) -> dict:
    """Ordered rectangular/ragged cell grids, exact coordinates; NOT TEDS."""
    target = {(r, c): v for r, row in enumerate(reference) for c, v in enumerate(row)}
    actual = {(r, c): v for r, row in enumerate(prediction) for c, v in enumerate(row)}
    keys = target.keys() | actual.keys()
    matches = sum(k in actual and k in target and actual[k] == target[k] for k in keys)
    return {
        "table_exact": prediction == reference,
        "table_cell_accuracy": matches / len(keys) if keys else 1.0,
    }


def reading_order_accuracy(prediction: list[str], reference: list[str]) -> float:
    """Pair order over unique gold block IDs; missing blocks lose their pairs."""
    if len(set(prediction)) != len(prediction) or len(set(reference)) != len(reference):
        raise ValueError("Block IDs must be unique")
    if len(reference) < 2:
        return float(prediction == reference)
    positions = {v: i for i, v in enumerate(prediction)}
    pairs = [(a, b) for i, a in enumerate(reference) for b in reference[i + 1 :]]
    return sum(
        a in positions and b in positions and positions[a] < positions[b]
        for a, b in pairs
    ) / len(pairs)


def cost_scenario(
    pages: int,
    *,
    api_usd_per_attempt: float = 0,
    hourly_usd: float = 0,
    hours: float = 24 * 30,
    pages_per_hour: float = 1000,
    utilization: float = 0.7,
    attempts_per_page: float = 1,
    completion_rate: float = 1,
    review_fraction: float = 0,
    review_usd_per_page: float = 0,
) -> dict:
    """Hypothetical monthly plan. Dedicated compute bills all provisioned hours."""
    values = [
        pages,
        api_usd_per_attempt,
        hourly_usd,
        hours,
        pages_per_hour,
        utilization,
        attempts_per_page,
        completion_rate,
        review_fraction,
        review_usd_per_page,
    ]
    if not all(math.isfinite(v) and v >= 0 for v in values) or attempts_per_page < 1:
        raise ValueError("Scenario values must be finite/nonnegative; attempts >= 1")
    if (
        not 0 < utilization <= 1
        or not 0 <= completion_rate <= 1
        or not 0 <= review_fraction <= 1
    ):
        raise ValueError("Invalid utilization, completion or review fraction")
    total = (
        hourly_usd * hours
        + pages * attempts_per_page * api_usd_per_attempt
        + pages * review_fraction * review_usd_per_page
    )
    successes = pages * completion_rate
    capacity = hours * pages_per_hour * utilization / attempts_per_page
    return {
        "kind": "hypothetical",
        "monthly_usd": total,
        "capacity_pages": capacity,
        "capacity_sufficient": pages <= capacity,
        "successful_pages": successes,
        "usd_per_success": total / successes if successes else None,
    }
