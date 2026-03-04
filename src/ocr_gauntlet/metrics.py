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
