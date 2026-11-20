from __future__ import annotations

import math

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict:
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    real_count, fake_count = int(np.sum(~labels)), int(np.sum(labels))
    fpr = fp / real_count if real_count else None
    fnr = fn / fake_count if fake_count else None
    f1_denominator = 2 * tp + fp + fn
    return {
        "videos": int(len(labels)),
        "real_videos": real_count,
        "fake_videos": fake_count,
        "accuracy": (tp + tn) / len(labels) if len(labels) else None,
        "balanced_accuracy": (
            1 - (fpr + fnr) / 2
            if fpr is not None and fnr is not None
            else None
        ),
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "auc": roc_auc(labels, probabilities),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "brier": float(np.mean((probabilities - labels.astype(float)) ** 2)),
    }


def finite_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None
