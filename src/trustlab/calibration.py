"""Platt scaling: fit confidence->accuracy on train, apply on test. Pure Python.

Learns p = sigmoid(a * confidence + b). Used in Stage 5 (correction).
"""

from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def platt_fit(confidences, outcomes, lr: float = 0.1, epochs: int = 3000) -> tuple[float, float]:
    """Fit logistic regression on (confidence in 0..100, outcome 0/1).

    Returns (a, b) where calibrated_prob = sigmoid(a * conf + b).
    """
    a = 0.0
    b = 0.0
    n = len(confidences)
    xs = [c / 100.0 for c in confidences]  # work in 0..1 for stable gradients
    for _ in range(epochs):
        da = 0.0
        db = 0.0
        for x, y in zip(xs, outcomes):
            p = _sigmoid(a * x + b)
            err = p - y
            da += err * x
            db += err
        a -= lr * da / n
        b -= lr * db / n
    return a, b


def platt_predict(confidences, a: float, b: float) -> list[float]:
    """Return calibrated confidence (0..100) for each input confidence."""
    return [_sigmoid(a * (c / 100.0) + b) * 100.0 for c in confidences]


def reliability_bins(confidences, outcomes, n_bins: int = 10) -> list[dict]:
    """Bin (confidence, outcome) pairs into reliability-diagram data.

    Returns list of {bin_center, avg_confidence, accuracy, count}.
    """
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, outcomes):
        idx = min(int(c / 100.0 * n_bins), n_bins - 1)
        bins[idx].append((c, y))

    out = []
    for i, bin_ in enumerate(bins):
        if not bin_:
            continue
        cs = [c for c, _ in bin_]
        ys = [y for _, y in bin_]
        out.append(
            {
                "bin_center": round((i + 0.5) / n_bins * 100.0, 1),
                "avg_confidence": round(sum(cs) / len(cs), 1),
                "accuracy": round(sum(ys) / len(ys), 3),
                "count": len(bin_),
            }
        )
    return out
