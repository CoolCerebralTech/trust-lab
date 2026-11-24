"""Plain-code scoring: accuracy, Brier score, calibration gap, abstain rate.

No AI judge — every number here is computable by hand from the logs.
"""

from __future__ import annotations


def accuracy(results) -> tuple[float, int]:
    """Fraction correct among non-abstained, answerable records."""
    scored = [r for r in results if r.get("correct") is not None and not r.get("abstain")]
    if not scored:
        return 0.0, 0
    return sum(1 for r in scored if r["correct"]) / len(scored), len(scored)


def brier_score(results) -> tuple[float | None, int]:
    """Mean squared error between predicted probability (conf/100) and outcome (0/1)."""
    pts = [
        r for r in results
        if r.get("confidence") is not None and r.get("correct") is not None and not r.get("abstain")
    ]
    if not pts:
        return None, 0
    s = sum((r["confidence"] / 100.0 - (1.0 if r["correct"] else 0.0)) ** 2 for r in pts)
    return s / len(pts), len(pts)


def calibration_gap(results) -> tuple[float | None, int]:
    """Average confidence minus accuracy (positive = overconfident)."""
    pts = [
        r for r in results
        if r.get("confidence") is not None and r.get("correct") is not None and not r.get("abstain")
    ]
    if not pts:
        return None, 0
    avg_conf = sum(r["confidence"] for r in pts) / len(pts) / 100.0
    acc = sum(1 for r in pts if r["correct"]) / len(pts)
    return avg_conf - acc, len(pts)


def abstain_rate(results) -> tuple[float, int]:
    """Fraction of records where the model abstained (or failed to parse)."""
    n = len(results)
    if n == 0:
        return 0.0, 0
    a = sum(1 for r in results if r.get("abstain"))
    return a / n, n
