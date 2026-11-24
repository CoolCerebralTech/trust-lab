"""Tests for scoring.py and calibration.py — plain code, no API needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlab.scoring import accuracy, brier_score, calibration_gap, abstain_rate  # noqa: E402
from trustlab.calibration import platt_fit, platt_predict, reliability_bins  # noqa: E402


def make_results(rows):
    """rows: list of (correct: bool|None, confidence: float|None, abstain: bool)"""
    out = []
    for c, conf, abst in rows:
        out.append({"correct": c, "confidence": conf, "abstain": abst})
    return out


def test_accuracy():
    r = make_results([(True, 80, False), (False, 60, False), (True, 90, False)])
    acc, n = accuracy(r)
    assert n == 3
    assert abs(acc - 2 / 3) < 1e-9

    # all abstained -> 0, 0
    r = make_results([(True, 80, True), (False, 60, True)])
    acc, n = accuracy(r)
    assert acc == 0.0 and n == 0


def test_brier():
    # perfect confidence: correct=1 conf=100, wrong=0 conf=0
    r = make_results([(True, 100, False), (False, 0, False)])
    b, n = brier_score(r)
    assert n == 2
    assert abs(b - 0.0) < 1e-9

    # completely wrong confident: correct=0 conf=100 -> error 1
    r = make_results([(False, 100, False)])
    b, n = brier_score(r)
    assert abs(b - 1.0) < 1e-9


def test_calibration_gap():
    # overconfident: avg conf high but accuracy low
    r = make_results([(False, 100, False), (False, 100, False), (True, 100, False)])
    gap, n = calibration_gap(r)
    assert n == 3
    avg_conf = 1.0
    acc = 1 / 3
    assert abs(gap - (avg_conf - acc)) < 1e-9


def test_abstain_rate():
    r = make_results([(True, 80, False), (False, 60, True), (True, 90, True)])
    rate, n = abstain_rate(r)
    assert n == 3
    assert abs(rate - 2 / 3) < 1e-9


def test_platt_roundtrip():
    confs = [10, 20, 30, 70, 80, 90]
    outcomes = [0, 0, 0, 1, 1, 1]
    a, b = platt_fit(confs, outcomes)
    preds = platt_predict(confs, a, b)
    # Low confs should predict low probs, high confs high probs
    assert preds[0] < preds[3]
    assert preds[-1] > preds[2]
    assert all(0 <= p <= 100 for p in preds)


def test_reliability_bins():
    confs = [10, 20, 30, 70, 80, 90, 50]
    outcomes = [0, 0, 0, 1, 1, 1, 1]
    bins = reliability_bins(confs, outcomes, n_bins=5)
    assert len(bins) <= 5
    for b in bins:
        assert "bin_center" in b and "avg_confidence" in b and "accuracy" in b and "count" in b
        assert b["count"] > 0


if __name__ == "__main__":
    test_accuracy(); test_brier(); test_calibration_gap(); test_abstain_rate()
    test_platt_roundtrip(); test_reliability_bins()
    print("scoring/calibration tests: all passed")
