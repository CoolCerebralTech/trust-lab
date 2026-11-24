"""Tests for parse.py — plain code, no API needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlab.parse import extract_answer, extract_confidence  # noqa: E402


def test_confidence():
    assert extract_confidence("Confidence: 85%") == 85.0
    assert extract_confidence("Confidence: 85") == 85.0
    assert extract_confidence("I'm 40% sure.") == 40.0
    assert extract_confidence("No confidence here.") is None
    assert extract_confidence("") is None
    assert extract_confidence("Confidence: 120%") == 100.0  # clamped
    assert extract_confidence("Confidence: -5%") == 0.0  # clamped


def test_yesno():
    assert extract_answer("yes", "yesno") == "yes"
    assert extract_answer("No.", "yesno") == "no"
    assert extract_answer("Yes\nConfidence: 80%", "yesno") == "yes"
    assert extract_answer("maybe", "yesno") is None


def test_mc():
    opts = ["Spain", "Argentina", "France", "England"]
    assert extract_answer("B", "mc", opts) == "Argentina"
    assert extract_answer("b", "mc", opts) == "Argentina"
    assert extract_answer("Spain", "mc", opts) == "Spain"
    assert extract_answer("Option B", "mc", opts) == "Argentina"


def test_numeric():
    assert extract_answer("4", "numeric") == 4.0
    assert extract_answer("The answer is 225.", "numeric") == 225.0
    assert extract_answer("unknown", "numeric") is None


def test_abstain_none():
    assert extract_answer("", "mc", ["a", "b"]) is None


if __name__ == "__main__":
    test_confidence(); test_yesno(); test_mc(); test_numeric(); test_abstain_none()
    print("parse tests: all passed")
