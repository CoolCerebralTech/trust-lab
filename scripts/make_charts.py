"""Charts & tables from experiment results.

Reads results/metrics.json + results.jsonl and writes:
  results/reliability.csv        reliability-diagram bins (for charting in Excel/Sheets)
  results/brier_table.md         Brier before/after Platt calibration
  results/agreement_table.md     cross-model agreement summary

Run after run_experiment.py:
  python scripts/make_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustlab.config import ROOT  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    results_dir = ROOT / "results"
    metrics = load_json(results_dir / "metrics.json")
    results = None
    rjl = results_dir / "results.jsonl"
    if rjl.exists():
        results = [json.loads(l) for l in rjl.read_text(encoding="utf-8").splitlines() if l.strip()]

    if not metrics:
        print("No results/metrics.json found. Run run_experiment.py first.")
        return

    # ---- Reliability CSV from platt reliability_bins
    platt = metrics.get("platt")
    if platt and platt.get("reliability_bins"):
        lines = ["bin_center,avg_confidence,accuracy,count"]
        for b in platt["reliability_bins"]:
            lines.append(f"{b['bin_center']},{b['avg_confidence']},{b['accuracy']},{b['count']}")
        (results_dir / "reliability.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote results/reliability.csv ({len(platt['reliability_bins'])} bins)")

    # ---- Brier table
    lines = ["# Brier score table (per model x style)", ""]
    lines.append("| model | style | accuracy | brier | n |")
    lines.append("|-------|-------|----------|-------|---|")
    for m in metrics.get("metrics", []):
        def v(x): return "n/a" if x is None else str(x)
        lines.append(f"| {m['model']} | {m['style']} | {v(m['accuracy'])} | {v(m['brier'])} | {m['brier_n']} |")
    if platt:
        lines.append("")
        lines.append("## Platt calibration (held-out test)")
        lines.append(f"- Brier before calibration: {platt['brier_before_calibration']}")
        lines.append(f"- Brier after calibration:  {platt['brier_after_calibration']}")
        lines.append(f"- Train n: {platt['train_n']} | Test n: {platt['test_n']}")
    (results_dir / "brier_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote results/brier_table.md")

    # ---- Agreement table
    d = metrics.get("disagreement", {})
    lines = ["# Cross-model agreement", ""]
    lines.append(f"- Questions compared: {d.get('questions_compared')}")
    lines.append(f"- Agree rate: {d.get('agree_rate')}")
    lines.append(f"- Agree when right: {d.get('agree_when_right')}")
    lines.append(f"- Agree when wrong: {d.get('agree_when_wrong')}")
    (results_dir / "agreement_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote results/agreement_table.md")
    print("\nDone. Open results/summary.md for the full picture.")


if __name__ == "__main__":
    main()
