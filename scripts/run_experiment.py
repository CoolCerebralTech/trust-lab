"""Stage 1 runner: run every model x prompt x question, parse, score, save results.

Usage:
  python scripts/run_experiment.py              # real run (needs .env keys + model IDs)
  python scripts/run_experiment.py --dry-run    # offline verification, no API keys
  python scripts/run_experiment.py --dry-run --limit 10

Outputs:
  results/results.jsonl   one record per model x style x question
  results/metrics.json    per model+style accuracy, Brier, calibration gap, abstain
  results/summary.md      human-readable table
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402  (settings loaded directly)

from trustlab.config import ROOT, load_models_config, model_provider_cfg, make_client  # noqa: E402
from trustlab.llm_client import LLMClient  # noqa: E402
from trustlab.prompts import prompt_for, STYLES  # noqa: E402
from trustlab.parse import extract_answer, extract_confidence  # noqa: E402
from trustlab.scoring import accuracy, brier_score, calibration_gap, abstain_rate  # noqa: E402
from trustlab.calibration import platt_fit, platt_predict, reliability_bins  # noqa: E402
from trustlab.disagreement import cross_model_agreement  # noqa: E402


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(path: Path) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def make_split(questions, seed, train_ratio, splits_dir: Path):
    splits_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    order = list(range(len(questions)))
    rnd.shuffle(order)
    n_train = int(len(order) * train_ratio)
    train = [questions[i] for i in order[:n_train]]
    test = [questions[i] for i in order[n_train:]]
    (splits_dir / "train.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in train), encoding="utf-8"
    )
    (splits_dir / "test.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in test), encoding="utf-8"
    )
    return train, test


def is_correct(pred, truth, qtype, tolerance=0.01):
    if pred is None or truth is None:
        return None
    if qtype == "numeric":
        try:
            return abs(float(pred) - float(truth)) <= tolerance
        except Exception:
            return False
    return str(pred).strip().lower() == str(truth).strip().lower()


def build_record(model, style, question, answer_text, conf, settings, api_record):
    pred = extract_answer(answer_text, question["type"], question.get("options"))
    correct = is_correct(pred, question["answer"], question["type"], settings.get("numeric_tolerance", 0.01))
    abstain = (pred is None) or (conf is None and settings.get("abstain_option"))
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "cutoff": question.get("cutoff"),
        "model": model["name"],
        "provider": model["provider"],
        "model_id": model["model_id"],
        "lab": model.get("lab"),
        "style": style,
        "pred": pred,
        "truth": question["answer"],
        "correct": correct,
        "confidence": conf,
        "answer_text": answer_text if settings.get("save_raw_responses") else None,
        "abstain": abstain,
        "api_record": api_record,
    }


def run_experiment(settings, models_cfg, questions, dry_run=False, limit=0):
    llm = LLMClient(settings)
    styles = settings.get("prompt_styles", list(STYLES))
    self_consistency = int(settings.get("self_consistency_samples", 1))

    train, test = make_split(
        questions,
        settings["seed"],
        settings["train_ratio"],
        ROOT / settings["splits_dir"],
    )
    pool = test if test else questions

    results = []
    per_question = {}  # qid -> {model: result}  (for disagreement)
    self_consistency_samples = {}  # qid -> [sample results]

    count = 0
    for q in pool:
        qid = q["id"]
        per_question[qid] = {}
        samples_list = []
        for model in models_cfg["models"]:
            mid = model["model_id"]
            if "FILL_ME_IN" in str(mid):
                if dry_run:
                    # Use a dummy id so the offline pipeline still runs end-to-end.
                    mid = f"dry-run-{model['provider']}"
                else:
                    print(f"SKIP  {model['name']}: model_id not filled in config/models.yaml")
                    continue
            provider_cfg = model_provider_cfg(models_cfg, model)
            client = make_client(provider_cfg)
            if client is None and not dry_run:
                print(f"SKIP  {model['name']}: no API key in .env for {provider_cfg['key_env']}")
                continue
            for style in styles:
                # self-consistency: repeat the same (model, style, question) N times
                for s in range(self_consistency):
                    prompt = prompt_for(style, q["question"])
                    answer_text, rec = llm.call(
                        model["name"],
                        provider_cfg,
                        mid,
                        prompt,
                        max_tokens=settings.get("max_tokens", 400),
                        temperature=settings.get("temperature", 0.0),
                        dry_run=dry_run,
                    )
                    conf = extract_confidence(answer_text)
                    rec_result = build_record(model, style, q, answer_text, conf, settings, rec)
                    results.append(rec_result)
                    samples_list.append(rec_result)
                    count += 1
                    if limit and count >= limit:
                        break
                if limit and count >= limit:
                    break
            # keep one representative per (model, style) for disagreement (last sample)
            for style in styles:
                style_results = [r for r in samples_list if r["style"] == style]
                if style_results:
                    per_question[qid][model["name"]] = style_results[-1]
        if limit and count >= limit:
            break

    # ---- Aggregate metrics per model x style
    metrics = []
    combos = {}
    for r in results:
        key = (r["model"], r["style"])
        combos.setdefault(key, []).append(r)
    for (model, style), res in sorted(combos.items()):
        acc, acc_n = accuracy(res)
        brier, brier_n = brier_score(res)
        gap, gap_n = calibration_gap(res)
        abst, abst_n = abstain_rate(res)
        metrics.append(
            {
                "model": model,
                "style": style,
                "accuracy": round(acc, 3) if acc is not None else None,
                "accuracy_n": acc_n,
                "brier": round(brier, 4) if brier is not None else None,
                "brier_n": brier_n,
                "calibration_gap": round(gap, 3) if gap is not None else None,
                "calibration_gap_n": gap_n,
                "abstain_rate": round(abst, 3) if abst is not None else None,
                "abstain_n": abst_n,
            }
        )

    # ---- Platt scaling: fit on train predictions, evaluate on test predictions
    platt = None
    train_res = [r for r in results if r.get("api_record", {}).get("dry_run") is True]  # all dry-run
    # Use confidence->correct on this run's data as a stand-in train/test split.
    all_conf = [r["confidence"] for r in results if r.get("confidence") is not None and r.get("correct") is not None and not r.get("abstain")]
    all_out = [1 if r["correct"] else 0 for r in results if r.get("confidence") is not None and r.get("correct") is not None and not r.get("abstain")]
    if len(all_conf) > 20:
        split = int(len(all_conf) * settings["train_ratio"])
        a, b = platt_fit(all_conf[:split], all_out[:split])
        cal_conf = all_conf[split:]
        cal_out = all_out[split:]
        pred_cal = platt_predict(cal_conf, a, b)
        # Brier before vs after calibration on held-out
        brier_before = sum((c / 100.0 - o) ** 2 for c, o in zip(cal_conf, cal_out)) / len(cal_conf)
        brier_after = sum((pc / 100.0 - o) ** 2 for pc, o in zip(pred_cal, cal_out)) / len(cal_conf)
        platt = {
            "train_n": split,
            "test_n": len(cal_conf),
            "brier_before_calibration": round(brier_before, 4),
            "brier_after_calibration": round(brier_after, 4),
            "reliability_bins": reliability_bins(cal_conf, cal_out),
        }

    # ---- Disagreement across models (per question)
    disagreement = cross_model_agreement(per_question)

    return {
        "settings": settings,
        "dry_run": dry_run,
        "n_results": len(results),
        "n_questions": len(pool),
        "models": [m["name"] for m in models_cfg["models"]],
        "metrics": metrics,
        "platt": platt,
        "disagreement": disagreement,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }, results


def write_outputs(summary: dict, results, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (results_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )

    # Human-readable summary table
    lines = []
    lines.append("# Trust Lab — Stage 1 Summary")
    lines.append(f"Generated: {summary['generated_utc']}  |  dry_run={summary['dry_run']}")
    lines.append(f"Questions: {summary['n_questions']}  |  Records: {summary['n_results']}")
    lines.append("")
    lines.append("## Per model x style metrics")
    lines.append("| model | style | accuracy | brier | calibration_gap | abstain_rate |")
    lines.append("|-------|-------|----------|-------|-----------------|--------------|")
    for m in summary["metrics"]:
        def v(x): return "n/a" if x is None else str(x)
        lines.append(
            f"| {m['model']} | {m['style']} | {v(m['accuracy'])} | {v(m['brier'])} | "
            f"{v(m['calibration_gap'])} | {v(m['abstain_rate'])} |"
        )
    lines.append("")
    if summary["platt"]:
        p = summary["platt"]
        lines.append("## Platt calibration (held-out)")
        lines.append(f"- train n: {p['train_n']}, test n: {p['test_n']}")
        lines.append(f"- Brier before calibration: {p['brier_before_calibration']}")
        lines.append(f"- Brier after calibration:  {p['brier_after_calibration']}")
        lines.append("")
    d = summary["disagreement"]
    lines.append("## Cross-model agreement")
    lines.append(f"- Questions compared: {d['questions_compared']}")
    lines.append(f"- Agree rate: {d['agree_rate']}")
    lines.append(f"- Agree when right: {d['agree_when_right']} | Agree when wrong: {d['agree_when_wrong']}")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Trust Lab Stage 1 experiment runner")
    ap.add_argument("--dry-run", action="store_true", help="Offline verification without API keys")
    ap.add_argument("--limit", type=int, default=0, help="Max records to run (quick smoke test)")
    ap.add_argument("--questions", default=None, help="Path to questions JSONL (default: settings)")
    args = ap.parse_args()

    settings = load_settings()
    models_cfg = load_models_config()
    qpath = Path(args.questions) if args.questions else ROOT / settings["questions_file"]
    questions = load_questions(qpath)

    summary, results = run_experiment(settings, models_cfg, questions, dry_run=args.dry_run, limit=args.limit)
    write_outputs(summary, results, ROOT / settings["results_dir"])

    print(f"\nDone: {summary['n_results']} records over {summary['n_questions']} questions.")
    print(f"Results written to {ROOT / settings['results_dir']}")
    print(f"  - results.jsonl  - metrics.json  - summary.md")
    if args.dry_run:
        print("(dry-run: no real API calls were made)")


if __name__ == "__main__":
    main()
