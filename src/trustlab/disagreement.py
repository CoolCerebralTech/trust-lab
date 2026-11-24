"""Cross-model agreement and self-consistency signals.

Compares disagreement vs. stated confidence vs. self-consistency as error
predictors (Stage 4). All plain code.
"""

from __future__ import annotations


def cross_model_agreement(per_question: dict) -> dict:
    """per_question: {qid: {model_name: {answer, correct, confidence}}}

    Returns metrics on whether models agree, and whether agreement correlates
    with being right vs. wrong.
    """
    agree_total = 0
    agree_right = 0
    agree_wrong = 0
    disagree_total = 0
    all_q = []

    for qid, model_results in per_question.items():
        # Only consider models that produced a parsable answer for this question.
        answers = {
            m: r["answer"]
            for m, r in model_results.items()
            if r.get("answer") is not None
        }
        if len(answers) < 2:
            continue
        uniq = set(answers.values())
        all_same = len(uniq) == 1
        # Is the common answer correct? (use correctness of first model)
        sample = next(iter(model_results.values()))
        correct = sample.get("correct")
        all_q.append({"qid": qid, "agree": all_same, "correct": correct})

        if all_same:
            agree_total += 1
            if correct is True:
                agree_right += 1
            elif correct is False:
                agree_wrong += 1
        else:
            disagree_total += 1

    total = agree_total + disagree_total
    return {
        "questions_compared": total,
        "agree_count": agree_total,
        "disagree_count": disagree_total,
        "agree_rate": round(agree_total / total, 3) if total else 0.0,
        "agree_when_right": agree_right,
        "agree_when_wrong": agree_wrong,
        "details": all_q,
    }


def self_consistency(samples_by_question: dict) -> dict:
    """samples_by_question: {qid: [ {answer, correct} ] from repeated sampling.

    Returns the fraction of questions where all samples agree (self-consistent)
    and whether those agree-answers tend to be right.
    """
    consistent = 0
    consistent_right = 0
    total = 0
    for qid, samples in samples_by_question.items():
        answers = [s.get("answer") for s in samples if s.get("answer") is not None]
        if len(answers) < 2:
            continue
        total += 1
        if len(set(answers)) == 1:
            consistent += 1
            if samples[0].get("correct") is True:
                consistent_right += 1
    return {
        "questions_compared": total,
        "consistent_count": consistent,
        "consistent_rate": round(consistent / total, 3) if total else 0.0,
        "consistent_when_right": consistent_right,
    }
