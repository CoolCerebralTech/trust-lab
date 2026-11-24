"""Prompt styles. Each style returns the full user text given a question.

All styles share one answer-format contract so parse.py can extract the answer
and confidence with plain code (no AI judge).
"""

ANSWER_FORMAT = (
    "\n\nReply with ONLY your answer, then on a new line write exactly:\n"
    "Confidence: <number between 0 and 100>\n"
    "Example:\nB\nConfidence: 85%"
)


def direct(question: str) -> str:
    return question + "\n\nAnswer with just the correct option or value." + ANSWER_FORMAT


def think_step_by_step(question: str) -> str:
    return (
        question
        + "\n\nThink step by step, but keep it brief. Then give your final answer."
        + ANSWER_FORMAT
    )


def concise(question: str) -> str:
    return question + "\n\nBe concise." + ANSWER_FORMAT


STYLES = {
    "direct": direct,
    "think_step_by_step": think_step_by_step,
    "concise": concise,
}


def prompt_for(style: str, question: str) -> str:
    if style not in STYLES:
        raise ValueError(f"Unknown prompt style: {style!r} (have {sorted(STYLES)})")
    return STYLES[style](question)
