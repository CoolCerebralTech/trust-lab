"""Extract answer + confidence from model output. Plain code — no AI judge."""

from __future__ import annotations

import re


def extract_confidence(text: str) -> float | None:
    """Return confidence in 0..100, or None if not found."""
    if not text:
        return None
    m = re.search(r"confidence\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*%?", text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return min(100.0, max(0.0, v))
    # Fallback: a lone percentage anywhere in the text.
    m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%", text)
    if m:
        v = float(m.group(1))
        return min(100.0, max(0.0, v))
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def extract_answer(text: str, qtype: str, options: list[str] | None = None) -> str | float | None:
    """Return the model's chosen answer, normalized by question type."""
    if not text:
        return None
    t = _norm(text)
    first = t.split("\n")[0].strip()
    first = re.sub(r"[^a-z0-9 ]", "", first)

    if qtype == "yesno":
        if first.startswith("yes") or first in ("y", "true"):
            return "yes"
        if first.startswith("no") or first in ("n", "false"):
            return "no"
        # Look at the very start of the text for a yes/no token.
        for token in ("yes", "no"):
            if t.startswith(token):
                return token
        return None

    if qtype == "mc":
        if options:
            # Match a bare letter.
            m = re.match(r"^([a-z])$", first)
            if m:
                idx = ord(m.group(1)) - ord("a")
                if 0 <= idx < len(options):
                    return options[idx]
            # Match full option text.
            for o in options:
                if _norm(o) == first:
                    return o
            # Match "option b" / letter anywhere.
            m = re.search(r"\b([a-d])\b", first)
            if m:
                idx = ord(m.group(1)) - ord("a")
                if 0 <= idx < len(options):
                    return options[idx]
        return first  # best-effort raw answer

    if qtype == "numeric":
        m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", first)
        if m:
            return float(m.group(0))
        return None

    return first  # unknown type: return best-effort string
