"""Stage 6 stub: designed forgetting agent.

Attack harness that compares pruning, expiring, and quarantining of model
memories/facts. Implemented in Stage 6; this module holds the interface so the
rest of the project can import it without breaking.
"""

from __future__ import annotations


class ForgettingAgent:
    """Placeholder. Strategies: 'prune', 'expire', 'quarantine'."""

    def __init__(self, strategy: str = "quarantine"):
        if strategy not in ("prune", "expire", "quarantine"):
            raise ValueError("strategy must be one of: prune, expire, quarantine")
        self.strategy = strategy

    def should_forget(self, fact, age_days: int, confidence: float) -> bool:
        """Decide whether to forget a fact. Override in Stage 6 implementation."""
        raise NotImplementedError("Stage 6 not yet implemented")
