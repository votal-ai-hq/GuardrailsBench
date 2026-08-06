"""GuardrailsBench: a benchmark for LLM guardrails that is hard to game.

The dataset is built in five stages (``build.py``), attacked with
intent-preserving transforms (``attacks.py``), scored on four axes at a fixed
over-block budget (``metrics.py``), and checked for the shortcuts and leaks that
make benchmarks look better than they are (``validate.py``).
"""
from .metrics import DEFAULT_WEIGHTS, robustness_delta, score
from .schema import (
    LABEL_ALLOWED,
    LABEL_VIOLATING,
    TIER_ADVERSARIAL,
    TIER_CORE,
    TIER_HARD_NEGATIVE,
    Decision,
    Item,
    Trace,
    load_items,
    load_traces,
)

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_WEIGHTS", "LABEL_ALLOWED", "LABEL_VIOLATING", "TIER_ADVERSARIAL",
    "TIER_CORE", "TIER_HARD_NEGATIVE", "Decision", "Item", "Trace", "load_items",
    "load_traces", "robustness_delta", "score",
]
