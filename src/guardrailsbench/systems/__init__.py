"""Guardrail registry.

Register a system here and it becomes available to ``python -m guardrailsbench eval
--system <name>`` and to the leaderboard. Entries are factories, not instances,
because cross-validation fits a fresh copy per fold.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import DEFAULT_MAX_OVERBLOCK, Guardrail, ScoreGuardrail
from .control import AllowAll, BlockAll, LengthStump
from .http_api import HttpConfig, HttpGuardrail
from .incumbent import IncumbentReplay
from .keyword import KeywordGuardrail
from .tfidf_lr import TfidfLRGuardrail

REGISTRY: dict[str, Callable[[], Guardrail]] = {
    TfidfLRGuardrail.name: TfidfLRGuardrail,
    KeywordGuardrail.name: KeywordGuardrail,
    LengthStump.name: LengthStump,
    BlockAll.name: BlockAll,
    AllowAll.name: AllowAll,
}

# Systems on the published leaderboard, in the order they are evaluated. The
# controls stay out of the default set — they are diagnostics, run them
# explicitly with --system.
DEFAULT_SYSTEMS = [TfidfLRGuardrail.name, KeywordGuardrail.name, LengthStump.name]


def get(name: str) -> Callable[[], Guardrail]:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown system {name!r}; known: {sorted(REGISTRY)}") from None


__all__ = [
    "DEFAULT_MAX_OVERBLOCK", "DEFAULT_SYSTEMS", "REGISTRY", "AllowAll", "BlockAll",
    "Guardrail", "HttpConfig", "HttpGuardrail", "IncumbentReplay",
    "KeywordGuardrail", "LengthStump", "ScoreGuardrail", "TfidfLRGuardrail", "get",
]
