"""Controls. None of these read the request; they exist to bound the leaderboard.

CONTROL-length-stump is the shortcut probe. Stage 2 of the build length-matches
the core tier on a joint (prompt, response) grid precisely so this thing cannot
work; if it ever climbs the leaderboard again, the dataset regressed, not the
state of the art.

CONTROL-block-all and CONTROL-allow-all are the degenerate corners. A metric on
which either of them looks respectable is a broken metric — block-all gets ASR
0.000 for free, and allow-all gets a perfect over-block rate for free.
"""
from __future__ import annotations

from ..schema import Decision, Item
from .base import Guardrail, ScoreGuardrail


class LengthStump(ScoreGuardrail):
    """Blocks long text. Reads nothing else."""

    name = "CONTROL-length-stump"

    def input_score(self, item: Item) -> float:
        return float(len(item.input_text))

    def output_score(self, item: Item) -> float:
        return float(len(item.response))


class BlockAll(Guardrail):
    name = "CONTROL-block-all"

    def screen_input(self, item: Item) -> Decision:
        return Decision(True, 1.0, "block_all")

    def screen_output(self, item: Item) -> Decision:
        return Decision(True, 1.0, "block_all")


class AllowAll(Guardrail):
    name = "CONTROL-allow-all"

    def screen_input(self, item: Item) -> Decision:
        return Decision(False, 0.0, "allow_all")

    def screen_output(self, item: Item) -> Decision:
        return Decision(False, 0.0, "allow_all")
