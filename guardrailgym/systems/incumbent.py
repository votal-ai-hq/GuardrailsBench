"""incumbent-replay: the shipped guardrail's recorded verdicts, replayed.

The seed corpus carries the incumbent's own decisions (``input_blocked`` /
``output_blocked`` / ``latency_ms``). Stage 0 of the build drops those columns —
they are the answer key for the core tier — but they are still the most useful
baseline available, because they are the thing under evaluation in production.

Replay is deliberately restricted to core items. Adversarial items inherit
``origin_id`` from the seed row they were derived from, and reusing the seed's
verdict for transformed text would be a fabricated result: the incumbent never
saw that string. Everything outside the core tier is reported as
``NOT_EVALUABLE``, which the scorer turns into a coverage gap and the leaderboard
marks PARTIAL. Getting a real number there means running the live system.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..schema import REASON_NOT_EVALUABLE, TIER_CORE, Decision, Item, Trace
from .base import Guardrail

_TRUE = {"1", "true", "t", "yes", "y"}


def _flag(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in _TRUE


class IncumbentReplay(Guardrail):
    name = "incumbent-replay"

    def __init__(self, verdict_csv: str | Path) -> None:
        self.verdict_csv = Path(verdict_csv)
        self.verdicts: dict[int, dict] = {}
        with open(self.verdict_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    self.verdicts[int(row["id"])] = row
                except (KeyError, TypeError, ValueError):
                    continue

    def _row(self, item: Item) -> dict | None:
        if item.tier != TIER_CORE or item.origin_id is None:
            return None
        return self.verdicts.get(int(item.origin_id))

    def screen_input(self, item: Item) -> Decision:
        row = self._row(item)
        if row is None:
            return Decision(False, None, REASON_NOT_EVALUABLE)
        return Decision(_flag(row, "input_blocked"), None,
                        row.get("fired_policy") or None)

    def screen_output(self, item: Item) -> Decision:
        row = self._row(item)
        if row is None:
            return Decision(False, None, REASON_NOT_EVALUABLE)
        return Decision(_flag(row, "output_blocked"), None,
                        row.get("fired_policy") or None)

    def trace(self, item: Item) -> Trace | None:
        """Replay carries its own recorded latency, so the runner's wall clock
        (which would measure a dict lookup) is overridden here."""
        row = self._row(item)
        if row is None:
            return Trace(item.item_id,
                         Decision(False, None, REASON_NOT_EVALUABLE), None, 0.0)
        inp = self.screen_input(item)
        out = None if inp.blocked else self.screen_output(item)
        try:
            latency = float(row.get("latency_ms") or 0.0)
        except ValueError:
            latency = 0.0
        return Trace(item.item_id, inp, out, latency)
