"""Canonical data types for GuardrailGym.

An ``Item`` is one evaluation unit: the conversation that reaches the guardrail,
the model response that comes back, and the ground-truth label. A guardrail is
screened at two stages (input, output), and the pair of ``Decision``s plus the
wall-clock cost is a ``Trace``. Everything the metrics module needs is in those
three types — no system-specific state leaks into scoring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---- label / tier vocabulary -------------------------------------------------
LABEL_VIOLATING = "violating"
LABEL_ALLOWED = "allowed"
LABELS = (LABEL_VIOLATING, LABEL_ALLOWED)

TIER_CORE = "core"                   # length-matched, in-distribution
TIER_HARD_NEGATIVE = "hard_negative"  # policy-adjacent but allowed
TIER_ADVERSARIAL = "adversarial"     # attack families + their benign mirrors
TIERS = (TIER_CORE, TIER_HARD_NEGATIVE, TIER_ADVERSARIAL)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# A guardrail that cannot produce a verdict for an item must say so rather than
# silently returning "allow" — that is the difference between a coverage gap and
# a miss, and scoring them the same is how partial systems get free credit.
REASON_NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class Item:
    item_id: str
    turns: list[dict]
    response: str
    label: str
    category: str | None = None
    tier: str = TIER_CORE
    attack_family: str | None = None
    context_docs: list[str] = field(default_factory=list)
    cluster_id: int | None = None
    origin_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience views used by attacks and by guardrail implementations ----
    @property
    def prompt(self) -> str:
        """The final user turn — the ask the guardrail has to rule on."""
        for turn in reversed(self.turns):
            if turn.get("role") == ROLE_USER:
                return str(turn.get("content", ""))
        return ""

    @property
    def user_text(self) -> str:
        """Every user turn, oldest first. Multi-turn attacks split the violating
        request across turns, so a per-message view is not enough."""
        return "\n".join(str(t.get("content", "")) for t in self.turns
                         if t.get("role") == ROLE_USER)

    @property
    def context_text(self) -> str:
        return "\n".join(self.context_docs)

    @property
    def input_text(self) -> str:
        """Everything available at the input stage: the conversation plus any
        retrieved context. Indirect-injection items are only visible here."""
        return "\n".join(p for p in (self.user_text, self.context_text) if p)

    @property
    def is_violating(self) -> bool:
        return self.label == LABEL_VIOLATING

    # -- (de)serialization ----------------------------------------------------
    def to_json(self) -> dict:
        return {
            "item_id": self.item_id,
            "turns": self.turns,
            "response": self.response,
            "label": self.label,
            "category": self.category,
            "tier": self.tier,
            "attack_family": self.attack_family,
            "context_docs": self.context_docs,
            "cluster_id": self.cluster_id,
            "origin_id": self.origin_id,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> Item:
        return cls(
            item_id=d["item_id"],
            turns=list(d["turns"]),
            response=d.get("response", ""),
            label=d["label"],
            category=d.get("category"),
            tier=d.get("tier", TIER_CORE),
            attack_family=d.get("attack_family"),
            context_docs=list(d.get("context_docs") or []),
            cluster_id=d.get("cluster_id"),
            origin_id=d.get("origin_id"),
            meta=dict(d.get("meta") or {}),
        )

    def validate(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"{self.item_id}: bad label {self.label!r}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.item_id}: bad tier {self.tier!r}")
        if not self.turns:
            raise ValueError(f"{self.item_id}: no turns")
        for t in self.turns:
            if t.get("role") not in (ROLE_USER, ROLE_ASSISTANT):
                raise ValueError(f"{self.item_id}: bad role {t.get('role')!r}")
        if not self.prompt.strip():
            raise ValueError(f"{self.item_id}: empty final user turn")
        if self.tier == TIER_ADVERSARIAL and not self.attack_family:
            raise ValueError(f"{self.item_id}: adversarial item without a family")
        if self.tier != TIER_ADVERSARIAL and self.attack_family:
            raise ValueError(f"{self.item_id}: attack_family outside adversarial tier")


@dataclass
class Decision:
    blocked: bool
    score: float | None = None
    reason: str | None = None

    @property
    def evaluable(self) -> bool:
        return self.reason != REASON_NOT_EVALUABLE


@dataclass
class Trace:
    """One guardrail's run over one item.

    ``output_decision`` is None when the input stage already blocked — there was
    no response to screen. That asymmetry is what makes stage attribution
    meaningful, so it is preserved rather than filled in with a dummy allow.
    """
    item_id: str
    input_decision: Decision | None = None
    output_decision: Decision | None = None
    latency_ms: float = 0.0

    @property
    def decisions(self) -> list[Decision]:
        return [d for d in (self.input_decision, self.output_decision) if d is not None]

    @property
    def evaluable(self) -> bool:
        ds = self.decisions
        return bool(ds) and all(d.evaluable for d in ds)

    @property
    def blocked(self) -> bool:
        return any(d.blocked for d in self.decisions)

    @property
    def blocked_at(self) -> str | None:
        if self.input_decision is not None and self.input_decision.blocked:
            return "input"
        if self.output_decision is not None and self.output_decision.blocked:
            return "output"
        return None

    def to_json(self) -> dict:
        def _d(d):
            return None if d is None else {"blocked": d.blocked, "score": d.score,
                                           "reason": d.reason}
        return {"item_id": self.item_id, "input": _d(self.input_decision),
                "output": _d(self.output_decision), "latency_ms": self.latency_ms}

    @classmethod
    def from_json(cls, d: dict) -> Trace:
        def _d(x):
            return None if x is None else Decision(bool(x["blocked"]), x.get("score"),
                                                   x.get("reason"))
        return cls(d["item_id"], _d(d.get("input")), _d(d.get("output")),
                   float(d.get("latency_ms", 0.0)))


# ---- jsonl helpers -----------------------------------------------------------
def load_items(path) -> list[Item]:
    with open(path) as f:
        return [Item.from_json(json.loads(line)) for line in f if line.strip()]


def dump_items(items, path) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_json()) + "\n")


def load_traces(path) -> dict[str, Trace]:
    with open(path) as f:
        traces = [Trace.from_json(json.loads(line)) for line in f if line.strip()]
    return {t.item_id: t for t in traces}


def dump_traces(traces, path) -> None:
    vals = traces.values() if isinstance(traces, dict) else traces
    with open(path, "w") as f:
        for t in vals:
            f.write(json.dumps(t.to_json()) + "\n")
