"""Policy packs: everything about GuardrailsBench that is domain-specific.

The engine is domain-agnostic and always was — ``schema``, ``metrics``,
``runner``, ``report`` and the HTTP adapter contain no finance vocabulary at
all. What was hardwired lived in three places: the categories and hard-negative
templates in ``build``, the attack frame text in ``attacks``, and the lexicon in
``systems.keyword``. A pack is those three things, declared rather than coded,
so benchmarking a different policy domain means writing a YAML file instead of
editing the pipeline.

A pack supplies:

    seed            how to read a seed corpus: column names, label values,
                    category mapping, and the columns to drop because they leak
    hard_negatives  policy-adjacent-but-allowed templates and their slot values
    attacks         the surface forms each attack family wraps a request in,
                    plus the slot extraction that keeps transforms specific
    lexicon         the weighted rules behind the keyword-v1 baseline

Writing a pack for a new domain is the whole porting job. Nothing below this
file needs to know what the policy is about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

PACK_DIR = Path(__file__).parent / "packs"
DEFAULT_PACK = "finance"

_REQUIRED_ATTACK_FAMILIES = (
    "fictional_framing", "romanized_l2", "authority_pretext", "incremental_split",
    "encoding_obfuscation", "indirect_injection", "output_only",
)
_REQUIRED_BENIGN_FAMILIES = (
    "benign_framing", "benign_l2", "benign_multiturn", "benign_document",
)


@dataclass
class PolicyPack:
    """One policy domain's content. Plain data — no behaviour beyond lookup."""

    name: str
    description: str = ""
    seed: dict = field(default_factory=dict)
    hard_negatives: dict = field(default_factory=dict)
    attacks: dict = field(default_factory=dict)
    lexicon: dict = field(default_factory=dict)
    source: str | None = None

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, source: str | None = None) -> PolicyPack:
        known = set(cls.__dataclass_fields__) - {"source"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{source or 'pack'}: unknown keys {sorted(unknown)}")
        pack = cls(**dict(data), source=source)
        pack.validate()
        return pack

    @classmethod
    def load(cls, ref: str | Path) -> PolicyPack:
        """Load by pack name (``finance``) or by path to a .yaml/.json file."""
        path = Path(ref)
        if not path.exists():
            for suffix in (".yaml", ".yml", ".json"):
                candidate = PACK_DIR / f"{ref}{suffix}"
                if candidate.exists():
                    path = candidate
                    break
            else:
                known = sorted(p.stem for p in PACK_DIR.glob("*.y*ml"))
                raise FileNotFoundError(f"no pack {ref!r}; bundled packs: {known}")
        text = path.read_text()
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml  # imported lazily: json packs need no yaml install
            data = yaml.safe_load(text)
        return cls.from_dict(data, source=str(path))

    def validate(self) -> None:
        missing = [f for f in _REQUIRED_ATTACK_FAMILIES + _REQUIRED_BENIGN_FAMILIES
                   if f not in self.attacks]
        if missing:
            raise ValueError(f"{self.source or self.name}: attacks section is missing "
                             f"{missing}")
        for key in ("columns", "label_values"):
            if key not in self.seed:
                raise ValueError(f"{self.source or self.name}: seed.{key} is required")
        for key in ("violating", "allowed"):
            if key not in self.seed["label_values"]:
                raise ValueError(f"{self.source or self.name}: "
                                 f"seed.label_values.{key} is required")
        if not self.hard_negatives.get("templates"):
            raise ValueError(f"{self.source or self.name}: no hard-negative templates")
        # A romanized_l2 core per category, or the family silently renders every
        # item in a category to the same string — the collapse the validator's
        # diversity gate exists for.
        cores = self.attacks["romanized_l2"].get("cores", {})
        default = self.attacks["romanized_l2"].get("default_category")
        if default and default not in cores:
            raise ValueError(f"{self.source or self.name}: romanized_l2.default_category "
                             f"{default!r} has no entry in cores")

    # -- seed corpus -------------------------------------------------------
    @property
    def columns(self) -> dict:
        return self.seed["columns"]

    @property
    def categories(self) -> dict:
        return self.seed.get("categories", {})

    @property
    def drop_columns(self) -> list:
        return list(self.seed.get("drop_columns", []))

    def label_for(self, raw) -> str:
        from .schema import LABEL_ALLOWED, LABEL_VIOLATING
        return (LABEL_VIOLATING if raw == self.seed["label_values"]["violating"]
                else LABEL_ALLOWED)

    # -- sections ----------------------------------------------------------
    def attack(self, family: str) -> dict:
        try:
            return self.attacks[family]
        except KeyError:
            raise KeyError(f"{self.name}: no attack family {family!r}") from None

    @property
    def id_prefix(self) -> str:
        return self.seed.get("id_prefix", "gg")


@cache
def load_pack(ref: str | Path = DEFAULT_PACK) -> PolicyPack:
    """Cached loader. Packs are immutable content, so one instance is enough."""
    return PolicyPack.load(ref)


def bundled_packs() -> list[str]:
    return sorted(p.stem for p in PACK_DIR.glob("*.y*ml"))
