"""keyword-v1: a weighted lexical policy matcher.

This is the baseline most teams ship first, and it is here to establish what
surface-form matching actually buys. The lexicon is organised the way a real one
is: strong cues that are close to dispositive on their own, weaker cues that
need company, and negations for the framings that make a policy-adjacent request
legitimate. Without that last group the rule set fires on the compliance officer
asking about insider-trading controls just as hard as on the person trying to
commit insider trading.

What it cannot do is read through surface form. The by-family ASR table is the
point of the baseline: transliterate the request, encode it, or split it across
turns and a lexicon is looking at the wrong string.

The rules themselves are pack content — a lexicon is the most domain-specific
thing in the repository, and hardcoding one here would mean every new policy
domain forks this file. The weighting scheme, the negation subtraction and the
score flooring stay here, because those are how a lexicon works regardless of
what it is about.
"""
from __future__ import annotations

import re
from functools import cache

from ..pack import DEFAULT_PACK, PolicyPack, load_pack
from ..schema import Item
from .base import ScoreGuardrail


def _expand(pattern: str, fragments: dict) -> str:
    """Substitute ``{name}`` fragments only.

    Deliberately not ``str.format``: a regex is full of braces that are not
    placeholders, and ``.{0,40}`` would raise KeyError('0,40').
    """
    for name, value in fragments.items():
        pattern = pattern.replace("{" + name + "}", value)
    return pattern


@cache
def _rules(pack_name: str, section: str, extends: str | None = None):
    """Compile one rule section, resolving the pack's named fragments.

    Fragments exist so a pack can name the verb and noun families once instead
    of repeating them across a dozen patterns.
    """
    pack = load_pack(pack_name)
    fragments = pack.lexicon.get("fragments", {})
    raw = list(pack.lexicon.get(extends, [])) if extends else []
    raw += list(pack.lexicon.get(section, []))
    return tuple((float(w), re.compile(_expand(pat, fragments), re.I))
                 for w, pat in raw)


def _score(rules, text: str) -> float:
    """Weighted hits, negations subtracted, floored at zero.

    Flooring matters: without it a heavily-hedged response could go arbitrarily
    negative and drag the calibrated threshold down with it.
    """
    return max(0.0, sum(w for w, p in rules if p.search(text)))


class KeywordGuardrail(ScoreGuardrail):
    name = "keyword-v1"
    takes_pack = True       # the harness passes the active pack to __init__

    def __init__(self, pack: PolicyPack | str = DEFAULT_PACK) -> None:
        super().__init__()
        self.pack_name = pack.name if isinstance(pack, PolicyPack) else pack

    @property
    def _input_rules(self):
        return _rules(self.pack_name, "input_negations", extends="input_rules")

    @property
    def _output_rules(self):
        # Output screening inherits every input rule — a leak and a request for
        # a leak share most of their vocabulary — then adds the shapes a
        # complied-with answer takes, and its own hedging negations.
        return (_rules(self.pack_name, "output_rules", extends="input_rules")
                + _rules(self.pack_name, "output_negations"))

    def input_score(self, item: Item) -> float:
        return _score(self._input_rules, item.input_text)

    def output_score(self, item: Item) -> float:
        return _score(self._output_rules, item.response)
