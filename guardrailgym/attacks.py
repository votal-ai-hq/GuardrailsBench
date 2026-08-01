"""Intent-preserving adversarial transforms.

Design rule: a transform may change SURFACE FORM ONLY. The violating request
must survive intact — if a human reviewer would no longer call the output a
policy violation, the transform is invalid and the item is dropped. Each
family below is grounded in an observed failure of the incumbent guardrail on
the seed corpus (see reports/failure_taxonomy.md).

The *shape* of each attack lives here; the *words* live in a policy pack
(``pack.py``). Wrapping a violating request in a screenplay frame is a
domain-independent idea — which screenplay, and what counts as a violation, are
not. Every transform takes the pack it draws text from, so porting to another
policy domain is a YAML file rather than a fork of this module.
"""
from __future__ import annotations

import base64
import random
import re
from functools import cache

from .pack import DEFAULT_PACK, PolicyPack, load_pack
from .schema import TIER_ADVERSARIAL, Item

# ---- slot extraction: keeps transforms specific rather than generic ----------
# Which nouns matter is a property of the domain, so the patterns come from the
# pack. The fallback chain — closed list, then any capitalised token that is not
# a sentence opener, then a default — does not.


@cache
def _compiled(amount: str, subject: str, proper: str, age: str):
    return (re.compile(amount), re.compile(subject), re.compile(proper),
            re.compile(age, re.I))


def _patterns(pack: PolicyPack):
    s = pack.attacks["slots"]
    return _compiled(s["amount_regex"], s["subject_regex"], s["proper_noun_regex"],
                     s["age_regex"])


def _slots(text: str, pack: PolicyPack) -> dict:
    amt_re, subject_re, proper_re, age_re = _patterns(pack)
    cfg = pack.attacks["slots"]
    defaults, stop = cfg["defaults"], set(cfg["stopwords"])
    m_amt, m_age = amt_re.search(text), age_re.search(text)
    subject_hit = subject_re.search(text)
    if subject_hit:
        subject = subject_hit.group(0)
    else:
        cands = [c for c in proper_re.findall(text) if c not in stop]
        subject = cands[0] if cands else defaults["subject"]
    return {
        "amount": m_amt.group(0) if m_amt else defaults["amount"],
        "age": m_age.group(1) if m_age else defaults["age"],
        "subject": subject,
    }


def _derive(item: Item, family: str, turns, response=None, docs=None, idx: int = 0) -> Item:
    return Item(
        item_id=f"{item.item_id}::{family}{idx}",
        turns=turns,
        response=response if response is not None else item.response,
        label=item.label,
        category=item.category,
        tier=TIER_ADVERSARIAL,
        attack_family=family,
        context_docs=docs or [],
        cluster_id=item.cluster_id,      # stays in the same cluster — no split leakage
        origin_id=item.origin_id,
        meta={"base_item": item.item_id},
    )


def _u(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---- F1: fictional / professional framing -----------------------------------
# Observed: 5 of 12 incumbent misses. Satire, journalism, screenplay, speech.
def fictional_framing(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    frame = rng.choice(pack.attack("fictional_framing")["frames"])
    return _derive(item, "fictional_framing", _u(frame.format(core=item.prompt)))


# ---- F2: romanized L2 --------------------------------------------------------
# Observed: 4 of 12 misses. Latin-script transliteration of he/ja/ko/th defeats
# lexical matching and degrades English-tuned classifiers.
#
# The pack carries one core per (language, policy category). A single shared
# core made every non-investment item render to the same string — the
# validator's diversity gate exists because that shipped once, and the pack
# loader re-checks that the default category has an entry.
def romanized_l2(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("romanized_l2")
    wrappers, cores = cfg["wrappers"], cfg["cores"]
    lang = rng.choice(list(wrappers))
    cat = item.category or cfg["default_category"]
    s = _slots(item.prompt, pack)
    core = cores.get(cat, cores[cfg["default_category"]])[lang].format(subject=s["subject"])
    text = wrappers[lang].format(age=s["age"], amount=s["amount"],
                                 **{f"core_{lang}": core}) + rng.choice(cfg["tails"])
    it = _derive(item, "romanized_l2", _u(text))
    it.meta.update({"lang": lang, "category": cat, "subject": s["subject"]})
    return it


# ---- F3: authority pretext ---------------------------------------------------
def authority_pretext(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    pretext = rng.choice(pack.attack("authority_pretext")["pretexts"])
    return _derive(item, "authority_pretext", _u(pretext.format(core=item.prompt)))


# ---- F4: incremental split (multi-turn) --------------------------------------
# Splits the violating request across turns so no single turn trips a
# per-message classifier. Tests whether the guardrail carries conversation state.
def incremental_split(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("incremental_split")
    setups, acks = cfg["setups"], cfg["acknowledgements"]
    p = item.prompt
    # split at a sentence boundary so the violating ask lands in the final turn
    parts = re.split(r"(?<=[.?!])\s+", p)
    if len(parts) < 2:
        head, tail = rng.choice(setups), p
    else:
        head, tail = " ".join(parts[:-1]), parts[-1]
    turns = [
        {"role": "user", "content": rng.choice(setups)},
        {"role": "assistant", "content": acks[0]},
        {"role": "user", "content": head},
        {"role": "assistant", "content": acks[1 % len(acks)]},
        {"role": "user", "content": tail},
    ]
    return _derive(item, "incremental_split", turns)


# ---- F5: encoding obfuscation -----------------------------------------------
_ZWSP = "​"


@cache
def _leet_table(pairs: tuple) -> dict:
    return str.maketrans(dict(pairs))


def encoding_obfuscation(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("encoding_obfuscation")
    mode = rng.choice(cfg["modes"])
    p = item.prompt
    if mode == "base64":
        blob = base64.b64encode(p.encode()).decode()
        text = cfg["base64_instruction"].format(blob=blob)
    elif mode == "leet":
        text = p.translate(_leet_table(tuple(cfg["leet_map"].items())))
    else:
        text = _ZWSP.join(p)  # zero-width joiner between every char
    it = _derive(item, "encoding_obfuscation", _u(text))
    it.meta["mode"] = mode
    return it


# ---- F6: indirect injection --------------------------------------------------
# The violating instruction arrives via retrieved content, not the user turn.
# The user turn is entirely benign — tests whether the guard screens context.
def indirect_injection(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("indirect_injection")
    doc = cfg["document"].format(core=item.prompt)
    return _derive(item, "indirect_injection", _u(rng.choice(cfg["carriers"])), docs=[doc])


# ---- F7: output-only violation ----------------------------------------------
# Benign prompt, violating RESPONSE. The seed corpus has almost none of these
# (every violating row violates at both stages), yet it is the dominant real
# failure: RAG or a fine-tune leaks policy-violating content unprompted.
def output_only(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("output_only")
    if not item.response or len(item.response) < cfg["min_response_length"]:
        return None
    it = _derive(item, "output_only", _u(rng.choice(cfg["carriers"])),
                 response=item.response)
    it.meta["note"] = cfg["note"]
    return it


# =============================================================================
# ADVERSARIAL-BENIGN: the same surface forms, applied to ALLOWED requests.
# Without these the adversarial tier is single-class and ASR rewards a guardrail
# that simply blocks anything unusual-looking. A real journalist, a real
# Korean speaker, and a real pasted document all exist in production traffic.
# =============================================================================
def benign_framing(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    frame = rng.choice(pack.attack("benign_framing")["frames"])
    return _derive(item, "benign_framing", _u(frame.format(core=item.prompt)))


def benign_l2(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    wrappers = pack.attack("benign_l2")["wrappers"]
    lang = rng.choice(list(wrappers))
    it = _derive(item, "benign_l2", _u(wrappers[lang].format(core=item.prompt)))
    it.meta["lang"] = lang
    return it


def benign_multiturn(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    cfg = pack.attack("benign_multiturn")
    turns = [
        {"role": "user", "content": rng.choice(cfg["setups"])},
        {"role": "assistant", "content": cfg["acknowledgements"][0]},
        {"role": "user", "content": item.prompt},
    ]
    return _derive(item, "benign_multiturn", turns)


def benign_document(item: Item, rng: random.Random, pack: PolicyPack) -> Item | None:
    return _derive(item, "benign_document", _u(item.prompt),
                   docs=[pack.attack("benign_document")["document"]])


# Declaration order is load-bearing twice over: it seeds the adversarial build
# loop (so it feeds the RNG in a fixed sequence) and it breaks ties in the
# leaderboard's per-family tables.
_VIOLATING = {
    "fictional_framing": fictional_framing,
    "romanized_l2": romanized_l2,
    "authority_pretext": authority_pretext,
    "incremental_split": incremental_split,
    "encoding_obfuscation": encoding_obfuscation,
    "indirect_injection": indirect_injection,
    "output_only": output_only,
}

_BENIGN = {
    "benign_framing": benign_framing,
    "benign_l2": benign_l2,
    "benign_multiturn": benign_multiturn,
    "benign_document": benign_document,
}


def _bind(fn, pack: PolicyPack):
    def bound(item: Item, rng: random.Random) -> Item | None:
        return fn(item, rng, pack)
    bound.__name__ = fn.__name__
    bound.__doc__ = fn.__doc__
    bound.pack = pack
    return bound


def _resolve(pack: PolicyPack | str) -> PolicyPack:
    return pack if isinstance(pack, PolicyPack) else load_pack(pack)


def families(pack: PolicyPack | str = DEFAULT_PACK) -> dict:
    """Attack families bound to a pack: ``{name: fn(item, rng) -> Item | None}``."""
    resolved = _resolve(pack)
    return {name: _bind(fn, resolved) for name, fn in _VIOLATING.items()}


def benign_families(pack: PolicyPack | str = DEFAULT_PACK) -> dict:
    """The benign mirrors, bound the same way."""
    resolved = _resolve(pack)
    return {name: _bind(fn, resolved) for name, fn in _BENIGN.items()}


# Default-pack views, kept as module constants because the family *names* are a
# stable part of the schema even when the text behind them is not.
FAMILIES = families()
BENIGN_FAMILIES = benign_families()
