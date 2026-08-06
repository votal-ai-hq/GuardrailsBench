"""GuardrailsBench dataset build pipeline.

Stages
  0 ingest        seed CSV -> canonical Items, drop leaky metadata
  1 cluster       near-dup detection -> cluster ids (splits are cluster-disjoint)
  2 neutralize    length-match allowed/violating pools to kill the length shortcut
  3 hard_negs     synthesize policy-adjacent-but-allowed items
  4 adversarial   apply intent-preserving attack families to violating items
  5 split         cluster-disjoint, tier-stratified dev/test
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .attacks import benign_families, families
from .pack import DEFAULT_PACK, PolicyPack, load_pack
from .schema import LABEL_ALLOWED, LABEL_VIOLATING, TIER_CORE, TIER_HARD_NEGATIVE, Item

SEED = 1337


def _pack(pack: PolicyPack | str | None) -> PolicyPack:
    if isinstance(pack, PolicyPack):
        return pack
    return load_pack(pack or DEFAULT_PACK)


# ---------------- stage 0: ingest --------------------------------------------
def ingest(csv_path: str, pack: PolicyPack | str | None = None) -> list[Item]:
    """Seed CSV -> canonical Items.

    Only the five columns the pack names are read. Everything else is dropped on
    the floor, which is the point: the seed export also carries the incumbent's
    own verdicts and a latency column that leaks the label (a stump on it hit
    70.7%). A column that is never read cannot leak.
    """
    pk = _pack(pack)
    cols = pk.columns
    df = pd.read_csv(csv_path)
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}; the pack expects "
                         f"{sorted(cols.values())} but the file has {list(df.columns)}")
    has_category = "category" in cols
    items = []
    for _, r in df.iterrows():
        label = pk.label_for(r[cols["label"]])
        raw_cat = r[cols["category"]] if has_category else None
        cat = pk.categories.get(raw_cat) if isinstance(raw_cat, str) else None
        items.append(Item(
            item_id=f"{pk.id_prefix}-{int(r[cols['id']]):06d}",
            turns=[{"role": "user", "content": str(r[cols["prompt"]])}],
            response=str(r[cols["response"]]),
            label=label,
            category=cat,
            tier=TIER_CORE,
            origin_id=int(r[cols["id"]]),
        ))
    return items


# ---------------- stage 1: near-dup clustering -------------------------------
def _shingles(text: str, k: int = 5) -> set:
    w = re.findall(r"[a-z0-9$]+", text.lower())
    if len(w) < k:
        return {tuple(w)}
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def cluster(items: list[Item], threshold: float = 0.5) -> list[Item]:
    """Union-find over 5-gram Jaccard. Prevents paraphrase leakage across splits."""
    sh = [_shingles(it.prompt) for it in items]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    inv = collections.defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            inv[g].append(i)
    seen = set()
    for _g, ids in inv.items():
        if len(ids) < 2 or len(ids) > 60:   # skip ultra-common shingles
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pair = (ids[a], ids[b])
                if pair in seen:
                    continue
                seen.add(pair)
                inter = len(sh[ids[a]] & sh[ids[b]])
                union_sz = len(sh[ids[a]] | sh[ids[b]])
                if union_sz and inter / union_sz >= threshold:
                    union(ids[a], ids[b])
    for i, it in enumerate(items):
        it.cluster_id = find(i)
    return items


# ---------------- stage 2: shortcut neutralization ---------------------------
def _bin_index(vals: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    return np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, n_bins - 1)


def length_match(items: list[Item], rng: random.Random,
                 n_bins_prompt: int = 8, n_bins_response: int = 4) -> list[Item]:
    """Subsample within JOINT (prompt-length, response-length) cells.

    Matching prompt length alone is not enough: in the seed corpus the allowed
    *responses* were short factual answers and the violating ones were long
    detailed advice, so a stump on response length still reached 0.846 and the
    output-stage guard could ride it. Both marginals have to be flattened, so
    the grid is 2-D. Cells that are single-class carry only shortcut signal and
    are dropped whole.

    The grid is asymmetric (8 x 4) because prompt length was the stronger
    confound. Finer grids flatten both probes further but shrink the tier fast;
    see docs/METRICS.md for the sweep.
    """
    plen = np.array([len(it.prompt) for it in items])
    rlen = np.array([len(it.response) for it in items])
    pb, rb = _bin_index(plen, n_bins_prompt), _bin_index(rlen, n_bins_response)
    keep: list[Item] = []
    for cell in {(a, b) for a, b in zip(pb, rb, strict=True)}:
        pool = [it for it, a, b in zip(items, pb, rb, strict=True) if (a, b) == cell]
        viol = [it for it in pool if it.label == LABEL_VIOLATING]
        allw = [it for it in pool if it.label == LABEL_ALLOWED]
        n = min(len(viol), len(allw))
        if n == 0:
            continue
        keep += rng.sample(viol, n) + rng.sample(allw, n)
    return keep


# ---------------- stage 3: hard negatives ------------------------------------
# Policy-adjacent and ALLOWED. The finance pack's templates are anchored on the
# three real over-blocks in the seed corpus, which were all "employee/bank asks
# about its own controls".
def make_hard_negatives(n: int, rng: random.Random,
                        pack: PolicyPack | str | None = None) -> list[Item]:
    pk = _pack(pack)
    templates = pk.hard_negatives["templates"]
    slots = pk.hard_negatives.get("slots", {})
    slot_names = list(slots)
    out = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        # Slot values are drawn in a fixed key order so the RNG sequence is a
        # property of the pack, not of dict iteration luck.
        values = {name: rng.choice(slots[name]) for name in slot_names}
        text = tmpl["prompt"].format(**values)
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        out.append(Item(
            item_id=f"{pk.id_prefix}-hn-{i:04d}",
            turns=[{"role": "user", "content": text}],
            response=tmpl["response"],
            label=LABEL_ALLOWED,
            category=None,
            tier=TIER_HARD_NEGATIVE,
            cluster_id=10_000_000 + (i % len(templates)),  # dup-safe by template
            meta={"template": i % len(templates), "hash": h},
        ))
    return out


# ---------------- stage 4: adversarial arena ---------------------------------
def make_adversarial(items: list[Item], per_family: int, rng: random.Random,
                     pack: PolicyPack | str | None = None) -> list[Item]:
    """Violating items get attack transforms; ALLOWED items get the same surface
    forms with benign intent. Both land in the adversarial tier so that ASR is
    measured against a two-class pool and paranoia is not rewarded."""
    pk = _pack(pack)
    viol = [it for it in items if it.label == LABEL_VIOLATING]
    allw = [it for it in items if it.label == LABEL_ALLOWED]
    out = []
    for fam, fn in families(pk).items():
        for base in rng.sample(viol, min(per_family, len(viol))):
            made = fn(base, rng)
            if made is not None:
                made.item_id = f"{base.item_id}::{fam}"
                out.append(made)
    for fam, fn in benign_families(pk).items():
        for base in rng.sample(allw, min(per_family, len(allw))):
            made = fn(base, rng)
            if made is not None:
                made.item_id = f"{base.item_id}::{fam}"
                out.append(made)
    return out


# ---------------- stage 5: split ---------------------------------------------
def split(items: list[Item], rng: random.Random, dev_frac: float = 0.3):
    """Cluster-disjoint split. No paraphrase of a test item may appear in dev."""
    clusters = sorted({it.cluster_id for it in items})
    rng.shuffle(clusters)
    n_dev = int(len(clusters) * dev_frac)
    dev_c = set(clusters[:n_dev])
    dev = [it for it in items if it.cluster_id in dev_c]
    test = [it for it in items if it.cluster_id not in dev_c]
    return dev, test


# ---------------- orchestration ----------------------------------------------
def build(csv_path: str, out_dir: str, per_family: int = 120, n_hard_neg: int = 400,
          pack: PolicyPack | str | None = None):
    pk = _pack(pack)
    rng = random.Random(SEED)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = ingest(csv_path, pk)
    raw = cluster(raw)
    core = length_match(raw, rng)
    hard = make_hard_negatives(n_hard_neg, rng, pk)
    adv = make_adversarial(core + hard, per_family, rng, pk)

    allitems = core + hard + adv
    dev, test = split(allitems, rng)

    for name, part in [("dev", dev), ("test", test), ("all", allitems)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for it in part:
                f.write(json.dumps(it.to_json()) + "\n")

    stats = {
        "pack": pk.name,
        "seed_rows": len(raw),
        "clusters": len({i.cluster_id for i in raw}),
        "core": len(core),
        "hard_negative": len(hard),
        "adversarial": len(adv),
        "total": len(allitems),
        "dev": len(dev), "test": len(test),
        "adv_by_family": dict(collections.Counter(i.attack_family for i in adv)),
        "dropped_columns": pk.drop_columns,
    }
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
