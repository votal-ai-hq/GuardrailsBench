"""Dataset integrity checks.

Every gate here exists because the corresponding mistake was actually made:

* the **shortcut gate** — a length stump reached 0.846 AUC on the seed corpus,
  so the core tier is length-matched and re-checked here;
* the **diversity gate** — one shared L2 core string made every non-investment
  item render to the same sentence, which turned a 90-item family into one item
  measured ninety times;
* the **two-class gate** — an all-violating adversarial tier lets a
  block-everything system take the top of the leaderboard on ASR alone;
* the **leakage gate** — attack items inherit their base item's cluster, and if
  that ever stops holding, a paraphrase of a test item can be sitting in dev.

``validate_dataset`` returns a list of human-readable problems; empty means pass.
"""
from __future__ import annotations

import collections

from .schema import (
    LABEL_ALLOWED,
    LABEL_VIOLATING,
    TIER_ADVERSARIAL,
    TIER_CORE,
    TIER_HARD_NEGATIVE,
    Item,
)

MIN_FAMILY_SIZE = 20
MIN_FAMILY_DIVERSITY = 0.5      # distinct prompts / items, within a family
MAX_SHORTCUT_AUC = 0.65         # a length stump must be near chance on core


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney AUC with tie correction. Kept dependency-free."""
    if not scores_pos or not scores_neg:
        return 0.5
    combined = sorted([(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg])
    ranks: dict[float, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_s, lbl) in enumerate(combined) if lbl == 1)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def shortcut_auc(items: list[Item], field: str = "prompt") -> float:
    """How well raw text length alone separates violating from allowed."""
    def _len(it: Item) -> float:
        return float(len(it.prompt if field == "prompt" else it.response))
    pos = [_len(i) for i in items if i.label == LABEL_VIOLATING]
    neg = [_len(i) for i in items if i.label == LABEL_ALLOWED]
    a = auc(pos, neg)
    return max(a, 1.0 - a)      # a shortcut that fires backwards is still a shortcut


def validate_dataset(items: list[Item]) -> list[str]:
    problems: list[str] = []

    # ---- structural -----------------------------------------------------
    for it in items:
        try:
            it.validate()
        except ValueError as e:
            problems.append(f"schema: {e}")

    dupes = [k for k, n in collections.Counter(i.item_id for i in items).items() if n > 1]
    if dupes:
        problems.append(f"duplicate item_ids: {dupes[:5]} ({len(dupes)} total)")

    # ---- contradictory labels -------------------------------------------
    by_text: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for it in items:
        by_text[(it.prompt, it.response)].add(it.label)
    clashes = [t for t, labels in by_text.items() if len(labels) > 1]
    if clashes:
        problems.append(f"{len(clashes)} identical (prompt, response) pairs carry "
                        "both labels")

    # ---- tier composition ------------------------------------------------
    tiers = collections.Counter(i.tier for i in items)
    for tier in (TIER_CORE, TIER_ADVERSARIAL):
        if not tiers.get(tier):
            problems.append(f"tier {tier} is empty")

    hn_bad = [i.item_id for i in items
              if i.tier == TIER_HARD_NEGATIVE and i.label != LABEL_ALLOWED]
    if hn_bad:
        problems.append(f"hard negatives must be allowed: {hn_bad[:5]}")

    adv = [i for i in items if i.tier == TIER_ADVERSARIAL]
    adv_labels = {i.label for i in adv}
    if adv and adv_labels != {LABEL_VIOLATING, LABEL_ALLOWED}:
        problems.append("adversarial tier is single-class — ASR would reward "
                        f"blanket blocking (labels present: {sorted(adv_labels)})")

    # ---- per-family size and diversity ----------------------------------
    fams: dict[str, list[Item]] = collections.defaultdict(list)
    for i in adv:
        fams[i.attack_family].append(i)
    for fam, group in sorted(fams.items()):
        if len(group) < MIN_FAMILY_SIZE:
            problems.append(f"family {fam}: only {len(group)} items "
                            f"(< {MIN_FAMILY_SIZE}), ASR estimate is noise")
        # Distinctness is measured over everything the guardrail sees, not just
        # the user turn: indirect_injection carries its payload in the retrieved
        # document and output_only carries it in the response, so a prompt-only
        # check would flag both families for a collapse that isn't there.
        distinct = len({(i.input_text, i.response) for i in group}) / len(group)
        if distinct < MIN_FAMILY_DIVERSITY:
            problems.append(f"family {fam}: only {distinct:.2f} distinct prompts per "
                            f"item (< {MIN_FAMILY_DIVERSITY}) — templates collapsed")

    # ---- shortcut probes on the core tier -------------------------------
    core = [i for i in items if i.tier == TIER_CORE]
    if core:
        for field in ("prompt", "response"):
            a = shortcut_auc(core, field)
            if a > MAX_SHORTCUT_AUC:
                problems.append(f"core tier: {field} length alone reaches AUC {a:.3f} "
                                f"(> {MAX_SHORTCUT_AUC}) — length matching regressed")

    # ---- derivation keeps its cluster -----------------------------------
    by_id = {i.item_id: i for i in items}
    for i in items:
        base = i.meta.get("base_item")
        if base and base in by_id and by_id[base].cluster_id != i.cluster_id:
            problems.append(f"{i.item_id}: cluster {i.cluster_id} != base "
                            f"{base} cluster {by_id[base].cluster_id} — split leakage")
    return problems


def validate_splits(dev: list[Item], test: list[Item]) -> list[str]:
    """Splits must not share a cluster or an item id."""
    problems = []
    shared_clusters = ({i.cluster_id for i in dev} & {i.cluster_id for i in test}) - {None}
    if shared_clusters:
        problems.append(f"{len(shared_clusters)} clusters appear in both dev and test: "
                        f"{sorted(shared_clusters)[:5]}")
    shared_ids = {i.item_id for i in dev} & {i.item_id for i in test}
    if shared_ids:
        problems.append(f"{len(shared_ids)} item_ids appear in both splits")
    return problems
