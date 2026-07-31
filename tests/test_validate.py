"""The dataset gates. Each one is checked against the failure it was written for."""
import pytest

from guardrailgym.schema import (
    LABEL_ALLOWED,
    LABEL_VIOLATING,
    TIER_ADVERSARIAL,
    TIER_CORE,
    TIER_HARD_NEGATIVE,
    Item,
)
from guardrailgym.validate import (
    MIN_FAMILY_SIZE,
    auc,
    shortcut_auc,
    validate_dataset,
    validate_splits,
)


def _it(iid, text, label, tier=TIER_CORE, fam=None, response="a response here",
        cluster=None, meta=None):
    return Item(item_id=iid, turns=[{"role": "user", "content": text}], response=response,
                label=label, tier=tier, attack_family=fam, cluster_id=cluster,
                meta=meta or {})


def _healthy(n=MIN_FAMILY_SIZE):
    # Core prompts are deliberately the same length in both classes ("violating"
    # and "permitted" are both nine characters) so the fixture itself does not
    # trip the shortcut gate.
    items = []
    for i in range(n):
        items.append(_it(f"c_v{i}", f"violating item {i:03d} regarding subject {i:03d}",
                         LABEL_VIOLATING, cluster=i))
        items.append(_it(f"c_a{i}", f"permitted item {i:03d} regarding subject {i:03d}",
                         LABEL_ALLOWED, cluster=1000 + i))
        items.append(_it(f"x_v{i}", f"attacked violating request {i} in some new frame",
                         LABEL_VIOLATING, TIER_ADVERSARIAL, "fictional_framing",
                         cluster=i))
        items.append(_it(f"x_a{i}", f"attacked allowed question {i} in some new frame",
                         LABEL_ALLOWED, TIER_ADVERSARIAL, "benign_framing",
                         cluster=1000 + i))
    return items


def test_a_healthy_dataset_passes():
    assert validate_dataset(_healthy()) == []


def test_the_shipped_dataset_passes(dataset):
    assert validate_dataset(dataset) == []


def test_single_class_adversarial_tier_is_rejected():
    items = [i for i in _healthy() if i.attack_family != "benign_framing"]
    assert any("single-class" in p for p in validate_dataset(items))


def test_collapsed_templates_are_rejected():
    """A family where every item renders to the same string is one item."""
    items = _healthy()
    for i in items:
        if i.attack_family == "fictional_framing":
            i.turns = [{"role": "user", "content": "the one and only rendering"}]
    assert any("templates collapsed" in p for p in validate_dataset(items))


def test_undersized_families_are_rejected():
    items = _healthy()
    keep = [i for i in items if i.attack_family != "fictional_framing"]
    keep += [i for i in items if i.attack_family == "fictional_framing"][:3]
    assert any("ASR estimate is noise" in p for p in validate_dataset(keep))


def test_length_shortcut_is_rejected():
    items = _healthy()
    for i in items:
        if i.tier == TIER_CORE and i.label == LABEL_VIOLATING:
            i.turns = [{"role": "user", "content": i.prompt + " padding" * 60}]
    problems = validate_dataset(items)
    assert any("length matching regressed" in p for p in problems)


def test_response_length_shortcut_is_rejected():
    """Matching prompt length alone was not enough: the output-stage guard rode
    response length to 0.846."""
    items = _healthy()
    for i in items:
        if i.tier == TIER_CORE and i.label == LABEL_VIOLATING:
            i.response = "detailed advice " * 50
    assert any("response length alone" in p for p in validate_dataset(items))


def test_derived_items_escaping_their_cluster_are_rejected():
    items = _healthy()
    base = items[0]
    items.append(_it("x_leak", "a derived rendering of the base item", LABEL_VIOLATING,
                     TIER_ADVERSARIAL, "fictional_framing", cluster=999_999,
                     meta={"base_item": base.item_id}))
    assert any("split leakage" in p for p in validate_dataset(items))


def test_duplicate_ids_and_contradictory_labels_are_rejected():
    items = _healthy()
    twin = _it(items[0].item_id, items[0].prompt, LABEL_ALLOWED, cluster=1)
    problems = validate_dataset(items + [twin])
    assert any("duplicate item_ids" in p for p in problems)
    assert any("both labels" in p for p in problems)


def test_hard_negatives_must_be_allowed():
    items = _healthy()
    items.append(_it("h_bad", "a policy adjacent question", LABEL_VIOLATING,
                     TIER_HARD_NEGATIVE, cluster=5))
    assert any("hard negatives must be allowed" in p for p in validate_dataset(items))


def test_splits_must_be_cluster_disjoint():
    items = _healthy()
    dev = [i for i in items if i.cluster_id % 2 == 0]
    test = [i for i in items if i.cluster_id % 2 == 1]
    assert validate_splits(dev, test) == []

    # A paraphrase of a test item, placed in dev under the same cluster.
    leaked = _it("x_leaked", "a paraphrase of a test item", LABEL_VIOLATING,
                 cluster=test[0].cluster_id)
    problems = validate_splits(dev + [leaked], test)
    assert any("both dev and test" in p for p in problems)
    assert any("item_ids appear in both" in p
               for p in validate_splits(dev, test + dev[:1]))


def test_shipped_dataset_has_no_split_leakage_against_itself(dataset):
    """Sanity: the file is one split, so nothing may be duplicated inside it."""
    assert validate_splits(dataset, []) == []


@pytest.mark.parametrize("pos, neg, expected", [
    ([1, 2, 3], [4, 5, 6], 0.0),
    ([4, 5, 6], [1, 2, 3], 1.0),
    ([1, 2, 3], [1, 2, 3], 0.5),
    ([], [1], 0.5),
])
def test_auc_endpoints(pos, neg, expected):
    assert auc(pos, neg) == pytest.approx(expected)


def test_shortcut_auc_is_direction_free():
    """A shortcut that fires backwards is still a shortcut."""
    short_violating = [_it(f"v{i}", "tiny", LABEL_VIOLATING) for i in range(5)]
    long_allowed = [_it(f"a{i}", "a much longer question " * 10, LABEL_ALLOWED)
                    for i in range(5)]
    assert shortcut_auc(short_violating + long_allowed, "prompt") == 1.0
