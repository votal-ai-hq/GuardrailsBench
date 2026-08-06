"""Regression tests for the two scoring bugs found during development."""
from guardrailsbench.metrics import robustness_delta, score
from guardrailsbench.schema import Decision, Item, Trace


def _it(iid, label, tier, fam=None):
    return Item(item_id=iid, turns=[{"role": "user", "content": "x"}], response="y",
                label=label, category=None, tier=tier, attack_family=fam)


def _tr(iid, blocked, sc=None):
    return Trace(iid, Decision(blocked, sc), None if blocked else Decision(False, sc), 1.0)


def test_asr_excludes_benign_mirror_items():
    """Regression: benign items were counted as attack successes, so a guardrail
    was penalised for correctly allowing legitimate traffic."""
    items = [_it("v", "violating", "adversarial", "fictional_framing"),
             _it("b", "allowed", "adversarial", "benign_framing")]
    # violating blocked (good), benign passed (also good) -> ASR must be 0
    r = score(items, {"v": _tr("v", True), "b": _tr("b", False)})
    assert r["asr_overall"] == 0.0
    assert "benign_framing" not in r["asr_by_family"]
    assert r["benign_overblock_by_family"]["benign_framing"]["overblock"] == 0.0


def test_blanket_blocking_is_penalised():
    """Regression: an all-violating adversarial tier let a block-everything
    system score ASR 0.000 and win. With the benign mirror it pays for it."""
    items = [_it("v", "violating", "adversarial", "fictional_framing"),
             _it("b", "allowed", "adversarial", "benign_framing")]
    r = score(items, {"v": _tr("v", True), "b": _tr("b", True)})
    assert r["asr_overall"] == 0.0
    assert r["benign_overblock_by_family"]["benign_framing"]["overblock"] == 1.0
    assert r["over_block"]["overall"] == 1.0
    assert r["components"]["utility"] == 0.0


def test_not_evaluable_items_become_coverage_gap_not_credit():
    items = [_it("a", "violating", "core"), _it("b", "violating", "core")]
    tr = {"a": _tr("a", True),
          "b": Trace("b", Decision(False, None, "NOT_EVALUABLE"), None, 0.0)}
    r = score(items, tr)
    assert r["coverage_gap"] == 0.5
    assert r["tiers"]["core"]["tpr"] == 1.0, "the skipped item must not count as a miss"


def test_stage_attribution_sums_to_one():
    items = [_it("a", "violating", "core"), _it("b", "violating", "core"),
             _it("c", "violating", "core")]
    tr = {"a": Trace("a", Decision(True), None, 1.0),
          "b": Trace("b", Decision(False), Decision(True), 1.0),
          "c": Trace("c", Decision(False), Decision(False), 1.0)}
    s = score(items, tr)["stage_attribution"]
    assert abs(sum(s.values()) - 1.0) < 1e-9
    assert s["input_caught"] == s["output_caught"] == s["leaked"]


def test_robustness_delta_is_clean_minus_adversarial():
    items = [_it("c1", "violating", "core"),
             _it("a1", "violating", "adversarial", "fictional_framing")]
    r = score(items, {"c1": _tr("c1", True), "a1": _tr("a1", False)})
    assert robustness_delta(r) == 1.0   # perfect clean, total adversarial failure
