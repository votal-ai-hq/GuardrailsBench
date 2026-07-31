"""CLI and leaderboard rendering.

The leaderboard is the artifact people read, so its shape is tested: the ranking
order, the em dash for metrics that were never measured, the PARTIAL tag, and the
family ordering that keeps two tables comparable row by row.
"""
import json

import pytest

from guardrailgym.cli import main
from guardrailgym.metrics import score
from guardrailgym.report import (
    DIVIDER,
    HEADER,
    family_table,
    leaderboard_markdown,
    partial_tag,
    summary_text,
)
from guardrailgym.schema import (
    REASON_NOT_EVALUABLE,
    TIER_ADVERSARIAL,
    TIER_CORE,
    Decision,
    Item,
    Trace,
)


def _it(iid, label, tier, fam=None):
    return Item(item_id=iid, turns=[{"role": "user", "content": "x"}], response="y",
                label=label, category=None, tier=tier, attack_family=fam)


def _corpus():
    items = [_it("c_v", "violating", TIER_CORE), _it("c_a", "allowed", TIER_CORE)]
    for fam in ("output_only", "fictional_framing", "romanized_l2"):
        items.append(_it(f"x_{fam}", "violating", TIER_ADVERSARIAL, fam))
    items.append(_it("x_b", "allowed", TIER_ADVERSARIAL, "benign_framing"))
    return items


def _result(blocked_ids):
    items = _corpus()
    traces = {i.item_id: Trace(i.item_id, Decision(i.item_id in blocked_ids), None, 1.0)
              for i in items}
    return score(items, traces)


def test_leaderboard_ranks_by_gg_score_and_keeps_the_published_header():
    strong = _result({"c_v", "x_output_only", "x_fictional_framing", "x_romanized_l2"})
    weak = _result(set())
    md = leaderboard_markdown([("weak-system", weak), ("strong-system", strong)])
    lines = md.splitlines()
    assert lines[0] == "# GuardrailGym leaderboard"
    assert lines[2] == HEADER and lines[3] == DIVIDER
    assert lines[4].startswith("| strong-system"), "best system goes first"
    assert lines[5].startswith("| weak-system")
    assert "## Attack success rate by family" in md
    assert md.endswith("\n")


def test_unmeasured_metrics_render_as_an_em_dash_not_a_zero():
    items = [_it("c_v", "violating", TIER_CORE), _it("c_a", "allowed", TIER_CORE)]
    traces = {i.item_id: Trace(i.item_id, Decision(True), None, 1.0) for i in items}
    row = leaderboard_markdown([("core-only", score(items, traces))]).splitlines()[4]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "—", "no adversarial items were scored, so ASR is unknown"
    assert cells[4] == "—" and cells[5] == "—"
    assert cells[2] == "1.000"


def test_partial_runs_are_labelled_with_what_they_covered():
    items = _corpus()
    traces = {}
    for i in items:
        if i.tier == TIER_CORE:
            traces[i.item_id] = Trace(i.item_id, Decision(i.label == "violating"),
                                      None, 1.0)
        else:
            traces[i.item_id] = Trace(i.item_id,
                                      Decision(False, None, REASON_NOT_EVALUABLE),
                                      None, 0.0)
    r = score(items, traces)
    assert partial_tag(r) == " (PARTIAL: core only)"
    assert " (PARTIAL: core only)" in leaderboard_markdown([("incumbent", r)])
    assert partial_tag(_result(set())) == ""


def test_family_tables_sort_by_asr_then_declaration_order():
    """Ties keep a fixed order so two systems' tables line up row by row."""
    r = _result({"x_fictional_framing"})
    rows = [line for line in family_table(r) if line.startswith("| `")]
    names = [line.split("`")[1] for line in rows]
    assert names[0] == "output_only" or names[0] == "romanized_l2"
    assert names[-1] == "fictional_framing", "the caught family sorts last"
    tied = list(names[:2])
    assert tied == ["romanized_l2", "output_only"], "declaration order breaks ties"


def test_summary_text_reports_every_axis():
    text = summary_text("sys", _result({"c_v"}))
    for field in ("GG-Score", "components", "robustness", "over-block", "stage",
                  "latency"):
        assert field in text


# ---- command line ----------------------------------------------------------
def test_validate_command_passes_on_the_shipped_dataset(capsys):
    assert main(["validate", "--data", "data/test.jsonl"]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_command_fails_loudly_on_a_broken_dataset(tmp_path, capsys):
    from guardrailgym.schema import dump_items
    broken = [_it("a", "violating", TIER_ADVERSARIAL, "fictional_framing")]
    path = tmp_path / "broken.jsonl"
    dump_items(broken, path)
    assert main(["validate", "--data", str(path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_build_command_writes_a_dataset(seed_csv, tmp_path, capsys):
    assert main(["build", "--seed-csv", seed_csv, "--out", str(tmp_path),
                 "--per-family", "8", "--hard-negatives", "24"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] > 0
    assert (tmp_path / "test.jsonl").exists()


def test_eval_command_writes_traces_and_metrics(seed_csv, tmp_path, capsys):
    data = tmp_path / "data"
    main(["build", "--seed-csv", seed_csv, "--out", str(data), "--per-family", "8",
          "--hard-negatives", "24"])
    out = tmp_path / "reports"
    assert main(["eval", "--system", "keyword-v1", "--data", str(data / "all.jsonl"),
                 "--folds", "3", "--out", str(out)]) == 0
    metrics = json.loads((out / "keyword-v1.metrics.json").read_text())
    assert metrics["system"].startswith("keyword-v1")
    assert 0.0 <= metrics["gg_score"] <= 1.0
    assert (out / "keyword-v1.traces.jsonl").exists()
    assert "GG-Score" in capsys.readouterr().out


def test_leaderboard_command_renders_a_file(seed_csv, tmp_path):
    data = tmp_path / "data"
    main(["build", "--seed-csv", seed_csv, "--out", str(data), "--per-family", "8",
          "--hard-negatives", "24"])
    out = tmp_path / "LEADERBOARD.md"
    assert main(["leaderboard", "--data", str(data / "all.jsonl"), "--out", str(out),
                 "--reports", str(tmp_path / "reports"), "--folds", "3",
                 "--systems", "keyword-v1", "CONTROL-length-stump"]) == 0
    md = out.read_text()
    assert md.startswith("# GuardrailGym leaderboard")
    assert "keyword-v1" in md and "CONTROL-length-stump" in md


def test_incumbent_replay_requires_its_verdict_csv(tmp_path):
    with pytest.raises(SystemExit, match="incumbent-csv"):
        main(["eval", "--system", "incumbent-replay", "--data", "data/test.jsonl"])


def test_unknown_system_is_rejected():
    with pytest.raises(SystemExit):
        main(["eval", "--system", "nope", "--data", "data/test.jsonl"])


def test_provenance_footer_states_the_operating_point_and_fitting():
    from guardrailgym.report import provenance
    cv = provenance("data/test.jsonl", 1300, 0.05)
    assert "cluster-disjoint 5-fold" in cv and "5%" in cv
    assert "incumbent-replay" in cv, "the missing row has to be explained"
    held_out = provenance("data/test.jsonl", 1300, 0.05, train_path="data/dev.jsonl")
    assert "data/dev.jsonl" in held_out and "cross-validation" not in held_out


def test_leaderboard_renders_the_footer_only_when_given_one():
    r = _result({"c_v"})
    plain = leaderboard_markdown([("s", r)]).splitlines()
    assert "---" not in plain, "the horizontal rule belongs to the footer"
    with_footer = leaderboard_markdown([("s", r)], footer="a note").splitlines()
    assert with_footer[-2:] == ["", "a note"]
