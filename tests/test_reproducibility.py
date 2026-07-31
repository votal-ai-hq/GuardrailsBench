"""The shipped dataset must be exactly what the pipeline produces from the seed.

This is the test that makes `data/` auditable: if `build.py`, `attacks.py`, the
item schema, or the JSON serialisation drifts, the committed dataset stops
matching its own provenance and this fails. It is a byte comparison on purpose —
a "close enough" dataset check is not a provenance check.
"""
import json

from conftest import ROOT

from guardrailgym.build import build
from guardrailgym.schema import load_items
from guardrailgym.validate import validate_splits


def test_build_reproduces_the_committed_dataset(seed_corpus, tmp_path):
    build(seed_corpus, str(tmp_path))
    for name in ("dev", "test"):
        rebuilt = (tmp_path / f"{name}.jsonl").read_text()
        committed = (ROOT / "data" / f"{name}.jsonl").read_text()
        assert rebuilt == committed, f"data/{name}.jsonl no longer matches the seed"


def test_committed_build_stats_match(seed_corpus, tmp_path):
    stats = build(seed_corpus, str(tmp_path))
    committed = json.loads((ROOT / "data" / "build_stats.json").read_text())
    assert stats == committed


def test_committed_splits_are_cluster_disjoint():
    dev = load_items(ROOT / "data" / "dev.jsonl")
    test = load_items(ROOT / "data" / "test.jsonl")
    assert validate_splits(dev, test) == []
