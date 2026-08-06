# Working in this repository

Read this before changing anything. Most of it is constraints that are cheap to
violate by accident and expensive to notice later.

## What this is

An evaluation harness, not an application and not an agent. It builds a
guardrail benchmark from a seed corpus, runs guardrails against it, and scores
them. There is no model client in the repo and nothing holds state across runs —
if you are adding one, that is a new component, not a change to an existing one.

```
src/guardrailsbench/
  schema.py     Item / Decision / Trace — the only vocabulary the engine has
  build.py      five stages: ingest -> cluster -> length-match -> hard negs -> attacks -> split
  attacks.py    attack SHAPES (the words live in packs/)
  metrics.py    GG-Score, ASR, over-block, coverage, stage attribution
  validate.py   the dataset gates
  runner.py     execution, latency, cluster-disjoint cross-validation
  pack.py       policy packs — the domain boundary
  systems/      the guardrails under test
```

## Hard constraints

**1. `data/dev.jsonl` and `data/test.jsonl` must stay byte-identical to what the
seed corpus produces.** `tests/test_reproducibility.py` rebuilds both from
`data/seed/` and compares byte for byte. This is what makes the committed
dataset auditable rather than a blob someone once generated.

Consequences, all of which have bitten:

- **Never reorder RNG consumption in `build.py` or `attacks.py`.** Every
  `rng.choice` / `rng.sample` call is positional. Adding a call, removing one, or
  moving one changes every downstream item. `make_hard_negatives` draws slot
  values in a fixed key order for exactly this reason — do not "simplify" it into
  something that depends on dict iteration.
- **Declaration order in `attacks._VIOLATING` is load-bearing twice**: it drives
  the build loop's RNG sequence, and it breaks ties in the leaderboard's
  per-family tables.
- Pack content is data, but changing `packs/finance.yaml` changes the dataset.
  Treat it as a data migration, not a config tweak.

**2. Never let a column that leaks the label into an `Item`.** The seed CSV
carries `latency_ms` (a stump on it alone hit 70.7%) and the incumbent's own
verdicts. `ingest` reads only the five columns the pack names; a column that is
never read cannot leak.

**3. Abstaining is not allowing.** A system that cannot rule on an item returns
`Decision(False, None, REASON_NOT_EVALUABLE)`. It becomes a coverage gap, the
score is scaled by coverage, and the row is tagged PARTIAL. Never return "allow"
for a failure — that is how a broken integration scores as a working one.

**4. The engine stays domain-free.** `schema`, `metrics`, `runner`, `report`,
`systems/base.py`, `systems/control.py`, `systems/http_api.py` must contain no
policy vocabulary. `test_pack.py::test_the_engine_carries_no_domain_vocabulary`
enforces it. Domain content goes in a pack.

## Commands

```bash
pip install -e ".[dev]"                             # src layout: install first
pytest                                              # 182 unit tests, ~30s, no network
python evals/run_evals.py                           # golden results, ~40s
ruff check .
python -m guardrailsbench validate --data data/test.jsonl
python -m guardrailsbench packs                        # bundled policy packs
python -m guardrailsbench build --seed-csv data/seed/custom_policy_5k_llmshield.csv --out data/
python -m guardrailsbench leaderboard --train data/dev.jsonl \
    --incumbent-csv data/seed/custom_policy_5k_llmshield.csv
```

CI runs all of these on every push.

`tests/` guards the code; `evals/` guards the results. If you change calibration,
the metric weights, or a pack, expect `evals/` to fail — that is it working.
Update the bounds in `evals/cases/` and say in the commit message why the
numbers moved.

## Adding things

**A guardrail** — subclass `ScoreGuardrail` (emits a risk score, gets calibrated
to the shared over-block budget) or `Guardrail` (fixed operating point).
Register in `systems/__init__.py`. Set `takes_pack = True` if construction needs
the active pack. For an HTTP service, do not write a class — write a JSON config
for the existing `http-api` system.

**A policy domain** — write a pack YAML. Nothing outside that file should need
to change; if it does, the pack boundary is wrong and that is the bug to fix.
The loader validates on load, so a missing attack family or a `romanized_l2`
core for the default category fails loudly rather than producing a quietly
degenerate dataset.

**An attack family** — the shape goes in `attacks.py`, the text goes in *every*
pack. A family present in one pack and absent from another fails pack
validation.

## Judgement calls already made

- **Scores, not thresholds.** Systems are compared at a fixed over-block budget
  because any detection number can be bought with over-block and vice versa.
  Over-block on the leaderboard is measured held-out and routinely exceeds the
  budget; that gap is a result, not a bug.
- **Utility carries the largest weight (0.35).** Detection and robustness are
  both maxed by blocking more; over-block cost is the only thing holding that in
  check.
- **Missing components are absent, not zero.** Weights renormalise over what was
  measured; the coverage factor charges for the gap. A partial run should look
  incomplete, not failing.

## Known limitations — do not "discover" these again

- **The hard-negative tier is thin.** 400 items, 85 distinct strings, 8 clusters
  (four templates carry no variable slots). Under a cluster-disjoint split, dev
  and test hold *disjoint* template sets, so `hard-neg over-block` is a rate over
  4 template groups and no held-out system can generalise there. Documented in
  README and surfaced by `validate` and the leaderboard footer. Fixing it means
  authoring more templates, which changes the dataset — see constraint 1.
- **`LEADERBOARD.md` diverges from the original reference run** for `keyword-v1`
  and the length stump, whose implementations were never part of the handover
  and were rewritten. `tfidf-lr-v1` reproduces closely.
- **`test_prepare_issues_calls_concurrently` is timing-sensitive** (12 slow HTTP
  calls under 3s). It can flake on a loaded runner.
