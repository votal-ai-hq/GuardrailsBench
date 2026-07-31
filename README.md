# GuardrailGym

A benchmark for LLM guardrails in a financial-assistant setting, built so that
the obvious ways of winning it don't work.

Three things make a guardrail benchmark say something useful, and most say
nothing because they skip one of them:

1. **Blocking more must cost something.** Every attack family has a *benign
   mirror* — the same surface form carrying a legitimate request. A
   block-everything system gets ASR 0.000 and still finishes below a keyword
   matcher, because it pays for the mirrors and the hard negatives.
2. **The shortcut has to be dead.** In the seed corpus a decision stump on
   response length alone reached 0.846 AUC. The core tier is length-matched on a
   joint (prompt, response) grid, and `guardrailgym validate` fails the dataset
   if that ever regresses.
3. **Abstaining is not passing.** A system that can't rule on an item says
   `NOT_EVALUABLE`; those items are excluded from every rate, counted as a
   coverage gap, and the score is scaled by coverage.

## Quick start

```bash
pip install -e ".[dev]"

python -m guardrailgym validate --data data/test.jsonl        # dataset gates
python -m guardrailgym eval --system keyword-v1 --train data/dev.jsonl
python -m guardrailgym leaderboard --train data/dev.jsonl \
    --incumbent-csv data/seed/custom_policy_5k_llmshield.csv
pytest
```

Current results are in [LEADERBOARD.md](LEADERBOARD.md); the scoring model is in
[docs/METRICS.md](docs/METRICS.md).

## What is in the dataset

`data/test.jsonl` (1300 items) and `data/dev.jsonl` (752), cluster-disjoint,
both built from `data/seed/custom_policy_5k_llmshield.csv`. The build is
reproducible byte for byte — `tests/test_reproducibility.py` rebuilds both files
from the seed and compares.

The test split, in three tiers:

| tier | n | what it is |
|---|---|---|
| `core` | 234 | clean, in-distribution, length-matched, both labels |
| `hard_negative` | 200 | policy-*adjacent* and allowed: the compliance officer asking about insider-trading controls. **Known limitation: 33 distinct strings in 4 clusters** — see below |
| `adversarial` | 866 | seven attack families (586 violating) and four benign mirror families (280 allowed) |

Each item is a conversation, a model response, and a label. Both stages are
scored, because a violation can arrive in the request or leave in the response.

### Attack families

| family | what it changes | what it tests |
|---|---|---|
| `fictional_framing` | wraps the ask in satire, journalism, screenplay, training material | whether framing defeats intent detection |
| `romanized_l2` | Latin-script Hebrew / Japanese / Korean / Thai | English-tuned lexical matching |
| `authority_pretext` | fake red-team ticket, fake CCO, fake dev mode | whether claimed authority relaxes policy |
| `incremental_split` | splits the request across turns | whether the guard carries conversation state |
| `encoding_obfuscation` | base64, leetspeak, zero-width joiners | character-level robustness |
| `indirect_injection` | payload arrives in a retrieved document; the user turn is benign | whether retrieved context is screened at all |
| `output_only` | benign prompt, violating response | whether the output stage exists |

Mirrors: `benign_framing`, `benign_l2`, `benign_multiturn`, `benign_document`.

A transform may change **surface form only**. If a human reviewer would no
longer call the result a violation, the transform is invalid and the item is
dropped — `tests/test_attacks.py` holds every family to that rule.

## Baselines

| system | what it is |
|---|---|
| `tfidf-lr-v1` | word + char n-gram TF-IDF into logistic regression, one model per stage |
| `keyword-v1` | weighted policy lexicon with negation rules for legitimate framings |
| `CONTROL-length-stump` | text length only — the shortcut probe |
| `CONTROL-block-all` / `CONTROL-allow-all` | the degenerate corners |
| `incumbent-replay` | the shipped guardrail's recorded verdicts (needs `--incumbent-csv`) |

All systems are compared at a fixed over-block budget (`@OB<=5%`), because any
detection number can be bought with over-block and vice versa.

`--train data/dev.jsonl` fits and calibrates on the held-out split. Without it
the harness falls back to cluster-disjoint 5-fold cross-validation over the eval
set, so nothing is ever scored by a model that saw its cluster — including its
paraphrases and its obfuscated restatements, which stage 1 of the build keeps in
the same cluster.

The two modes give materially different answers, and the published leaderboard
uses held-out. Under cross-validation `tfidf-lr-v1` leads on 0.755; under
held-out training it over-blocks 89% of benign mirrors and drops to 0.682,
behind `keyword-v1`. Generalising an operating point across splits is most of
the problem, and cross-validation hides how much.

### Known limitation: the hard-negative tier is thin

`make_hard_negatives` renders 400 items from 8 templates, but four of those
templates carry no variable slots, so each is duplicated 50 times. The result is
85 distinct strings overall, and because a template is one cluster, the split
gives dev 4 templates and test the other 4.

Two consequences: `hard-neg over-block` is a rate over 4 template groups, not
200 items; and no held-out system can generalise there, which is most of why
`tfidf-lr-v1` over-blocks 100% of them. `guardrailgym validate` prints
items/distinct/clusters per tier, and the leaderboard footer carries the caveat.
Fixing it means authoring more templates with real variation, which changes
`data/test.jsonl` and invalidates any prior leaderboard.

## Adding a system

```python
from guardrailgym.systems.base import ScoreGuardrail

class MyGuard(ScoreGuardrail):
    name = "my-guard-v1"

    def input_score(self, item):    # item.input_text = turns + retrieved docs
        return my_model(item.input_text)

    def output_score(self, item):
        return my_model(item.response)
```

Register it in `guardrailgym/systems/__init__.py` and it is available to `eval`
and `leaderboard`. Calibration and cross-validation come for free; override
`fit` if it trains. If your system can't rule on some items, return
`Decision(False, None, REASON_NOT_EVALUABLE)` rather than allowing them.

## Rebuilding the dataset

```bash
python -m guardrailgym build --seed-csv path/to/seed.csv --out data/
```

Five stages: ingest (drops the columns that leak the label, including the
incumbent's own verdicts), near-dup clustering, length matching, hard negatives,
attack families, cluster-disjoint split. The seed CSV is not in this repository;
`tests/fixtures/seed_sample.csv` is a small stand-in that exercises the pipeline
end to end.

## Layout

```
guardrailgym/
  schema.py     Item / Decision / Trace
  build.py      the five build stages
  attacks.py    attack families and their benign mirrors
  metrics.py    GG-Score, ASR, over-block, coverage, stage attribution
  validate.py   the dataset gates
  runner.py     execution, latency, cluster-disjoint cross-validation
  report.py     leaderboard rendering
  systems/      baselines
tests/          130 tests; test_metrics.py pins the two scoring regressions
docs/METRICS.md the scoring model
```
