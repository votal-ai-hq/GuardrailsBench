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
pip install -e ".[dev]"        # src layout: install once, then the CLI works

python -m guardrailgym validate --data data/test.jsonl        # dataset gates
python -m guardrailgym eval --system keyword-v1 --train data/dev.jsonl
python -m guardrailgym leaderboard --train data/dev.jsonl \
    --incumbent-csv data/seed/custom_policy_5k_llmshield.csv

pytest                      # the code
python evals/run_evals.py   # the results
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
| `http-api` | any guardrail behind an HTTP endpoint (needs `--api-config`) |

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

## Porting to another policy domain

The engine is domain-agnostic: `schema`, `metrics`, `runner`, `report` and the
HTTP adapter contain no policy vocabulary at all, and a test asserts they stay
that way. Everything domain-specific lives in a **policy pack** — one YAML file
holding the seed-column mapping, the categories, the hard-negative templates,
the text each attack family wraps a request in, and the keyword lexicon.

```bash
python -m guardrailgym packs                      # what is bundled
python -m guardrailgym packs --pack healthcare    # what is in one
python -m guardrailgym build --pack healthcare --seed-csv my_corpus.csv --out data/
python -m guardrailgym eval --system keyword-v1 --pack healthcare --data data/test.jsonl
```

Two packs ship. `finance` is the original content, extracted unchanged —
rebuilding from it reproduces `data/` byte for byte. `healthcare` exists to
prove the boundary is in the right place: its seed CSV shares no column names,
no label values and no categories with the finance one, and `tests/test_pack.py`
builds a complete, gate-passing dataset from it.

What a pack contains:

| section | what it decides |
|---|---|
| `seed` | column names, label values, category mapping, which columns to drop because they leak |
| `hard_negatives` | the policy-adjacent-but-allowed templates and their slot values |
| `attacks` | per family: the frames, transliterated cores, pretexts, carriers, documents — plus the slot regexes that keep transforms specific rather than generic |
| `lexicon` | the weighted rules and negations behind `keyword-v1` |

What stays in code: the *shape* of each attack. Wrapping a violating request in
a screenplay frame is domain-independent; which screenplay, and what counts as a
violation, are not. Attack shape, scoring, calibration, the dataset gates and
the harness are all pack-blind.

The loader validates on load — a pack missing an attack family, a label value,
or a `romanized_l2` core for its default category is rejected with the reason,
rather than producing a quietly degenerate dataset.

## Benchmarking a guardrail behind an HTTP API

Point it at the endpoint with a JSON config — no code:

```bash
export GUARDRAIL_API_KEY=...
cp configs/http_api.example.json configs/mine.json   # edit urls + json paths
python -m guardrailgym eval --system http-api --api-config configs/mine.json \
    --data data/test.jsonl --train data/dev.jsonl
```

```json
{
  "name": "my-guardrail",
  "input_url":  "https://guard.example.com/v1/screen/input",
  "output_url": "https://guard.example.com/v1/screen/output",
  "headers":    {"Authorization": "Bearer ${GUARDRAIL_API_KEY}"},
  "input_body":  {"messages": "{turns}", "context": "{context_text}"},
  "output_body": {"prompt": "{prompt}", "completion": "{response}"},
  "score_path":   "result.risk_score",
  "blocked_path": "result.blocked",
  "timeout_s": 20, "retries": 2, "concurrency": 8,
  "cache_dir": ".cache/my-guardrail"
}
```

### The request/response contract

**Request** — you write the body, so it is whatever your endpoint expects. Slots:
`{turns}` and `{turns_with_response}` become the real message *list* (the second
appends the completion, which is what an output-stage screen has to see);
`{prompt}`, `{user_text}`, `{context_text}`, `{input_text}`, `{response}`,
`{item_id}` are strings. Anything that isn't a placeholder passes through, so
static fields like `model` and `temperature` survive.

**Response** — dotted paths into the decoded JSON: `results.0.flagged`,
`output.risk.score`. `unwrap_json_at` first parses a JSON *string* at a path,
for envelopes that carry the verdict as text; `blocked_pattern` regex-matches a
string, for guards that answer in prose.

`${VAR}` in headers is read from the environment and fails loudly if unset —
credentials never enter the repo, and never enter the cache either.

### OpenAI-compatible endpoints

All three common shapes work; ready-made configs are in `configs/`.

| your endpoint | config | example |
|---|---|---|
| `/v1/moderations` | `blocked_path: results.0.flagged` | `openai_moderations.example.json` |
| `/v1/chat/completions` returning JSON | `unwrap_json_at: choices.0.message.content` then `blocked_path: blocked` | `openai_chat.example.json` |
| `/v1/chat/completions` returning prose | `blocked_path: choices.0.message.content` + `blocked_pattern` | `openai_chat_prose.example.json` |
| tool/function call | `unwrap_json_at: choices.0.message.tool_calls.0.function.arguments` | — |

Nothing is hardcoded to OpenAI: it is a URL, headers, a body template and some
paths, so a bespoke JSON API on your own infra needs no more work than a
provider-shaped one. `tests/test_http_api.py` exercises every row above against
a real localhost server.

Four things this handles that a naive loop does not:

* **A failed call is not an allow.** Timeouts, 5xx, rate limits, and responses
  that don't match `score_path` come back `NOT_EVALUABLE` and land in the
  coverage gap. An endpoint that falls over on a third of the set cannot score
  as though it passed them, and `error_summary()` says which failure it was.
* **Latency is the real round trip**, recorded per call and fed to the
  efficiency term. Cached replays report the *original* latency, never the cache
  lookup.
* **Re-runs are free.** Successful responses are cached on disk by request hash;
  failures are not cached, so a transient outage doesn't freeze into a permanent
  gap.
* **Calls are concurrent** (`concurrency`), so 1300 items against a 500 ms
  endpoint is under a minute rather than eleven.

Set `score_path` if the endpoint returns a continuous risk score — that is what
lets it be calibrated to the same `@OB<=5%` budget as every other row. With only
`blocked_path` it has a fixed operating point, is not calibrated, and its row is
labelled `@fixed`. Omitting `output_url` means the output stage is never
screened, which is a finding, not a gap: `output_only` becomes uncatchable.

## Running the tests

```bash
pytest                       # 182 unit tests, ~30s, no network
python evals/run_evals.py    # 33 golden checks on the benchmark's own results
ruff check .                 # lint
python -m guardrailgym validate --data data/test.jsonl
```

Two suites, because they catch different failures. **`tests/`** asserts the code
is correct — fast, offline, per-function. **`evals/`** asserts the *results* have
not drifted: that a change to calibration, the metric weights, or a policy pack
has not silently moved the leaderboard. Cases are YAML declaring metric ranges,
per-family ASR bounds, and outranking invariants (`keyword-v1` must beat
`CONTROL-block-all`, and so on). Ranges rather than exact values, so a
scikit-learn bump does not fail the build but a real drift does.

CI (`.github/workflows/ci.yml`) runs all four on every push and PR against
Python 3.11. Nothing reaches the network: the HTTP adapter tests start a real
`ThreadingHTTPServer` on localhost, so timeouts, retries, concurrency and
recorded latency are exercised for real rather than mocked.

| file | what it covers |
|---|---|
| `test_metrics.py` | the two scoring regressions, unchanged as delivered |
| `test_scoring.py` | GG-Score bounds, coverage scaling, per-tier rates, percentiles |
| `test_schema.py` | item views, round-trips, malformed-item rejection |
| `test_attacks.py` | every family: label/cluster preservation, diversity, reversibility |
| `test_build.py` | all five build stages against a fixture seed CSV |
| `test_reproducibility.py` | rebuilds `data/` from the real seed and compares byte for byte |
| `test_systems.py` | calibration budgets, cross-validation disjointness, replay |
| `test_http_api.py` | the HTTP adapter against a live localhost server |
| `test_validate.py` | each dataset gate against the failure it was written for |
| `test_cli.py` | every subcommand, leaderboard rendering, provenance footer |
| `test_pack.py` | the domain boundary: a full dataset built from a second policy pack |

Tests needing `data/` skip rather than fail when it is absent, so the suite
still runs on a checkout without the corpus.

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

Register it in `src/guardrailgym/systems/__init__.py` and it is available to `eval`
and `leaderboard`. Calibration and cross-validation come for free; override
`fit` if it trains. If your system can't rule on some items, return
`Decision(False, None, REASON_NOT_EVALUABLE)` rather than allowing them.

## Rebuilding the dataset

```bash
python -m guardrailgym build \
    --seed-csv data/seed/custom_policy_5k_llmshield.csv --out data/
```

Five stages: ingest (drops the columns that leak the label, including the
incumbent's own verdicts), near-dup clustering, length matching, hard negatives,
attack families, cluster-disjoint split. Deterministic under `SEED = 1337` —
this rewrites `data/dev.jsonl` and `data/test.jsonl` byte for byte.

The seed corpus is 4350 rows and confirms what the pipeline is defending
against: the incumbent misses 12 and over-blocks 3, `latency_ms` leaks the label
(6572 ms mean on safe rows vs 4400 on unsafe), and prompt length leaks it the
other way (92 vs 213 characters).

## Layout

```
AGENTS.md         constraints to read before changing anything
.env.example      the one credential the repo can consume
src/guardrailgym/
  schema.py       Item / Decision / Trace
  build.py        the five build stages
  attacks.py      attack shapes; the words live in packs/
  metrics.py      GG-Score, ASR, over-block, coverage, stage attribution
  validate.py     the dataset gates
  runner.py       execution, latency, cluster-disjoint cross-validation
  report.py       leaderboard rendering
  pack.py         policy packs: the domain boundary
  packs/          finance.yaml, healthcare.yaml
  systems/        baselines + http_api.py for a guardrail behind an endpoint
configs/          example http-api config
data/             dev.jsonl, test.jsonl, and seed/ — the corpus they build from
tests/            182 unit tests: the code
                  test_metrics.py pins the two scoring regressions
                  test_reproducibility.py pins data/ to the seed corpus
                  test_pack.py builds a whole dataset from a second domain
evals/            33 golden checks: the results
                  cases/*.yaml + run_evals.py
docs/METRICS.md   the scoring model
```
