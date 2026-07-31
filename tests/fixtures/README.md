# Test fixtures

`seed_sample.csv` is a **test fixture, not a data source**. It exists so the
build pipeline and the incumbent-replay system can be exercised end to end
without the real seed corpus.

* `id`, `prompt`, `ai_response`, `expected`, `category` — 160 core-tier rows
  round-tripped out of `data/test.jsonl`, so the text is real.
* `latency_ms`, `llmshield_verdict`, `input_blocked`, `output_blocked`,
  `fired_policy`, `outcome` — **synthetic**, generated with a fixed seed. They
  are the columns stage 0 of the build drops (they are the incumbent's own
  predictions), and they are present here only so tests can assert that the
  dropping happens and that replay parses them.

Nothing in this file should be read as a measurement of any real system. The
`incumbent-replay` row on the leaderboard requires the real verdict CSV, which
is not in this repository.
