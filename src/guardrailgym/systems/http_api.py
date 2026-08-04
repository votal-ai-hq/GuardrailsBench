"""http-api: benchmark a guardrail that lives behind an HTTP endpoint.

Every other system in this repo is an in-process Python object. A real guardrail
usually is not — it is a service, and benchmarking one has three problems that
a local class does not have:

* **Failures are not verdicts.** A timeout, a 500, or a rate-limit is not the
  guardrail saying "allow". Those items come back ``NOT_EVALUABLE`` and land in
  the coverage gap, so an endpoint that falls over on a third of the set cannot
  quietly score as though it passed them.
* **Latency is the product.** The p95 that feeds the efficiency term is the real
  round trip, recorded per call. Cached replays report the latency of the
  original call, never the cache lookup.
* **Runs cost money and rate limit.** Responses are cached on disk by request
  hash, so a re-run of 1300 items is free, and calls are issued concurrently.

Configured by a JSON file rather than code, so pointing the benchmark at a new
endpoint does not mean writing a system:

    {
      "name": "llmshield-live",
      "input_url":  "https://guard.example.com/v1/screen/input",
      "output_url": "https://guard.example.com/v1/screen/output",
      "headers":    {"Authorization": "Bearer ${GUARDRAIL_API_KEY}"},
      "input_body":  {"text": "{input_text}"},
      "output_body": {"prompt": "{prompt}", "completion": "{response}"},
      "score_path":   "result.risk_score",
      "blocked_path": "result.blocked",
      "timeout_s": 20, "retries": 2, "concurrency": 8,
      "cache_dir": ".cache/llmshield"
    }

Works with anything reachable over HTTPS, on any infrastructure: it is a URL,
headers, a body template and some JSON paths. OpenAI-shaped endpoints are the
common case and all three variants are covered — ``/v1/moderations`` reads
directly (``results.0.flagged``); ``/v1/chat/completions`` carries its verdict as
a JSON *string*, so name that path in ``unwrap_json_at`` and the fields inside
become addressable; a guard that answers in prose gets ``blocked_pattern``. See
``configs/openai_*.example.json``.

``${VAR}`` in headers is expanded from the environment, so credentials stay out
of the repo. Set ``score_path`` when the endpoint returns a continuous risk
score — that is what lets it be calibrated to the same over-block budget as
everything else. With only ``blocked_path`` the endpoint has a fixed operating
point, it is not calibrated, and its row is labelled ``@fixed`` to say so.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import REASON_NOT_EVALUABLE, Decision, Item, Trace
from .base import DEFAULT_MAX_OVERBLOCK, ScoreGuardrail

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    """Expand ${VAR} from the environment. Missing variables are an error rather
    than an empty header, which would otherwise read as an auth failure."""
    def sub(m):
        try:
            return os.environ[m.group(1)]
        except KeyError:
            raise KeyError(f"environment variable {m.group(1)} is not set") from None
    return _ENV.sub(sub, value)


def dig(payload, path: str):
    """Follow a dotted path into decoded JSON. Supports list indices: a.0.b"""
    cur = payload
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "block", "blocked",
                                         "unsafe", "deny"}
    return bool(value)


@dataclass
class HttpConfig:
    input_url: str
    output_url: str | None = None
    name: str = "http-api"
    method: str = "POST"
    headers: dict = field(default_factory=dict)
    input_body: dict = field(default_factory=lambda: {"text": "{input_text}"})
    output_body: dict = field(default_factory=lambda: {"text": "{response}"})
    score_path: str | None = None
    blocked_path: str | None = None
    # OpenAI chat-completions and tool calls carry their payload as a JSON
    # *string* inside the envelope. Naming that path unwraps it, after which
    # score_path / blocked_path address the guardrail's own fields.
    unwrap_json_at: str | None = None
    # For guards that answer in prose ("UNSAFE", "block") rather than a boolean.
    blocked_pattern: str | None = None
    timeout_s: float = 20.0
    retries: int = 2
    backoff_s: float = 1.0
    concurrency: int = 8
    cache_dir: str | None = None

    @classmethod
    def load(cls, path) -> HttpConfig:
        raw = json.loads(Path(path).read_text())
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        cfg = cls(**raw)
        if not (cfg.score_path or cfg.blocked_path):
            raise ValueError(f"{path}: set score_path, blocked_path, or both")
        if cfg.blocked_pattern and not cfg.blocked_path:
            raise ValueError(f"{path}: blocked_pattern needs blocked_path to match against")
        return cfg


def render(template: dict, item: Item) -> dict:
    """Fill a body template from one item. Values that are not strings pass
    through untouched, so static fields (model ids, policy names) survive."""
    slots = {
        "item_id": item.item_id,
        "prompt": item.prompt,
        "user_text": item.user_text,
        "context_text": item.context_text,
        "input_text": item.input_text,
        "response": item.response,
    }
    # A body value that is exactly one of these placeholders gets the structured
    # conversation, not a string — OpenAI-shaped endpoints need the real list.
    # `turns_with_response` appends the completion, which is what an
    # output-stage screen has to see.
    structured = {
        "{turns}": lambda: item.turns,
        "{turns_with_response}": lambda: [
            *item.turns, {"role": "assistant", "content": item.response}],
    }

    def fill(v):
        if isinstance(v, str):
            if v in structured:
                return structured[v]()
            return v.format(**slots)
        if isinstance(v, dict):
            return {k: fill(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fill(x) for x in v]
        return v
    return {k: fill(v) for k, v in template.items()}


class HttpClient:
    """POSTs a body, returns (decoded json, latency_ms). Caches by request hash."""

    def __init__(self, cfg: HttpConfig) -> None:
        self.cfg = cfg
        self._headers = {"Content-Type": "application/json",
                         **{k: _expand(v) for k, v in cfg.headers.items()}}
        self.cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0

    def request_key(self, url: str, body: dict) -> str:
        """Identity of a request. Two items with identical text share one.

        Headers are deliberately excluded: a cache file must never be a place an
        API key can end up, and rotating a key should not void the cache.
        """
        blob = json.dumps({"url": url, "body": body}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def call(self, url: str, body: dict) -> tuple[dict | None, float, str | None]:
        """Returns (payload, latency_ms, error). payload is None on failure."""
        key = self.request_key(url, body)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            cached = json.loads(path.read_text())
            self.cache_hits += 1
            return cached.get("payload"), cached.get("latency_ms", 0.0), cached.get("error")

        payload = error = None
        latency = 0.0
        for attempt in range(self.cfg.retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode(), headers=self._headers,
                    method=self.cfg.method)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    raw = resp.read()
                latency = (time.perf_counter() - t0) * 1000.0
                payload, error = json.loads(raw), None
                break
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
                latency = (time.perf_counter() - t0) * 1000.0
                error = f"{type(e).__name__}: {e}"
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.backoff_s * (2 ** attempt))
        self.calls += 1

        if path and payload is not None:
            # Only successes are cached. A cached failure would silently freeze a
            # transient outage into a permanent coverage gap.
            #
            # Written via a unique temp file and renamed, because os.replace is
            # atomic: a reader either sees the old file or the whole new one,
            # never a half-written mixture. Two threads writing the same path
            # directly produced torn JSON that only failed under load.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"payload": payload, "latency_ms": latency,
                                       "error": None}))
            os.replace(tmp, path)
        return payload, latency, error


class HttpGuardrail(ScoreGuardrail):
    """A guardrail behind an HTTP endpoint."""

    # Class-level name is the CLI selector; the instance overrides it with the
    # config's name, which is what appears on the leaderboard.
    name = "http-api"

    def __init__(self, config: str | Path | HttpConfig) -> None:
        super().__init__()
        self.cfg = config if isinstance(config, HttpConfig) else HttpConfig.load(config)
        self.name = self.cfg.name
        self.client = HttpClient(self.cfg)
        self._results: dict[tuple[str, str], tuple[dict | None, float, str | None]] = {}
        # prepare() runs these concurrently, so everything below is shared state.
        # Items with identical text render to an identical request; without the
        # per-request lock they would issue N calls for one answer and race each
        # other writing the same cache file.
        self._lock = threading.Lock()
        self._request_locks: dict[str, threading.Lock] = {}
        self._by_request: dict[str, tuple[dict | None, float, str | None]] = {}
        # Why each item went unevaluable. A coverage gap you cannot diagnose is
        # not actionable — "the endpoint returned 429" and "the config points at
        # a field the endpoint does not return" need different fixes.
        self.errors: dict[tuple[str, str], str] = {}

    # -- transport ---------------------------------------------------------
    def _fetch(self, item: Item, stage: str):
        url = self.cfg.input_url if stage == "input" else self.cfg.output_url
        if not url:
            return None, 0.0, None      # stage not configured; nothing was asked
        key = (item.item_id, stage)
        with self._lock:
            if key in self._results:
                return self._results[key]

        template = self.cfg.input_body if stage == "input" else self.cfg.output_body
        body = render(template, item)
        request_id = self.client.request_key(url, body)
        with self._lock:
            request_lock = self._request_locks.setdefault(request_id, threading.Lock())

        with request_lock:
            with self._lock:
                result = self._by_request.get(request_id)
            if result is None:
                result = self.client.call(url, body)
                with self._lock:
                    self._by_request[request_id] = result
        with self._lock:
            self._results[key] = result
        return result

    def _stages(self) -> list[str]:
        return ["input"] + (["output"] if self.cfg.output_url else [])

    def prepare(self, items: list[Item]) -> None:
        """Warm every call concurrently before scoring. 1300 items against a
        500 ms endpoint is 11 minutes sequentially and under a minute at 8."""
        jobs = [(it, stage) for it in items for stage in self._stages()
                if (it.item_id, stage) not in self._results]
        if not jobs:
            return
        with concurrent.futures.ThreadPoolExecutor(self.cfg.concurrency) as pool:
            futures = {pool.submit(self._fetch, it, stage): (it, stage)
                       for it, stage in jobs}
            concurrent.futures.wait(futures)

    # -- decisions ---------------------------------------------------------
    def _decision(self, item: Item, stage: str) -> tuple[Decision, float]:
        payload, latency, error = self._fetch(item, stage)
        if payload is None:
            # The endpoint did not answer. That is a coverage gap, not an allow.
            self.errors[(item.item_id, stage)] = error or "no response"
            return Decision(False, None, REASON_NOT_EVALUABLE), latency
        try:
            if self.cfg.unwrap_json_at:
                payload = json.loads(dig(payload, self.cfg.unwrap_json_at))
            score = float(dig(payload, self.cfg.score_path)) if self.cfg.score_path else None
            blocked = None
            if self.cfg.blocked_path:
                raw = dig(payload, self.cfg.blocked_path)
                blocked = (bool(re.search(self.cfg.blocked_pattern, str(raw)))
                           if self.cfg.blocked_pattern else _as_bool(raw))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
            # The endpoint answered in a shape the config does not match.
            self.errors[(item.item_id, stage)] = f"response did not match config: {e}"
            return Decision(False, None, REASON_NOT_EVALUABLE), latency
        if score is not None:
            threshold = self.t_in if stage == "input" else self.t_out
            return Decision(score >= threshold, score), latency
        return Decision(bool(blocked), None), latency

    def input_score(self, item: Item) -> float:
        return self._score_of(self._fetch(item, "input")[0])

    def output_score(self, item: Item) -> float:
        if not self.cfg.output_url:
            return 0.0                  # no output endpoint: nothing to threshold
        return self._score_of(self._fetch(item, "output")[0])

    def _score_of(self, payload) -> float:
        if payload is None or not self.cfg.score_path:
            return 0.0
        try:
            return float(dig(payload, self.cfg.score_path))
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0

    def screen_input(self, item: Item) -> Decision:
        return self._decision(item, "input")[0]

    def screen_output(self, item: Item) -> Decision:
        if not self.cfg.output_url:
            # No output endpoint configured: the output stage is not screened at
            # all, which is a real finding, not a missing measurement.
            return Decision(False, None)
        return self._decision(item, "output")[0]

    def trace(self, item: Item) -> Trace:
        """Report the endpoint's own round trip, not the cache lookup."""
        inp, in_ms = self._decision(item, "input")
        out, out_ms = (None, 0.0)
        if not inp.blocked and inp.evaluable:
            if self.cfg.output_url:
                out, out_ms = self._decision(item, "output")
            else:
                out = Decision(False, None)
        return Trace(item.item_id, inp, out, in_ms + out_ms)

    # -- operating point ---------------------------------------------------
    def calibrate(self, items: list[Item],
                  max_overblock: float = DEFAULT_MAX_OVERBLOCK) -> HttpGuardrail:
        if not self.cfg.score_path:
            return self          # a boolean endpoint has no dial to turn
        self.prepare(items)
        return super().calibrate(items, max_overblock)

    @property
    def label(self) -> str:
        if not self.cfg.score_path:
            return f"{self.name} @fixed"
        return f"{self.name} @OB<={round(DEFAULT_MAX_OVERBLOCK * 100)}%"

    def error_summary(self, limit: int = 3) -> str:
        """One line per distinct failure, with counts. Empty when nothing failed."""
        if not self.errors:
            return ""
        counts: dict[str, int] = {}
        for msg in self.errors.values():
            counts[msg] = counts.get(msg, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
        head = (f"{len(self.errors)} unevaluable call(s) across "
                f"{len({i for i, _ in self.errors})} items:")
        return "\n".join([head] + [f"  {n:>5}x {msg}" for msg, n in top])
