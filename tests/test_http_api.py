"""The HTTP adapter, exercised against a real server on localhost.

A stubbed transport would prove the parsing and nothing else. These tests start
an actual http.server, so the timeout path, the retry path, the concurrency and
the recorded latency are all the real ones.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from guardrailsbench.metrics import score
from guardrailsbench.runner import evaluate, run
from guardrailsbench.schema import TIER_CORE, Item
from guardrailsbench.systems.http_api import HttpConfig, HttpGuardrail, dig, render


def _item(iid="a", text="Should I buy Tesla stock now?", response="I'd put it all in.",
          label="violating"):
    return Item(item_id=iid, turns=[{"role": "user", "content": text}],
                response=response, label=label, tier=TIER_CORE, cluster_id=hash(iid) % 13)


class _Handler(BaseHTTPRequestHandler):
    """Scores by keyword. `behaviour` on the server drives the failure modes."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, body, dict(self.headers)))
        mode = self.server.behaviour

        if mode == "500":
            self.send_error(500, "boom")
            return
        if mode == "slow":
            time.sleep(0.5)
        if mode == "flaky" and len(self.server.seen) <= self.server.flaky_calls:
            self.send_error(503, "try again")
            return

        text = json.dumps(body)
        risk = 0.9 if ("buy" in text.lower() or "put it all" in text.lower()) else 0.1
        payload = ({"unexpected": "shape"} if mode == "badshape"
                   else {"result": {"risk_score": risk, "blocked": risk >= 0.5}})
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    # Threading, or the concurrency test would measure the server's own
    # single-request serialisation rather than the client's.
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.seen, srv.behaviour, srv.flaky_calls = [], "ok", 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _cfg(server, **kw):
    host, port = server.server_address
    base = dict(
        name="stub-guard",
        input_url=f"http://{host}:{port}/input",
        output_url=f"http://{host}:{port}/output",
        input_body={"text": "{input_text}"},
        output_body={"text": "{response}"},
        score_path="result.risk_score",
        blocked_path="result.blocked",
        retries=0, backoff_s=0.0, timeout_s=5.0, concurrency=4,
    )
    base.update(kw)
    return HttpConfig(**base)


# ---- config and templating --------------------------------------------------
def test_config_rejects_unknown_keys_and_missing_extraction(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"input_url": "http://x", "typo_here": 1,
                                "score_path": "a"}))
    with pytest.raises(ValueError, match="unknown config keys"):
        HttpConfig.load(path)

    path.write_text(json.dumps({"input_url": "http://x"}))
    with pytest.raises(ValueError, match="score_path, blocked_path"):
        HttpConfig.load(path)


def test_headers_expand_environment_variables(server, monkeypatch):
    monkeypatch.setenv("GG_TEST_KEY", "s3cret")
    guard = HttpGuardrail(_cfg(server, headers={"Authorization": "Bearer ${GG_TEST_KEY}"}))
    run(guard, [_item()])
    assert server.seen[0][2]["Authorization"] == "Bearer s3cret"


def test_a_missing_credential_fails_loudly(server, monkeypatch):
    """An unset key must not silently become an empty header and read as a 401."""
    monkeypatch.delenv("GG_ABSENT_KEY", raising=False)
    with pytest.raises(KeyError, match="GG_ABSENT_KEY"):
        HttpGuardrail(_cfg(server, headers={"Authorization": "${GG_ABSENT_KEY}"}))


def test_render_fills_slots_and_passes_structure_through():
    item = _item(text="hello", response="world")
    item.context_docs = ["a retrieved doc"]
    body = render({"messages": "{turns}", "ctx": "{context_text}", "p": "{prompt}",
                   "model": "guard-v2", "nested": {"r": "{response}"}, "n": 7}, item)
    assert body["messages"] == item.turns, "chat endpoints need the real list"
    assert body["ctx"] == "a retrieved doc"
    assert body["p"] == "hello" and body["nested"]["r"] == "world"
    assert body["model"] == "guard-v2" and body["n"] == 7


def test_dig_walks_objects_and_lists():
    assert dig({"a": {"b": [{"c": 5}]}}, "a.b.0.c") == 5
    with pytest.raises(KeyError):
        dig({"a": 1}, "b")


# ---- happy path -------------------------------------------------------------
def test_it_screens_both_stages_and_records_latency(server):
    guard = HttpGuardrail(_cfg(server))
    guard.calibrate([_item(f"ok{i}", "What are the overdraft fees?",
                           "Check the schedule.", "allowed") for i in range(20)])
    traces = run(guard, [_item()])
    tr = traces["a"]
    assert tr.evaluable and tr.blocked
    assert tr.latency_ms > 0, "the endpoint round trip has to be measured"
    assert {path for path, _, _ in server.seen} == {"/input", "/output"}


def test_calibration_puts_a_scoring_endpoint_on_the_shared_operating_point(server):
    guard = HttpGuardrail(_cfg(server))
    allowed = [_item(f"a{i}", "What are the overdraft fees?", "Check the schedule.",
                     "allowed") for i in range(20)]
    guard.calibrate(allowed, max_overblock=0.05)
    assert "@OB<=5%" in guard.label
    traces = run(guard, allowed)
    assert sum(t.blocked for t in traces.values()) / len(traces) <= 0.05


def test_a_boolean_endpoint_is_labelled_fixed_and_not_calibrated(server):
    guard = HttpGuardrail(_cfg(server, score_path=None))
    guard.calibrate([_item(f"x{i}", "What are the overdraft fees?", "Fine.", "allowed")
                     for i in range(20)])
    assert guard.label == "stub-guard @fixed"
    assert run(guard, [_item()])["a"].blocked, "the endpoint's own verdict is used"


def test_no_output_url_means_the_output_stage_is_unscreened(server):
    """Not a missing measurement — a real finding. output_only is uncatchable."""
    guard = HttpGuardrail(_cfg(server, output_url=None))
    guard.calibrate([_item(f"x{i}", "overdraft fees", "fine", "allowed")
                     for i in range(20)])
    run(guard, [_item()])
    assert {path for path, _, _ in server.seen} == {"/input"}


# ---- failure is a coverage gap, not an allow --------------------------------
def test_a_failing_endpoint_produces_coverage_gaps_not_passes(server):
    server.behaviour = "500"
    guard = HttpGuardrail(_cfg(server))
    items = [_item("a"), _item("b")]
    traces = run(guard, items)
    assert not any(t.evaluable for t in traces.values())

    result = score(items, traces)
    assert result["coverage_gap"] == 1.0
    assert result["gg_score"] == 0.0, "an endpoint that never answered scores nothing"
    assert "500" in guard.error_summary()


def test_a_response_shape_the_config_does_not_match_is_a_gap(server):
    server.behaviour = "badshape"
    guard = HttpGuardrail(_cfg(server))
    traces = run(guard, [_item()])
    assert not traces["a"].evaluable
    assert "did not match config" in guard.error_summary()


def test_a_timeout_is_a_gap_not_an_allow(server):
    server.behaviour = "slow"
    guard = HttpGuardrail(_cfg(server, timeout_s=0.05))
    traces = run(guard, [_item()])
    assert not traces["a"].evaluable


def test_retries_recover_a_transient_failure(server):
    server.behaviour, server.flaky_calls = "flaky", 2
    guard = HttpGuardrail(_cfg(server, retries=3, backoff_s=0.0))
    traces = run(guard, [_item()])
    assert traces["a"].evaluable, "a 503 that clears on retry is not a coverage gap"


def test_error_summary_is_empty_when_nothing_failed(server):
    guard = HttpGuardrail(_cfg(server))
    run(guard, [_item()])
    assert guard.error_summary() == ""


# ---- caching and concurrency ------------------------------------------------
def test_responses_are_cached_across_runs_and_replay_recorded_latency(server, tmp_path):
    cfg = _cfg(server, cache_dir=str(tmp_path / "cache"))
    first = HttpGuardrail(cfg)
    traces = run(first, [_item()])
    calls_after_first = len(server.seen)
    recorded = traces["a"].latency_ms
    assert calls_after_first == 2 and recorded > 0

    second = HttpGuardrail(cfg)
    replayed = run(second, [_item()])
    assert len(server.seen) == calls_after_first, "a cached re-run must not re-bill"
    assert second.client.cache_hits == 2
    assert replayed["a"].latency_ms == pytest.approx(recorded), \
        "cached runs report the original round trip, not the cache lookup"


def test_failures_are_not_cached(server, tmp_path):
    """A transient outage must not freeze into a permanent coverage gap."""
    cfg = _cfg(server, cache_dir=str(tmp_path / "cache"))
    server.behaviour = "500"
    assert not run(HttpGuardrail(cfg), [_item()])["a"].evaluable
    server.behaviour = "ok"
    guard = HttpGuardrail(cfg)
    guard.calibrate([_item(f"x{i}", "overdraft fees", "fine", "allowed")
                     for i in range(20)])
    assert run(guard, [_item()])["a"].evaluable


def test_the_cache_never_contains_credentials(server, tmp_path, monkeypatch):
    monkeypatch.setenv("GG_TEST_KEY", "super-secret-value")
    cfg = _cfg(server, cache_dir=str(tmp_path / "cache"),
               headers={"Authorization": "Bearer ${GG_TEST_KEY}"})
    run(HttpGuardrail(cfg), [_item()])
    written = list((tmp_path / "cache").glob("*.json"))
    assert written
    assert not any("super-secret-value" in f.read_text() for f in written)


def test_prepare_issues_calls_concurrently(server):
    server.behaviour = "slow"          # 0.5s each; 12 calls serially is 6s
    guard = HttpGuardrail(_cfg(server, concurrency=8, timeout_s=10))
    # Distinct text per item: identical requests are deduplicated to a single
    # call by design, which would leave nothing here to run in parallel.
    items = [_item(f"i{i}", f"Should I buy stock number {i} right now?",
                   f"I would put all of it into number {i} today, without question.")
             for i in range(6)]
    t0 = time.perf_counter()
    guard.prepare(items)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"12 calls took {elapsed:.1f}s — not running concurrently"
    assert len(server.seen) == 12


def test_it_plugs_into_the_standard_harness(server):
    """The point of the adapter: a remote endpoint scores like any other system."""
    guard = HttpGuardrail(_cfg(server))
    items = [_item("v1"), _item("v2"),
             _item("a1", "What are the overdraft fees?", "Check the schedule.", "allowed"),
             _item("a2", "How is a P/E ratio calculated?", "Price over earnings.",
                   "allowed")]
    calib = items + [_item(f"pad{i}", "What are the overdraft fees?",
                           "Check the schedule.", "allowed") for i in range(16)]
    traces, returned = evaluate(guard, items, train_items=calib)
    assert returned is guard
    result = score(items, traces)
    assert result["coverage"] == 1.0
    assert result["tiers"][TIER_CORE]["tpr"] == 1.0
    assert result["latency"]["p95_ms"] > 0


def test_identical_items_issue_one_call_and_do_not_race_the_cache(server, tmp_path):
    """Regression: N items with the same text render to the same request, so
    prepare() had N threads write the same cache file at once. The torn result
    was invalid JSON that only failed under load — CI caught it, local runs did
    not. Fixed by an atomic rename plus one in-flight call per distinct request.
    """
    cache = tmp_path / "cache"
    guard = HttpGuardrail(_cfg(server, cache_dir=str(cache), concurrency=8))
    # Same text, different ids — exactly the shape of a templated tier.
    items = [_item(f"same{i}", "overdraft fees", "fine", "allowed") for i in range(24)]

    guard.prepare(items)
    assert len(server.seen) == 2, \
        f"24 identical items should be 1 input + 1 output call, got {len(server.seen)}"

    written = list(cache.glob("*.json"))
    assert len(written) == 2
    for f in written:
        json.loads(f.read_text())          # torn writes raise JSONDecodeError here
    assert not list(cache.glob("*.tmp")), "temp files must be renamed away"

    traces = run(guard, items)
    assert all(t.evaluable for t in traces.values())


def test_a_second_process_reads_the_cache_written_by_the_first(server, tmp_path):
    """The atomic write has to be readable, not just well-formed in memory."""
    cfg = _cfg(server, cache_dir=str(tmp_path / "cache"), concurrency=8)
    first = HttpGuardrail(cfg)
    items = [_item(f"same{i}", "overdraft fees", "fine", "allowed") for i in range(12)]
    first.prepare(items)
    calls = len(server.seen)

    second = HttpGuardrail(cfg)
    second.prepare(items)
    assert len(server.seen) == calls, "everything should have come from the cache"
    assert second.client.cache_hits == 2


# ---- OpenAI-shaped endpoints ------------------------------------------------
class _OpenAIHandler(BaseHTTPRequestHandler):
    """Speaks the two OpenAI shapes a guardrail is usually deployed behind."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, body, dict(self.headers)))
        blob = json.dumps(body).lower()
        risky = "buy" in blob or "put it all" in blob

        if self.path.endswith("/moderations"):
            payload = {"id": "modr-1", "results": [
                {"flagged": risky, "category_scores": {"financial_advice": 0.9 if risky else 0.1}}]}
        elif self.server.style == "prose":
            verdict = "UNSAFE — this violates the advice policy." if risky else "Safe."
            payload = {"choices": [{"message": {"role": "assistant", "content": verdict}}]}
        else:  # a guard told to answer in JSON: the payload is a *string*
            verdict = json.dumps({"blocked": risky, "risk_score": 0.9 if risky else 0.1})
            payload = {"id": "chatcmpl-1", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": verdict},
                 "finish_reason": "stop"}]}

        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def openai_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    srv.seen, srv.style = [], "json"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _url(server, path):
    host, port = server.server_address
    return f"http://{host}:{port}{path}"


def test_openai_chat_completions_with_a_json_verdict(openai_server):
    """The verdict arrives as a JSON *string* inside choices[0].message.content,
    so it has to be unwrapped before score_path/blocked_path can address it."""
    guard = HttpGuardrail(HttpConfig(
        name="chat-guard",
        input_url=_url(openai_server, "/v1/chat/completions"),
        output_url=_url(openai_server, "/v1/chat/completions"),
        input_body={"model": "guard-1", "temperature": 0, "messages": "{turns}"},
        output_body={"model": "guard-1", "messages": "{turns_with_response}"},
        unwrap_json_at="choices.0.message.content",
        score_path="risk_score", blocked_path="blocked",
        retries=0, timeout_s=5,
    ))
    allowed = [_item(f"a{i}", "What are the overdraft fees?", "Check the schedule.",
                     "allowed") for i in range(20)]
    guard.calibrate(allowed)
    traces = run(guard, [_item()])
    assert traces["a"].evaluable and traces["a"].blocked
    assert traces["a"].input_decision.score == pytest.approx(0.9)

    # The request really was OpenAI-shaped: a messages list, not a blob.
    # seen[0] is a calibration call, so find the one carrying the scored item.
    sent = next(b for _, b, _ in openai_server.seen
                if b.get("messages") == [{"role": "user", "content": _item().prompt}])
    assert sent["model"] == "guard-1"
    assert sent["temperature"] == 0


def test_turns_with_response_appends_the_completion(openai_server):
    """Output-stage screening of a chat-shaped guard needs the assistant turn."""
    guard = HttpGuardrail(HttpConfig(
        name="chat-guard",
        input_url=_url(openai_server, "/v1/chat/completions"),
        output_url=_url(openai_server, "/v1/chat/completions"),
        input_body={"messages": "{turns}"},
        output_body={"messages": "{turns_with_response}"},
        unwrap_json_at="choices.0.message.content", blocked_path="blocked",
        retries=0, timeout_s=5,
    ))
    item = _item("x", "What are the overdraft fees?", "Check the fee schedule.", "allowed")
    run(guard, [item])
    output_call = [b for _, b, _ in openai_server.seen if len(b["messages"]) == 2]
    assert output_call, "the output stage never sent the completion"
    assert output_call[0]["messages"][-1] == {"role": "assistant",
                                             "content": "Check the fee schedule."}


def test_openai_moderations_shape_needs_no_unwrapping(openai_server):
    """This one is plain nested JSON, so dotted paths reach it directly."""
    guard = HttpGuardrail(HttpConfig(
        name="moderation-guard",
        input_url=_url(openai_server, "/v1/moderations"),
        input_body={"model": "mod-1", "input": "{input_text}"},
        blocked_path="results.0.flagged",
        score_path="results.0.category_scores.financial_advice",
        retries=0, timeout_s=5,
    ))
    allowed = [_item(f"a{i}", "What are the overdraft fees?", "Check the schedule.",
                     "allowed") for i in range(20)]
    guard.calibrate(allowed)
    assert run(guard, [_item()])["a"].blocked


def test_a_guard_that_answers_in_prose(openai_server):
    """No JSON at all — match the text. Labelled @fixed: no score to calibrate."""
    openai_server.style = "prose"
    guard = HttpGuardrail(HttpConfig(
        name="prose-guard",
        input_url=_url(openai_server, "/v1/chat/completions"),
        input_body={"messages": "{turns}"},
        blocked_path="choices.0.message.content",
        blocked_pattern=r"(?i)\b(unsafe|block(ed)?|violat)",
        retries=0, timeout_s=5,
    ))
    assert guard.label == "prose-guard @fixed"
    assert run(guard, [_item()])["a"].blocked
    safe = _item("s", "What are the overdraft fees?", "Check the schedule.", "allowed")
    assert not run(guard, [safe])["s"].blocked


def test_a_wrong_unwrap_path_is_a_diagnosable_gap(openai_server):
    guard = HttpGuardrail(HttpConfig(
        name="misconfigured",
        input_url=_url(openai_server, "/v1/chat/completions"),
        input_body={"messages": "{turns}"},
        unwrap_json_at="choices.0.message.role",   # a plain string, not JSON
        blocked_path="blocked", retries=0, timeout_s=5,
    ))
    assert not run(guard, [_item()])["a"].evaluable
    assert "did not match config" in guard.error_summary()


def test_blocked_pattern_requires_something_to_match_against(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"input_url": "http://x", "score_path": "s",
                                "blocked_pattern": "unsafe"}))
    with pytest.raises(ValueError, match="needs blocked_path"):
        HttpConfig.load(path)
