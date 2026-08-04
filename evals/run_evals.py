#!/usr/bin/env python3
"""Golden-case evals for the benchmark's own results.

`tests/` asserts the code is correct. This asserts the *results* are what they
were — that a change to calibration, to the metric weights, or to a policy pack
has not silently moved the leaderboard. Those are different failures and they
want different runs: the unit suite is fast, offline and per-function; this one
is slow, whole-dataset, and reads like a report.

Cases are YAML in `cases/`, each declaring three kinds of expectation:

    expect     a metric lands inside a range, by dotted path into score()'s result
    outrank    system A must finish above system B on GG-Score
    families   per-family ASR lands inside a range

Ranges rather than exact values, because the point is to catch a drift that
matters, not to fail on the third decimal of a scikit-learn bump.

    python evals/run_evals.py                # every case
    python evals/run_evals.py --case anti-gaming
    python evals/run_evals.py --verbose      # show every check, not just failures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardrailgym.metrics import score  # noqa: E402
from guardrailgym.runner import evaluate  # noqa: E402
from guardrailgym.schema import load_items  # noqa: E402
from guardrailgym.systems import get  # noqa: E402
from guardrailgym.systems.http_api import dig  # noqa: E402

CASES = Path(__file__).parent / "cases"

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


class Results:
    """Scores each system once per (dataset, train) pair and hands them out."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None, str], dict] = {}

    def get(self, system: str, dataset: str, train: str | None) -> dict:
        key = (dataset, train, system)
        if key not in self._cache:
            items = load_items(ROOT / dataset)
            train_items = load_items(ROOT / train) if train else None
            factory = get(system)
            traces, _ = evaluate(lambda _f=factory: _f(), items, train_items)
            self._cache[key] = score(items, traces)
        return self._cache[key]


def _check(label: str, value, bounds) -> tuple[bool, str]:
    lo, hi = bounds
    ok = value is not None and lo <= value <= hi
    shown = "—" if value is None else f"{value:.3f}"
    return ok, f"{label:<52} {shown:>7}   expected [{lo:.2f}, {hi:.2f}]"


def run_case(case: dict, results: Results, verbose: bool) -> list[str]:
    """Returns the list of failure lines; empty means the case passed."""
    dataset, train = case["dataset"], case.get("train")
    failures, checks = [], []

    for entry in case.get("expect", []):
        system = entry["system"]
        result = results.get(system, dataset, train)
        for path, bounds in entry.get("metrics", {}).items():
            try:
                value = dig(result, path)
            except (KeyError, IndexError, TypeError):
                value = None
            checks.append(_check(f"{system}  {path}", value, bounds))

    for system, families in case.get("families", {}).items():
        result = results.get(system, dataset, train)
        for family, bounds in families.items():
            value = result["asr_by_family"].get(family)
            checks.append(_check(f"{system}  ASR[{family}]", value, bounds))

    for better, worse in case.get("outrank", []):
        a = results.get(better, dataset, train)["gg_score"]
        b = results.get(worse, dataset, train)["gg_score"]
        ok = a > b
        checks.append((ok, f"{better + ' > ' + worse:<52} "
                           f"{a:>7.3f}   must exceed {b:.3f}"))

    for ok, line in checks:
        if not ok:
            failures.append(line)
        if verbose or not ok:
            mark = f"{GREEN}pass{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"  {mark}  {line}")
    if not verbose and not failures:
        print(f"  {GREEN}pass{RESET}  {len(checks)} checks")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", action="append", help="run only these case names")
    ap.add_argument("--verbose", action="store_true", help="show passing checks too")
    args = ap.parse_args(argv)

    paths = sorted(CASES.glob("*.yaml"))
    if not paths:
        print(f"no cases in {CASES}", file=sys.stderr)
        return 1

    results = Results()
    total_failures = 0
    for path in paths:
        case = yaml.safe_load(path.read_text())
        if args.case and case["name"] not in args.case:
            continue
        print(f"\n{case['name']}  {DIM}({path.name}){RESET}")
        print(f"  {DIM}{' '.join(case['description'].split())[:100]}{RESET}")
        total_failures += len(run_case(case, results, args.verbose))

    print()
    if total_failures:
        print(f"{RED}{total_failures} expectation(s) out of range{RESET}")
        print("If the change was intended, update the bounds in evals/cases/ "
              "and say so in the commit message.")
        return 1
    print(f"{GREEN}all cases passed{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
