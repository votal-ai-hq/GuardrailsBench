"""Command line entry point: ``python -m guardrailgym <command>``.

    build        seed CSV -> dev/test jsonl (see build.py for the stages)
    validate     structural + leakage checks on a built dataset
    eval         run one guardrail, print a summary, write traces + metrics
    leaderboard  run several and render LEADERBOARD.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report, systems
from . import validate as validate_mod
from .metrics import score
from .runner import evaluate
from .schema import dump_traces, load_items
from .systems import (
    DEFAULT_MAX_OVERBLOCK,
    DEFAULT_SYSTEMS,
    HttpGuardrail,
    IncumbentReplay,
)


def _load(path: str):
    items = load_items(path)
    if not items:
        raise SystemExit(f"no items in {path}")
    return items


def _run_one(name: str, eval_items, train_items, args):
    if name == IncumbentReplay.name:
        if not args.incumbent_csv:
            raise SystemExit("incumbent-replay needs --incumbent-csv")
        system = IncumbentReplay(args.incumbent_csv)
        traces, system = evaluate(system, eval_items)
    elif name == HttpGuardrail.name:
        if not args.api_config:
            raise SystemExit("http-api needs --api-config")
        system = HttpGuardrail(args.api_config)
        traces, system = evaluate(system, eval_items, train_items,
                                  max_overblock=args.max_overblock)
        if system.error_summary():
            print(system.error_summary(), file=sys.stderr)
    else:
        traces, system = evaluate(systems.get(name), eval_items, train_items,
                                  n_folds=args.folds, max_overblock=args.max_overblock)
    return traces, system, score(eval_items, traces)


def cmd_build(args) -> int:
    from .build import build
    stats = build(args.seed_csv, args.out, per_family=args.per_family,
                  n_hard_neg=args.hard_negatives)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_validate(args) -> int:
    items = _load(args.data)
    print(f"{'tier':<16}{'items':>8}{'distinct':>10}{'clusters':>10}")
    for tier, n in validate_mod.effective_n(items).items():
        print(f"{tier:<16}{n['items']:>8}{n['distinct']:>10}{n['clusters']:>10}")
    print()
    problems = validate_mod.validate_dataset(items)
    for p in problems:
        print(f"FAIL {p}")
    print(f"{'FAIL' if problems else 'OK'}: {args.data} ({len(problems)} problems)")
    return 1 if problems else 0


def cmd_eval(args) -> int:
    eval_items = _load(args.data)
    train_items = _load(args.train) if args.train else None
    traces, system, result = _run_one(args.system, eval_items, train_items, args)
    label = getattr(system, "label", args.system)
    print(report.summary_text(label, result))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        slug = args.system.replace("/", "_")
        dump_traces(traces, out / f"{slug}.traces.jsonl")
        report.write_result_json(out / f"{slug}.metrics.json", label, result)
        print(f"\nwrote {out / f'{slug}.traces.jsonl'} and "
              f"{out / f'{slug}.metrics.json'}")
    return 0


def cmd_leaderboard(args) -> int:
    eval_items = _load(args.data)
    train_items = _load(args.train) if args.train else None
    names = args.systems or list(DEFAULT_SYSTEMS)
    if args.incumbent_csv and IncumbentReplay.name not in names:
        names = names + [IncumbentReplay.name]
    if args.api_config and HttpGuardrail.name not in names:
        names = names + [HttpGuardrail.name]

    entries = []
    out_dir = Path(args.reports) if args.reports else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        traces, system, result = _run_one(name, eval_items, train_items, args)
        label = getattr(system, "label", name)
        entries.append((label, result))
        print(report.summary_text(label, result), file=sys.stderr)
        print("", file=sys.stderr)
        if out_dir:
            slug = name.replace("/", "_")
            dump_traces(traces, out_dir / f"{slug}.traces.jsonl")
            report.write_result_json(out_dir / f"{slug}.metrics.json", label, result)

    footer = report.provenance(args.data, len(eval_items), args.max_overblock,
                               args.train, args.folds,
                               validate_mod.effective_n(eval_items),
                               args.incumbent_csv)
    md = report.leaderboard_markdown(entries, footer)
    Path(args.out).write_text(md)
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guardrailgym")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the dataset from the seed CSV")
    b.add_argument("--seed-csv", required=True)
    b.add_argument("--out", default="data")
    b.add_argument("--per-family", type=int, default=120)
    b.add_argument("--hard-negatives", type=int, default=400)
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="check a built dataset")
    v.add_argument("--data", default="data/test.jsonl")
    v.set_defaults(func=cmd_validate)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="data/test.jsonl")
    common.add_argument("--train", default=None,
                        help="held-out split to fit and calibrate on; without it, "
                             "cluster-disjoint cross-validation over --data")
    common.add_argument("--folds", type=int, default=5)
    common.add_argument("--max-overblock", type=float, default=DEFAULT_MAX_OVERBLOCK)
    common.add_argument("--incumbent-csv", default=None,
                        help="seed CSV carrying the incumbent's recorded verdicts")
    common.add_argument("--api-config", default=None,
                        help="JSON config for the http-api system (see "
                             "configs/http_api.example.json)")

    e = sub.add_parser("eval", parents=[common], help="run one guardrail")
    e.add_argument("--system", required=True,
                   choices=sorted(systems.REGISTRY)
                   + [IncumbentReplay.name, HttpGuardrail.name])
    e.add_argument("--out", default=None, help="directory for traces and metrics")
    e.set_defaults(func=cmd_eval)

    lb = sub.add_parser("leaderboard", parents=[common], help="run several and render")
    lb.add_argument("--systems", nargs="*", default=None)
    lb.add_argument("--out", default="LEADERBOARD.md")
    lb.add_argument("--reports", default="reports")
    lb.set_defaults(func=cmd_leaderboard)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
