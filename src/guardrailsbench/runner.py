"""Running a guardrail over a dataset and getting traces back.

Two things here are worth more than the plumbing:

* **Latency is measured, not declared.** Both stages are timed with a
  perf counter, and the p95 across items feeds the efficiency term. A guardrail
  that cannot answer inside a request budget is not deployable however accurate
  it is.

* **Nothing is scored by a model that has seen it.** Systems that need training
  or threshold calibration get either an explicit held-out split
  (``train_items``) or, when only one file is available, cluster-disjoint
  cross-validation. Clusters come from stage 1 of the build and follow attack
  derivations, so a fold never contains a paraphrase — or an obfuscated
  restatement — of an item it is being asked to rule on.
"""
from __future__ import annotations

import random
import time

from .schema import Item, Trace
from .systems import DEFAULT_MAX_OVERBLOCK
from .systems.base import Guardrail

SEED = 1337


def run(system: Guardrail, items: list[Item]) -> dict[str, Trace]:
    """Screen every item. Input stage first; output stage only if input allowed."""
    traces: dict[str, Trace] = {}
    system.prepare(items)
    for it in items:
        custom = getattr(system, "trace", None)
        if callable(custom):
            tr = custom(it)
            if tr is not None:
                traces[tr.item_id] = tr
                continue
        t0 = time.perf_counter()
        inp = system.screen_input(it)
        out = None if inp.blocked else system.screen_output(it)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        traces[it.item_id] = Trace(it.item_id, inp, out, elapsed_ms)
    return traces


def _needs_training(factory) -> bool:
    # `factory` may be a class, a callable returning one, or an already-built
    # instance — incumbent-replay arrives built, because it is constructed
    # around a verdict file.
    if isinstance(factory, Guardrail):
        cls = type(factory)
    elif isinstance(factory, type):
        cls = factory
    else:
        cls = type(factory())
    return (cls.fit is not Guardrail.fit) or (cls.calibrate is not Guardrail.calibrate)


def cluster_folds(items: list[Item], n_folds: int, seed: int = SEED) -> list[list[Item]]:
    """Partition items into folds that share no cluster."""
    buckets: dict[object, list[Item]] = {}
    for it in items:
        key = it.cluster_id if it.cluster_id is not None else f"__solo__{it.item_id}"
        buckets.setdefault(key, []).append(it)
    keys = sorted(buckets, key=str)
    random.Random(seed).shuffle(keys)
    folds: list[list[Item]] = [[] for _ in range(n_folds)]
    for i, key in enumerate(keys):
        folds[i % n_folds].extend(buckets[key])
    return [f for f in folds if f]


def evaluate(factory, eval_items: list[Item], train_items: list[Item] | None = None,
             n_folds: int = 5, max_overblock: float = DEFAULT_MAX_OVERBLOCK):
    """Produce traces for every eval item, plus the system instance used.

    With ``train_items``: fit and calibrate once on that split, then run.
    Without: cluster-disjoint K-fold, fitting on the other folds each time.
    """
    system = factory() if callable(factory) and not isinstance(factory, Guardrail) else factory

    if not _needs_training(factory):
        return run(system, eval_items), system

    if train_items:
        system.prepare(train_items)
        system.fit(train_items)
        system.calibrate(train_items, max_overblock)
        return run(system, eval_items), system

    traces: dict[str, Trace] = {}
    folds = cluster_folds(eval_items, n_folds)
    last = system
    for i, fold in enumerate(folds):
        held_in = [it for j, f in enumerate(folds) if j != i for it in f]
        last = factory()
        last.prepare(held_in)
        last.fit(held_in)
        last.calibrate(held_in, max_overblock)
        traces.update(run(last, fold))
    return traces, last
