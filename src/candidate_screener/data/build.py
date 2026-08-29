"""Build every derived artefact — the Phase 3 entry point.

Each task is a pure function of (`data/raw/`, seed): running this on a fresh clone
after `fetch --all` reproduces the artefacts that `data/` is git-ignored for.

Usage:
    uv run python -m candidate_screener.data.build --all --seed 0
    uv run python -m candidate_screener.data.build --task fit-split
"""
from __future__ import annotations

import argparse

from candidate_screener.annotation import queue as judging_queue
from candidate_screener.annotation import sample as indomain
from candidate_screener.data import a2_finetune, fit_split, pools

#: The hold-out fraction signed off on 23 Aug 2026 — see `fit-split-survey.json`.
#: 30% yields 31 test JDs carrying a Good Fit against 24 at 25%; the val fold was
#: dropped in favour of doubly-disjoint CV inside train (D16's stated fallback).
FIT_TEST_FRAC, FIT_VAL_FRAC = 0.30, 0.0

#: A2 eval-region reservation (D19) — matches the A1 precedent since no fraction was
#: specified when this task was scoped. Re-cuttable via `a2_finetune --eval-fraction`
#: up until the first document is labelled.
A2_EVAL_FRACTION = 0.25


def build_fit_split(seed: int) -> None:
    fit_split.print_build(fit_split.build(FIT_TEST_FRAC, FIT_VAL_FRAC, seed))


def build_pools(seed: int) -> None:
    pools.print_report(pools.build(seed))


def build_a2_partition(seed: int) -> None:
    a2_finetune.print_partition(a2_finetune.build_partition(A2_EVAL_FRACTION, seed))


def build_a2_shortlist(seed: int) -> None:
    a2_finetune.print_shortlist(a2_finetune.build(
        a2_finetune.DATASCIENCE_KEYWORDS, jds_per_cell=15, per_jd=5, seed=seed))


def build_indomain(seed: int) -> None:
    indomain.print_report(indomain.build(n_jds=40, per_jd=5, seed=seed))


def build_judging_queue(seed: int) -> None:
    judging_queue.print_report(judging_queue.build(seed))


#: Task -> builder, **in dependency order**, which is the order `--all` runs them in.
#: `indomain` reads `a2-partition.csv` and `judging-queue` reads `indomain-pairs.csv`,
#: so this dict's insertion order is load-bearing and not cosmetic. 3.1 and 3.5 register
#: as they land.
TASKS = {
    "fit-split": build_fit_split,
    "pools": build_pools,
    "a2-partition": build_a2_partition,
    "a2-shortlist": build_a2_shortlist,
    "indomain": build_indomain,
    "judging-queue": build_judging_queue,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=sorted(TASKS), default=[])
    ap.add_argument("--all", action="store_true", help="every registered task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Insertion order, not `sorted` — the dict above is a dependency order, and
    # alphabetical agreement with it was a coincidence of the first four task names.
    tasks = list(TASKS) if args.all else [t for t in TASKS if t in args.task]
    if not tasks:
        ap.error("pass --all or --task <name> [...]")
    for name in tasks:
        TASKS[name](args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
