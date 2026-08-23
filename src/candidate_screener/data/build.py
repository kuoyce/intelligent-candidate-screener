"""Build every derived artefact — the Phase 3 entry point.

Each task is a pure function of (`data/raw/`, seed): running this on a fresh clone
after `fetch --all` reproduces the artefacts that `data/` is git-ignored for.

Usage:
    uv run python -m candidate_screener.data.build --all --seed 0
    uv run python -m candidate_screener.data.build --task fit-split
"""
from __future__ import annotations

import argparse

from candidate_screener.data import fit_split, pools

#: The hold-out fraction signed off on 23 Aug 2026 — see `fit-split-survey.json`.
#: 30% yields 31 test JDs carrying a Good Fit against 24 at 25%; the val fold was
#: dropped in favour of doubly-disjoint CV inside train (D16's stated fallback).
FIT_TEST_FRAC, FIT_VAL_FRAC = 0.30, 0.0


def build_fit_split(seed: int) -> None:
    fit_split.print_build(fit_split.build(FIT_TEST_FRAC, FIT_VAL_FRAC, seed))


def build_pools(seed: int) -> None:
    pools.print_report(pools.build(seed))


#: Task -> builder, in dependency order. 3.1, 3.4 and 3.5 register as they land.
TASKS = {"fit-split": build_fit_split, "pools": build_pools}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=sorted(TASKS), default=[])
    ap.add_argument("--all", action="store_true", help="every registered task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = sorted(TASKS) if args.all else args.task
    if not tasks:
        ap.error("pass --all or --task <name> [...]")
    for name in tasks:
        TASKS[name](args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
