"""End-to-end latency of `pipeline.screen` on 500-candidate roles (S4, P7.3).

Wall-clock time depends on the machine, so this record is informational and has no
`--check`: it states the hardware it was measured on. Measured on the pilot pools (the
same 500-candidate pools as the retrieval pilot), median over the JDs timed.

Usage:
    uv run python -m candidate_screener.evaluation.latency --n-jds 5
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from candidate_screener.config import OUTPUT

RECORD = OUTPUT / "matching" / "latency.json"
PROPOSAL_BUDGET_S_PER_CANDIDATE = 120.0


def measure(n_jds: int = 5, seed: int = 0) -> dict:
    from candidate_screener import pipeline
    from candidate_screener.evaluation.retrieval_pilot import build_pools
    from candidate_screener.extraction.rules import default_matcher
    pools = build_pools(seed)
    matcher = default_matcher()
    ranker = pipeline.default_ranker(True)
    t0 = time.perf_counter()
    ranker.fit(None)
    fit_s = time.perf_counter() - t0
    timings = []
    for jd_id in sorted(pools.jd_id.unique())[:n_jds]:
        pool = pools[pools.jd_id == jd_id]
        cvs = pool[["cv_id", "cv_text", "cv_keyword", "cv_exp_band"]]
        meta = {"jd_id": jd_id, "keyword": pool.jd_keyword.iloc[0], "exp_band": pool.jd_exp_band.iloc[0]}
        start = time.perf_counter()
        pipeline.screen(pool.jd_text.iloc[0], cvs, jd_meta=meta, ranker=ranker, matcher=matcher)
        timings.append(time.perf_counter() - start)
    per_role = float(np.median(timings))
    return {"pipeline": pipeline.SELECTED, "candidates_per_role": int(len(pool)),
            "roles_timed": len(timings), "fit_seconds_once": round(fit_s, 2),
            "median_seconds_per_role": round(per_role, 3),
            "median_ms_per_candidate": round(1000 * per_role / len(pool), 3),
            "proposal_budget_seconds_per_candidate": PROPOSAL_BUDGET_S_PER_CANDIDATE,
            "machine": {"platform": platform.platform(), "cpus": os.cpu_count(),
                        "python": platform.python_version()},
            "note": ("System time only (ranking + evidence, no semantic support encoder). "
                     "Recruiter time saved is not measured: that needs a user study (S4).")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jds", type=int, default=5)
    args = ap.parse_args(argv)
    rec = measure(args.n_jds)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
