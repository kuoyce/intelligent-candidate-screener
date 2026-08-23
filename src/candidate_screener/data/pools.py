"""Task 3.3 — retrieval pools over the leak-free test split *(decisions D1, D9)*.

A pool turns the pairwise fit corpus into a retrieval benchmark: for each test JD
*q*, one ranked candidate set containing every resume A1 judged against *q* plus
enough unjudged resumes to reach a realistic pool depth.

Two things about these pools must travel with every number computed on them:

1. **Distractors are *assumed* non-relevant, not judged.** A1 labels a median of 4
   resumes per test JD; a 100-deep pool is therefore ~96% assumption. A relevant
   resume hiding among the distractors is scored as a false positive, so precision
   is biased *downward* by construction. The pools are a comparison instrument
   between systems, not an estimate of production precision.
2. **A judged `No Fit` is not the same as an unjudged distractor.** The schema keeps
   `source` alongside `relevance` so an analysis can restrict itself to judged
   documents when that distinction matters.

Pool variants are **nested** — N20 ⊂ N100 ⊂ Nfull — so a sensitivity run differs
from the primary run only by depth, never by which distractors were drawn.

Usage:
    uv run python -m candidate_screener.data.pools --build --seed 0
    uv run python -m candidate_screener.data.pools --feasibility
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.data.fit_split import MANIFESTS, SPLIT_MANIFEST, FIT_OUT

POOLS_OUT = PROCESSED / "pools"
POOL_MANIFEST = MANIFESTS / "pools.csv"
POOL_YIELD = MANIFESTS / "pools-yield.json"

#: Target pool depth per variant. `None` = every resume held out into the split.
#: N100 is primary; the others are sensitivity runs. N is a *floor*: every judged
#: resume stays in the pool even where that pushes it past N, because discarding a
#: human judgement to hit a round number would be the wrong trade.
VARIANTS: dict[str, int | None] = {"N20": 20, "N100": 100, "Nfull": None}

#: A1's 3-class scheme -> the graded relevance nDCG consumes directly.
RELEVANCE = {"Good Fit": "good", "Potential Fit": "potential", "No Fit": "none"}

ASSUMPTION = (
    "Distractors are UNJUDGED and assumed non-relevant. A1 judges a median of 4 "
    "resumes per test JD, so a 100-deep pool is ~96% assumption. Precision is biased "
    "downward by construction: a relevant resume among the distractors scores as a "
    "false positive. These pools compare systems; they do not estimate production "
    "precision. Report this wherever a pool-derived figure appears."
)


def load_test_pool() -> tuple[pd.DataFrame, np.ndarray]:
    """The judged test pairs, and the full universe of test-side resumes.

    The universe is every resume *assigned* to test — 193 — not only the 158 that
    happen to appear in a judged pair. The other 35 are held out from training just
    as firmly, and a distractor's job is to be a plausible unjudged candidate.
    """
    judged = pd.read_parquet(FIT_OUT / "test.parquet")
    manifest = pd.read_csv(SPLIT_MANIFEST)
    universe = manifest[(manifest.doc_type == "resume") & (manifest.split == "test")].doc_id
    return judged, np.array(sorted(universe))


def build_pools(judged: pd.DataFrame, universe: np.ndarray, seed: int) -> pd.DataFrame:
    queries = sorted(judged.jd_id.unique())
    seeds = np.random.SeedSequence(seed).spawn(len(queries))
    rows = []

    for query, query_seed in zip(queries, seeds):
        labelled = judged[judged.jd_id == query]
        judged_ids = set(labelled.resume_id)
        # One permutation per query, prefixed per variant — this is what makes the
        # variants nested rather than three unrelated draws.
        pool_universe = np.array([r for r in universe if r not in judged_ids])
        order = pool_universe[np.random.default_rng(query_seed).permutation(len(pool_universe))]

        for variant, target in VARIANTS.items():
            n_distractors = (len(order) if target is None
                             else max(0, min(target - len(labelled), len(order))))
            for _, pair in labelled.iterrows():
                rows.append((query, pair.resume_id, RELEVANCE[pair.label], "labelled", variant))
            for resume in order[:n_distractors]:
                rows.append((query, resume, "none", "distractor", variant))

    return pd.DataFrame(rows, columns=["query_jd_id", "candidate_resume_id",
                                       "relevance", "source", "pool_variant"])


def summarise(pools: pd.DataFrame) -> dict:
    out: dict = {"queries": int(pools.query_jd_id.nunique()), "variants": {}}
    for variant, group in pools.groupby("pool_variant"):
        size = group.groupby("query_jd_id").size()
        good = group[group.relevance == "good"].groupby("query_jd_id").size()
        graded = group[group.relevance.isin(["good", "potential"])].groupby("query_jd_id").size()
        out["variants"][variant] = {
            "rows": int(len(group)),
            "pool_size": {"min": int(size.min()), "median": int(size.median()),
                          "max": int(size.max())},
            "judged_per_pool_median": int(
                group[group.source == "labelled"].groupby("query_jd_id").size().median()),
            "queries_with_relevant": {"strict": int(len(good)), "graded": int(len(graded))},
            "relevant_per_query_median": {"strict": int(good.median()),
                                          "graded": int(graded.median())},
        }
    return out


def attainability(pools: pd.DataFrame, variant: str = "N100") -> dict:
    """The ceiling each Recall@k carries on *these* pools — Q17's evidence.

    Recall@k cannot exceed min(k, R)/R. The predecessor plan banned Recall@10 on a
    median of 18 relevant resumes per query; that was the *leaky* split's density.
    Held-out queries keep only their held-out resumes, so the median falls and the
    ceiling rises. Measured here rather than assumed either way.
    """
    group = pools[pools.pool_variant == variant]
    out: dict = {"variant": variant, "definitions": {}}
    for name, grades in (("strict", ["good"]), ("graded", ["good", "potential"])):
        R = group[group.relevance.isin(grades)].groupby("query_jd_id").size()
        entry = {"n_queries": int(len(R)), "relevant_per_query": {
            "min": int(R.min()), "median": int(R.median()), "max": int(R.max()),
            "mean": round(float(R.mean()), 1)}, "recall_ceiling": {}}
        for k in (5, 10, 20, 50):
            cap = np.minimum(k, R) / R
            entry["recall_ceiling"][f"@{k}"] = {
                "median_cap": round(float(np.median(cap)), 3),
                "pct_queries_cap_ge_0.90": round(100 * float((cap >= 0.9).mean()), 1),
                "pct_queries_cap_eq_1.0": round(100 * float((cap == 1).mean()), 1)}
        out["definitions"][name] = entry
    return out


def build(seed: int) -> dict:
    judged, universe = load_test_pool()
    pools = build_pools(judged, universe, seed)
    pools = pools.sort_values(["pool_variant", "query_jd_id", "source", "candidate_resume_id"],
                              kind="stable").reset_index(drop=True)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    pools.to_csv(POOL_MANIFEST, index=False, lineterminator="\n")

    report = {"seed": seed, "universe_resumes": int(len(universe)),
              "note_distractors": ASSUMPTION,
              "note_variants": "Nested: N20 subset of N100 subset of Nfull.",
              "summary": summarise(pools), "attainability": attainability(pools)}
    POOL_YIELD.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    POOLS_OUT.mkdir(parents=True, exist_ok=True)
    text = pd.concat([
        judged[["jd_id", "job_description_text"]].drop_duplicates("jd_id"),
    ]).set_index("jd_id").job_description_text
    resumes = judged[["resume_id", "resume_text"]].drop_duplicates("resume_id").set_index("resume_id").resume_text
    joined = pools.assign(
        job_description_text=pools.query_jd_id.map(text),
        resume_text=pools.candidate_resume_id.map(resumes))
    joined.to_parquet(POOLS_OUT / "pools.parquet", index=False)
    report["unjoinable_resume_text"] = int(joined.resume_text.isna().sum())
    return report


def print_report(r: dict) -> None:
    s = r["summary"]
    print(f"\n=== Pools over {s['queries']} test queries, {r['universe_resumes']} candidate resumes")
    for variant, v in s["variants"].items():
        print(f"  {variant:<6} pool size {v['pool_size']['min']}-{v['pool_size']['max']} "
              f"(median {v['pool_size']['median']}), judged/pool median "
              f"{v['judged_per_pool_median']}, queries w/ relevant: "
              f"strict {v['queries_with_relevant']['strict']} "
              f"graded {v['queries_with_relevant']['graded']}")
    print(f"\n=== Q17 — Recall@k ceiling on these pools ({r['attainability']['variant']})")
    for name, d in r["attainability"]["definitions"].items():
        rp = d["relevant_per_query"]
        print(f"  {name:<7} n={d['n_queries']:<3} relevant/query median={rp['median']} "
              f"(min {rp['min']}, max {rp['max']})")
        for k, c in d["recall_ceiling"].items():
            print(f"      Recall{k:<4} median cap {c['median_cap']:.2f}  "
                  f"queries reaching 0.90: {c['pct_queries_cap_ge_0.90']:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--feasibility", action="store_true", help="attainability only, writes nothing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.build or args.feasibility):
        ap.error("pass --build or --feasibility")

    if args.feasibility:
        judged, universe = load_test_pool()
        pools = build_pools(judged, universe, args.seed)
        print_report({"summary": summarise(pools), "attainability": attainability(pools),
                      "universe_resumes": len(universe)})
    if args.build:
        print_report(build(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
