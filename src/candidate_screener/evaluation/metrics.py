"""Retrieval metrics for the A1 pools — with the guards that stop a figure being
reported as more than it is *(decision D9)*.

Three rules are enforced in code rather than left to reviewer discipline:

1. **No figure without its *n*.** At 31 scoreable queries the confidence interval is
   the finding, so `Figure` refuses to exist without a query count, and every
   rendering carries *n* and a bootstrap CI.
2. **No k deeper than the pool.** Recall@50 over a 20-deep pool is not a hard number,
   it is a soft lie — it silently equals Recall@20. `score` raises instead.
3. **Queries with no relevant document are excluded, not scored as zero.** Recall and
   nDCG are undefined there; scoring them 0 would report the pool's label sparsity
   as a property of the system.

The graded scheme is A1's own: `Good Fit` = 2, `Potential Fit` = 1, `No Fit` = 0.
That is nDCG's gain input directly — no separate relevance judgement is invented.

Usage:
    uv run python -m candidate_screener.evaluation.metrics --sanity
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: A1's 3-class scheme as nDCG gains. `strict` counts only Good Fit as relevant;
#: `graded` counts Potential too. They answer different questions and both are
#: reported — strict is "would a recruiter shortlist this", graded is "is this
#: worth a human's time".
GAINS = {"good": 2.0, "potential": 1.0, "none": 0.0}
DEFINITIONS = {"strict": {"good"}, "graded": {"good", "potential"}}

#: Travels with every Precision@k figure *(decision D26)*. Q18 closes as a standing
#: limitation rather than a judging wave: the ceiling is recorded and reported, and
#: Recall@10 — which carries no such ceiling at a median of 6 relevant per query — is
#: the headline retrieval metric the project's success measures actually name.
CEILING_NOTE = (
    "Precision@k on these pools is bounded below 1.0 by judged supply: many queries "
    "have fewer than k judged-relevant resumes, so no ranker can fill the top k. The "
    "ceiling is a property of the split, identical at every pool variant. Report the "
    "raw figure AND this ceiling; a system at 0.45 against a 0.781 ceiling is at 58% "
    "of attainable, not 45%."
)


@dataclass(frozen=True)
class Figure:
    """One reported number. Cannot be constructed without the *n* behind it."""
    metric: str
    definition: str
    variant: str
    value: float
    n_queries: int
    ci_low: float
    ci_high: float

    def __post_init__(self) -> None:
        if not self.n_queries:
            raise ValueError(
                f"{self.metric} has no scoreable queries — a figure without its n is "
                "not reportable. Check the relevance definition against the pool.")

    def __str__(self) -> str:
        return (f"{self.metric:<12} {self.definition:<7} {self.variant:<6} "
                f"{self.value:.3f}  [{self.ci_low:.3f}, {self.ci_high:.3f}]  n={self.n_queries}")


def bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI resampling **queries**, which is the unit of independence here.

    Resampling pairs instead would treat 659 correlated judgements as 659
    observations and produce an interval several times too narrow.
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def recall_at_k(ranked: list[str], k: int, definition: str) -> float:
    rel = DEFINITIONS[definition]
    total = sum(r in rel for r in ranked)
    return sum(r in rel for r in ranked[:k]) / total if total else float("nan")


def precision_at_k(ranked: list[str], k: int, definition: str) -> float:
    rel = DEFINITIONS[definition]
    return sum(r in rel for r in ranked[:k]) / k


def ndcg_at_k(ranked: list[str], k: int) -> float:
    """Graded nDCG over A1's own 3-class gains, ideal taken from the same pool."""
    def dcg(grades: list[str]) -> float:
        return sum(GAINS[g] / np.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = dcg(sorted(ranked, key=lambda g: -GAINS[g]))
    return dcg(ranked) / ideal if ideal else float("nan")


def rank(pool: pd.DataFrame, scores: pd.Series) -> list[str]:
    """Rank one pool by score, breaking ties on candidate ID so runs are comparable."""
    ordered = pool.assign(score=pool.candidate_resume_id.map(scores)).sort_values(
        ["score", "candidate_resume_id"], ascending=[False, True], kind="stable")
    return ordered.relevance.tolist()


def score(pools: pd.DataFrame, scores: pd.DataFrame, variant: str, metric: str,
          k: int, definition: str = "strict", seed: int = 0) -> Figure:
    """Compute one figure over every scoreable query in one pool variant.

    `scores` is long-form `query_jd_id, candidate_resume_id, score` — whatever
    produced it, a lexical baseline or a cross-encoder, is not this module's business.
    """
    group = pools[pools.pool_variant == variant]
    depth = group.groupby("query_jd_id").size()
    if k > int(depth.min()):
        raise ValueError(
            f"{metric}@{k} is deeper than the shallowest {variant} pool ({int(depth.min())} "
            f"candidates). It would silently collapse to {metric}@{int(depth.min())}. "
            f"Use a k within the pool, or the Nfull variant.")

    per_query = []
    for query, pool in group.groupby("query_jd_id"):
        relevant = sum(r in DEFINITIONS[definition] for r in pool.relevance)
        if not relevant:      # undefined, not zero
            continue
        ranked = rank(pool, scores[scores.query_jd_id == query]
                      .set_index("candidate_resume_id").score)
        per_query.append(
            ndcg_at_k(ranked, k) if metric == "nDCG"
            else recall_at_k(ranked, k, definition) if metric == "Recall"
            else precision_at_k(ranked, k, definition))

    values = np.array(per_query, dtype=float)
    if not len(values):
        raise ValueError(f"{metric}@{k} ({definition}): no query carries a relevant "
                         "document — nothing to report")
    low, high = bootstrap_ci(values, seed=seed)
    return Figure(f"{metric}@{k}", definition, variant, float(values.mean()),
                  len(values), low, high)


def attainable_ceiling(pools: pd.DataFrame, metric: str, k: int,
                       definition: str = "strict", variant: str = "N100",
                       seed: int = 0) -> Figure:
    """The best value **any** ranker could reach on these pools *(Q18a, decision D26)*.

    A second bias, distinct from the unjudged-distractor assumption `pools.ASSUMPTION`
    describes: for many queries there are not *k* judged-relevant resumes in existence
    to put in the top *k*, so a perfect ranker scores below 1.0 by construction. With a
    graded median of 4 relevant resumes per query, 90 of the 320 top-5 slots are
    unwinnable — reporting a raw Precision@5 without this understates every system
    uniformly and silently.

    Per query the ceiling is `min(k, R) / k` for Precision and `min(k, R) / R` for
    Recall, where R is the judged-relevant count under `definition`.

    **nDCG has no ceiling below 1.0** and asking for one raises: `ndcg_at_k` normalises
    by an ideal taken from the same pool, so a perfect ranker scores exactly 1.0 however
    sparse the labels are. That is not the ceiling being absent, it is nDCG hiding the
    same sparsity in its denominator.

    The ceiling is a property of the **split**, not of pool depth — it is identical at
    N20, N100 and Nfull, because adding distractors changes precision's denominator but
    not the supply of judged-relevant documents. `test_metrics.py` asserts that, because
    it is the fact that stops someone "fixing" the ceiling by choosing another variant.
    """
    if metric == "nDCG":
        raise ValueError(
            "nDCG@k has no attainable ceiling below 1.0 — it normalises by an ideal "
            "drawn from the same pool, so a perfect ranker reaches 1.0 at any label "
            "density. Use Precision or Recall, and report nDCG's sparsity separately.")
    if metric not in ("Precision", "Recall"):
        raise ValueError(f"unknown metric {metric!r} — Precision or Recall")

    group = pools[pools.pool_variant == variant]
    depth = group.groupby("query_jd_id").size()
    if k > int(depth.min()):
        raise ValueError(
            f"{metric}@{k} is deeper than the shallowest {variant} pool "
            f"({int(depth.min())} candidates) — the same guard `score` applies.")

    relevant = group[group.relevance.isin(DEFINITIONS[definition])].groupby(
        "query_jd_id").size()
    if not len(relevant):      # undefined, not zero — the rule `score` enforces
        raise ValueError(f"{metric}@{k} ({definition}): no query carries a relevant "
                         "document — nothing to bound")

    caps = np.minimum(k, relevant.to_numpy()) / (
        k if metric == "Precision" else relevant.to_numpy())
    low, high = bootstrap_ci(caps.astype(float), seed=seed)
    return Figure(f"ceiling {metric}@{k}", definition, variant, float(caps.mean()),
                  len(caps), low, high)


def ceilings(pools: pd.DataFrame, variant: str = "N100", seed: int = 0) -> dict:
    """Every precision ceiling these pools carry, shaped for `pools-yield.json`."""
    out: dict = {"variant": variant, "note": CEILING_NOTE, "definitions": {}}
    for definition in DEFINITIONS:
        entry: dict = {}
        for k in (5, 10):
            figure = attainable_ceiling(pools, "Precision", k, definition, variant, seed)
            group = pools[pools.pool_variant == variant]
            relevant = group[group.relevance.isin(DEFINITIONS[definition])].groupby(
                "query_jd_id").size()
            entry[f"precision@{k}"] = {
                "max_attainable_mean": round(figure.value, 4),
                "n_queries": figure.n_queries,
                "queries_able_to_reach_1.0": int((relevant >= k).sum()),
                "unwinnable_slots": int((k - np.minimum(k, relevant)).sum()),
                "total_slots": int(k * len(relevant))}
        out["definitions"][definition] = entry
    return out


def random_scores(pools: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The floor every real system must beat, and the metric code's own sanity check."""
    unique = pools[["query_jd_id", "candidate_resume_id"]].drop_duplicates()
    return unique.assign(score=np.random.default_rng(seed).random(len(unique)))


def main() -> int:
    from candidate_screener.data.pools import POOL_MANIFEST

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity", action="store_true",
                    help="score the pools with a random ranker — the floor, and a self-test")
    ap.add_argument("--variant", default="N100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.sanity:
        ap.error("pass --sanity")

    pools = pd.read_csv(POOL_MANIFEST)
    scores = random_scores(pools, args.seed)
    print(f"\n=== Random ranker on {args.variant} — the floor, not a result\n")
    print(f"  {'metric':<12} {'defn':<7} {'pool':<6} {'value':<7} {'95% CI':<18} n")
    for metric, k in (("Recall", 10), ("Recall", 20), ("Precision", 5),
                      ("Precision", 10), ("nDCG", 10)):
        for definition in ("strict", "graded"):
            print("  " + str(score(pools, scores, args.variant, metric, k, definition, args.seed)))

    print("\n=== The guards")
    for variant, metric, k in (("N20", "Recall", 50), ("N100", "nDCG", 193)):
        try:
            score(pools, scores, variant, metric, k, "strict", args.seed)
            print(f"  [FAIL] {metric}@{k} on {variant} was allowed")
        except ValueError as exc:
            print(f"  [ok] {metric}@{k} on {variant} refused: {str(exc).split('.')[0]}.")
    try:
        Figure("Recall@10", "strict", "N100", 0.5, 0, 0.1, 0.9)
        print("  [FAIL] a figure with n=0 was constructed")
    except ValueError:
        print("  [ok] a figure with n=0 cannot be constructed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
