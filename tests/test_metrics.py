"""The attainable-ceiling guards *(Q18a, decision D26)*.

Q18 closes as a standing limitation rather than a judging wave, which puts the whole
weight of the closure on this computation being right and staying right. Three
invariants carry it, and each was shown to fail against the code it guards:

1. **Depth-invariance.** The ceiling is a property of the *split* — it is identical at
   N20, N100 and Nfull. Without this test, someone reading "the pool is 96% assumption"
   could reasonably try to mitigate the ceiling by changing `pool_variant`, get a
   different-looking number by chance, and conclude it worked. Adding pool depth to the
   ceiling's denominator makes this test fail and nothing else.
2. **nDCG refuses.** nDCG normalises by an ideal drawn from the same pool, so a perfect
   ranker reaches 1.0 at any label density. Returning a ceiling for it would silently
   report 1.0 and read as "nDCG has no bias", which inverts the truth.
3. **The four figures.** Golden values, but pinned to the *committed manifest*, not to a
   run — `docs/data/manifests/pools.csv` is in git, so this runs on a fresh clone with
   no `data/`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from candidate_screener.evaluation import metrics

#: Measured 29 Aug 2026 and previously unreproducible from the package — pinning them
#: here is what task 5.2 of the Phase 5 plan existed to do.
CEILINGS = {("strict", 5): 0.7806, ("strict", 10): 0.5677,
            ("graded", 5): 0.7188, ("graded", 10): 0.4734}

VARIANTS = ("N20", "N100", "Nfull")


def synthetic_pools() -> pd.DataFrame:
    """Three queries with 1, 3 and 6 relevant documents, padded to depth 10.

    Chosen so the P@5 ceiling is hand-checkable: min(5,R)/5 = 0.2, 0.6, 1.0 -> mean 0.6.
    """
    rows = []
    for variant, depth in (("N20", 10), ("N100", 10)):
        for query, (good, potential) in enumerate([(1, 0), (3, 0), (4, 2)]):
            grades = (["good"] * good + ["potential"] * potential
                      + ["none"] * (depth - good - potential))
            for i, grade in enumerate(grades):
                rows.append((f"j_{query}", f"r_{query}_{i}", grade,
                             "labelled" if grade != "none" else "distractor", variant))
    return pd.DataFrame(rows, columns=["query_jd_id", "candidate_resume_id",
                                       "relevance", "source", "pool_variant"])


def test_ceiling_is_hand_checkable_on_a_synthetic_pool():
    pools = synthetic_pools()
    figure = metrics.attainable_ceiling(pools, "Precision", 5, "strict", "N100")
    assert figure.value == pytest.approx((1 / 5 + 3 / 5 + 4 / 5) / 3)
    assert figure.n_queries == 3


def test_recall_ceiling_differs_from_precision_ceiling():
    """min(k,R)/R, not min(k,R)/k — a query with 1 relevant caps recall at 1.0, not 0.2."""
    pools = synthetic_pools()
    assert metrics.attainable_ceiling(pools, "Recall", 5, "strict", "N100").value == 1.0


def test_queries_with_no_relevant_document_are_excluded_not_zeroed():
    """The rule `score` enforces. Scoring them 0 would report label sparsity as a ceiling."""
    pools = synthetic_pools()
    pools.loc[pools.query_jd_id == "j_0", "relevance"] = "none"
    figure = metrics.attainable_ceiling(pools, "Precision", 5, "strict", "N100")
    assert figure.n_queries == 2
    assert figure.value == pytest.approx((3 / 5 + 4 / 5) / 2)


def test_ndcg_has_no_ceiling_and_says_why():
    with pytest.raises(ValueError, match="normalises by an ideal"):
        metrics.attainable_ceiling(synthetic_pools(), "nDCG", 5, "strict", "N100")


def test_ceiling_honours_the_same_depth_guard_as_score():
    with pytest.raises(ValueError, match="deeper than the shallowest"):
        metrics.attainable_ceiling(synthetic_pools(), "Precision", 20, "strict", "N100")


def test_ceiling_is_invariant_to_pool_depth():
    """The load-bearing property: adding distractors cannot move the ceiling.

    The synthetic N20 and N100 pools carry identical judgements at identical depth, so
    any dependence on the *variant* — pool size, distractor count, row count — breaks
    this rather than shifting it.
    """
    pools = synthetic_pools()
    values = {v: metrics.attainable_ceiling(pools, "Precision", 5, "strict", v).value
              for v in ("N20", "N100")}
    assert values["N20"] == values["N100"]


# --- against the committed manifest ---------------------------------------

@pytest.fixture(scope="module")
def committed_pools() -> pd.DataFrame:
    from candidate_screener.data.pools import POOL_MANIFEST
    if not POOL_MANIFEST.exists():          # committed, so this should not happen
        pytest.skip(f"{POOL_MANIFEST} missing")
    return pd.read_csv(POOL_MANIFEST)


@pytest.mark.parametrize("definition,k", sorted(CEILINGS))
def test_committed_ceilings_reproduce(committed_pools, definition, k):
    figure = metrics.attainable_ceiling(committed_pools, "Precision", k, definition)
    assert round(figure.value, 4) == CEILINGS[(definition, k)]
    assert figure.n_queries == (31 if definition == "strict" else 64)


@pytest.mark.parametrize("definition,k", sorted(CEILINGS))
def test_committed_ceilings_are_identical_at_every_variant(committed_pools, definition, k):
    values = {v: round(metrics.attainable_ceiling(
        committed_pools, "Precision", k, definition, v).value, 4) for v in VARIANTS}
    assert len(set(values.values())) == 1, values


def test_recall_at_10_carries_no_meaningful_ceiling(committed_pools):
    """Why D26 headlines Recall@10 rather than Precision@5.

    Q17 reinstated Recall@10 on a median of 6 relevant per query. This asserts the
    consequence that decision rests on: the *ceiling* bias Q18a found is a precision
    artefact and does not reach the metric the proposal's success measures name.
    """
    for definition in ("strict", "graded"):
        assert metrics.attainable_ceiling(
            committed_pools, "Recall", 10, definition).value > 0.93
