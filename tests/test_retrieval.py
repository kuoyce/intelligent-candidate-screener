"""Top-k exposure — the Q18b measurement *(decisions D26, D27)*.

`top_k_exposure` is the number Q18 waited two phases for: of the slots a system puts
at the top, how many are unjudged distractors that precision counts as misses without
anyone having looked? Q18 could not be closed without it and it needed no annotator —
only a scoring pass.

Because a single figure now carries a decision, the arithmetic is pinned rather than
trusted: the three slot classes must partition the top-k exactly, and the classes must
not silently merge. `judged No Fit` is a slot precision is *right* to count as a miss;
`unjudged distractor` is one it *may* be wrong about. Folding them together would
report the bias as larger than it is and would be invisible in the total.
"""
from __future__ import annotations

import pandas as pd
import pytest

from candidate_screener.evaluation import retrieval


def pool_with(grades: list[tuple[str, str]]) -> pd.DataFrame:
    """One query; `grades` is (relevance, source) in the order the scores will rank."""
    return pd.DataFrame(
        [("j_0", f"r_{i}", relevance, source, "N100")
         for i, (relevance, source) in enumerate(grades)],
        columns=["query_jd_id", "candidate_resume_id", "relevance", "source",
                 "pool_variant"])


def descending_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Score so that rank order equals row order — the ranking itself is `score`'s job."""
    return pool[["query_jd_id", "candidate_resume_id"]].assign(
        score=[float(len(pool) - i) for i in range(len(pool))])


def test_slot_classes_partition_the_top_k_exactly():
    pool = pool_with([("good", "labelled"), ("potential", "labelled"),
                      ("none", "labelled"), ("none", "distractor"),
                      ("none", "distractor"), ("good", "labelled")])
    out = retrieval.top_k_exposure(pool, descending_scores(pool), "N100", k=5)
    slots = out["slots"]
    assert slots["judged_relevant_graded"] + slots["judged_none"] \
        + slots["unjudged_distractor"] == out["total_slots"] == 5
    assert slots["judged_relevant_strict"] == 1        # only the leading `good`
    assert slots["judged_relevant_graded"] == 2        # plus the `potential`
    assert slots["judged_none"] == 1                   # A1 looked and said No Fit
    assert slots["unjudged_distractor"] == 2           # nobody looked


def test_a_judged_no_fit_is_not_counted_as_an_unjudged_distractor():
    """The distinction `pools.py` keeps `source` alongside `relevance` to preserve.

    Both are precision misses, but only one is a miss the judging wave could have
    overturned. Merging them inflates the reported bias.
    """
    pool = pool_with([("none", "labelled")] * 5)
    out = retrieval.top_k_exposure(pool, descending_scores(pool), "N100", k=5)
    assert out["slots"]["judged_none"] == 5
    assert out["slots"]["unjudged_distractor"] == 0
    assert out["distractor_share_mean"] == 0.0


def test_distractor_share_is_a_per_query_mean_not_a_slot_total():
    """Two queries, unequal exposure: the CI resamples queries, so the mean must too.

    A slot-weighted total would let one query with many distractors dominate, and the
    figure would no longer match the interval reported beside it.
    """
    pool = pd.concat([
        pool_with([("good", "labelled")] * 5),
        pool_with([("none", "distractor")] * 5).assign(query_jd_id="j_1"),
    ], ignore_index=True)
    out = retrieval.top_k_exposure(pool, descending_scores(pool), "N100", k=5)
    assert out["queries"] == 2
    assert out["distractor_share_mean"] == pytest.approx(0.5)


def test_figure_row_reports_percent_of_attainable_against_the_ceiling():
    """A raw 0.45 against a 0.781 ceiling is 58% of attainable — the D26 reporting rule."""
    from candidate_screener.evaluation.metrics import Figure
    row = retrieval.figure_row(
        Figure("Precision@5", "strict", "N100", 0.45, 31, 0.3, 0.6), ceiling=0.7806)
    assert row["pct_of_attainable"] == pytest.approx(57.6, abs=0.1)
