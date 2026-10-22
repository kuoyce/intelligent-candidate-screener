"""C2 validation (plan 2026-10-22): the weighted cascade order, the tie rule (no data, no model)."""
import numpy as np
import pandas as pd

from candidate_screener.evaluation import c2_validation as cv


def _frame(n=6):
    return pd.DataFrame({"pair_id": [f"p{i}" for i in range(n)], "jd_id": ["j"] * n})


def test_fused_order_reorders_only_the_top_k_with_the_c2_weight():
    f = _frame()
    ids = f.pair_id.to_numpy()
    l1 = pd.Series([6, 5, 4, 3, 2, 1], index=ids, dtype=float)       # top 3 = p0, p1, p2
    c2 = pd.Series({"p0": 0.0, "p1": 0.0, "p2": 1.0})                 # C2 likes p2 only
    low = cv.fused_order(f, l1, c2, k=3, w=0.25)                      # p0 1 + 0, p1 .5 + 0, p2 0 + .25
    assert list(low.sort_values(ascending=False).index[:3]) == ["p0", "p1", "p2"]
    high = cv.fused_order(f, l1, c2, k=3, w=1.0)                      # p2 0 + 1 ties p0 1 + 0
    assert high["p2"] > high["p1"]
    assert (high.loc[ids[:3]] >= 10).all() and (high.loc[ids[3:]] <= 1).all()
    assert list(high.loc[ids[3:]].sort_values(ascending=False).index) == list(ids[3:])


def test_choose_takes_the_best_and_breaks_ties_to_the_smaller_k_then_w():
    totals = {(k, w): 0.5 for k in cv.KS for w in cv.WEIGHTS}
    assert cv.choose(totals) == (20, 0.25)                              # all tie: cheapest
    totals[(50, 1.0)] = 0.5004                                          # 0.500 at 3 dp: still a tie
    assert cv.choose(totals) == (20, 0.25)
    totals[(30, 0.5)] = 0.502
    assert cv.choose(totals) == (30, 0.5)


def test_top_lists_cuts_each_jd():
    f = pd.DataFrame({"pair_id": [f"{j}{i}" for j in "ab" for i in range(5)], "jd_id": [j for j in "ab" for _ in range(5)]})
    s = pd.Series(np.arange(10, dtype=float), index=f.pair_id.to_numpy())
    got = cv.top_lists(f, s, 2)
    assert sorted(got.pair_id) == ["a3", "a4", "b3", "b4"]


def test_fresh_plan_pads_each_jd_with_anchors_to_whole_batches():
    from candidate_screener.rerank import chat_rerank as cr
    scored = pd.DataFrame({"pair_id": [f"{j}-s{i:02d}" for j in "ab" for i in range(40)],
                           "jd_id": [j for j in "ab" for _ in range(40)], "cv_id": [f"c{i}" for i in range(80)],
                           "score": list(range(40)) * 2})
    new = pd.DataFrame({"pair_id": ["a-n1", "a-n2", "b-n1"], "jd_id": ["a", "a", "b"], "cv_id": ["x", "y", "z"]})
    plan = cr.fresh_plan(scored, new, cr.CHAT4_RUN, cr.CHAT4_PREFIX, pd.Series({"a": 1, "b": 2}),
                         pd.Series({"a": "generic", "b": "targeted"}), seed=0)
    for jd, g in plan.groupby("jd_id"):
        assert len(g) % cr.BATCH == 0 and (g.source == "anchor").sum() >= cr.MIN_ANCHORS
        assert set(g[g.source == "pooled"].pair_id) == set(new[new.jd_id == jd].pair_id)
        assert g.short_id.str.startswith("H").all() and g.short_id.is_unique and g.batch.eq(1).all()
    assert set(plan.run) == {cr.CHAT4_RUN} and plan.pair_id.is_unique
    assert cr.paths(chat4=True) == (cr.CHAT4_PLAN, cr.CHAT4_REPLIES)
