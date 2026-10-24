"""chat-3 (D62): the JD draw, the pools and the guards (no data, no model)."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import chat3
from candidate_screener.rerank import chat_rerank as cr


def _docs():
    kws = ["Python", "Java", "Business Analyst", "Data Science"]
    jd = pd.DataFrame({"id": [f"j{i:03d}" for i in range(80)],
                       "Primary Keyword": [kws[i % 4] for i in range(80)],
                       "title": ["" if i == 0 else f"title {i}" for i in range(80)]})
    cv = pd.DataFrame({"id": [f"c{i:03d}" for i in range(300)],
                       "Primary Keyword": [kws[i % 3] if i < 297 else "Data Science" for i in range(300)]})
    return jd, cv


def test_eligible_needs_a_title_and_enough_keyword_cvs():
    jd, cv = _docs()
    ok = chat3.eligible_jds(jd, cv, n_same=8)
    assert "j000" not in set(ok["id"])                                   # no title
    assert not ok["Primary Keyword"].eq("Data Science").any()           # 3 CVs < 8
    assert len(ok) == 59


def test_draw_meets_quotas_and_never_takes_an_excluded_jd():
    jd, cv = _docs()
    ok = chat3.eligible_jds(jd, cv, n_same=8)
    excluded = {"j001", "j002", "j006"}
    d = chat3.draw_jds(ok, excluded, n_generic=10, targeted={"Business Analyst": 4}, seed=0)
    assert d.stratum.value_counts().to_dict() == {"generic": 10, "targeted": 4}
    assert not set(d.jd_id) & excluded and d.jd_id.is_unique
    kw = dict(zip(jd["id"], jd["Primary Keyword"]))
    assert all(kw[j] == "Business Analyst" for j in d[d.stratum == "targeted"].jd_id)
    pd.testing.assert_frame_equal(d, chat3.draw_jds(ok, excluded, n_generic=10,
                                                     targeted={"Business Analyst": 4}, seed=0))
    with pytest.raises(ValueError, match="quota"):
        chat3.draw_jds(ok, excluded, n_generic=10, targeted={"Business Analyst": 40}, seed=0)


def test_pools_follow_the_dev_recipe():
    jd, cv = _docs()
    jds = pd.DataFrame({"jd_id": ["j001", "j005"], "stratum": ["generic", "targeted"]})
    p = chat3.draw_pools(jds, jd, cv, pool_size=50, n_same=8, n_other=2, seed=0)
    kw_cv = dict(zip(cv["id"], cv["Primary Keyword"]))
    kw_jd = dict(zip(jd["id"], jd["Primary Keyword"]))
    for j, g in p.groupby("jd_id"):
        assert len(g) == 50 and g.cv_id.is_unique
        assert (g[g.slot == "same_keyword"].cv_id.map(kw_cv) == kw_jd[j]).sum() == 8
        assert (g[g.slot == "other_keyword"].cv_id.map(kw_cv) != kw_jd[j]).sum() == 2
    pd.testing.assert_frame_equal(p, chat3.draw_pools(jds, jd, cv, pool_size=50, n_same=8, n_other=2, seed=0))


def test_overlaps_counts_forbidden_jds_and_cvs():
    frame = pd.DataFrame({"jd_id": ["a", "a", "b"], "cv_id": ["x", "y", "z"]})
    hits = chat3.overlaps(frame, {"eval": {"jd": {"b"}, "cv": {"y", "q"}}, "d50": {"jd": {"c"}}})
    assert hits == {"eval": 2, "d50": 0}


def test_chat3_keeps_its_own_files_and_never_mixes_runs():
    assert cr.paths(True) == (cr.CHAT3_PLAN, cr.CHAT3_REPLIES)
    assert cr.paths(False) == (cr.PLAN, cr.REPLIES)
    assert not {cr.CHAT3_PLAN, cr.CHAT3_REPLIES} & {cr.PLAN, cr.REPLIES}
    mixed = pd.DataFrame({"pair_id": ["p1", "p2"], "run": [cr.CHAT3_RUN, cr.RUN]})
    with pytest.raises(ValueError, match="mixes"):
        cr.texts(mixed)


def _plan_and_replies():
    plan = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "jd_id": ["j", "j", "k"], "cv_id": ["a", "b", "c"],
                         "stratum": ["generic", "generic", "targeted"], "source": ["x2_top100"] * 3})
    replies = pd.DataFrame({"pair_id": ["p1", "p2", "p3", "p1"], "score": [80, 40, 60, 10],
                            "label": ["Good Fit", "No Fit", "Potential Fit", "No Fit"],
                            "run": [cr.CHAT3_RUN] * 3 + [cr.RUN], "instrument_sha256": ["i"] * 4})
    return plan, replies


def test_join_scores_takes_chat3_only_and_refuses_gaps_or_another_instrument():
    plan, replies = _plan_and_replies()
    got = chat3.join_scores(plan, replies, "i")
    assert got.set_index("pair_id").score.to_dict() == {"p1": 80, "p2": 40, "p3": 60}   # chat-1's p1 ignored
    with pytest.raises(ValueError, match="no score yet"):
        chat3.join_scores(plan, replies[replies.pair_id != "p3"], "i")
    with pytest.raises(ValueError, match="instrument"):
        chat3.join_scores(plan, replies, "other")


def test_choose_takes_the_best_and_breaks_ties_by_menu_order():
    t = {("L1", "F9"): 0.4501, ("L2", "F9"): 0.4504, ("L1", "F12"): 0.44, ("L2", "F12"): 0.46}
    assert chat3.choose(t) == ("L2", "F12")
    t[("L2", "F12")] = 0.4499                                   # 0.450 all round: the first arm wins
    assert chat3.choose(t) == ("L1", "F9")


def test_rules_apply_the_preset_bars():
    def d(v, lo):
        return {"recall": {"total": {"value": v, "ci": [lo, v + 0.03], "n_jd": 60}}}
    results = {chat3.STAGE2: {"delta_vs_X2": d(0.06, 0.02), f"delta_vs_{chat3.S1_ORDER}": d(0.01, -0.01)}}
    human = {"systems": {chat3.STAGE2: {"auc_delta_vs_X2": [
        {"scope": "generic", "subset": "all", "delta": -1, "ci": [-2, 0], "n_jd": 40},
        {"scope": "total", "subset": "all", "delta": 0.05, "ci": [0.01, 0.09], "n_jd": 48}]}}}
    got = [r["passes"] for r in chat3.rules(results, human).values()]
    assert got == [True, False, True]
    results[chat3.STAGE2]["delta_vs_X2"] = d(0.049, 0.01)        # under the +0.05 bar
    assert not next(iter(chat3.rules(results, human).values()))["passes"]


def test_pool_coverage_ranks_the_whole_pool_and_counts_unscored_cvs_as_misses():
    pool = pd.DataFrame({"pair_id": [f"p{i:02d}" for i in range(30)], "jd_id": ["j"] * 30})
    ref = pd.DataFrame({"pair_id": [f"p{i:02d}" for i in range(20)], "jd_id": ["j"] * 20,
                        "score": list(range(100, 0, -5))})                 # chat top 10 = p00..p09
    best_first = pd.Series(-np.arange(30, dtype=float), index=pool.pair_id.to_numpy())
    got = chat3.pool_coverage(pool, ref, best_first, 10).loc["j"]
    assert got.coverage == 1.0 and got.scored == 1.0
    unscored_first = pd.Series(np.arange(30, dtype=float), index=pool.pair_id.to_numpy())  # p29.. on top
    got = chat3.pool_coverage(pool, ref, unscored_first, 20).loc["j"]
    assert got.coverage == 0.0 and got.scored == 0.5                       # p10..p29: none of the top 10
