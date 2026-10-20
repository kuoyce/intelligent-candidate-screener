"""chat-3 (D62): the JD draw, the pools and the guards (no data, no model)."""
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
