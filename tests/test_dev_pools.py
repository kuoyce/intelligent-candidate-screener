"""Shortlist-recall development set (plan 2026-10-07, Q1): synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import dev_pools as dp
from conftest import reads_final_labels


def _docs(n: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"{prefix}{i:03d}" for i in range(n)],
                         "text": [f"text {i}" for i in range(n)]})


def test_dev_documents_are_disjoint_from_fit_calibration_weak_pairs_and_eval():
    """Q1 acceptance: a dev document was never fitted on (role classifier's fit part, its
    calibration draw, SBERT weak's pairs) and is never eval-region."""
    docs = {"jd": _docs(20, "j"), "cv": _docs(30, "c")}
    parts = {"jd": {"fit": np.arange(0, 10), "val": np.arange(10, 20), "calib": np.array([10, 11])},
             "cv": {"fit": np.arange(0, 15), "val": np.arange(15, 30), "calib": np.array([15])}}
    excluded = {"jd": {"j012", "j003"}, "cv": {"c020", "c029"}}
    dev = dp.select_dev_documents(docs, parts, excluded)
    for s in ("jd", "cv"):
        ids = set(dev[s]["id"])
        fit = set(docs[s]["id"].iloc[parts[s]["fit"]])
        calib = set(docs[s]["id"].iloc[parts[s]["calib"]])
        assert not ids & fit
        assert not ids & calib
        assert not ids & excluded[s]
    assert set(dev["jd"]["id"]) == {f"j{i:03d}" for i in range(13, 20)}
    assert len(dev["cv"]) == 15 - 1 - 2


def _pool_inputs():
    rng = np.random.default_rng(0)
    kws = ["Python", "Java", "QA", "Design"]
    cv = pd.DataFrame({"id": [f"c{i:04d}" for i in range(600)],
                       "Primary Keyword": [kws[i % 4] for i in range(600)]})
    jd = pd.DataFrame({"id": [f"j{i:03d}" for i in range(40)],
                       "Primary Keyword": [kws[i % 4] for i in range(40)],
                       "title": ["" if i % 10 == 0 else f"Dev {i}" for i in range(40)]})
    return jd, cv.iloc[rng.permutation(600)]


def test_pools_follow_the_generic_design_and_split_by_jd():
    jd, cv = _pool_inputs()
    p = dp.draw_pools(jd, cv, n_jds=20, n_dev_train=12, pool_size=50, seed=0)
    assert p.jd_id.nunique() == 20 and p.groupby("jd_id").cv_id.nunique().eq(50).all()
    assert not set(p.jd_id) & {"j000", "j010", "j020", "j030"}          # empty title
    kw = dict(zip(cv["id"], cv["Primary Keyword"]))
    jkw = dict(zip(jd["id"], jd["Primary Keyword"]))
    for jd_id, g in p.groupby("jd_id"):
        assert (g.slot == "same_keyword").sum() == 8
        assert all(kw[c] == jkw[jd_id] for c in g[g.slot == "same_keyword"].cv_id)
        assert all(kw[c] != jkw[jd_id] for c in g[g.slot == "other_keyword"].cv_id)
    split = p.drop_duplicates("jd_id").split.value_counts()
    assert split["dev-train"] == 12 and split["dev-select"] == 8
    assert p.equals(dp.draw_pools(jd, cv.sample(frac=1, random_state=3), n_jds=20,
                                  n_dev_train=12, pool_size=50, seed=0))   # order-free


def test_pairs_frame_carries_no_label_or_metadata_column():
    jd, cv = _pool_inputs()
    pools = dp.draw_pools(jd, cv, n_jds=4, n_dev_train=2, pool_size=20, seed=0)
    dev = {"jd": jd.assign(text="t", family="F", Position="p"),
           "cv": cv.assign(text="t", family="F", Position="p"), "pools": pools}
    frame = dp.pairs(dev)
    assert set(frame.columns) == {"pair_id", "jd_id", "cv_id", "split", "jd_title", "jd_text",
                                  "cv_text"}
    assert dp.labels(dev, frame).eq(1).all()


def test_module_never_reads_final_labels_or_judgements():
    """D46: the development set is where options are chosen; it never sees the final set."""
    assert reads_final_labels(dp) == []
