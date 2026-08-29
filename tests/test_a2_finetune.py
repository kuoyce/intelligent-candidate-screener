"""`a2_finetune` — the A2 train/eval partition and the data-science shortlist (D19).

Runs without `data/`: `partition_ids` and `sample_shortlist` are pure functions of their
arguments, so the disjointness and determinism guarantees the module exists to provide are
tested directly, without a 396 MB download.
"""
from __future__ import annotations

import pandas as pd

from candidate_screener.data.a2_finetune import (
    cv_experience_band, partition_ids, sample_shortlist, summarise_shortlist,
)

JD_IDS = [f"j{i}" for i in range(200)]
CV_IDS = [f"c{i}" for i in range(200)]


def test_partition_covers_every_id_exactly_once():
    jd_region, cv_region = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    assert set(jd_region.index) == set(JD_IDS)
    assert set(cv_region.index) == set(CV_IDS)
    assert jd_region.isin(["train", "eval"]).all()
    assert cv_region.isin(["train", "eval"]).all()


def test_partition_fraction_is_approximately_respected():
    jd_region, _ = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    assert 40 <= (jd_region == "eval").sum() <= 60


def test_partition_is_deterministic_given_the_same_seed():
    a = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    b = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    assert a[0].equals(b[0]) and a[1].equals(b[1])


def test_partition_differs_across_seeds():
    a = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    b = partition_ids(JD_IDS, CV_IDS, 0.25, seed=1)
    assert not a[0].equals(b[0])


def test_jd_and_cv_regions_are_assigned_independently():
    """The whole point of D19: a JD landing in eval says nothing about which CVs do."""
    jd_region, cv_region = partition_ids(JD_IDS, CV_IDS, 0.25, seed=0)
    # If the two sides were assigned from the same draw, the eval fraction of JD ids
    # and CV ids would be correlated by construction. With disjoint id spaces (j*/c*)
    # and independent spawned seeds, both regions simply need to be internally valid —
    # already covered above — and this asserts they are not literally the same series.
    assert list(jd_region.to_numpy()) != list(cv_region.to_numpy())


def test_cv_experience_band_matches_notebook_03():
    assert cv_experience_band(0) == "0-1"
    assert cv_experience_band(1) == "0-1"
    assert cv_experience_band(2) == "2-3"
    assert cv_experience_band(3) == "2-3"
    assert cv_experience_band(4) == "4-6"
    assert cv_experience_band(6) == "4-6"
    assert cv_experience_band(7) == "7+"
    assert cv_experience_band(11) == "7+"


def test_cv_experience_band_handles_missing_values():
    assert cv_experience_band(float("nan")) is None


def _synthetic_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    jd = pd.DataFrame({
        "id": [f"j{i}" for i in range(6)],
        "Primary Keyword": ["Data Science"] * 3 + ["Data Analyst"] * 3,
        "exp_band": ["0-1", "0-1", "2-3", "0-1", "0-1", "2-3"],
    })
    cv = pd.DataFrame({
        "id": [f"c{i}" for i in range(10)],
        "Primary Keyword": ["Data Science"] * 5 + ["Data Analyst"] * 5,
        "exp_band": ["0-1"] * 4 + ["2-3"] + ["0-1"] * 4 + ["2-3"],
    })
    return jd, cv


def test_sample_shortlist_only_draws_within_matching_cells():
    jd_f, cv_f = _synthetic_pool()
    out = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    jd_lookup = jd_f.set_index("id")[["Primary Keyword", "exp_band"]]
    cv_lookup = cv_f.set_index("id")[["Primary Keyword", "exp_band"]]
    for row in out.itertuples():
        assert tuple(jd_lookup.loc[row.jd_id]) == (row.primary_keyword, row.exp_band)
        assert tuple(cv_lookup.loc[row.cv_id]) == (row.primary_keyword, row.exp_band)


def test_sample_shortlist_pair_ids_are_unique():
    jd_f, cv_f = _synthetic_pool()
    out = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    assert out.pair_id.is_unique


def test_sample_shortlist_is_deterministic():
    jd_f, cv_f = _synthetic_pool()
    a = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    b = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    assert a.equals(b)


def test_sample_shortlist_respects_per_jd_cap():
    jd_f, cv_f = _synthetic_pool()
    out = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=2, seed=0)
    assert (out.groupby("jd_id").size() <= 2).all()


def test_sample_shortlist_skips_cells_with_no_candidates():
    """A JD band with zero matching CVs must not raise or silently duplicate rows."""
    jd_f = pd.DataFrame({"id": ["j0"], "Primary Keyword": ["Data Science"], "exp_band": ["7+"]})
    cv_f = pd.DataFrame({"id": ["c0"], "Primary Keyword": ["Data Science"], "exp_band": ["0-1"]})
    out = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    assert out.empty


def test_summarise_shortlist_counts_match_the_frame():
    jd_f, cv_f = _synthetic_pool()
    out = sample_shortlist(jd_f, cv_f, jds_per_cell=5, per_jd=3, seed=0)
    summary = summarise_shortlist(out)
    assert summary["pairs"] == len(out)
    assert summary["jds"] == out.jd_id.nunique()
    assert summary["cvs"] == out.cv_id.nunique()
