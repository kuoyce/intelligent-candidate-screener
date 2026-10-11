"""Ordering evaluation (D52): named invariants on inline synthetic data, no `data/` needed."""
import itertools

import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import final_set, ordering


def _frame(rows):
    """rows: (jd_id, stratum, yc_raw, van_raw, tier, final_label)."""
    out = []
    for i, (jd, stratum, yc, van, tier, final) in enumerate(rows):
        out.append({"pair_id": f"{jd}__c{i}", "stratum": stratum, "jd_id": jd, "yc_raw": yc,
                    "van_raw": van, "tier": tier, "final_label": final})
    return pd.DataFrame(out)


G, P, N = "Good Fit", "Potential Fit", "No Fit"


def _grades():
    rows = []
    for jd, stratum, labels in (("j1", "generic", [G, P, N, N, P]),
                                ("j2", "generic", [P, G, N, G, N]),
                                ("j3", "targeted", [G, N, P, N, N]),
                                ("j4", "targeted", [N, N, N, N, N])):     # all equal
        for k, lab in enumerate(labels):
            van = lab if k < 3 else None
            tier = "humans_agree" if van else "single_human"
            rows.append((jd, stratum, lab, van, tier, "No Fit" if lab == N else "Fit"))
    return ordering.grade_frame(_frame(rows))


# --- the named invariants ------------------------------------------------------------------

def test_score_ties_give_no_tie_break_luck():
    """Permuting tied items cannot change nDCG: a tie is worth its expected value.
    (A naive argsort nDCG gives 1.0 for one permutation of the tie and less for another.)"""
    gains = np.array([2.0, 0.0, 1.0, 0.0, 2.0])
    scores = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    values = {round(ordering.ndcg(gains[list(p)], scores[list(p)]), 12)
              for p in itertools.permutations(range(5))}
    assert len(values) == 1
    # an all-tied scorer equals the random expectation exactly
    assert ordering.ndcg(gains, np.zeros(5)) == pytest.approx(
        ordering.expected_random(gains, "ndcg"))


def test_a_jd_with_all_equal_grades_is_excluded_and_counted():
    g = _grades()
    s = pd.Series(np.arange(len(g), dtype=float), index=g.index)
    for metric in ordering.METRICS:
        values, excluded = ordering.per_jd(g, "yc", s, metric)
        assert "j4" not in values.index and excluded == 1
        assert sorted(values.index) == ["j1", "j2", "j3"]
    fig = [f for f in ordering.figures("s", s, g) if f["reference"] == "yc"
           and f["subset"] == "full" and f["scope"] == "targeted" and f["metric"] == "ndcg"][0]
    assert fig["n_jd"] == 1 and fig["n_jd_excluded"] == 1


def test_a_constant_scorer_scores_exactly_one_half_pairwise():
    g = _grades()
    values, _ = ordering.per_jd(g, "yc", pd.Series(0.3, index=g.index), "pairwise")
    assert (values == 0.5).all()
    assert ordering.pairwise([2, 1, 0, 0], [7, 7, 7, 7]) == 0.5


def test_a_perfect_scorer_scores_one_on_both_metrics():
    g = _grades()
    perfect = g.yc + np.linspace(0, 1e-3, len(g))     # orders by grade, arbitrary within grade
    for metric in ordering.METRICS:
        values, _ = ordering.per_jd(g, "yc", perfect, metric)
        assert np.allclose(values, 1.0)
    # a reversed scorer is 0 pairwise
    assert ordering.pairwise([2, 1, 0], [0, 1, 2]) == 0.0


def test_the_mean_reference_is_nan_where_only_one_human_exists():
    g = _grades()
    one = g.van.isna()
    assert g.loc[one, "mean"].isna().all()
    assert g.loc[~one, "mean"].notna().all()
    assert (g.loc[one, "yc_single"] == g.loc[one, "yc"]).all()
    assert g.loc[~one, "yc_single"].isna().all()


def test_the_llm_is_rejected_as_a_reference():
    g = _grades()
    for bad in ("llm", "final_label", "fit"):
        with pytest.raises(ValueError, match="not a reference"):
            ordering.reference(g, bad)


# --- supporting checks ---------------------------------------------------------------------

def test_expected_random_matches_the_average_over_all_permutations():
    gains = np.array([2.0, 1.0, 0.0, 0.0])
    perms = [ordering.ndcg(gains, np.array(p, dtype=float))
             for p in itertools.permutations(range(4))]
    assert ordering.expected_random(gains, "ndcg") == pytest.approx(np.mean(perms))
    assert ordering.expected_random(gains, "pairwise") == 0.5


def test_figures_always_carry_generic_targeted_and_total_with_n():
    g = _grades()
    figs = ordering.figures("s", g.yc, g)
    ordering.assert_total_has_strata(figs)
    keys = {(f["reference"], f["subset"], f["metric"]) for f in figs}
    assert len(keys) == len(ordering.REFERENCES) * len(ordering.SUBSETS) * len(ordering.METRICS)
    for key in keys:
        assert {f["scope"] for f in figs if (f["reference"], f["subset"], f["metric"]) == key} \
            == set(final_set.SCOPES)
    assert all("n_jd" in f and "n_jd_excluded" in f for f in figs)
    with pytest.raises(ValueError):
        ordering.assert_total_has_strata([{**figs[0], "scope": "total"}])


def test_subsets_select_llm_free_tiers_and_final_fit():
    g = _grades()
    assert set(ordering.restrict(g, "total", "llm_free").tier) <= set(final_set.LLM_FREE_TIERS)
    assert ordering.restrict(g, "total", "fit_only").fit.all()


def test_paired_delta_is_the_mean_per_jd_difference():
    g = _grades()
    perfect = g.yc + np.linspace(0, 1e-3, len(g))
    d = ordering.paired_delta(perfect, pd.Series(0.0, index=g.index), g, "yc", "pairwise")
    assert d["delta"] == pytest.approx(0.5) and d["n_jd"] == 3
    d0 = ordering.paired_delta(pd.Series(0.0, index=g.index), None, g, "yc", "pairwise")
    assert d0["delta"] == 0.0                      # a constant scorer is exactly random


def test_llm_series_must_be_exactly_600_unique_pairs():
    base = {"agent_sha256": final_set.LLM_AGENT_SHA256, "model": final_set.LLM_MODEL}
    ok = pd.DataFrame([{**base, "pair_id": f"p{i}", "run": 9 if i < 400 else 10, "score": i % 100}
                       for i in range(600)])
    other = pd.DataFrame([{**base, "pair_id": "p0", "run": 5, "score": 1},
                          {**base, "model": "other", "pair_id": "p1", "run": 9, "score": 1}])
    assert len(ordering.llm_scores([ok, other])) == 600
    with pytest.raises(ValueError):
        ordering.llm_scores([ok, ok.iloc[:1]])


def test_unknown_raw_label_is_refused():
    with pytest.raises(ValueError):
        ordering.grade_frame(_frame([("j", "generic", "Great Fit", None, "single_human", "Fit")]))


def test_model_dependent_systems_need_sentence_transformers():
    pytest.importorskip("sentence_transformers")
    from candidate_screener.matching import sbert
    assert hasattr(sbert, "SbertZeroShot") and hasattr(sbert, "SbertWeak")
