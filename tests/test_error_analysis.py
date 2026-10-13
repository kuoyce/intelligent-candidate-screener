"""Error analysis 2 (plan 2026-10-13-error-analysis-2): synthetic pairs only."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import error_analysis as ea

W = pd.Series({"role_match": 2.0, "experience_compatible": 1.0, "tfidf_scaled": 1.0})


def frame():
    X = pd.DataFrame({"role_match": [0.9, 0.1, 0.0, 0.8], "experience_compatible": [1.0, 1.0, 0.0, 1.0],
                      "tfidf_scaled": [0.2, 0.9, 0.1, 0.3], "coverage": [0, 0, 0, 0],
                      "exp_known": [1.0, 0.0, 0.0, 1.0], "coverage_unweighted": [0, 0, 0, 0],
                      "log_jd_concepts": [np.log1p(2)] * 4}, index=["f1", "n1", "f2", "n2"])
    y = pd.Series([1, 0, 1, 0], index=X.index)
    jd = pd.Series(["j1", "j1", "j2", "j2"], index=X.index)
    return X, y, jd


def test_kind_of_uses_the_half_threshold_on_both_sides():
    k = ea.kind_of([0.9, 0.2, 0.2, 0.9], [0.9, 0.9, 0.2, 0.2])
    assert list(k) == ["both in role", "Fit out of role, No Fit in", "both out of role", "Fit in role, No Fit out"]


def test_culprit_is_the_term_pushing_hardest_the_wrong_way():
    X, _, _ = frame()
    # j1: Fit f1 (role .9, tfidf .2) vs NoFit n1 (role .1, tfidf .9): tfidf pushes +0.7, role pushes -1.6
    assert list(ea.culprit_of(W, X, ["f1"], ["n1"])) == ["tfidf_scaled"]
    # NoFit n2 beats Fit f2 on role (+1.6) and tfidf (+0.2): role is the culprit
    assert list(ea.culprit_of(W, X, ["f2"], ["n2"])) == ["role_match"]


def test_exact_tie_has_no_culprit():
    X = pd.DataFrame({"role_match": [0.5, 0.5], "experience_compatible": [1.0, 1.0], "tfidf_scaled": [0.3, 0.3]},
                     index=["a", "b"])
    assert list(ea.culprit_of(W, X, ["a"], ["b"])) == ["tie"]


def test_mistakes_lists_only_wrongly_ordered_pairs():
    X, y, jd = frame()
    s = pd.Series((X[list(W.index)] * W).sum(axis=1))      # j1 is ordered right (3.0 vs 2.1); j2 is not (0.1 vs 2.9)
    m = ea.mistakes(s, y, jd, X, W)
    assert list(zip(m.fit_id, m.nofit_id)) == [("f2", "n2")]
    assert list(m.kind) == ["Fit out of role, No Fit in"] and list(m.culprit) == ["role_match"]


def test_shares_sum_to_one_and_follow_the_order():
    inv = pd.DataFrame({"inv": [1.0, 1.0, 0.5], "culprit": ["a", "a", "b"]})
    s = ea.shares(inv, "culprit", ["b", "a", "c"])
    assert list(s) == ["b", "a", "c"] and s["c"] == 0.0 and sum(s.values()) == pytest.approx(1.0)


def test_sample_takes_at_most_n_per_bucket_and_is_reproducible():
    inv = pd.DataFrame({"inv": 1.0, "culprit": ["a"] * 30 + ["b"] * 5, "fit_id": range(35), "nofit_id": range(35)})
    s1, s2 = ea.sample_mistakes(inv, 20, 0), ea.sample_mistakes(inv, 20, 0)
    assert s1.groupby("culprit").size().to_dict() == {"a": 20, "b": 5} and s1.equals(s2)
