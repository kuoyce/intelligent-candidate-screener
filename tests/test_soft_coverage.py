"""Soft-matched coverage columns (plan 2026-10-13-soft-matching-in-the-score): synthetic only."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import soft_coverage as sc


class Ex:
    def __init__(self, by_text): self.by_text = by_text
    def concepts(self, text, side="cv"): return frozenset(self.by_text[text])


class Freq:
    def weight(self, c): return {"a": 1.0, "b": 3.0}.get(c, 1.0)


VEC = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
       "a2": np.array([0.8, 0.6]), "b2": np.array([0.0, 1.0])}        # cos(a, a2) = 0.8


def pairs():
    return pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "jd_text": ["J", "J", "J"], "cv_text": ["C1", "C2", "C3"]})


def sims():
    ex = Ex({"J": {"a", "b"}, "C1": {"a2", "b2"}, "C2": {"a2"}, "C3": set()})
    return sc.SoftSims(pairs(), ex, Freq(), VEC)


def test_columns_count_a_concept_when_its_best_similarity_reaches_the_threshold():
    s = sims()
    c = s.columns(0.8)
    assert c.loc["p1"].tolist() == pytest.approx([1.0, 1.0, np.log1p(2)])          # a~a2 (0.8), b~b2 (1.0)
    assert c.loc["p2"].tolist() == pytest.approx([1.0 / 4.0, 0.5, np.log1p(1)])    # only a matches; weight 1 of 1+3
    assert c.loc["p3"].tolist() == [0.0, 0.0, 0.0]                                 # CV with no concepts


def test_a_higher_threshold_can_only_remove_matches():
    s = sims()
    assert s.columns(0.9).loc["p1"].tolist() == pytest.approx([3.0 / 4.0, 0.5, np.log1p(1)])
    assert (s.columns(0.9).to_numpy() <= s.columns(0.8).to_numpy() + 1e-12).all()


def test_replace_columns_touches_only_the_three_coverage_columns():
    X = pd.DataFrame({"role_match": [0.1, 0.2, 0.3], "coverage": [9, 9, 9], "coverage_unweighted": [9, 9, 9],
                      "log_matched": [9, 9, 9]}, index=["p1", "p2", "p3"])
    out = sc.replace_columns(X, sims().columns(0.8))
    assert out.role_match.tolist() == [0.1, 0.2, 0.3] and X.coverage.tolist() == [9, 9, 9]
    assert out.coverage.tolist() == pytest.approx([1.0, 0.25, 0.0])
