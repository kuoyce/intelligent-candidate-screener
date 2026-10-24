"""Plan 2026-10-18 steps 1-2: the S1 classifier and the S1 / S2 measures. Synthetic data only.

Each test names the silent bug it guards."""
import numpy as np
import pandas as pd
import pytest

from conftest import reads_final_labels

from candidate_screener.evaluation import stage1_classifier as sc
from candidate_screener.matching import learned


def _pools():
    """Two JDs of four CVs; pair ids sort in row order within a JD."""
    return pd.DataFrame({"jd_id": ["a"] * 4 + ["b"] * 4,
                         "pair_id": [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]})


def test_ranks_are_within_jd_with_ties_broken_by_pair_id():
    """A rank over the whole frame (not per JD) or an unstable tie order would move CVs in and
    out of a top k between runs."""
    f = _pools()
    r = sc.ranks(f, [0.5, 0.9, 0.5, 0.1, 3.0, 2.0, 1.0, 0.0])
    assert r.tolist() == [1, 0, 2, 3, 0, 1, 2, 3]
    assert sc.top_k(f, [0.5, 0.9, 0.5, 0.1, 3.0, 2.0, 1.0, 0.0], {"a": 1, "b": 3}).tolist() == [
        False, True, False, False, True, True, True, False]


def test_x2_is_cut_at_s1s_size_per_jd():
    """Size matching: X2 keeps as many CVs per JD as S1 passed there, not a fixed k. With a
    fixed k = 2 X2 would keep both of JD a's Fits and the delta would flip sign."""
    f = _pools()
    passed = [True, False, False, False, True, True, True, False]      # sizes 1 and 3
    x2 = [0.9, 0.8, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6]
    rel = [0, 1, 1, 0, 1, 0, 0, 1]
    d = sc.size_matched(f, passed, x2, rel)
    assert d.n_pass.to_dict() == {"a": 1, "b": 3}
    assert d.s1.to_dict() == {"a": 0, "b": 1} and d.x2.to_dict() == {"a": 0, "b": 1}
    assert d.recall_s1.tolist() == [0.0, 0.5] and d.recall_x2.tolist() == [0.0, 0.5]


def test_an_empty_passed_set_scores_zero_for_both_and_no_fit_jd_is_left_out():
    """An empty passed set must not let X2 keep anything; a JD without a Fit has no recall."""
    f = _pools()
    d = sc.size_matched(f, [False] * 8, [4, 3, 2, 1, 1, 2, 3, 4], [1, 0, 0, 0, 0, 0, 0, 0])
    assert d.loc["a", "recall_s1"] == 0 and d.loc["a", "recall_x2"] == 0
    assert np.isnan(d.loc["b", "recall_s1"]) and np.isnan(d.loc["b", "recall_x2"])


def test_overlap_is_the_shortlist_share_inside_the_reference_and_nan_when_empty():
    """Dividing by the reference size (100) instead of the shortlist would hide a shortlist that
    lies mostly outside X2's top 100; an empty shortlist must not count as overlap 0 or 1."""
    f = _pools()
    o = sc.overlap(f, [True, True, True, False, False, False, False, False],
                   [True, False, False, True, True, True, False, False])
    assert o.loc["a", ["size", "inside", "outside"]].tolist() == [3, 1, 2]
    assert o.loc["a", "overlap"] == pytest.approx(1 / 3)
    assert o.loc["b", "size"] == 0 and np.isnan(o.loc["b", "overlap"])


def test_open_question_6_switches_to_the_top_100_only_past_125():
    """Past 125 a passed set cannot be 80% inside a top 100, whatever it contains."""
    assert sc.s2_shortlist(125) == sc.PASSED
    assert sc.s2_shortlist(125.5) == sc.TOP100


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(learned.F12))), columns=list(learned.F12),
                     index=[f"p{i}" for i in range(n)])
    y = pd.Series((X.role_match + 0.3 * rng.normal(size=n) > 0).astype(int), index=X.index)
    X["role_match"] *= 3.0                                     # an unequal scale to standardise
    return X, y


def test_classifier_fits_its_own_columns_and_standardises_coefficients():
    """The base-4 arm must not see F12's extra columns; a standardised coefficient is the raw
    one times the training SD, so a feature on a large scale does not look weak."""
    X, y = _synthetic()
    m = learned.LearnedStage1("base 4", features=object()).fit_features(X, y)
    c = m.coefficients()
    assert list(c["raw"]) == list(learned.BASE4)
    sd = X["role_match"].std(ddof=0)
    assert c["standardised"]["role_match"] == pytest.approx(c["raw"]["role_match"] * sd)
    assert max(c["standardised"], key=lambda k: abs(c["standardised"][k])) == "role_match"
    p = m.proba(X)
    assert (m.passes(X) == (p >= learned.THRESHOLD)).all()
    assert m.passes(X).loc[y.index[y == 1]].mean() > 0.8


def test_classifier_refuses_any_training_frame_but_d50():
    with pytest.raises(ValueError, match="D50"):
        learned.LearnedStage1("F12", features=object()).fit(pd.DataFrame({"x": [1]}))


def test_learned_module_never_reads_final_labels_or_judgements():
    """D46: the module that fits S1 never names the final labels or the judgements."""
    assert reads_final_labels(learned) == []


def test_the_chat_guard_compares_x2s_top_100_only(monkeypatch):
    """The S1 extension adds chat-scored pairs outside X2's top 100; counting them made the
    'X2's top 100 is what chat-1 scored' guard fail (18 Oct 2026)."""
    from candidate_screener.rerank import chat_targets
    monkeypatch.setattr(chat_targets, "load", lambda: pd.DataFrame({
        "jd_id": ["a", "a", "a"], "pair_id": ["a__1", "a__2", "a__9"],
        "source": ["x2_top100", "x2_top100", "s1_passed"]}))
    assert sc._chat_pairs() == {"a": {"a__1", "a__2"}}
