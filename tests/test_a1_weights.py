"""`evaluation/a1_weights.py`: the pieces that carry the learned-weights notebooks (18-20).

Synthetic inputs only; the data-dependent loaders (`d50_pairs`, `llm_notes`, `build_scorer`)
are exercised by the notebooks. Each test names the silent bug it guards.
"""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import a1_weights as aw


def _frame(n=60, seed=0):
    """Two JDs' worth of pairs; the first feature separates Fit from No Fit."""
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n), index=[f"p{i}" for i in range(n)])
    X = pd.DataFrame(rng.normal(size=(n, len(aw.ALL_FEATURES))), index=y.index, columns=list(aw.ALL_FEATURES))
    X["role_match"] = y.to_numpy() * 2.0 + rng.normal(scale=0.3, size=n)
    jd = pd.Series(np.where(np.arange(n) < n // 2, "jdA", "jdB"), index=y.index)
    return X, y, jd


def test_fit_weights_is_indexed_by_feature_and_finds_the_separating_feature():
    X, y, _ = _frame()
    w = aw.fit_weights(X, y)
    assert list(w.index) == list(aw.FEATURES)
    assert w["role_match"] == w.max() and w["role_match"] > 0


def test_linear_score_is_the_dot_product_in_feature_order():
    X, _, _ = _frame(6)
    s = aw.linear_score(X, aw.HAND_WEIGHTS)
    expected = X[list(aw.FEATURES)].to_numpy() @ aw.HAND_WEIGHTS
    assert np.allclose(s.to_numpy(), expected) and list(s.index) == list(X.index)


def test_mean_jd_auc_skips_jds_with_one_class():
    s = pd.Series([3.0, 1.0, 2.0, 5.0, 4.0], index=list("abcde"))
    y = pd.Series([1, 0, 1, 1, 1], index=s.index)             # jd2 has no No Fit
    jd = pd.Series(["jd1", "jd1", "jd1", "jd2", "jd2"], index=s.index)
    auc, n = aw.mean_jd_auc(s, y, jd)
    assert n == 1 and auc == 1.0                              # jd1: both Fits outrank the No Fit


def test_pair_inversions_counts_wrong_order_and_half_for_ties():
    s = pd.Series([3.0, 1.0, 2.0, 2.0], index=["f1", "f2", "n1", "n2"])
    y = pd.Series([1, 1, 0, 0], index=s.index)
    jd = pd.Series("jd", index=s.index)
    inv = aw.pair_inversions(s, y, jd).set_index(["fit_id", "nofit_id"]).inv
    assert inv[("f1", "n1")] == 0 and inv[("f2", "n1")] == 1
    s.loc[["n1", "n2"]] = [3.0, 0.0]
    inv = aw.pair_inversions(s, y, jd).set_index(["fit_id", "nofit_id"]).inv
    assert inv[("f1", "n1")] == 0.5                           # a tie is half an inversion
    assert inv.mean() == pytest.approx(1 - aw.mean_jd_auc(s, y, jd)[0])


def test_pair_inversions_never_crosses_jds():
    s = pd.Series([1.0, 2.0], index=["f", "n"])
    y = pd.Series([1, 0], index=s.index)
    assert aw.pair_inversions(s, y, pd.Series(["a", "b"], index=s.index)).empty   # no JD has both classes


def test_llm_labels_refuses_pooled_runs(tmp_path, monkeypatch):
    path = tmp_path / "llm.csv"
    pd.DataFrame({"pair_id": ["a", "a"], "run": [aw.LLM_RUN, aw.LLM_RUN], "parsed_ok": [True, True],
                  "label": ["Fit", "No Fit"]}).to_csv(path, index=False)
    monkeypatch.setattr(aw, "LLM_FILE", path)
    with pytest.raises(ValueError, match="pooled"):
        aw.llm_labels()


def test_llm_labels_keeps_only_the_chosen_run(tmp_path, monkeypatch):
    path = tmp_path / "llm.csv"
    pd.DataFrame({"pair_id": ["a", "b", "a"], "run": [aw.LLM_RUN, aw.LLM_RUN, aw.LLM_RUN + 1],
                  "parsed_ok": [True, True, True],
                  "label": ["Good Fit", "No Fit", "No Fit"]}).to_csv(path, index=False)
    monkeypatch.setattr(aw, "LLM_FILE", path)
    assert aw.llm_labels().to_dict() == {"a": 1, "b": 0}


@pytest.mark.parametrize("kind", ["logit", "gbm"])
def test_fit_learner_scores_unseen_rows_on_the_same_index(kind):
    X, y, _ = _frame()
    f = aw.fit_learner(kind, aw.ALL_FEATURES, X.iloc[:40], y)
    s = f(X.iloc[40:])
    assert list(s.index) == list(X.index[40:]) and s[y.iloc[40:] == 1].mean() > s[y.iloc[40:] == 0].mean()


def test_learning_curve_trains_only_on_the_requested_number_of_jds():
    X, y, jd = _frame(240)
    jd = pd.Series([f"jd{i % 12}" for i in range(240)], index=X.index)
    curve = aw.learning_curve(X, y, jd, X, y, jd, sizes=(3, 6), reps=2, n_test_jds=4)
    assert set(curve["train JDs"]) == {3, 6} and set(curve.learner) == set(aw.LEARNERS)
    assert curve[["A2 AUC", "held-out AUC"]].notna().all().all()
