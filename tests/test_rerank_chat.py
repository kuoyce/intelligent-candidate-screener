"""Plan 2026-10-18 step 3: chat-1 targets, folds, top-10 metrics, L1/L2, L3 windows. Synthetic only.

Each test names the silent bug it guards."""
import numpy as np
import pandas as pd
import pytest

from conftest import reads_final_labels

from candidate_screener.evaluation import rerank_chat as rc
from candidate_screener.rerank import chat_targets as ct, chunked, learners


# --- folds ----------------------------------------------------------------------------------

def test_folds_are_by_jd_stratified_and_seeded():
    """8 generic + 4 targeted JDs per fold, every JD in exactly one fold, the same draw twice.
    A pair-level split would let a model train on the JD it is scored on."""
    strata = pd.Series(["generic"] * 40 + ["targeted"] * 20,
                       index=[f"j{i:02d}" for i in range(60)])
    f = ct.folds(strata)
    assert f.index.tolist() == sorted(strata.index) and f.equals(ct.folds(strata))
    counts = pd.crosstab(f, strata.loc[f.index])
    assert (counts["generic"] == 8).all() and (counts["targeted"] == 4).all()
    assert not f.equals(ct.folds(strata, seed=1))


# --- top-10 metrics -------------------------------------------------------------------------

def test_chat_weights_share_the_tenth_place_among_ties():
    """Three CVs tied at the 10th chat score for two slots count 2/3 each, so the weights sum to
    10. Counting every tie as 1 would let a model score above 1."""
    chat = np.array([90] * 8 + [50, 50, 50] + [10] * 4, dtype=float)
    w = ct.chat_weights(chat)
    assert w[:8].tolist() == [1.0] * 8 and w[8:11] == pytest.approx([2 / 3] * 3)
    assert w.sum() == pytest.approx(10) and (w[11:] == 0).all()


def test_recall_counts_order_inside_the_top_10_as_irrelevant():
    ids = np.array([f"p{i:02d}" for i in range(20)])
    chat = np.arange(20, 0, -1, dtype=float)                    # p00 best
    perfect = chat.copy()
    shuffled_inside = np.r_[np.arange(10)[::-1] + 100, np.zeros(10)]
    one_out = perfect.copy()
    one_out[9], one_out[10] = 0, 100                            # p10 replaces p09
    assert ct.recall_at_k(ids, chat, perfect) == 1.0
    assert ct.recall_at_k(ids, chat, shuffled_inside) == 1.0
    assert ct.recall_at_k(ids, chat, one_out) == pytest.approx(0.9)


def test_a_short_returned_list_counts_its_misses():
    """A stage-1 list of 4 CVs can hold at most 4 of the chat's top 10: recall <= 0.4, not 1."""
    ids = np.array([f"p{i:02d}" for i in range(20)])
    chat = np.arange(20, 0, -1, dtype=float)
    returned = np.zeros(20, bool)
    returned[:4] = True
    assert ct.recall_at_k(ids, chat, chat, returned) == pytest.approx(0.4)
    assert ct.no_fits_at_k(ids, np.array(["Good Fit"] * 4 + ["No Fit"] * 16), chat, returned) == 0
    per = ct.per_jd(pd.DataFrame({"pair_id": ids, "jd_id": "j", "score": chat,
                                  "label": ["No Fit"] * 20}),
                    pd.Series(chat, index=ids), pd.Series(returned, index=ids))
    assert per.loc["j", "returned"] == 4 and per.loc["j", "no_fits"] == 4


def test_ndcg_at_10_uses_the_chat_score_as_gain_and_is_one_for_the_chat_order():
    ids = np.array([f"p{i:02d}" for i in range(12)])
    chat = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0], dtype=float)
    assert ct.ndcg_at_k(ids, chat, chat) == pytest.approx(1.0)
    swap = chat.copy()
    swap[[0, 9]] = swap[[9, 0]]                                 # best CV to 10th place
    assert ct.ndcg_at_k(ids, chat, swap) < 0.9


# --- learners -------------------------------------------------------------------------------

def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["role_match", "log_jd_concepts", "sbert"],
                     index=[f"p{i}" for i in range(n)])
    y = 10 * X.role_match - 5 * X.log_jd_concepts - 3 * X.sbert + rng.normal(size=n)
    return X, y


def test_bounded_ridge_holds_match_signals_non_negative_and_frees_descriptors():
    """`sbert` truly has a negative effect here: the bound must hold it at 0, while the free
    `log_jd_concepts` keeps its negative sign."""
    X, y = _xy()
    c = learners.BoundedRidge(X.columns).fit(X, y).coefficients()
    assert c["sbert"] == pytest.approx(0, abs=1e-9) and c["role_match"] > 0
    assert c["log_jd_concepts"] < 0


def test_bounded_ridge_is_sklearn_ridge_when_no_bound_binds():
    from sklearn.linear_model import Ridge
    X, y = _xy()
    y = y + 6 * X.sbert                                          # every match effect positive
    m = learners.BoundedRidge(X.columns).fit(X, y)
    Z = (X - X.mean()) / X.std(ddof=0)
    assert m.coef == pytest.approx(Ridge(alpha=learners.L1_ALPHA).fit(Z, y).coef_, rel=1e-6)


def test_boosting_is_monotone_in_a_constrained_feature():
    X, y = _xy()
    y = y - 8 * X.role_match ** 2                                # a falling region to resist
    m = learners.MonotoneBoosting(X.columns).fit(X, y)
    grid = pd.DataFrame({"role_match": np.linspace(-3, 3, 50), "log_jd_concepts": 0.0, "sbert": 0.0})
    assert (np.diff(m.predict(grid)) >= -1e-12).all()


def test_out_of_fold_never_scores_a_pair_with_a_model_that_saw_its_jd():
    """Each fold's model fits only the other folds' training rows; extension rows never train."""
    chat = pd.DataFrame({"pair_id": [f"p{i}" for i in range(8)], "fold": [0, 0, 1, 1, 0, 0, 1, 1],
                         "score": [10, 20, 30, 40, 50, 60, 70, 80],
                         "source": ["x2_top100"] * 6 + ["s1_passed"] * 2})
    X = pd.DataFrame({"role_match": np.arange(8.0)}, index=chat.pair_id)
    seen = []

    class Spy(learners.BoundedRidge):
        def fit(self, X, y):
            seen.append(sorted(X.index))
            return super().fit(X, y)
    learners.LEARNERS["spy"] = Spy
    try:
        s, _ = learners.out_of_fold("spy", X, chat, ["role_match"])
    finally:
        del learners.LEARNERS["spy"]
    assert seen == [["p2", "p3"], ["p0", "p1", "p4", "p5"]]
    assert s.notna().all() and list(s.index) == list(chat.pair_id)


# --- L3 windows -----------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(0, [0]), (317, [0]), (318, [0, 256]), (900, [0, 256, 512, 768])])
def test_windows_cover_every_cv_token(n, expected):
    starts = chunked.window_starts(n)
    assert starts == expected
    covered = set()
    for s in starts:
        covered |= set(range(s, min(n, s + chunked.CONFIG["window"])))
    assert covered == set(range(n))


def test_sequences_cut_the_jd_and_fit_512_tokens():
    seqs = chunked.sequences(list(range(1000, 1400)), list(range(2000, 2700)), cls=101, sep=102)
    assert len(seqs) == len(chunked.window_starts(700))
    for ids, tt in seqs:
        assert len(ids) <= chunked.CONFIG["max_length"] and len(ids) == len(tt)
        assert ids[0] == 101 and ids[1:193] == list(range(1000, 1192)) and ids[193] == 102
        assert tt[:194] == [0] * 194 and set(tt[194:]) == {1}
    assert seqs[-1][0][-2] == 2699                              # the CV's last token is read


# --- the evaluation's pure pieces -----------------------------------------------------------

def test_the_bar_needs_both_the_point_estimate_and_the_ci():
    def d(v, lo):
        return {"recall": {"total": {"value": v, "ci": [lo, v + 0.1]}}}
    assert rc.passes(d(0.05, 0.001)) and not rc.passes(d(0.05, 0.0)) and not rc.passes(d(0.049, 0.01))


def test_best_breaks_ties_to_the_cheaper_candidate():
    order = ["S1", "L1", "L3"]
    assert rc.pick_best({"S1": 0.5, "L1": 0.5000001, "L3": 0.6}, order) == "L3"
    assert rc.pick_best({"S1": 0.6, "L1": 0.6000001, "L3": 0.6}, order) == "S1"


def test_stage2_training_modules_never_read_final_labels_or_judgements():
    """D46 / D61: the code that fits on chat-1 never names the final labels or the judgements."""
    for module in (ct, chunked, learners):
        assert reads_final_labels(module) == [], module.__name__
