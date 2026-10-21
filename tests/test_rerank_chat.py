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


def test_step_3_stays_on_chat_1_and_step_4_takes_the_latest_score(tmp_path, monkeypatch):
    """An anchor re-scored in chat-2 must change step 4's reference (latest wins, owner 18 Oct
    2026) and must not change chat-1, on which step 3 was trained and scored."""
    from candidate_screener.rerank import chat_rerank as cr
    plan = pd.DataFrame({"jd_id": "j", "cv_id": ["a", "b", "a", "n"], "pair_id": ["j__a", "j__b", "j__a", "j__n"],
                         "stratum": "generic", "short_id": ["C001", "C002", "F001", "F002"],
                         "batch": [1, 1, 101, 101], "jd_order": 1,
                         "source": ["x2_top100", "x2_top100", "anchor", "s1_t030"],
                         "run": ["chat-1", "chat-1", "chat-2", "chat-2"]})
    replies = pd.DataFrame({"jd_id": "j", "short_id": ["C001", "C002", "F001", "F002"],
                            "pair_id": ["j__a", "j__b", "j__a", "j__n"], "batch": [1, 1, 101, 101],
                            "label": ["Good Fit", "No Fit", "Potential Fit", "No Fit"], "score": [80, 10, 55, 5],
                            "run": ["chat-1", "chat-1", "chat-2", "chat-2"], "instrument_sha256": "x",
                            "saved_at": ["2026-10-17T10:00:00", "2026-10-17T10:00:00",
                                         "2026-10-18T10:00:00", "2026-10-18T10:00:00"]})
    plan.to_csv(tmp_path / "plan.csv", index=False)
    replies.to_csv(tmp_path / "replies.csv", index=False)
    monkeypatch.setattr(cr, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(cr, "REPLIES", tmp_path / "replies.csv")
    monkeypatch.setattr(cr, "latest", lambda path=tmp_path / "replies.csv": pd.read_csv(path))
    one = ct.load()
    assert one.set_index("pair_id").score.to_dict() == {"j__a": 80, "j__b": 10}
    both = ct.load_latest().set_index("pair_id")
    assert both.score.to_dict() == {"j__a": 55, "j__b": 10, "j__n": 5}
    assert both.loc["j__a", "source"] == "x2_top100" and both.loc["j__n", "source"] == "s1_t030"
    assert ct.anchor_pairs().set_index("pair_id")[["first", "again"]].loc["j__a"].tolist() == [80, 55]


def test_ceiling_is_the_recall_of_a_perfect_stage_2():
    """The ceiling counts only what stage 1 hands over: half the chat's top 10 handed over is 0.5,
    whatever order stage 2 would give it."""
    from candidate_screener.evaluation import stage1_threshold as st
    chat = np.arange(20, 0, -1, dtype=float)
    returned = np.zeros(20, bool)
    returned[[0, 2, 4, 6, 8, 11, 13]] = True                    # 5 of the top 10, 2 below it
    assert st.ceiling(np.arange(20), chat, returned) == pytest.approx(0.5)
    ids = np.array([f"p{i:02d}" for i in range(20)])
    assert ct.recall_at_k(ids, chat, chat, returned) == pytest.approx(0.5)


def test_drift_reports_the_offset_and_agreement_per_jd():
    from candidate_screener.evaluation import stage1_threshold as st
    a = pd.DataFrame({"pair_id": [f"p{i}" for i in range(6)], "jd_id": ["a"] * 3 + ["b"] * 3,
                      "first": [10, 50, 90, 20, 40, 60], "again": [15, 55, 95, 10, 30, 50],
                      "first_label": ["No Fit"] * 6, "again_label": ["No Fit"] * 5 + ["Good Fit"]})
    d = st.drift(a, pd.Series({"a": "generic", "b": "targeted"}))
    assert d["offset"]["generic"]["value"] == 5 and d["offset"]["targeted"]["value"] == -10
    assert d["offset"]["total"]["value"] == -2.5 and d["spearman"]["total"]["value"] == 1.0
    assert d["label_agreement"]["targeted"]["value"] == pytest.approx(2 / 3)


def test_feature_adoption_follows_the_plans_rule():
    """F9 survives a loss up to 0.005; a change needs a positive delta AND a CI above 0."""
    from candidate_screener.evaluation import feature_arms as fa
    d = lambda v, lo: {"value": v, "ci": [lo, v + 0.05]}
    out = fa.adopt({"A2": d(0.02, 0.001), "A3": d(0.03, -0.01), "A5": d(-0.01, -0.03)}, d(-0.004, -0.02))
    assert out == {"base": "F9", "changes": ["A2"]}
    assert fa.adopt({}, d(-0.006, -0.02))["base"] == "F12"


def test_d50_folds_hold_each_jd_once_and_are_balanced():
    from candidate_screener.evaluation import feature_arms as fa
    f = fa.jd_folds([f"j{i:03d}" for i in range(100)] * 10)
    assert len(f) == 100 and f.value_counts().eq(20).all() and f.equals(fa.jd_folds(list(f.index)))


def test_top10_digest_ignores_last_bit_noise_but_not_a_changed_top10():
    # D55 cold run, 21 Oct 2026: re-scoring R1 / R2 moved their floats in the last bits only
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"pair_id": [f"j{j}-c{c:02d}" for j in range(3) for c in range(30)],
                          "jd_id": [f"j{j}" for j in range(3) for _ in range(30)]})
    s = pd.Series(rng.normal(size=90), index=frame.pair_id.to_numpy())
    noisy = s * (1 + 1e-7 * rng.normal(size=90))
    assert rc._sha(s) != rc._sha(noisy)
    assert rc._top10_sha(frame, s) == rc._top10_sha(frame, noisy)
    g = s.loc[[p for p in s.index if p.startswith("j0-")]].sort_values(ascending=False)
    swapped = s.copy()
    swapped[g.index[9]], swapped[g.index[10]] = g.iloc[10], g.iloc[9]   # ranks 10 and 11 trade places
    assert rc._top10_sha(frame, swapped) != rc._top10_sha(frame, s)
