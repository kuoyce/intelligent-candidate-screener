"""Step 5a: the embedder's query text and the pilot's sample, fusion and metrics (no data, no model)."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import rerank_embed as re_
from candidate_screener.rerank import embedder


def test_query_text_variants():
    jd = "We need a Python developer."
    assert embedder.query_text(jd, "I0", "Backend Engineer") == jd
    q1 = embedder.query_text(jd, "I1", "Backend Engineer")
    assert q1 == f"Instruct: {embedder.TASK_FIT}\nQuery:{jd}"
    assert "Backend Engineer" not in embedder.query_text(jd, "I2", "Backend Engineer")
    q3 = embedder.query_text(jd, "I3", "Backend Engineer")
    assert q3 == f"Instruct: {embedder.TASK_CRITERIA}\nQuery:Job title: Backend Engineer\n{jd}"
    assert embedder.query_text(jd, "I3", None).endswith(f"Query:{jd}")


def test_cosine_matches_definition():
    a = np.array([[1.0, 0.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [4.0, 3.0]])
    assert np.allclose(embedder.cosine(a, b), [1.0, 24 / 25])


def _chat(n_jd=4, n_cv=15):
    rows = []
    for j in range(n_jd):
        for c in range(n_cv):
            rows.append({"pair_id": f"p{j:02d}-{c:02d}", "jd_id": f"jd{j}", "cv_id": f"cv{j}-{c}",
                         "fold": j % 2, "score": 100 - 5 * c})
    return pd.DataFrame(rows)


def test_pilot_sample_keeps_the_chat_top10_and_n_other():
    chat = _chat()
    s = re_.pilot_sample(chat, fold=0, n_other=3, seed=0)
    assert set(s.jd_id) == {"jd0", "jd2"}
    for _, g in s.groupby("jd_id"):
        assert g.top.sum() == 10 and (~g.top).sum() == 3
        assert g[g.top].score.min() > g[~g.top].score.max()
    pd.testing.assert_frame_equal(s, re_.pilot_sample(chat, fold=0, n_other=3, seed=0))


def test_pilot_sample_counts_ties_at_the_10th_score_as_top():
    chat = _chat(n_jd=2)
    chat.loc[chat.pair_id.isin(["p00-09", "p00-10", "p00-11"]), "score"] = 55
    s = re_.pilot_sample(chat, fold=0, n_other=2, seed=0)
    assert s[s.top].pair_id.str.startswith("p00").sum() == 12


def test_minmax_fuse_scales_within_jd():
    sample = pd.DataFrame({"pair_id": ["a", "b", "c", "d"], "jd_id": ["x", "x", "y", "y"]})
    a = pd.Series([0.0, 10.0, 5.0, 5.0], index=sample.pair_id)
    b = pd.Series([1.0, 3.0, 100.0, 200.0], index=sample.pair_id)
    fused = re_.minmax_fuse(sample, a, b)
    assert np.allclose(fused.to_numpy(), [0.0, 2.0, 0.0, 1.0])   # a constant JD scales to 0


def test_per_jd_auc_and_best_variant():
    sample = pd.DataFrame({"pair_id": list("abcdef"), "jd_id": list("xxxyyy"),
                           "top": [True, False, False, True, True, False], "score": [90, 10, 20, 80, 70, 0]})
    perfect = pd.Series([3, 1, 2, 3, 3, 1], index=sample.pair_id, dtype=float)
    assert re_.per_jd_auc(sample, perfect).tolist() == [1.0, 1.0]
    assert re_.best_variant({"I0": 0.7, "I1": 0.8, "I2": 0.8}) == "I1"
    with pytest.raises(KeyError):
        embedder.query_text("jd", "I9")


def test_coverage_is_the_share_of_the_chat_top10_in_the_system_top_k():
    chat = _chat(n_jd=1, n_cv=20)                       # chat's top 10 = cv 0..9
    system = pd.Series(np.arange(20, dtype=float), index=chat.pair_id)   # reversed: best = cv 19
    assert re_.coverage(chat, system, 10).iloc[0] == 0.0
    assert re_.coverage(chat, system, 15).iloc[0] == 0.5
    assert re_.coverage(chat, system, 20).iloc[0] == 1.0
    assert re_.coverage(chat, -system, 10).iloc[0] == 1.0


def test_cascade_reorders_only_the_first_stage_top_k():
    chat = _chat(n_jd=1, n_cv=6)
    ids = chat.pair_id.to_numpy()
    first = pd.Series([6, 5, 4, 3, 2, 1], index=ids, dtype=float)          # top 3 = ids[0:3]
    second = pd.Series({ids[0]: 0.0, ids[1]: 1.0, ids[2]: 2.0})             # only the top 3 scored
    fused, alone = re_.cascade_scores(chat, first, second, k=3)
    assert list(alone.sort_values(ascending=False).index[:3]) == [ids[2], ids[1], ids[0]]
    assert (fused.loc[ids[:3]] >= 10).all() and (fused.loc[ids[3:]] <= 1).all()
    assert list(fused.loc[ids[3:]].sort_values(ascending=False).index) == list(ids[3:])
    assert np.allclose(fused.loc[ids[:3]].to_numpy(), [11.0, 11.0, 11.0])  # min-max sums tie here


def test_capped_list_applies_threshold_then_cap():
    chat = _chat(n_jd=2, n_cv=6)
    p = pd.Series([0.9, 0.8, 0.7, 0.35, 0.2, 0.1] * 2, index=chat.pair_id.to_numpy())
    got = re_.capped_list(chat, p, t=0.3, cap=3)
    assert got == {"p00-00", "p00-01", "p00-02", "p01-00", "p01-01", "p01-02"}
    assert re_.capped_list(chat, p, t=0.3, cap=30) == got | {"p00-03", "p01-03"}
