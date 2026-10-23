import numpy as np
import pandas as pd

from candidate_screener.evaluation import pool_scaling as ps


def test_rescale_minmaxes_the_raw_cosine_over_the_subpool_only():
    X = pd.DataFrame({"tfidf_scaled": [0.9, 0.5, 0.1], "role_match": [1.0, 0.2, 0.3]})
    out = ps.rescale(X, np.array([0.30, 0.20, 0.25]))
    assert np.allclose(out.tfidf_scaled, [1.0, 0.0, 0.5])
    assert out.role_match.tolist() == [1.0, 0.2, 0.3]          # every other column untouched


def test_rescale_of_a_constant_subpool_is_zero_not_nan():
    out = ps.rescale(pd.DataFrame({"tfidf_scaled": [0.4, 0.7]}), np.array([0.2, 0.2]))
    assert out.tfidf_scaled.tolist() == [0.0, 0.0]


def test_compare_metrics_on_a_known_swap():
    ids = np.array(["a", "b", "c", "d"])
    a = np.array([4.0, 3.0, 2.0, 1.0])
    b = np.array([3.0, 4.0, 2.0, 1.0])                         # top two swapped
    m = ps.compare(ids, a, b)
    assert m["top1_agree"] == 0.0
    assert m["topk_overlap"] == 1.0                            # k = 2: same set {a, b}
    assert np.isclose(m["kendall_tau"], 4 / 6)                 # 5 concordant, 1 discordant
    assert np.isclose(m["mean_abs_delta"], 0.5)


def test_relevant_draws_come_from_the_top_and_cannot_exceed_it():
    ids = np.array([f"c{i:03d}" for i in range(200)])
    scores = np.arange(200, dtype=float)                       # c199 best
    rng = np.random.default_rng(0)
    pos = ps.draws(ids, scores, 20, "relevant", rng)
    assert set(pos) <= set(range(200 - ps.TOP_RELEVANT, 200))
    assert ps.draws(ids, scores, ps.TOP_RELEVANT + 1, "relevant", rng) is None
