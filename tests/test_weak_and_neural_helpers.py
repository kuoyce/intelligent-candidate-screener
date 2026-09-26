"""Weak pairs and the model-free helpers of the torch rungs."""
import numpy as np
import pandas as pd

from candidate_screener.data import weak_pairs
from candidate_screener.matching import cross_encoder, sbert


def _docs():
    jd = pd.DataFrame({"id": ["j1", "j2", "j3"], "Primary Keyword": ["Java", "Java", "QA"],
                       "exp_band": ["2-3", "4-6", "0-1"], "jd_text": ["a", "b", "c"]})
    cv = pd.DataFrame({"id": [f"c{i}" for i in range(6)],
                       "Primary Keyword": ["Java", "Java", "Java", "QA", "QA", "PHP"],
                       "exp_band": ["0-1", "4-6", "7+", "0-1", "2-3", "4-6"],
                       "cv_text": list("uvwxyz")})
    return jd, cv


def test_weak_labels_follow_keyword_and_experience():
    jd, cv = _docs()
    pairs = weak_pairs.draw(jd, cv, n_jds=3, seed=0)
    kw = dict(zip(cv["id"], cv["Primary Keyword"])); band = dict(zip(cv["id"], cv.exp_band))
    jkw = dict(zip(jd["id"], jd["Primary Keyword"])); jband = dict(zip(jd["id"], jd.exp_band))
    order = {"0-1": 0, "2-3": 1, "4-6": 2, "7+": 3}
    for _, p in pairs.iterrows():
        same = kw[p.cv_id] == jkw[p.jd_id]
        meets = order[band[p.cv_id]] >= order[jband[p.jd_id]]
        assert p.label == int(same and meets)
        assert p.kind == ("positive" if same and meets else "hard_negative" if same else "easy_negative")


def test_the_validation_split_is_order_independent():
    ids = pd.Series([f"x{i}" for i in range(200)])
    assert weak_pairs.split_ids(ids, 0.1, 0) == weak_pairs.split_ids(ids.sample(frac=1, random_state=3), 0.1, 0)


def test_triplets_prefer_the_hard_negative():
    frame = pd.DataFrame({
        "jd_id": ["j1", "j1", "j1", "j2", "j2"], "kind": ["positive", "hard_negative", "easy_negative",
                                                          "positive", "easy_negative"],
        "jd_text": ["J1"] * 3 + ["J2"] * 2, "cv_text": ["pos1", "hard1", "easy1", "pos2", "easy2"]})
    t = sbert.weak_triplets(frame)
    assert t.to_dict("records") == [{"anchor": "J1", "positive": "pos1", "negative": "hard1"},
                                    {"anchor": "J2", "positive": "pos2", "negative": "easy2"}]


def test_chunks_cover_the_text():
    text = "word " * 500
    parts = sbert.chunks(text, 800)
    assert "".join(parts) == " ".join(text.split()) and all(len(p) <= 800 for p in parts)


def test_best_threshold_maximises_macro_f1():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.2, 0.4, 0.35, 0.8, 0.9])
    assert cross_encoder.best_threshold(y, p) in (0.35, 0.4, 0.8)
    t = cross_encoder.best_threshold(np.array([0, 1]), np.array([0.2, 0.7]))
    assert 0.2 < t <= 0.7
