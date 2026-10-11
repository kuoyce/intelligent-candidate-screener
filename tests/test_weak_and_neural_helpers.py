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


class _FakeModel:
    """Stands in for a loaded CrossEncoder: a deterministic score, and a call counter."""
    max_seq_length = 384

    def __init__(self, offset: float = 0.0):
        self.calls, self.offset = 0, offset

    def predict(self, texts, batch_size, show_progress_bar):
        self.calls += 1
        return [len(j) * 0.01 + len(c) * 0.001 + self.offset for j, c in texts]


def _ce(model, weights):
    ce = cross_encoder.CrossEncoderWeak()
    ce.model, ce.weights_path = model, weights
    return ce


def test_cross_encoder_call_key_distinguishes_order_subset_weights_and_length():
    """W4 (plan 2026-10-08): only an identical call shares a key."""
    texts = [("jd a", "cv one"), ("jd b", "cv two"), ("jd a", "cv three")]
    key = cross_encoder.call_key("w1", 384, 32, texts)
    assert key == cross_encoder.call_key("w1", 384, 32, list(texts))
    others = [cross_encoder.call_key("w1", 384, 32, texts[::-1]),       # reordered
              cross_encoder.call_key("w1", 384, 32, texts[:2]),         # a subset
              cross_encoder.call_key("w2", 384, 32, texts),             # other weights
              cross_encoder.call_key("w1", 256, 32, texts),             # other length
              cross_encoder.call_key("w1", 384, 16, texts),             # other batch
              cross_encoder.call_key("w1", 384, 32, [("jd acv", " one"), *texts[1:]]),
              cross_encoder.call_key("w1", 384, 32, [("jd a", None), *texts[1:]])]
    assert len({key, *others}) == len(others) + 1


def test_cross_encoder_predict_replays_only_identical_calls(monkeypatch):
    monkeypatch.setattr(cross_encoder, "_PREDICTIONS", {})
    frame = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "jd_text": ["jd a", "jd bb", "jd a"],
                          "cv_text": ["cv one", "cv two", "cv three"]})
    model = _FakeModel()
    ce = _ce(model, "w1")
    ce.threshold = 0.01
    first = ce._predict(frame)
    np.testing.assert_array_equal(first, model.predict(
        list(zip(frame.jd_text, frame.cv_text)), 32, False))
    model.calls = 0
    first[:] = -1.0                                          # a caller mutating its copy
    again = ce._predict(frame)
    calls = ce.classify(frame)
    assert model.calls == 0                                  # score then classify: one predict
    assert (again > 0).all() and list(calls) == ["Fit", "Fit", "Fit"]
    ce._predict(frame.iloc[::-1])                            # reordered: predicted anew
    assert model.calls == 1
    other = _ce(_FakeModel(offset=1.0), "w2")                # other weights: never shared
    assert (other._predict(frame) > 1).all()


def test_cross_encoder_store_is_written_cold_and_read_only_warm(monkeypatch, tmp_path):
    """W7 (plan 2026-10-08, D55): cold re-infers and writes; warm replays the stored call."""
    monkeypatch.setattr(cross_encoder, "SCORE_STORE", tmp_path / "ce-scores")
    monkeypatch.setattr(cross_encoder, "_PERSISTENT", False)
    frame = pd.DataFrame({"pair_id": ["p1", "p2"], "jd_text": ["jd a", "jd bb"],
                          "cv_text": ["cv one", "cv two"]})

    def fresh(model, digest):
        monkeypatch.setattr(cross_encoder, "_PREDICTIONS", {})   # a new process
        ce = _ce(model, "same/path")
        ce.weights_digest = digest
        return ce

    cold = _FakeModel()
    first = fresh(cold, "d1")._predict(frame)
    assert cold.calls == 1 and len(list((tmp_path / "ce-scores").glob("*.npy"))) == 1
    again = _FakeModel(offset=5.0)                               # cold never reads the store
    assert (fresh(again, "d1")._predict(frame) > 5).all() and again.calls == 1
    cross_encoder.set_persistent(True)
    warm = _FakeModel(offset=9.0)
    replay = fresh(warm, "d1")._predict(frame)
    assert warm.calls == 0                                       # warm replays the last write
    np.testing.assert_array_equal(replay, first + 5.0)
    other = _FakeModel()                                         # other weight bytes: a miss
    fresh(other, "d2")._predict(frame)
    assert other.calls == 1
    undigested = _FakeModel()                                    # no content digest: no disk
    monkeypatch.setattr(cross_encoder, "_PREDICTIONS", {})
    _ce(undigested, "same/path")._predict(frame)
    assert undigested.calls == 1
    assert len(list((tmp_path / "ce-scores").glob("*.npy"))) == 2


def test_weights_digest_follows_file_content(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    (tmp_path / "config.json").write_text("{}")
    d = cross_encoder.weights_digest(tmp_path)
    assert d == cross_encoder.weights_digest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"abd")
    assert cross_encoder.weights_digest(tmp_path) != d
