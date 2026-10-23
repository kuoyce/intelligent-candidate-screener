import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from candidate_screener.baselines.tfidf import TFIDF_CONFIG
from candidate_screener.inference.model import L1Model
from candidate_screener.inference.parts import load_parts, save_parts
from candidate_screener.matching.entity_weighted import ConceptExtractor, DocumentFrequency
from candidate_screener.rerank.learners import BoundedRidge

COLUMNS = ["role_match", "tfidf_scaled", "exp_gap", "log_cv_concepts"]


def _fit():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(COLUMNS))), columns=COLUMNS)
    X["exp_gap"] = X.exp_gap * 3 + 1
    y = 30 + 8 * X.role_match - 5 * X.tfidf_scaled + 2 * X.exp_gap + rng.normal(size=60)
    return X, BoundedRidge(COLUMNS).fit(X, y)


def test_frozen_l1_predicts_as_the_ridge_it_came_from():
    X, ridge = _fit()
    model = L1Model.from_ridge(ridge)
    assert np.allclose(model.predict(X), ridge.predict(X), rtol=0, atol=1e-12)
    # column order of the frame must not matter
    assert np.allclose(model.predict(X[COLUMNS[::-1]]), ridge.predict(X), rtol=0, atol=1e-12)


def test_contributions_sum_to_score_minus_intercept():
    X, ridge = _fit()
    model = L1Model.from_ridge(ridge)
    parts = model.contributions(X)
    assert list(parts.columns) == COLUMNS and parts.index.equals(X.index)
    assert np.allclose(parts.sum(axis=1).to_numpy(), model.predict(X) - model.intercept, atol=1e-12)


def test_json_round_trip_is_exact(tmp_path):
    X, ridge = _fit()
    model = L1Model.from_ridge(ridge, {"plan": "p"})
    back = L1Model.load(model.save(tmp_path / "l1.json"))
    assert back == model and back.meta == {"plan": "p"}
    assert np.array_equal(back.predict(X), model.predict(X))
    assert json.loads((tmp_path / "l1.json").read_text())["columns"] == COLUMNS


def _parts():
    ex = ConceptExtractor(aliases={"python": "Python", "sql": "SQL", "java": "Java", "spark": "Spark"})
    docs = pd.DataFrame({"id": list("abcde"), "region": "train", "text": [
        "python and sql", "java", "python spark", "sql", "python"]})
    freq = DocumentFrequency(ex).fit(docs)
    vec = TfidfVectorizer(**{**TFIDF_CONFIG, "min_df": 1, "max_df": 1.0}).fit(
        ["python developer with sql", "java developer spring", "qa engineer manual testing python",
         "data engineer spark sql python"])
    return ex, freq, vec


def test_parts_round_trip(tmp_path):
    ex, freq, vec = _parts()
    fp = save_parts(tmp_path, freq, vec, {"role_model": "rm"})
    got = load_parts(root=tmp_path, extractor=ex)
    assert got.fingerprint == fp == (tmp_path / "LATEST").read_text().strip()
    for c in ("Python", "SQL", "Java", "Spark", "unknown:x", "Go"):
        assert got.frequencies.weight(c) == freq.weight(c)
    jd, cv = frozenset({"Python", "Spark", "Go"}), frozenset({"Python"})
    assert got.frequencies.coverage(jd, cv) == freq.coverage(jd, cv)
    texts = ["python sql developer", "an unseen text", "spark engineer"]
    assert np.array_equal(got.vectorizer.transform(texts).toarray(), vec.transform(texts).toarray())


def test_parts_refuse_edited_files(tmp_path):
    ex, freq, vec = _parts()
    fp = save_parts(tmp_path, freq, vec, {})
    np.save(tmp_path / fp / "tfidf_idf.npy", np.asarray(vec.idf_) * 2)
    with pytest.raises(ValueError, match="digest"):
        load_parts(root=tmp_path, extractor=ex)
