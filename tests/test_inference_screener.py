"""The inference API's invariants (plan 2026-10-23-inference-api), on injected synthetic parts."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from candidate_screener.extraction import experience
from candidate_screener.inference import CVProfile, JDProfile, L1Model, Screener
from candidate_screener.matching.learned import F12
from candidate_screener.matching.role_filter import per_jd_minmax

VOCAB = ("python", "sql", "docker", "java")
JD = "Python developer with SQL and Docker, 2+ years"


def featurize(calls):
    def f(pairs, reference):
        calls.append((len(pairs), reference))
        rows = []
        for jd, cv in zip(pairs.jd_text, pairs.cv_text):
            j, c = set(jd.lower().split()), set(cv.lower().split())
            overlap = len(j & c) / max(len(j), 1)
            rows.append({"role_match": 0.9 if "python" in c else 0.1, "experience_compatible": 1.0,
                         "raw": overlap, "coverage": overlap, "coverage_unweighted": overlap,
                         "log_matched": np.log1p(len(j & c)), "log_jd_concepts": np.log1p(len(j)),
                         "log_cv_concepts": np.log1p(len(c)), "skill_overlap": overlap,
                         "exp_gap": 0.0, "exp_known": 0.0, "sbert": (len(cv) % 7) / 7})
        X = pd.DataFrame(rows, index=pairs.pair_id.to_numpy())
        X["tfidf_scaled"] = per_jd_minmax(pairs.jd_id.to_numpy(), X.raw.to_numpy())
        return X[list(F12)]
    return f


def extract_jds(texts, titles):
    return [JDProfile(None, t, "Python", 0.9, (("Python", 0.9), ("Java", 0.1)),
                      tuple(w for w in VOCAB if w in x.lower().split()), (), 2.0, None, None)
            for x, t in zip(texts, titles)]


def extract_cvs(texts, reference):
    out = []
    for x in texts:
        e = experience.cv_experience(x, reference)
        role = "Python" if "python" in x.lower().split() else "Java"
        out.append(CVProfile(None, role, 0.8, ((role, 0.8),), tuple(w for w in VOCAB if w in x.lower().split()),
                             (), e.years, e.source, None, None))
    return out


L1 = L1Model(tuple(F12), (0.0,) * len(F12), (1.0,) * len(F12),
             tuple(float(i + 1) for i in range(len(F12))), 10.0, 1.0)
BACKGROUND = pd.DataFrame({"id": [f"b{i}" for i in range(20)],
                           "cv_text": [" ".join(["python"] * (i % 3) + ["sql"] * (i % 4) + ["word", str(i)])
                                       for i in range(20)]})


def screener(calls=None, c2=None, c2_k=50):
    return Screener(featurize([] if calls is None else calls), L1, extract_jds, extract_cvs,
                    c2=c2, background=lambda: BACKGROUND, c2_k=c2_k, c2_w=0.25)


def cvs(n=6):
    texts = ["python sql docker developer", "java developer", "python only", "sql docker ops",
             "python sql", "docker", "python docker sql java", "nothing relevant"]
    return pd.DataFrame({"cv_id": [f"c{i}" for i in range(n)], "cv_text": texts[:n]})


@pytest.mark.parametrize("column", ["position", "final_label", "primary_keyword"])
def test_rank_refuses_any_cv_column_beyond_id_and_text(column):
    frame = cvs().assign(**{column: "x"})
    with pytest.raises(ValueError, match="D53"):
        screener().rank(JD, frame)


def test_identical_cvs_tie_and_break_by_cv_id():
    frame = pd.DataFrame({"cv_id": ["b", "a", "c"], "cv_text": ["python sql", "python sql", "java"]})
    res = screener().rank(JD, frame)
    assert [r.cv_id for r in res.ranked[:2]] == ["a", "b"]
    assert res.ranked[0].score == res.ranked[1].score
    assert all(np.isfinite(r.score) for r in res.ranked)


def test_pool_of_one_gives_a_finite_score():
    res = screener().rank(JD, cvs(1))
    assert len(res.ranked) == 1 and np.isfinite(res.ranked[0].score) and res.pool_size == 1


def test_d64_reorders_only_l1s_top_k_and_scores_only_those():
    seen = []

    def c2(pairs, progress=None):
        seen.append(len(pairs))
        return pd.Series(np.arange(len(pairs)) * 10.0, index=pairs.pair_id.to_numpy())
    s = screener(c2=c2, c2_k=3)
    d63, d64 = s.rank(JD, cvs(8)), s.rank(JD, cvs(8), rerank=True)
    assert seen == [3] and d64.system == "D64"
    assert {r.cv_id for r in d63.top(3)} == {r.cv_id for r in d64.top(3)}
    assert [r.cv_id for r in d63.ranked[3:]] == [r.cv_id for r in d64.ranked[3:]]
    assert [r.c2_score is not None for r in d64.ranked] == [True] * 3 + [False] * 5


def test_no_label_reaches_a_screening():
    def keys(d):
        if isinstance(d, dict):
            for k, v in d.items():
                yield k
                yield from keys(v)
        elif isinstance(d, list):
            for v in d:
                yield from keys(v)
    res = screener().rank(JD, cvs())
    assert not [k for k in keys(res.as_dict()) if "label" in k]


def test_as_dict_is_json_and_to_frame_has_a_row_per_cv():
    s = screener()
    res = s.rank(JD, cvs(), reference_date=dt.date(2026, 1, 1))
    doc = json.loads(json.dumps(res.as_dict()))
    assert doc["reference_date"] == "2026-01-01" and len(doc["ranked"]) == 6
    assert len(res.to_frame()) == 6 and set(F12) <= set(res.to_frame().columns)
    json.dumps(s.explain(res, "c0"))


def test_background_rows_are_never_returned_and_match_scoring_them_as_one_pool():
    up = cvs(4)
    s = screener()
    bg = s.rank(JD, up, scaling="background")
    assert [r.cv_id for r in bg.ranked if r.cv_id.startswith(("b", "bg"))] == []
    assert len(bg.ranked) == 4 and bg.pool_size == 4 + len(BACKGROUND)
    together = pd.concat([up, BACKGROUND.rename(columns={"id": "cv_id"})], ignore_index=True)
    whole = s.rank(JD, together, scaling="pool")
    for r in bg.ranked:
        assert r.features == whole.get(r.cv_id).features and r.l1_score == whole.get(r.cv_id).l1_score
    alone = s.rank(JD, up, scaling="pool")
    assert any(alone.get(r.cv_id).features["tfidf_scaled"] != r.features["tfidf_scaled"] for r in bg.ranked)


def test_explain_shares_sum_to_the_l1_score():
    s = screener()
    res = s.rank(JD, cvs())
    for r in res.ranked:
        why = s.explain(res, r.cv_id)
        assert sum(f["share"] for f in why["feature_shares"]) + why["l1_intercept"] == pytest.approx(r.l1_score, abs=1e-5)
        assert why["reasons"] and set(why["matched_skills"]) | set(why["missing_skills"]) == set(res.jd.skills)


def test_reference_date_reaches_the_years_and_the_features():
    text = "Python developer\n2024 - present: backend at a startup"
    calls = []
    s = screener(calls)
    old = s.extract_cv(text, reference_date=experience.REFERENCE_DATE).years
    new = s.extract_cv(text, reference_date=dt.date(2026, 10, 23)).years
    assert old != new and new is not None
    assert s.extract_cv(text).years == s.extract_cv(text, reference_date=dt.date.today()).years
    s.rank(JD, cvs(2), reference_date=dt.date(2025, 1, 1))
    assert calls[-1][1] == dt.date(2025, 1, 1)


def test_rank_on_injected_parts_never_fits(monkeypatch):
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.matching import entity_weighted, lexical

    def boom(*a, **k):
        raise AssertionError("inference must not fit")
    monkeypatch.setattr(dev_pools, "build", boom)
    monkeypatch.setattr(lexical, "a2_vectorizer", boom)
    monkeypatch.setattr(entity_weighted.DocumentFrequency, "fit", boom)
    assert len(screener().rank("A brand new job: Go and Kubernetes", cvs(3)).ranked) == 3


def _real_parts_available() -> bool:
    from candidate_screener.extraction import role_family
    from candidate_screener.inference import parts
    return (parts.ROOT / "LATEST").exists() and (role_family.CACHE / "LATEST").exists()


@pytest.mark.skipif(not _real_parts_available(), reason="needs data/ (run `inference --cache-parts` and the role model)")
def test_real_load_and_rank_on_new_texts_never_fit(monkeypatch, tmp_path):
    pytest.importorskip("sentence_transformers")
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.matching import entity_weighted, lexical, sbert

    def boom(*a, **k):
        raise AssertionError("inference must not fit")
    monkeypatch.setattr(dev_pools, "build", boom)
    monkeypatch.setattr(lexical, "a2_vectorizer", boom)
    monkeypatch.setattr(entity_weighted.DocumentFrequency, "fit", boom)
    monkeypatch.setattr(sbert, "EMBED_CACHE", tmp_path)          # new texts do not touch data/
    s = Screener.load()
    res = s.rank("Senior Go engineer: Kubernetes, PostgreSQL, 5+ years.",
                 [{"cv_id": "a", "cv_text": "Go developer, 6 years. Kubernetes, PostgreSQL, gRPC."},
                  {"cv_id": "b", "cv_text": "Accountant. Excel, SAP. 2015 - present."}])
    assert [r.cv_id for r in res.ranked] == ["a", "b"] and res.system == "D63"
