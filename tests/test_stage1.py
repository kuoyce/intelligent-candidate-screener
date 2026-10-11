"""Stage 1 (plan 2026-10-06 P7): the X rungs, the Retriever interface, S2 and selection."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import stage1
from candidate_screener.extraction import experience as ex
from candidate_screener.extraction import role_family
from candidate_screener.matching import extracted
from candidate_screener.matching.base import BaseSystem
from candidate_screener.matching.retriever import SystemRetriever

JDS = ["3+ years of experience with Python", "3+ years of experience with Python",
       "We value curiosity.", "1+ year of experience in QA"]
CVS = ["I have 5 years of experience in backend.", "I have 1 year of experience.",
       "I have 5 years of experience in backend.", "Hobbies: chess."]


def _pairs(with_metadata: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame({"pair_id": [f"j{i // 2}__{i}" for i in range(4)],
                          "jd_id": ["j0", "j0", "j1", "j1"], "cv_id": list("abcd"),
                          "jd_text": JDS, "cv_text": CVS})
    if with_metadata:   # true fields chosen to contradict the extracted ones
        frame = frame.assign(jd_keyword="Java", cv_keyword=["Java", "Java", "Go", "Go"],
                             jd_exp_band="7+", cv_exp_band="0-1")
    return frame


@pytest.fixture
def fake_role(monkeypatch):
    """role_match = 0.25 per pair index (no model on a fresh clone)."""
    monkeypatch.setattr(role_family, "role_match",
                        lambda jd, cv, model=None: np.array([0.0, 0.25, 0.5, 1.0])[:len(jd)])
    return SimpleNamespace(fingerprint="test")


def test_unknown_experience_is_compatible_only_when_asked():
    """Amendment A1. Fails if `unknown_compatible` is ignored (unknown -> 0)."""
    assert list(ex.experience_compatible_extracted(JDS, CVS)) == [1, 0, 0, 0]
    assert list(ex.experience_compatible_extracted(JDS, CVS, unknown_compatible=True)) == [1, 0, 1, 1]


def test_x1_is_the_1a_formula_on_extracted_fields(fake_role):
    x1 = extracted.ExtractedRule(fake_role)
    x1.fit(None)
    assert x1.score(_pairs()).tolist() == [0 + 1, 0.5 + 0, 1.0 + 1, 2.0 + 1]


def test_x_rungs_read_no_metadata(fake_role):
    """D53. Scores with and without the true-field columns are identical. Fails if X1 or
    an X+text rung computes its level from `keyword_match` / `experience_compatible`."""
    text = _Text()
    for system in (extracted.ExtractedRule(fake_role),
                   extracted.ExtractedPlus(text, "X2", extracted.ExtractedRule(fake_role))):
        system.fit(None)
        np.testing.assert_array_equal(system.score(_pairs(True)).to_numpy(),
                                      system.score(_pairs(False)).to_numpy())


class _Text(BaseSystem):
    name = "text"

    def _score(self, pairs):
        return [float(len(c)) for c in pairs.cv_text]


def test_x_plus_text_scales_text_per_jd_under_x1(fake_role):
    plus = extracted.ExtractedPlus(_Text(), "X2", extracted.ExtractedRule(fake_role))
    plus.fit(None)
    got = plus.score(_pairs()).to_numpy()
    level = np.array([1.0, 0.5, 2.0, 3.0])
    # per JD the longer CV gets 0.999, the shorter 0
    assert np.allclose(got - level, [0.999, 0.0, 0.999, 0.0])
    assert plus.name == "extracted rule + text"


def test_retriever_shortlist_is_top_k_ties_in_pool_order():
    class Fixed(BaseSystem):
        name = "fixed"
        def _score(self, pairs):
            return [0.5, 0.9, 0.5, 0.1, 0.9]
    r = SystemRetriever(Fixed())
    pool = list("abcde")
    assert r.shortlist("jd", pool, 3) == [1, 4, 0]
    assert r.shortlist("jd", pool, 10) == [1, 4, 0, 2, 3]
    assert r.shortlist("jd", [], 3) == []


def _s2(values: dict) -> dict:
    retrieval = {n: [{"metric": "recall@50", "scope": "total", "value": v,
                      "ci_low": v - 0.05, "ci_high": v + 0.05}] for n, v in values.items()}
    return stage1.s2_verdicts(retrieval)


def test_s2_judges_each_group_on_its_best_member():
    s2 = _s2({stage1.TFIDF: 0.78, stage1.X2: 0.83, stage1.SBERT_WEAK: 0.88, stage1.X4: 0.89})
    assert s2["classical"]["best"] == stage1.X2 and s2["classical"]["verdict"] == "pass"
    assert s2["classical"]["ci_above_bar"] is False
    assert s2["advanced"]["verdict"] == "miss"


def _verdicts(**justified):
    keys = [(b, a) for b, a, _ in stage1.BEATS]
    return {k: "justified" if justified.get(k[0]) else "not resolved" for k in keys}


def test_selection_adopts_the_cheapest_unbeaten_qualifier():
    s2 = _s2({stage1.TFIDF: 0.85, stage1.X2: 0.85, stage1.SBERT_WEAK: 0.95, stage1.X4: 0.95})
    assert stage1.select(s2, _verdicts())["selected"] == stage1.TFIDF
    assert stage1.select(s2, {**_verdicts(), (stage1.X2, stage1.TFIDF): "justified"}
                         )["selected"] == stage1.X2
    beaten_twice = {**_verdicts(), (stage1.X2, stage1.TFIDF): "justified",
                    (stage1.SBERT_WEAK, stage1.X2): "justified"}
    assert stage1.select(s2, beaten_twice)["selected"] == stage1.SBERT_WEAK


def test_a_rung_failing_s2_neither_qualifies_nor_beats():
    """Fails if an S2-missing rung can block a cheaper one."""
    s2 = _s2({stage1.TFIDF: 0.85, stage1.X2: 0.70, stage1.SBERT_WEAK: 0.95, stage1.X4: 0.95})
    out = stage1.select(s2, {**_verdicts(), (stage1.X2, stage1.TFIDF): "justified"})
    assert out["selected"] == stage1.TFIDF
    s2 = _s2({stage1.TFIDF: 0.70, stage1.X2: 0.70, stage1.SBERT_WEAK: 0.85, stage1.X4: 0.95})
    assert stage1.select(s2, _verdicts())["selected"] == stage1.X4


def test_registered_comparisons_are_k1_to_k5_and_only_k2_k5_decide_nothing():
    assert [c[0] for c in stage1.COMPARISONS] == ["K1", "K2", "K3", "K4", "K5"]
    assert {c[0] for c in stage1.COMPARISONS if not c[3]} == {"K2", "K5"}
