"""Evidence layer and pipeline: verbatim quotes only, coverage, ranking order."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener import pipeline
from candidate_screener.evidence import evidence as ev
from candidate_screener.evidence import requirements as req
from candidate_screener.matching.base import BaseSystem


class _Match:
    def __init__(self, start, end, text, canonical):
        self.start, self.end, self.text, self.canonical = start, end, text, canonical


class StubMatcher:
    """Finds a fixed vocabulary, case-insensitively, with char offsets."""
    VOCAB = {"python": "Python", "sql": "SQL", "airflow": "Apache Airflow", "spark": "Apache Spark"}

    def find(self, text):
        out = []
        low = text.lower()
        for alias, canonical in self.VOCAB.items():
            i = low.find(alias)
            if i >= 0:
                out.append(_Match(i, i + len(alias), text[i:i + len(alias)], canonical))
        return sorted(out, key=lambda m: m.start)

    def canonicals(self, text):
        return {m.canonical for m in self.find(text)}


class StubEncoder:
    """Only 'Airflow' has a direction, and only sentences about pipelines share it."""
    def encode(self, texts):
        def vec(t):
            t = t.lower()
            if t == "airflow" or "pipelines" in t:
                return [1.0, 0.0]
            return [0.0, 0.0] if len(t.split()) == 1 else [0.0, 1.0]
        return np.array([vec(t) for t in texts])


JD = "We need Python and SQL. Airflow is a plus. 3+ years of experience."
CV = "Data engineer, 5 years.\nBuilt ETL pipelines in Python.\nReporting with Excel."


def test_requirements_are_distinct_skills_in_order_plus_years():
    reqs = req.skill_requirements(JD, StubMatcher())
    assert [r.canonical for r in reqs] == ["Python", "SQL", "Apache Airflow"]
    assert req.years_required(JD) == 3 and req.years_stated(CV) == 5


def test_evidence_statuses_and_every_quote_is_verbatim():
    found = ev.gather(req.skill_requirements(JD, StubMatcher()), CV, StubMatcher(),
                      encoder=StubEncoder(), threshold=0.5)
    status = {e.requirement: e.status for e in found}
    assert status == {"Python": "matched", "SQL": "missing", "Apache Airflow": "supported"}
    for e in found:
        if e.status != "missing":
            assert CV[e.start:e.end] == e.quote
    assert ev.coverage(found) == pytest.approx(2 / 3)


def test_a_fabricated_quote_is_refused():
    fake = [ev.Evidence("Python", "matched", "Python expert", 0, 13, None)]
    with pytest.raises(ValueError, match="verbatim"):
        ev.assert_faithful(fake, CV)


def test_without_an_encoder_only_name_matches_count():
    found = ev.gather(req.skill_requirements(JD, StubMatcher()), CV, StubMatcher())
    assert [e.status for e in found] == ["matched", "missing", "missing"]


class StubTagger:
    def predict_spans(self, text):
        out = []
        for phrase in ("Python and SQL", "team player", "strong written communication in English across many teams"):
            i = text.find(phrase)
            if i >= 0:
                out.append((i, i + len(phrase)))
        return out


def test_the_tagger_adds_only_short_free_text_requirements():
    jd = JD + " A team player with strong written communication in English across many teams."
    reqs = req.requirements(jd, StubMatcher(), StubTagger())
    kinds = {r.canonical: r.kind for r in reqs}
    assert kinds == {"Python": "skill", "SQL": "skill", "Apache Airflow": "skill",
                     "team player": "free_text"}      # vocab span skipped; long span dropped
    free = [r for r in reqs if r.kind == "free_text"][0]
    assert jd[free.start:free.end] == free.text


def test_sentences_keep_offsets():
    for start, end, body in ev.sentences(CV):
        assert CV[start:end] == body


class _Len(BaseSystem):
    name = "len"
    def _score(self, pairs):
        return pairs.cv_text.str.len().to_numpy()


def test_screen_ranks_quotes_and_ignores_metadata_fit_calls():
    cvs = pd.DataFrame({"cv_id": ["a", "b"], "cv_text": [CV, "Excel only."],
                        "cv_keyword": ["Data Engineer", "Marketing"]})
    out = pipeline.screen(JD, cvs, jd_meta={"keyword": "Data Engineer"}, ranker=_Len(),
                          matcher=StubMatcher())
    assert [r.cv_id for r in out] == ["a", "b"] and [r.rank for r in out] == [1, 2]
    assert out[0].fit_call is None and out[1].fit_call is None
    assert out[0].years_stated is None and out[0].coverage == pytest.approx(1 / 3)


def test_screen_without_metadata_makes_no_fit_call():
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": [CV]})
    out = pipeline.screen(JD, cvs, ranker=_Len(), matcher=StubMatcher())
    assert out[0].fit_call is None


@pytest.mark.parametrize("text", ["No experience with Python", "Learning Python", "Never used Python"])
def test_negative_and_learning_mentions_are_not_positive_evidence(text):
    found = ev.gather(req.skill_requirements("Python", StubMatcher()), text, StubMatcher(),
                      encoder=StubEncoder(), threshold=0)
    assert found[0].status == "missing"


def test_denial_applies_to_whole_skill_list_but_not_following_clause():
    found = ev.gather(req.skill_requirements("Python SQL", StubMatcher()),
                      "No experience with Python and SQL", StubMatcher())
    assert all(e.status == "missing" for e in found)
    found = ev.gather(req.skill_requirements("SQL", StubMatcher()),
                      "No experience with Python, but SQL in production", StubMatcher())
    assert found[0].status == "matched"
