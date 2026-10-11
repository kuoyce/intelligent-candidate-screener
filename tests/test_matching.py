"""Ladder rungs (matching/): the interface, the rules, the hybrid."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.matching import corpus, lexical, rules
from candidate_screener.matching.base import BaseSystem


def _pairs():
    return pd.DataFrame({
        "pair_id": ["j1__a", "j1__b", "j1__c", "j2__a"], "jd_id": ["j1", "j1", "j1", "j2"],
        "cv_id": ["a", "b", "c", "a"],
        "jd_text": ["python data engineer spark"] * 3 + ["java backend"],
        "cv_text": ["python spark airflow", "java spring", "python", "java backend spring"],
        "jd_keyword": ["Data Engineer"] * 3 + ["Java"],
        "cv_keyword": ["Data Engineer", "Java", "Data Engineer", "Java"],
        "jd_exp_band": ["2-3"] * 3 + ["4-6"], "cv_exp_band": ["4-6", "4-6", "0-1", "2-3"]})


class _SeriesReturner(BaseSystem):
    name = "series"

    def _score(self, pairs):
        return pd.Series([0.1, 0.2, 0.3, 0.4])          # RangeIndex, not pair ids


def test_scores_are_values_not_realigned_series():
    """Regression: `pd.Series(series, index=pair_ids)` re-aligned on the RangeIndex and
    returned all-NaN scores for string pair ids. Caught by the NaN guard on first run."""
    s = _SeriesReturner().score(_pairs())
    assert list(s.index) == list(_pairs().pair_id)
    assert s.tolist() == [0.1, 0.2, 0.3, 0.4]


def test_a_nan_score_is_refused():
    class Bad(BaseSystem):
        name = "bad"
        def _score(self, pairs):
            return [np.nan] * len(pairs)
    with pytest.raises(ValueError, match="NaN"):
        Bad().score(_pairs())


def test_metadata_rules():
    p = _pairs()
    assert corpus.keyword_match(p).tolist() == [1, 0, 1, 1]
    assert corpus.experience_compatible(p).tolist() == [1, 1, 0, 0]
    assert rules.Metadata().score(p).tolist() == [3, 1, 2, 2]
    assert rules.metadata_classifier(p).tolist() == ["Fit", "No Fit", "Fit", "Fit"]


def test_random_is_independent_of_row_order():
    p = _pairs()
    a = rules.Random(0).score(p)
    b = rules.Random(0).score(p.iloc[::-1])
    assert a.sort_index().equals(b.sort_index())


def test_the_hybrid_orders_by_metadata_first_then_text():
    class Text(BaseSystem):
        name = "text"
        def _score(self, pairs):
            return [0.0, 0.9, 1.0, 0.5]             # prefers the off-keyword CV b
    hybrid = lexical.MetadataPlus(Text(), "test")
    s = hybrid.score(_pairs())
    assert s["j1__a"] > s["j1__c"] > s["j1__b"]     # metadata level beats text


def test_skill_overlap_is_coverage_of_the_jds_skills():
    class Stub:
        def canonicals(self, text):
            return {w for w in text.split() if w in {"python", "spark", "java", "spring"}}
    s = rules.SkillOverlap(matcher=Stub()).score(_pairs())
    assert s.tolist() == [1.0, 0.0, 0.5, 1.0]


def test_selection_checks_every_more_expensive_rung():
    """A cheaper rung is selected only when no more expensive rung is justified over it;
    a missing registered comparison is not an acquittal."""
    from candidate_screener.matching import selection
    names = {"random": 0, "metadata rules": 0, "TF-IDF": 0, "SBERT (zero-shot)": 0}
    auc = {"random": 0.5, "metadata rules": 0.7, "TF-IDF": 0.8, "SBERT (zero-shot)": 0.81}

    def delta(b, a, scope, subset):
        up = auc[b] - auc[a]
        ci = [up - 0.02, up + 0.02]
        return {"delta": up, "ci": ci, "half_width": 0.02, "n_jd": 50}

    out = selection.select(names, auc, delta, "metadata")
    assert out["selected"] == "TF-IDF"                 # SBERT's +0.01 is not resolved
    assert out["trail"][1]["beaten_by"] == ["TF-IDF", "SBERT (zero-shot)"]
    text = selection.select(names, auc, delta, "text_only")
    assert "metadata rules" not in [s["rung"] for s in text["trail"]]
