"""Synthetic invariants for bounded, train-only entity scoring."""
import math

import numpy as np
import pandas as pd
import pytest

from candidate_screener.matching.entity_weighted import (
    ConceptExtractor, DocumentFrequency, EntityWeightedX2,
)
from candidate_screener.matching.extracted import ExtractedPlus
from candidate_screener.matching.base import BaseSystem


def extractor():
    return ConceptExtractor(aliases={"Python": "python", "SQL": "sql", "K8s": "kubernetes",
                                     "Kubernetes": "kubernetes"})


def training(texts):
    return pd.DataFrame({"id": [f"c{i}" for i in range(len(texts))],
                         "text": texts, "region": "train", "doc_type": "cv"})


def test_document_frequency_counts_documents_not_mentions_or_repeated_rows():
    frame = training(["Python Python SQL", "Python", "SQL"])
    df = DocumentFrequency(extractor()).fit(pd.concat([frame, frame.iloc[:1]]))
    assert df.n_documents == 3
    assert df.df == {"python": 2, "sql": 2}
    assert df.weight("python") == pytest.approx(1 + math.log(4 / 3))


@pytest.mark.parametrize("region", ["eval", "dev-select", "validation"])
def test_idf_refuses_non_training_frequencies(region):
    frame = training(["Python", "SQL"])
    frame.loc[1, "region"] = region
    with pytest.raises(ValueError, match="train-region"):
        DocumentFrequency(extractor()).fit(frame)


def test_idf_refuses_conflicting_ids_and_jd_documents():
    frame = training(["Python", "SQL"])
    frame["id"] = "same"
    with pytest.raises(ValueError, match="conflicting"):
        DocumentFrequency(extractor()).fit(frame)
    frame = training(["Python"]).assign(doc_type="jd")
    with pytest.raises(ValueError, match="CV documents"):
        DocumentFrequency(extractor()).fit(frame)


def test_coverage_discounts_common_skills_ignores_unrelated_and_repeated_mentions():
    ex = extractor()
    df = DocumentFrequency(ex).fit(training(["Python SQL"] + ["Python"] * 8))
    wanted = ex.concepts("Python SQL", "jd")
    common = df.coverage(wanted, ex.concepts("Python Python Kubernetes"))
    assert common == df.coverage(wanted, ex.concepts("Python"))
    assert common < .5 < df.coverage(wanted, ex.concepts("SQL"))
    assert df.coverage(wanted, ex.concepts("SQL Python")) == 1
    assert df.coverage(wanted, ex.concepts("SQL"), weighted=False) == .5


def test_exact_alias_unknown_and_affirmative_context_only():
    rows = [{"label": "KNOWLEDGE", "surface": "K8s", "context_status": "unqualified"},
            {"label": "KNOWLEDGE", "surface": "SQL", "context_status": "negated"},
            {"label": "SKILL", "surface": "Learning Python", "context_status": "learning"},
            {"label": "SKILL", "surface": "  rare   process ", "context_status": "unqualified"},
            {"label": "DEGREE", "surface": "Python", "context_status": "unqualified"}]
    ex = ConceptExtractor(entities=lambda text, side: rows, aliases={"K8s": "kubernetes"})
    assert ex.concepts("text") == {"kubernetes", "unknown:rare process"}
    df = DocumentFrequency(ex).fit(training(["text"]))
    assert df.weight("unknown:rare process") == 1
    assert extractor().concepts("No SQL; learning Python; Kubernetes") == {"kubernetes"}


def test_weights_cap_and_zero_entities_are_not_excluded():
    df = DocumentFrequency(extractor()).fit(training([""] * 100))
    assert df.weight("sql") == 5
    assert df.coverage(set(), {"sql"}) == 0
    assert df.coverage({"sql"}, set()) == 0


class Text(BaseSystem):
    trained_on = "synthetic"

    def _score(self, pairs):
        return np.array([len(t) for t in pairs.cv_text], dtype=float)


class Rule(BaseSystem):
    trained_on = "synthetic"
    role_model = True

    def components(self, pairs):
        return pd.DataFrame({"role_match": [.2] * len(pairs),
                             "experience_compatible": [1.] * len(pairs)})

    def _score(self, pairs):
        c = self.components(pairs)
        return 2 * c.role_match.to_numpy() + c.experience_compatible.to_numpy()


def pairs():
    return pd.DataFrame({"pair_id": ["a", "b", "c"], "jd_id": ["j"] * 3,
                         "jd_text": ["Python SQL"] * 3, "jd_title": ["Developer"] * 3,
                         "cv_text": ["Python", "SQL", ""]})


def test_alpha_zero_exact_x2_control_even_without_entities_or_frequencies():
    class NoExtraction:
        def concepts(self, *args):
            raise AssertionError("alpha zero should not require extraction")

    baseline = ExtractedPlus(Text(), "X2", Rule())
    baseline.fit()
    candidate = EntityWeightedX2(rule=Rule(), text_system=Text(), extractor=NoExtraction())
    candidate.fit()
    assert np.array_equal(baseline.score(pairs()).to_numpy(), candidate.score(pairs()).to_numpy())
    subset = pairs().iloc[1:]
    assert np.array_equal(baseline.score(subset).to_numpy(), candidate.score(subset).to_numpy())


def test_bounded_blend_retains_zero_entity_rows_and_reports_missingness():
    ex = extractor()
    df = DocumentFrequency(ex).fit(training(["Python", "SQL"]))
    model = EntityWeightedX2(alpha=.5, extractor=ex, frequencies=df,
                             rule=Rule(), text_system=Text(), title=True)
    model.fit()
    result = model.score(pairs())
    assert len(result) == 3 and np.isfinite(result).all()
    assert result.between(1.4, 2.399).all()
    c = model.components(pairs())
    assert c.cv_concepts_available.tolist() == [True, True, False]
    assert model.inputs(pairs()).jd_text.iloc[0] == "Developer\nPython SQL"
    assert c.coverage.iloc[2] == 0


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan"), float("inf")])
def test_blend_weight_must_be_bounded(alpha):
    with pytest.raises(ValueError, match="alpha"):
        EntityWeightedX2(alpha=alpha)
