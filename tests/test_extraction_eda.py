"""Equal-population and denominator guards using synthetic evidence only."""
import pandas as pd
import pytest

from candidate_screener.extraction.eda import (analyze, bounded_examples, exact_comparison,
                                               per_document_counts, save_tables, threshold_counts, validate_population)
from candidate_screener.extraction.run import ENTITY_COLUMNS


def synthetic_analysis():
    docs = pd.DataFrame([{"doc_id": "a", "source": "djinni-cv", "doc_type": "cv", "fields": {"Position": "Python"}},
                         {"doc_id": "b", "source": "djinni-cv", "doc_type": "cv", "fields": {"Position": "No mentions"}},
                         {"doc_id": "c", "source": "djinni-jd", "doc_type": "jd", "fields": {"Position": "Python"}}])
    base = {"entity_id": "synthetic", "doc_id": "a", "source": "djinni-cv", "doc_type": "cv",
            "field": "Position", "text_version": "raw-v1", "start": 0, "end": 6, "label": "KNOWLEDGE",
            "surface": "Python", "canonical_id": None, "method": "gliner-zero-shot", "extractor_version": "test",
            "confidence": .75, "context_status": "unqualified"}
    neural = pd.DataFrame([base], columns=ENTITY_COLUMNS)
    rules = pd.DataFrame([{**base, "confidence": None, "method": "rules"}], columns=ENTITY_COLUMNS)
    return {"documents": docs, "neural": neural, "rules": rules}


def test_zero_entity_documents_remain_in_population():
    data = synthetic_analysis()
    counts = per_document_counts(data["documents"], data["neural"])
    assert len(counts) == 3
    assert counts.set_index("doc_id").entities.to_dict() == {"a": 1, "b": 0, "c": 0}


def test_threshold_coverage_uses_entire_population():
    data = synthetic_analysis()
    thresholds = threshold_counts(data["documents"], data["neural"])
    low = thresholds.query("doc_type == 'cv' and label == 'KNOWLEDGE' and threshold == .5").iloc[0]
    high = thresholds.query("doc_type == 'cv' and label == 'KNOWLEDGE' and threshold == .9").iloc[0]
    assert low.population_documents == high.population_documents == 2
    assert low.descriptive_coverage == .5
    assert high.retained_entities == 0
    assert high.descriptive_coverage == 0


def test_entities_from_other_population_refused():
    data = synthetic_analysis()
    data["neural"].loc[0, "doc_id"] = "outside"
    with pytest.raises(ValueError, match="outside"):
        validate_population(data["documents"], data["neural"])


def test_duplicate_entity_keys_refused():
    data = synthetic_analysis()
    duplicate = pd.concat([data["neural"], data["neural"]])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_population(data["documents"], duplicate)


def test_duplicate_document_population_refused():
    data = synthetic_analysis()
    duplicate = pd.concat([data["documents"], data["documents"].iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_population(duplicate, data["neural"])


def test_shared_requires_exact_offsets_and_label():
    data = synthetic_analysis()
    assert len(exact_comparison(data["neural"], data["rules"])["shared"]) == 1
    data["rules"].loc[0, "label"] = "SKILL"
    result = exact_comparison(data["neural"], data["rules"])
    assert len(result["shared"]) == 0
    assert len(result["neural_only"]) == len(result["rule_only"]) == 1


def test_analysis_includes_empty_classes_and_cooccurrence():
    tables = analyze(synthetic_analysis())
    assert len(tables["counts"]) == 16
    assert len(tables["label_coverage"]) == 16
    assert len(tables["cooccurrence"]) == 64
    assert len(tables["document_counts"]) == 6


def test_preview_redacts_contacts_and_caps_rows():
    data = synthetic_analysis()
    data["documents"].at[0, "fields"] = {"Position": "Python contact person@example.com"}
    example = bounded_examples(data, context_chars=120)
    assert "example.com" not in example.context.iloc[0]
    assert "[EMAIL]" in example.context.iloc[0]
    with pytest.raises(ValueError):
        bounded_examples(data, limit=6)


@pytest.mark.parametrize("threshold", [.4, float("nan"), float("inf")])
def test_threshold_cannot_recover_predictions_below_saved_floor(threshold):
    data = synthetic_analysis()
    with pytest.raises(ValueError, match="saved score floor"):
        threshold_counts(data["documents"], data["neural"], thresholds=[threshold])


def test_saved_document_distribution_excludes_ids(tmp_path):
    paths = save_tables(analyze(synthetic_analysis()), tmp_path)
    saved = pd.read_csv(paths["document_counts"])
    assert "doc_id" not in saved.columns
    assert saved.documents.sum() == 6
