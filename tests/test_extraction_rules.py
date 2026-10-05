"""Implementation invariants on synthetic text, never corpus accuracy claims."""
from copy import deepcopy

import pytest

from candidate_screener.extraction import extract_document, score_exact_spans, validate_entity


def document(text, field="Moreinfo"):
    return {"doc_id": "synthetic-1", "source": "synthetic", "doc_type": "cv", "fields": {field: text}}


def test_punctuation_sensitive_aliases_and_longest_overlap():
    rows = extract_document(document("C++ C# .NET Node.js React.js PostgreSQL"))
    assert [r["surface"] for r in rows] == ["C++", "C#", ".NET", "Node.js", "React.js", "PostgreSQL"]
    assert len({r["entity_id"] for r in rows}) == len(rows)


def test_ordinary_go_and_r_require_programming_context():
    assert not extract_document(document("We Go to work. R is an initial. go home."))
    assert [r["surface"] for r in extract_document(document("Go programming; R statistical analysis; Golang"))] == ["Go", "R", "Golang"]


def test_negation_and_learning_are_clause_bounded():
    rows = extract_document(document("No Python experience; learning Kubernetes; SQL. No AWS but Docker"))
    assert [(r["surface"], r["context_status"]) for r in rows] == [
        ("Python", "negated"), ("Kubernetes", "learning"), ("SQL", "unqualified"),
        ("AWS", "negated"), ("Docker", "unqualified")]
    assert extract_document(document("Python", "Looking For"))[0]["context_status"] == "learning"
    for text in ("not only Python", "no problem with Python"):
        assert extract_document(document(text))[0]["context_status"] == "unqualified"
    assert extract_document(document("No experience with Python"))[0]["context_status"] == "negated"


def test_original_unicode_offsets_and_separate_fields():
    doc = document("🙂 café Python and C++")
    doc["fields"]["Position"] = "Software engineer"
    original = deepcopy(doc)
    rows = extract_document(doc)
    assert doc == original
    assert {r["label"] for r in rows} == {"KNOWLEDGE", "TITLE"}
    for row in rows:
        assert doc["fields"][row["field"]][row["start"]:row["end"]] == row["surface"]
    assert rows == extract_document(doc)


def test_null_fields_and_cross_field_phrases():
    doc = document("software")
    doc["fields"].update({"Position": "engineer", "Highlights": None})
    assert extract_document(doc) == []
    with pytest.raises(ValueError, match="strings"):
        extract_document(document(3))


def test_offset_validation_refuses_shifted_evidence():
    text = "Python"
    row = extract_document(document(text))[0]
    row["start"] += 1
    with pytest.raises(ValueError):
        validate_entity(row, text)


def test_exact_span_label_scoring_keeps_documents_and_layers_distinct():
    gold = [{"doc_id": "a", "field": "body", "start": 0, "end": 6, "label": "KNOWLEDGE"}]
    assert score_exact_spans(gold, gold)["micro"]["f1"] == 1.0
    wrong = [{**gold[0], "end": 5}, {**gold[0], "doc_id": "b"}, {**gold[0], "label": "SKILL"}]
    scores = score_exact_spans(gold, wrong)
    assert scores["micro"]["true_positive"] == 0
    assert scores["per_label"]["KNOWLEDGE"]["support"] == 1
    assert score_exact_spans(gold, [{**gold[0], "layer": "other"}])["micro"]["f1"] == 0
    assert score_exact_spans(gold, gold + gold)["micro"]["precision"] == 0.5


def test_rule_confidence_null_and_seed_namespace():
    rows = extract_document(document("Master of Science; data analysis; SQL"))
    assert {r["label"] for r in rows} == {"DEGREE", "SKILL", "KNOWLEDGE"}
    assert all(r["confidence"] is None and r["canonical_id"].startswith("seed:") for r in rows)
