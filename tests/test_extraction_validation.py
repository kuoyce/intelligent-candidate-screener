"""Saved evidence traceability guards, no data or model dependencies."""
import pandas as pd
import pytest

from candidate_screener.extraction.validation import audit_entities, audit_saved_run, examples


def fixture():
    documents = [{"doc_id": "a", "source": "djinni-cv", "doc_type": "cv", "fields": {"CV": "Python person@example.com"}}]
    entities = pd.DataFrame([{"entity_id": "entity", "doc_id": "a", "source": "djinni-cv", "doc_type": "cv",
                              "field": "CV", "start": 0, "end": 6, "surface": "Python", "label": "KNOWLEDGE", "confidence": .8}])
    coverage = [{"doc_id": "a", "source": "djinni-cv", "field": "CV", "complete": True, "intervals": [[0, 25]]}]
    # The exact fixture string has 25 characters.
    coverage[0]["intervals"] = [[0, len(documents[0]["fields"]["CV"])]]
    return documents, entities, coverage


def audit(documents, entities, coverage):
    return audit_entities(documents, entities, coverage, labels={"KNOWLEDGE"}, threshold=.5)


def test_exact_original_source_span_passes():
    assert audit(*fixture())["passed"]


@pytest.mark.parametrize("column,value,failure", [("surface", "Java", "surface_mismatch"),
    ("field", "Looking For", "unknown_document_or_field"), ("end", 999, "invalid_offsets"),
    ("confidence", float("nan"), "invalid_model_score"), ("label", "SKILL", "label_outside_saved_prompt")])
def test_saved_entity_corruption_reported(column, value, failure):
    documents, entities, coverage = fixture()
    entities.loc[0, column] = value
    report = audit(documents, entities, coverage)
    assert not report["passed"]
    assert report["failures"][failure] == 1


def test_non_numeric_offset_reports_failure_without_crashing():
    documents, entities, coverage = fixture()
    entities["start"] = "broken"
    assert audit(documents, entities, coverage)["failures"]["invalid_offsets"] == 1


def test_duplicate_saved_spans_reported():
    documents, entities, coverage = fixture()
    report = audit(documents, pd.concat([entities, entities]), coverage)
    assert report["failures"]["duplicate_entity_ids"] == 1
    assert report["failures"]["duplicate_span_label_keys"] == 1


def test_coverage_cannot_claim_complete_with_missing_source_tail():
    documents, entities, coverage = fixture()
    coverage[0]["intervals"] = [[0, 6]]
    assert audit(documents, entities, coverage)["failures"]["false_complete_coverage"] == 1


def test_missing_field_coverage_reported():
    documents, entities, _ = fixture()
    assert audit(documents, entities, [])["failures"]["missing_field_coverage"] == 1


def test_real_context_display_redacts_contacts():
    documents, entities, coverage = fixture()
    result = examples({"documents": documents, "entities": entities}, context_chars=120)
    assert "example.com" not in result.context.iloc[0]
    assert "[EMAIL]" in result.context.iloc[0]
    with pytest.raises(ValueError):
        examples({"documents": documents, "entities": entities}, limit=6)


def test_corrupt_coverage_intervals_report_without_crashing():
    documents, entities, coverage = fixture()
    coverage[0]["intervals"] = [[None, 5], [0], "invalid"]
    assert audit(documents, entities, coverage)["failures"]["invalid_coverage_offsets"]


def test_saved_run_requires_all_artifact_hashes(tmp_path):
    import json
    from candidate_screener.extraction.inputs import sha256
    root = tmp_path / "runs"
    folder = root / "run"
    folder.mkdir(parents=True)
    partition = tmp_path / "partition.csv"
    partition.write_text("doc_id,doc_type,region\n")
    provenance = {"partition_sha256": sha256(partition), "fields": {}, "labels": {"tool": "KNOWLEDGE"}, "threshold": .5}
    metadata = {"run_id": "wrong", "provenance": provenance, "artifact_sha256": {},
                "summary": {"status": "complete", "completed_documents": 0}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (folder / "sample-manifest.json").write_text(json.dumps(provenance))
    (folder / "field-coverage.json").write_text("[]")
    _, entities, _ = fixture()
    entities.iloc[:0].to_parquet(folder / "entities.parquet")
    result = audit_saved_run("run", run_root=root, raw_root=tmp_path, partition_path=partition)
    assert result["report"]["failures"]["missing_artifact_hashes"] == 1
    assert result["report"]["failures"]["run_id_mismatch"] == 1
