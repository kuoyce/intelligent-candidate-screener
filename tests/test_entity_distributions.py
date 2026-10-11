"""Frozen evidence provenance and anonymous distribution invariants, synthetic only."""
import json
from pathlib import Path

import pandas as pd
import pytest

from candidate_screener.extraction import entity_distributions as ed
from candidate_screener.extraction.run import ENTITY_COLUMNS


@pytest.fixture
def frozen(tmp_path):
    docs = [
        {"source": "synthetic-cv", "doc_id": "person@example.com", "doc_type": "cv", "fields": {"body": "Python Python"}},
        {"source": "synthetic-cv", "doc_id": "empty-cv", "doc_type": "cv", "fields": {"body": ""}},
        {"source": "synthetic-jd", "doc_id": "empty-jd", "doc_type": "jd", "fields": {"body": "No named technology"}},
    ]
    provenance = {"sample_size": 2, "seed": 0, "region": "train",
                  "selected_ids": {"cv": ["person@example.com", "empty-cv"], "jd": ["empty-jd"]},
                  "labels": {label.lower(): label for label in ed.LABELS},
                  "code_sha256": {"historical.py": "old-producing-digest"},
                  "transformer_version": "historical-adapter"}
    base = {"source": "synthetic-cv", "doc_id": "person@example.com", "doc_type": "cv", "field": "body",
            "label": "KNOWLEDGE", "surface": "Python", "method": "gliner-zero-shot",
            "confidence": .8, "context_status": "unqualified", "canonical_id": None,
            "text_version": "raw-v1", "extractor_version": "historical-adapter"}
    entities = [{**base, "entity_id": "one", "start": 0, "end": 6},
                {**base, "entity_id": "two", "start": 7, "end": 13}]
    fields = [{"source": doc["source"], "doc_id": doc["doc_id"], "doc_type": doc["doc_type"],
               "field": "body", "characters": len(doc["fields"]["body"]), "complete": True,
               "intervals": [[0, len(doc["fields"]["body"])]] if doc["fields"]["body"] else []} for doc in docs]
    coverage = {}
    for kind in ("cv", "jd"):
        coverage[kind] = {"requested_documents": sum(doc["doc_type"] == kind for doc in docs),
                          "entities": 2 if kind == "cv" else 0,
                          "by_label": {label: 2 if kind == "cv" and label == "KNOWLEDGE" else 0 for label in ed.LABELS},
                          "documents_with_label": {label: 1 if kind == "cv" and label == "KNOWLEDGE" else 0 for label in ed.LABELS}}
    summary = {"status": "complete", "completed_documents": 3, "coverage": coverage}
    folder = tmp_path / "runs" / "frozen-run"
    folder.mkdir(parents=True)
    pd.DataFrame(entities, columns=ENTITY_COLUMNS).to_parquet(folder / "entities.parquet", index=False)
    (folder / "sample-manifest.json").write_text(json.dumps(provenance))
    (folder / "field-coverage.json").write_text(json.dumps(fields))
    metadata = {"run_id": "frozen-run", "summary": summary, "provenance": provenance,
                "artifact_sha256": {name: ed.sha256(folder / name) for name in ("entities.parquet", "sample-manifest.json", "field-coverage.json")}}
    record = tmp_path / "metrics.json"
    record.write_text(json.dumps(metadata))
    (folder / "metadata.json").write_text(json.dumps(metadata))
    loader = lambda size, seed: {"documents": docs, "provenance": {key: value for key, value in provenance.items() if key not in {"labels", "code_sha256", "transformer_version"}}}
    return record, folder, loader


def load(frozen):
    record, folder, loader = frozen
    return ed.load_frozen(record, artifact_root=folder.parent, input_loader=loader)


def refresh_hashes(frozen):
    record, folder, _ = frozen
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["artifact_sha256"] = {name: ed.sha256(folder / name) for name in metadata["artifact_sha256"]}
    record.write_text(json.dumps(metadata))
    (folder / "metadata.json").write_text(json.dumps(metadata))


def test_frozen_historical_code_is_reported_without_current_replay(frozen):
    analysis = load(frozen)
    assert analysis["validation"]["current_code_replay"] is False
    assert analysis["provenance"]["code_sha256"] == {"historical.py": "old-producing-digest"}


def test_zero_documents_and_repetition_remain_visible(frozen):
    tables = ed.distribution_tables(load(frozen))
    frame = tables["per_document"]
    assert len(frame) == 3 * 5
    knowledge = frame.query("doc_type == 'cv' and label == 'KNOWLEDGE'")
    assert sorted(knowledge.mentions) == [0, 2]
    assert knowledge.unique_mentions.sum() == 1
    assert knowledge.repeated_mentions.sum() == 1
    summary = tables["summary"].query("doc_type == 'cv' and label == 'KNOWLEDGE'").iloc[0]
    assert summary.documents == 2 and summary.descriptive_coverage == .5
    assert (frame.query("label == 'EXPERIENCE_YEARS'").mentions == 0).all()


def test_changed_artifact_is_refused_before_display(frozen):
    (frozen[1] / "field-coverage.json").write_text("[]")
    with pytest.raises(ValueError, match="digest"):
        load(frozen)


def test_changed_sample_membership_is_refused(frozen):
    original = frozen[2]
    def changed(size, seed):
        payload = original(size, seed)
        payload["provenance"] = {**payload["provenance"], "selected_ids": {"cv": ["different"], "jd": ["empty-jd"]}}
        return payload
    with pytest.raises(ValueError, match="membership"):
        load((frozen[0], frozen[1], changed))


def test_exact_original_surface_required_even_with_consistent_hashes(frozen):
    path = frozen[1] / "entities.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "surface"] = "False!"
    frame.to_parquet(path, index=False)
    refresh_hashes(frozen)
    with pytest.raises(ValueError, match="surface"):
        load(frozen)


def test_gap_in_complete_frozen_coverage_is_refused(frozen):
    path = frozen[1] / "field-coverage.json"
    fields = json.loads(path.read_text())
    fields[0]["intervals"] = [[0, 5], [7, 13]]
    path.write_text(json.dumps(fields))
    refresh_hashes(frozen)
    with pytest.raises(ValueError, match="gaps"):
        load(frozen)


def test_exports_never_contain_ids_or_source_strings(frozen, tmp_path):
    record = ed.save_distributions(load(frozen), tmp_path / "exports")
    assert record["validation"]["current_code_replay"] is False
    for path in (tmp_path / "exports").glob("*"):
        if path.suffix in {".json", ".csv"}:
            text = path.read_text()
            assert "person@example.com" not in text
            assert "Python" not in text
            assert "empty-cv" not in text
    assert len(list((tmp_path / "exports").glob("*.png"))) == 5
