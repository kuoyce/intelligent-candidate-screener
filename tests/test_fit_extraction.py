"""Fit campaign identities and state guards without model/data dependencies."""
import hashlib
import json
import pandas as pd
import pytest
from candidate_screener.extraction import fit_run as fit
from candidate_screener.extraction.run import ENTITY_COLUMNS, _json


def pairs():
    return pd.DataFrame([{"resume_id": "r", "jd_id": "j", "resume_text": "Python", "job_description_text": "Python",
                          "label": "Good Fit", "split": "train"},
                         {"resume_id": "z", "jd_id": "j", "resume_text": "Nothing", "job_description_text": "Python",
                          "label": "No Fit", "split": "train"}])


def test_unique_documents_extract_once_without_labels():
    docs = fit.documents_from_pairs(pairs())
    assert len(docs) == 3
    assert all(set(d) == {"doc_id", "source", "doc_type", "fields"} for d in docs)
    assert all(set(d["fields"]) == {"text"} for d in docs)


def test_conflicting_texts_for_document_identity_refused():
    frame = pairs()
    frame.loc[1, "job_description_text"] = "Different"
    with pytest.raises(ValueError, match="Conflicting"):
        fit.documents_from_pairs(frame)


def checkpoint(tmp_path, status="complete_zero", entities=None, complete=True):
    doc = {"source": "fit-cv", "doc_id": "r", "doc_type": "cv", "fields": {"text": "Python"}}
    record = {"campaign_id": "campaign", "source": "fit-cv", "doc_id": "r", "status": status,
              "entities": entities or [], "coverage": {"text": {"complete": complete, "intervals": [[0, 6]]}}}
    record["sha256"] = hashlib.sha256(_json(record).encode()).hexdigest()
    path = tmp_path / "checkpoint.json"
    path.write_text(_json(record))
    return path, doc


def test_complete_zero_remains_distinct_from_pending(tmp_path):
    path, doc = checkpoint(tmp_path)
    assert fit._load_checkpoint(path, doc, "campaign")["status"] == "complete_zero"


def test_checkpoint_corruption_refused(tmp_path):
    path, doc = checkpoint(tmp_path)
    record = json.loads(path.read_text())
    record["status"] = "failed"
    path.write_text(_json(record))
    with pytest.raises(ValueError, match="digest"):
        fit._load_checkpoint(path, doc, "campaign")


@pytest.mark.parametrize("status", ["unknown", "complete"])
def test_unknown_or_inconsistent_checkpoint_state_refused(tmp_path, status):
    path, doc = checkpoint(tmp_path, status=status)
    with pytest.raises(ValueError):
        fit._load_checkpoint(path, doc, "campaign")


def test_incomplete_saved_document_never_claims_complete(tmp_path):
    path, doc = checkpoint(tmp_path, status="incomplete", complete=False)
    assert fit._load_checkpoint(path, doc, "campaign")["status"] == "incomplete"
    path, doc = checkpoint(tmp_path, complete=False)
    with pytest.raises(ValueError, match="integrity"):
        fit._load_checkpoint(path, doc, "campaign")


def test_reporting_includes_zero_and_pending_denominator():
    statuses = pd.DataFrame([{"source": "fit-cv", "doc_id": "r", "doc_type": "cv", "status": "complete_zero", "entities": 0, "runtime_seconds": 1},
                            {"source": "fit-cv", "doc_id": "z", "doc_type": "cv", "status": "pending", "entities": None, "runtime_seconds": None},
                            {"source": "fit-jd", "doc_id": "j", "doc_type": "jd", "status": "failed", "entities": 0, "runtime_seconds": 1}])
    tables = fit.report_tables({"pairs": pairs(), "entities": pd.DataFrame(columns=ENTITY_COLUMNS), "statuses": statuses})
    cover = tables["label_coverage"].query("split == 'train' and doc_type == 'cv'")
    assert set(cover.requested_documents) == {2}
    assert set(cover.completed_documents) == {1}
    assert len(tables["document_counts"].query("split == 'train'")) == 3
    assert tables["document_counts"].query("status == 'pending'").entities.isna().all()


def test_campaign_directory_traversal_refused(tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        fit.load_campaign("../outside", campaign_root=tmp_path)


def test_exact_source_offset_corruption_rejected_before_resume(tmp_path):
    entity = {"entity_id": "e", "source": "fit-cv", "doc_id": "r", "doc_type": "cv", "field": "text",
              "start": 0, "end": 6, "surface": "Different", "label": "KNOWLEDGE", "confidence": .8}
    path, doc = checkpoint(tmp_path, status="complete", entities=[entity])
    with pytest.raises(ValueError, match="surface_mismatch"):
        fit._load_checkpoint(path, doc, "campaign")


def test_fit_conditioned_pair_denominator_requires_both_sides_complete():
    statuses = pd.DataFrame([{"source": "fit-cv", "doc_id": "r", "doc_type": "cv", "status": "complete_zero", "entities": 0, "runtime_seconds": 1},
                            {"source": "fit-cv", "doc_id": "z", "doc_type": "cv", "status": "complete_zero", "entities": 0, "runtime_seconds": 1},
                            {"source": "fit-jd", "doc_id": "j", "doc_type": "jd", "status": "pending", "entities": None, "runtime_seconds": None}])
    tables = fit.report_tables({"pairs": pairs(), "entities": pd.DataFrame(columns=ENTITY_COLUMNS), "statuses": statuses})
    assert set(tables["fit_conditioned"].completed_document_pairs) == {0}


def test_checkpoint_identity_cannot_be_reused_for_another_document(tmp_path):
    path, doc = checkpoint(tmp_path)
    doc["doc_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        fit._load_checkpoint(path, doc, "campaign")


@pytest.mark.parametrize("defect", ["manifest", "source"])
def test_campaign_resume_refuses_manifest_or_source_change(tmp_path, monkeypatch, defect):
    provenance = {"source_root": str(tmp_path), "source_sha256": {"train": "original"}, "document_ids": {}}
    identity = hashlib.sha256(_json(provenance).encode()).hexdigest()[:24]
    folder = tmp_path / identity
    folder.mkdir()
    metadata = {"campaign_id": identity, "provenance": provenance, "inventory": {}}
    if defect == "manifest":
        metadata["provenance"] = {**provenance, "source_sha256": {"train": "changed"}}
    (folder / "manifest.json").write_text(_json(metadata))
    current = {**provenance, "source_sha256": {"train": "changed"}} if defect == "source" else provenance
    monkeypatch.setattr(fit, "inventory", lambda _: {"provenance": current})
    with pytest.raises(ValueError):
        fit.load_campaign(identity, campaign_root=tmp_path)
