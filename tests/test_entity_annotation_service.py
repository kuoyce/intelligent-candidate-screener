"""Human-only review state, original spans and stale-write invariants."""
import json

import pytest

from candidate_screener.entity_annotation.campaign import build_manifest, save_manifest
from candidate_screener.entity_annotation.service import AnnotationService, RevisionConflict, _digest
from candidate_screener.entity_annotation.text import python_to_utf16


@pytest.fixture
def service(tmp_path):
    docs = [{"doc_id": side, "source": "synthetic", "side": side,
             "text": f"{side} 😀 contact private@example.org. Python developer."}
            for side in ("cv", "jd")]
    manifest = build_manifest(docs, labels=["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"],
                              counts={"train": 2, "dev": 0, "test": 0})
    path = tmp_path / "data" / "manifest.json"
    save_manifest(manifest, path)
    return AnnotationService(path)


def payload(service, **changes):
    doc = service.document(service.session("alice")["documents"][0]["doc_id"], "alice")
    a = doc["text"].index("Python")
    result = {"doc_id": doc["doc_id"], "annotator": "alice", "revision": doc["revision"],
              "status": "reviewed", "reviewed_all": True, "offset_unit": "utf16",
              "spans": [{"start": python_to_utf16(doc["text"], a),
                         "end": python_to_utf16(doc["text"], a + 6), "label": "KNOWLEDGE"}]}
    return {**result, **changes}


def test_display_never_leaks_original_or_other_annotator(service):
    p = payload(service)
    service.save(p)
    doc = service.document(p["doc_id"], "bob")
    assert not doc["spans"]
    assert doc["revision"] == 0
    assert "private@example.org" not in json.dumps(doc)
    assert "source_text" not in doc and "boundary_map" not in doc
    assert service.session("bob")["progress"]["reviewed"] == 0


def test_unicode_and_redaction_map_exact_original_slice(service):
    p = payload(service)
    service.save(p)
    record = service.records()[0]
    span = record["spans"][0]
    doc = next(d for d in service.campaign()["documents"] if d["doc_id"] == p["doc_id"])
    assert span["surface"] == doc["source_text"][span["start"]:span["end"]] == "Python"
    assert service.document(p["doc_id"], "alice")["spans"] == p["spans"]


def test_empty_review_requires_explicit_complete_confirmation(service):
    with pytest.raises(ValueError, match="whole-document"):
        service.save(payload(service, spans=[], reviewed_all=False))
    service.save(payload(service, spans=[]))
    assert service.session("alice")["progress"]["reviewed"] == 1
    assert service.records()[0]["spans"] == []


def test_stale_revision_cannot_overwrite_and_history_retained(service):
    p = payload(service)
    service.save(p)
    with pytest.raises(RevisionConflict):
        service.save(p)
    service.save(payload(service, spans=[], status="uncertain", reviewed_all=False))
    assert [r["revision"] for r in service.records()] == [1, 2]
    assert len(service.records()[0]["spans"]) == 1


def test_masked_entity_rejected(service):
    p = payload(service)
    doc = service.document(p["doc_id"], "alice")
    a = doc["text"].index("[EMAIL]")
    p["spans"] = [{"start": python_to_utf16(doc["text"], a),
                   "end": python_to_utf16(doc["text"], a + 7), "label": "KNOWLEDGE"}]
    with pytest.raises(ValueError, match="masked"):
        service.save(p)
    assert not service.records()


def test_overlaps_and_unapproved_category_rejected(service):
    p = payload(service)
    with pytest.raises(ValueError, match="Overlapping"):
        service.save({**p, "spans": p["spans"] * 2})
    p["spans"][0]["label"] = "EXPERIENCE_YEARS"
    with pytest.raises(ValueError, match="category"):
        service.save(p)


def test_journal_corruption_refused(service):
    service.save(payload(service))
    text = service.journal_path.read_text()
    service.journal_path.write_text(text.replace('"reviewed"', '"draft"'))
    with pytest.raises(ValueError, match="integrity"):
        service.session("alice")


def test_draft_is_not_completed_gold(service):
    service.save(payload(service, status="draft", reviewed_all=False))
    progress = service.session("alice")["progress"]
    assert progress["reviewed"] == 0 and progress["draft"] == 1


def test_journal_replay_rejects_rehashed_revision_jump(service):
    service.save(payload(service))
    record = service.records()[0]
    record.pop("record_sha256")
    record["revision"] = 9
    record["record_sha256"] = _digest(record)
    service.journal_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="revision sequence"):
        service.session("alice")


def test_registered_annotator_assignment_is_enforced(service, tmp_path):
    campaign = service.campaign()
    ids = [doc["doc_id"] for doc in campaign["documents"]]
    campaign["annotators"] = ["alice", "bob"]
    campaign["assignments"] = {"alice": ids, "bob": ids[:1]}
    campaign.pop("campaign_id")
    campaign["campaign_id"] = _digest(campaign)[:24]
    path = tmp_path / "data" / "assigned" / "manifest.json"
    save_manifest(campaign, path)
    assigned = AnnotationService(path)
    assert assigned.session("bob")["progress"]["total"] == 1
    with pytest.raises(ValueError, match="registered"):
        assigned.session("eve")
    with pytest.raises(ValueError, match="assigned"):
        assigned.document(ids[1], "bob")
    request = payload(assigned)
    request.update(annotator="bob", doc_id=ids[1])
    with pytest.raises(ValueError, match="assigned"):
        assigned.save(request)
