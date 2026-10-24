"""Suggestions stay separate from gold and hidden from independent evaluation."""
import json

import pytest

from candidate_screener.entity_annotation.campaign import build_manifest, save_manifest
from candidate_screener.entity_annotation.service import AnnotationService, _digest
from candidate_screener.entity_annotation import suggestions


@pytest.fixture
def service(tmp_path, monkeypatch):
    docs = [{"source": "synthetic", "doc_id": f"{side}{i}", "side": side,
             "text": f"Python developer {side}{i}."} for side in ("cv", "jd") for i in range(3)]
    c = build_manifest(docs, labels=["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"],
                       counts={"train": 2, "dev": 2, "test": 2})
    ids = [d["doc_id"] for d in c["documents"]]
    c.update(annotators=["annotator1", "annotator2"], assignments={"annotator1": ids, "annotator2": ids})
    c.pop("campaign_id")
    c["campaign_id"] = _digest(c)[:24]
    p = tmp_path / "data/manifest.json"
    save_manifest(c, p)
    monkeypatch.setattr(suggestions, "load_for_document", lambda *args:
                        {"available": True, "suggestion_set_id": "set123", "model_id": "model123",
                         "spans": [{"start": 0, "end": 6, "label": "KNOWLEDGE", "score": .9}]})
    return AnnotationService(p)


def doc(service, split="train"):
    return next(d for d in service.session("annotator1")["documents"] if d["split"] == split)


def payload(service, **changes):
    d = doc(service)
    return {"doc_id": d["doc_id"], "annotator": "annotator1", "revision": d["revision"],
            "offset_unit": "utf16", "spans": [], "status": "draft", "reviewed_all": False,
            **changes}


def test_loading_model_suggestions_never_saves_gold(service):
    assert service.suggestions(doc(service)["doc_id"], "annotator1")["available"]
    assert service.records() == []
    assert service.session("annotator1")["progress"]["reviewed"] == 0


@pytest.mark.parametrize("split", ["dev", "test"])
def test_heldout_suggestions_hidden_server_side(service, split):
    with pytest.raises(ValueError, match="hidden"):
        service.suggestions(doc(service, split)["doc_id"], "annotator1")
    with pytest.raises(ValueError, match="restricted"):
        service.save(payload(service, doc_id=doc(service, split)["doc_id"],
                             assistance={"suggestion_set_id": "set123", "model_id": "model123"}))


def test_independent_annotator_cannot_access_or_save_assistance(service):
    with pytest.raises(ValueError, match="hidden"):
        service.suggestions(doc(service)["doc_id"], "annotator2")
    with pytest.raises(ValueError, match="restricted"):
        service.save(payload(service, annotator="annotator2",
                             assistance={"suggestion_set_id": "set123", "model_id": "model123"}))


def test_assistance_survives_deletion_and_later_revisions(service):
    assistance = {"suggestion_set_id": "set123", "model_id": "model123"}
    service.save(payload(service, assistance=assistance))
    service.save(payload(service, revision=1, status="reviewed", reviewed_all=True))
    last = service.records()[-1]
    assert last["spans"] == []
    assert last["assistance"] == assistance
    assert last["annotation_mode"] == "model_assisted"


def test_existing_draft_cannot_be_replaced_with_suggestions(service):
    service.save(payload(service))
    with pytest.raises(ValueError, match="replaced"):
        service.suggestions(doc(service)["doc_id"], "annotator1")
    with pytest.raises(ValueError, match="replaced"):
        service.save(payload(service, revision=1, assistance={"suggestion_set_id": "set123", "model_id": "model123"}))


def test_forged_model_identity_rejected(service):
    with pytest.raises(ValueError, match="identity"):
        service.save(payload(service, assistance={"suggestion_set_id": "set123", "model_id": "forged"}))
