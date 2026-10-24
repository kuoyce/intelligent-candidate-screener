"""Original-source and completion invariants without optional neural packages."""
import pytest

from candidate_screener.extraction import medium_inference as inference
from candidate_screener.extraction import sentence_chunking


class Model:
    def predict_entities(self, text, labels, **kwargs):
        position = text.find("Python")
        return [] if position < 0 else [{"start": position, "end": position + 6,
                                         "text": "Python", "label": labels[0], "score": .9}]


def document():
    return {"doc_id": "r", "doc_type": "cv", "source": "fit-cv",
            "fields": {"text": "Intro. Python experience."}}


def plan(monkeypatch, windows, quarantine=None):
    monkeypatch.setattr(sentence_chunking, "plan_chunks", lambda *a, **k:
                        {"windows": windows, "quarantined_intervals": quarantine or [],
                         "diagnostics": {"sentence_count": 2, "fallback_windows": 0}})


def test_original_offsets_and_overlap_deduplication(monkeypatch):
    plan(monkeypatch, [(0, 25), (7, 25)])
    records, coverage = inference.extract_document(document(), Model())
    assert len(records) == 1
    assert records[0]["start"] == 7
    assert records[0]["end"] == 13
    assert records[0]["surface"] == document()["fields"]["text"][7:13]
    assert coverage["text"]["complete"]


def test_unsubmitted_tail_cannot_claim_complete(monkeypatch):
    plan(monkeypatch, [(0, 13)])
    _, coverage = inference.extract_document(document(), Model())
    assert not coverage["text"]["complete"]


def test_deadline_retains_unprocessed_state(monkeypatch):
    plan(monkeypatch, [(0, 25)])
    records, coverage = inference.extract_document(document(), Model(), deadline=0)
    assert not records
    assert coverage["text"]["interrupted"]
    assert not coverage["text"]["complete"]


def test_invalid_predictions_quarantined(monkeypatch):
    plan(monkeypatch, [(0, 25)])
    class Bad(Model):
        def predict_entities(self, text, labels, **kwargs):
            result = super().predict_entities(text, labels, **kwargs)
            result[0]["text"] = "invented"
            return result
    records, coverage = inference.extract_document(document(), Bad())
    assert not records
    assert coverage["text"]["quarantined_predictions"] == 1


def test_complete_coverage_requires_no_quarantine(monkeypatch):
    plan(monkeypatch, [(0, 25)], [[0, 7]])
    _, coverage = inference.extract_document(document(), Model())
    assert not coverage["text"]["complete"]


def test_fields_processed_separately(monkeypatch):
    plan(monkeypatch, [(0, 6)])
    doc = document()
    doc["fields"] = {"one": "Python", "two": "Python"}
    records, coverage = inference.extract_document(doc, Model())
    assert {r["field"] for r in records} == {"one", "two"}
    assert set(coverage) == {"one", "two"}
