"""Relevant-skill lookup uses the requested review scope and current source spans."""
import hashlib
from types import SimpleNamespace

import pytest

from candidate_screener.app.conversation_v3 import ConversationV3Service
from candidate_screener.app.portal_extractive import PortalExtractiveService


class Entities:
    """Synthetic current-source entities: no neural workers, weights or downloads."""
    def prepare(self, snapshot, selected):
        sources = [(snapshot.jd_id, "jd", snapshot.jd_text)]
        sources += [(row["cv_id"], "cv", row["preview_text"])
                    for row in snapshot.candidates if row["cv_id"] in selected]
        documents = {}
        for owner, kind, text in sources:
            digest = hashlib.sha256(text.encode()).hexdigest()
            entities = []
            for term in ("Python", "SQL", "C++"):
                if term not in text:
                    continue
                start = text.index(term)
                entities.append({"owner_id": owner, "owner": owner, "doc_id": owner,
                    "candidate_id": owner if kind == "cv" else None,
                    "snapshot_id": snapshot.snapshot_id, "source_digest": digest,
                    "coordinate_space": "redacted_preview", "start": start, "end": start + len(term),
                    "surface": term, "text": term,
                    "label": ("SKILL" if term == "Python" else "KNOWLEDGE") if kind == "jd" else
                             ("KNOWLEDGE" if term == "Python" else "SKILL"),
                    "context_status": "unqualified", "context": {"start": 0, "end": len(text), "text": text}})
            documents[owner] = {"owner_id": owner, "doc_type": kind, "source_digest": digest,
                "coordinate_space": "redacted_preview", "status": "complete", "entities": entities,
                "coverage": {"text": {"characters": len(text), "intervals": [[0, len(text)]], "complete": True}}}
        return {"status": "complete", "documents": documents}

    bundle = prepare


class NoQa:
    def answer(self, *args):
        pytest.fail("Relevant-skill overlap must not start a QA model")


class NoComposer:
    def compose_before(self, *args):
        pytest.fail("Relevant-skill overlap must not invoke conversational generation")


def snapshot():
    return SimpleNamespace(snapshot_id="skills-review", jd_id="JD-SKILLS",
        jd_title="Invented developer", jd_text="Python programming skills and SQL knowledge required.",
        candidates=tuple({"cv_id": f"CV-{rank}", "rank": rank,
                          "preview_text": "Python programming skills." if rank == 1 else
                                          "SQL knowledge." if rank == 2 else "C++ development skills.",
                          "preview_status": "available_redacted_text"} for rank in range(1, 11)))


def service():
    return ConversationV3Service(PortalExtractiveService(Entities(), NoQa()), NoComposer())


def prior(current, selected):
    return {"snapshot_id": current.snapshot_id, "jd_id": current.jd_id,
            "selected_candidates": selected, "intent": "candidate_qa" if len(selected) == 1 else "comparison",
            "category": "SKILL", "criteria": ["Python"]}


@pytest.mark.parametrize("question", ["Any relevant skills for any of these candidates?",
                                      "Any relevant skillset for all of these candidates?"])
def test_any_of_these_candidates_checks_all_ten_despite_prior_single_selection(question):
    current = snapshot()
    answer = service().answer(current, question, context=prior(current, ["CV-1"]))
    assert answer["context"]["selected_candidates"] == [f"CV-{rank}" for rank in range(1, 11)]
    assert "Candidate 1" in answer["text"] and "Python" in answer["text"]
    assert "Candidate 2" in answer["text"] and "SQL" in answer["text"]
    assert answer["generation_mode"] == "controlled"
    assert "do not have" not in answer["text"] and "lacks" not in answer["text"]
    owners = {citation["owner"] for citation in answer["citations"]}
    assert {current.jd_id, "CV-1", "CV-2"} <= owners


@pytest.mark.parametrize("question", ["Any relevant skills for candidate 1?",
                                      "Any relevant skills for candidate 1 among any of these candidates?"])
def test_explicit_candidate_reference_does_not_widen_to_review(question):
    current = snapshot()
    answer = service().answer(current, question, prior(current, ["CV-2"]))
    assert answer["context"]["selected_candidates"] == ["CV-1"]
    assert "Python" in answer["text"]
    assert {citation["owner"] for citation in answer["citations"]} <= {current.jd_id, "CV-1"}


@pytest.mark.parametrize("question", ["What are their relevant skills?", "Any relevant skills for these two?"])
def test_plural_skill_followup_keeps_active_pair(question):
    current = snapshot()
    answer = service().answer(current, question, prior(current, ["CV-1", "CV-2"]))
    assert answer["context"]["selected_candidates"] == ["CV-1", "CV-2"]
    assert "Python" in answer["text"] and "SQL" in answer["text"]


@pytest.mark.parametrize("question", ["Any relevant skills for candidate 11?",
                                      "Any relevant skills for candidate 11 among any of these candidates?"])
def test_unknown_explicit_rank_asks_clarification_without_widening(question):
    current = snapshot()
    answer = service().answer(current, question, prior(current, ["CV-1"]))
    assert answer["status"] == "clarification"
    assert not answer["citations"]


def test_skill_overlap_citations_slice_current_jd_and_selected_cv():
    current = snapshot()
    answer = service().answer(current, "Any relevant skills for candidates 1 and 2?")
    sources = {current.jd_id: current.jd_text, **{row["cv_id"]: row["preview_text"] for row in current.candidates}}
    for citation in answer["citations"]:
        text = sources[citation["owner"]]
        assert citation["quote"] == text[citation["start"]:citation["end"]]
        assert citation["document_digest"] == hashlib.sha256(text.encode()).hexdigest()
        assert citation["snapshot_id"] == current.snapshot_id


def test_skill_specific_duration_question_keeps_focused_reader_path(monkeypatch):
    current, engine, calls = snapshot(), service(), []
    question = "How many years of relevant Python skills does candidate 1 have?"

    def focused(snapshot, value, context=None):
        calls.append(value)
        return {"text": "Please specify the relevant employment period.", "status": "clarification",
                "citations": [], "claims": [], "comparison": [], "context": prior(snapshot, ["CV-1"])}

    monkeypatch.setattr(engine.reader, "answer", focused)
    answer = engine.answer(current, question)
    assert calls == [question]
    assert answer["status"] == "clarification"
