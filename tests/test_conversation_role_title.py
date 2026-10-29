"""Role-title questions read the selected JD title, not a requirements heading."""
from types import SimpleNamespace

import pytest

from candidate_screener.app.conversation_v3 import ConversationV3Service
from candidate_screener.app.portal_extractive import PortalExtractiveService
from candidate_screener.app.conversation_context import scoped_history


def snapshot(title="Senior Platform Engineer"):
    return SimpleNamespace(snapshot_id="title-review", jd_id="TITLE-JD", jd_title=title,
                           jd_text="Required experience and skills:\nPython and Linux.", candidates=())


class NeverQA:
    def answer(self, *args):
        raise AssertionError("Title lookup must not call QA")


class NeverComposer:
    def compose_before(self, *args):
        raise AssertionError("Title lookup must not call generation")


@pytest.mark.parametrize("question", [
    "What is the role required for this JD?", "What role does this JD require?",
    "What is the job title?", "What is the position advertised in this job description?",
    "What is the title of the selected JD?",
])
def test_role_title_is_answered_from_saved_jd_title(question):
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()), NeverComposer())
    answer = service.answer(snapshot(), question)
    assert answer["text"] == "The role required for this JD is Senior Platform Engineer."
    assert answer["generation_mode"] == "controlled"
    assert answer["qa_status"] == "not_run"
    assert answer["model"] is None
    assert answer["generation_provenance"]["source_field"] == "jd_title"
    assert answer["citations"] == []  # No fabricated body-text offsets for a metadata field.


@pytest.mark.parametrize("question", [
    "What skills does this role require?", "What does this JD require?",
    "What role does candidate 1 have?", "What is the role required for this JD and its skills?",
])
def test_requirements_and_candidate_roles_are_not_title_lookups(question):
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()))
    assert service._controlled(snapshot(), question, None) is None


@pytest.mark.parametrize("title", ["", None])
def test_missing_title_does_not_invent_one(title):
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()))
    assert service._controlled(snapshot(title), "What role does this JD require?", None) is None


def test_title_question_clears_prior_candidate_evidence_and_selection():
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()))
    current = snapshot()
    current.candidates = ({"cv_id": "C1", "rank": 1, "preview_text": "Python", "preview_status": "available_redacted_text"},)
    context = {"snapshot_id": current.snapshot_id, "jd_id": current.jd_id,
               "selected_candidates": ["C1"], "intent": "candidate_qa", "category": "SKILL",
               "last_question": "What skills does candidate 1 have?", "last_evidence": []}
    answer = service.answer(current, "What is the job title?", context=context)
    assert answer["context"]["selected_candidates"] == []
    assert answer["context"]["last_evidence"] == []
    assert answer["context"]["category"] == "TITLE"


def test_title_change_invalidates_prior_title_discourse():
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()))
    current = snapshot()
    answer = service.answer(current, "What role does this JD require?")
    history = [{"question": "What role does this JD require?", "answer": answer}]
    assert len(scoped_history(current, history)) == 1
    current.jd_title = "Cloud Support Engineer"
    assert scoped_history(current, history) == []


def test_title_is_redacted_before_display():
    service = ConversationV3Service(PortalExtractiveService(qa_client=NeverQA()))
    answer = service.answer(snapshot("Engineer contact synthetic@example.com"), "What is the job title?")
    assert "synthetic@example.com" not in answer["text"]
