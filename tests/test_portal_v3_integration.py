"""V3 transport boundaries use synthetic authorized workspaces only."""
from dataclasses import replace
import time

import pytest

from candidate_screener.app.portal_controller import PortalController, StaleReview
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_store import PortalStore, Principal


@pytest.fixture
def controller(tmp_path):
    principal = Principal("reviewer", ("demo",), time.time() + 3600, "demo")
    return PortalController(synthetic_demo(), PortalStore(tmp_path / "store.sqlite"), principal)


class HistoryService:
    accepts_history = True

    def __init__(self):
        self.received = []

    def answer(self, snapshot, question, context=None, *, history=None):
        self.received.append(history)
        return {"text": "Let's review the role.", "context": {"snapshot_id": snapshot.snapshot_id},
                "mode": "conversation_v3"}


def test_v3_history_is_explicit_current_session_and_duplicate_is_not_generated(controller):
    controller.qa_service = HistoryService()
    session = controller.open_job(controller.jobs()[0]["jd_id"])["session_id"]
    controller.chat(session, "Hello", "one")
    controller.chat(session, "What about this role?", "two")
    controller.chat(session, "What about this role?", "two")
    assert len(controller.qa_service.received) == 2
    assert controller.qa_service.received[0] == []
    assert controller.qa_service.received[1][0]["question"] == "Hello"
    other = controller.open_job(controller.jobs()[0]["jd_id"])["session_id"]
    controller.chat(other, "Hello", "three")
    assert controller.qa_service.received[-1] == []


def test_jd_documents_do_not_contain_candidate_text(controller):
    docs = controller.job_documents()
    assert docs
    assert set(docs[0]) == {"jd_id", "title", "text", "source_version"}
    assert docs[0]["text"] == controller.dataset.snapshot(docs[0]["jd_id"]).jd_text


def test_jd_documents_reject_changed_sources(controller):
    controller.freshness_guard = lambda: False
    with pytest.raises(StaleReview):
        controller.job_documents()


def test_saved_jd_search_rechecks_other_job_sources(controller, monkeypatch):
    docs = [{"jd_id": "synthetic-jd-99", "title": "Mobile developer", "text": "Flutter required.", "source_version": "v1"}]
    monkeypatch.setattr(controller, "job_documents", lambda: docs)
    controller.qa_service = HistoryService()
    session = controller.open_job(controller.jobs()[0]["jd_id"])["session_id"]
    answer = controller.chat(session, "Which JD requires Flutter?", "search")
    assert answer["citations"]
    docs.clear()
    reopened = controller.chat_history(session)[0]["answer"]
    assert reopened["status"] == "stale_sources"
    assert not reopened["citations"]
    assert "Mobile developer" not in str(reopened)
    duplicate = controller.chat(session, "Which JD requires Flutter?", "search")
    assert duplicate == reopened


def test_operational_jd_text_does_not_require_ranking(controller):
    from candidate_screener.app.portal_workspace import attach_services
    attach_services(controller)
    job = controller.job_service.create({"title": "Mobile developer", "department": "Engineering",
        "responsibilities": {"required": ["Build mobile apps"]}, "skills": {"required": ["Flutter"]}}, workspace="demo")
    job = controller.job_service.review(job["job_id"], job["revision"])
    job = controller.job_service.publish(job["job_id"], job["revision"])
    result = controller.search_job_requirements("Which JD requires Flutter?")
    assert any(item["jd_id"] == job["jd_id"] for item in result["matches"])
    controller.qa_service = HistoryService()
    assert controller.ask_job(job["jd_id"], "What skills does this role require?")["mode"] == "conversation_v3"


def test_jd_only_question_does_not_bypass_scope(controller):
    controller.qa_service = HistoryService()
    with pytest.raises(PermissionError):
        controller.ask_job("private-other-job", "What skills are required?")


def test_jd_only_service_runs_without_candidates(controller):
    from candidate_screener.app.conversation_v3 import ConversationV3Service
    from candidate_screener.app.portal_extractive import PortalExtractiveService
    from candidate_screener.app.conversation_smoke import SourceOnlyQA
    controller.qa_service = ConversationV3Service(PortalExtractiveService(qa_client=SourceOnlyQA()))
    answer = controller.ask_job(controller.jobs()[0]["jd_id"], "What skills does this role require?")
    assert answer["mode"] == "conversation_v3"
    assert answer["context"]["selected_candidates"] == []


def test_numeric_v3_digests_survive_persistence(controller):
    session = controller.open_job(controller.jobs()[0]["jd_id"])
    fields = {key: "1" * 64 for key in ("scope_digest", "prompt_sha256", "source_scope_sha256")}
    saved = controller.store.save_session(controller.principal, session["session_id"], session["revision"], fields)
    assert saved["state"] == fields
