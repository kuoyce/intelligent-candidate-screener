"""V2's real controller/UI happy flow, with explicit offline model substitutes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest


@pytest.mark.parametrize("count", [0, 1])
def test_trimmed_review_menu_does_not_block_chat_for_small_candidate_sets(workspace, count):
    dataset = workspace.dataset
    snapshots = json.loads(dataset._snapshots_json)
    for record in snapshots.values():
        payload = json.loads(record["_payload_json"])
        payload["candidates"] = payload["candidates"][:count]
        record["_payload_json"] = json.dumps(payload)
        record["digest"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    workspace.dataset = replace(dataset, _snapshots_json=json.dumps(snapshots))
    at = app()
    assert at.chat_input
    assert not any("unavailable" in item.value for item in at.error)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from portal_model_stubs import stub_portal_models
    stub_portal_models(monkeypatch)
    from candidate_screener.app.portal_controller import PortalController
    from candidate_screener.app.portal_data import synthetic_demo
    from candidate_screener.app.portal_store import PortalStore, Principal
    from candidate_screener.app import portal_ui as ui
    original = synthetic_demo()
    snapshots = json.loads(original._snapshots_json)
    for snapshot in snapshots.values():
        payload = json.loads(snapshot["_payload_json"])
        payload["jd_text"] += " Python programming skills required. Bachelor degree required."
        payload["candidates"][0]["preview_text"] += " Completed Bachelor degree."
        payload["candidates"][1]["preview_text"] += " Master degree in progress."
        snapshot["_payload_json"] = json.dumps(payload)
        snapshot["digest"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    dataset = replace(original, is_demo=False, _snapshots_json=json.dumps(snapshots))
    principal = Principal("reviewer", ("benchmark",), time.time() + 86400, "local_token")
    controller = PortalController(dataset, PortalStore(tmp_path / "workspace.sqlite"), principal,
                                  auth_guard=lambda: True)
    monkeypatch.setattr(ui, "_controller", lambda *args: controller)
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v2")
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    return controller


def app():
    from streamlit.testing.v1 import AppTest
    script = Path(__file__).resolve().parents[1] / "src/candidate_screener/app/streamlit_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert not at.exception
    return at


def test_v2_ui_comparison_followup_and_exact_citation_reopening(workspace):
    from candidate_screener.app.conversation_service import ConversationService
    at = app()
    assert isinstance(workspace.qa_service, ConversationService)
    assert workspace.qa_service.composer is None
    assert any("Hiring Desk v2" in caption.value for caption in at.caption)
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    session = at.session_state["portal_session_id"]
    first = workspace.chat_history(session)[-1]["answer"]
    assert first["mode"] == "conversation_v2"
    assert first["generation_mode"] == "controlled"
    assert first["candidate_labels"]
    tables = [item.value for item in at.dataframe if {"Candidate 1", "Candidate 2"} <= set(item.value.columns)]
    assert tables, "Chat's comparison table must use friendly candidate labels"
    assert not any(column.startswith("synthetic-cv") for table in tables for column in table.columns)

    at.chat_input[0].set_value("What about their qualifications?").run()
    assert not at.exception
    second = workspace.chat_history(session)[-1]["answer"]
    assert second["status"] != "clarification"
    assert second["context"]["selected_candidates"] == first["context"]["selected_candidates"]
    assert second["context"]["category"] == "DEGREE"
    assert "progress" in second["text"]
    assert "Completed Bachelor degree" in second["text"]

    at.chat_input[0].set_value("Why is 1 better than 2?").run()
    assert not at.exception
    explained = workspace.chat_history(session)[-1]["answer"]
    assert explained["context"]["intent"] == "comparison_explanation"
    assert explained["show_comparison_table"] is True
    assert len(explained["comparison"]) == len(second["comparison"])
    for after, before in zip(explained["comparison"], second["comparison"], strict=True):
        assert after["category"] == before["category"]
        assert after["requirement_evidence_ids"] == before["requirement_evidence_ids"]
        for owner, cell in after["answers"].items():
            assert cell["context"] == before["answers"][owner]["context"]
            assert cell["evidence_ids"] == before["answers"][owner]["evidence_ids"]
    assert any({"Candidate 1", "Candidate 2"} <= set(item.value.columns) for item in at.dataframe)

    at.chat_input[0].set_value("Where does it say the Master degree is in progress?").run()
    assert not at.exception
    reopened = workspace.chat_history(session)[-1]["answer"]
    assert reopened["generation_mode"] == "citation_reopen"
    assert reopened["citations"]
    assert all(item in explained["citations"] for item in reopened["citations"])
    snapshot = workspace.snapshot(session)
    sources = {row["cv_id"]: row["preview_text"] for row in snapshot.candidates}
    for citation in reopened["citations"]:
        assert citation["quote"] == sources[citation["owner"]][citation["start"]:citation["end"]]
    assert any("Master degree in progress" in item.value for item in at.text)


def test_v1_flag_keeps_extract_service_and_original_chat_mode(workspace, monkeypatch):
    from candidate_screener.app.portal_extractive import PortalExtractiveService
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v1")
    at = app()
    assert isinstance(workspace.qa_service, PortalExtractiveService)
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    result = workspace.chat_history(at.session_state["portal_session_id"])[-1]["answer"]
    assert result["mode"] == "comparison"
    assert any("Extractive QA: RoBERTa" in caption.value for caption in at.caption)


def test_v2_controller_preserves_rank_order_and_duplicate_turn_identity(workspace):
    from candidate_screener.app.portal_workspace import attach_services
    attach_services(workspace)
    job = workspace.jobs()[0]["jd_id"]
    session = workspace.open_job(job)["session_id"]
    before = [(row["cv_id"], row["rank"], row["score"]) for row in workspace.snapshot(session).candidates]
    first = workspace.chat(session, "Compare candidates 1 and 2", "stable-turn")
    duplicate = workspace.chat(session, "Compare candidates 1 and 2", "stable-turn")
    assert duplicate == first
    assert len(workspace.chat_history(session)) == 1
    after = [(row["cv_id"], row["rank"], row["score"]) for row in workspace.snapshot(session).candidates]
    assert before == after
