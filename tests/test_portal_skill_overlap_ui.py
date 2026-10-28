"""Real Streamlit relevant-skill lookup with invented source-owned entities."""
from dataclasses import replace
import hashlib
import json

from test_portal_v2_ui import workspace, app
from test_conversation_skill_overlap import Entities


def test_review_skill_overlap_covers_candidates_and_has_owned_supporting_sources(workspace, monkeypatch):
    from candidate_screener.app import portal_entities

    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    monkeypatch.setattr(portal_entities, "PortalEntityService", lambda *args, **kwargs: Entities())
    dataset = workspace.dataset
    snapshots = json.loads(dataset._snapshots_json)
    for record in snapshots.values():
        payload = json.loads(record["_payload_json"])
        payload["jd_text"] += " SQL knowledge required."
        payload["candidates"][1]["preview_text"] += " SQL knowledge."
        record["_payload_json"] = json.dumps(payload)
        record["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    workspace.dataset = replace(dataset, _snapshots_json=json.dumps(snapshots))
    at = app()
    at.chat_input[0].set_value("Summarise why candidate 1 is a good fit to this JD").run()
    assert not at.exception
    session = at.session_state["portal_session_id"]
    assert workspace.chat_history(session)[-1]["answer"]["context"]["selected_candidates"] == [
        workspace.snapshot(session).candidates[0]["cv_id"]]

    at.chat_input[0].set_value("Any relevant skills for any of these candidates?").run()
    assert not at.exception
    result = workspace.chat_history(session)[-1]["answer"]
    assert result["generation_mode"] == "controlled"
    assert result["context"]["selected_candidates"] == [row["cv_id"] for row in workspace.snapshot(session).candidates]
    assert "Candidate 1" in result["text"] and "Candidate 2" in result["text"]
    assert "Python" in result["text"] and "SQL" in result["text"]
    show = [item for item in at.checkbox if item.label == "Show supporting sources"][-1]
    show.check().run()
    assert not at.exception
    assert any(item.label == "View supporting source: The role" for item in at.expander)
    assert any(item.label == "View supporting source: Candidate 2" for item in at.expander)
