"""CSV review remains available in the production shell without a live model."""
from dataclasses import replace
from pathlib import Path
import time

import pytest


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from portal_model_stubs import stub_portal_models
    stub_portal_models(monkeypatch)
    from candidate_screener.app.portal_controller import PortalController
    from candidate_screener.app.portal_data import synthetic_demo
    from candidate_screener.app.portal_store import PortalStore, Principal
    from candidate_screener.app import portal_ui as ui
    dataset = replace(synthetic_demo(), is_demo=False)
    principal = Principal("reviewer", ("benchmark",), time.time() + 86400, "local_token")
    controller = PortalController(dataset, PortalStore(tmp_path / "workspace.sqlite"), principal,
                                  auth_guard=lambda: True)
    monkeypatch.setattr(ui, "_controller", lambda *args: controller)
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    return controller


def app():
    from streamlit.testing.v1 import AppTest
    script = Path(__file__).resolve().parents[1] / "src/candidate_screener/app/streamlit_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert not at.exception
    return at


def test_production_shell_has_no_demo_switch_and_keeps_csv_review(workspace):
    at = app()
    assert not any(item.label == "Workspace" for item in at.selectbox)
    assert any(item.label == "Review action" for item in at.selectbox)
    assert at.chat_input
    assert any(item.label == "Next JD" for item in at.button)
    assert any(item.label == "Log out" for item in at.sidebar.button)
    assert any(item.value == "Full job description" for item in at.subheader)


def test_compare_action_uses_extractive_table_and_never_qwen(workspace, monkeypatch):
    from candidate_screener.app import portal_ui as ui
    monkeypatch.setattr(ui, "_optional_provider", lambda: pytest.fail("Primary review must not invoke Qwen"))
    at = app()
    next(item for item in at.selectbox if item.label == "Review action").select("Compare candidates").run()
    assert not at.exception and at.dataframe
    assert any(item.label == "Refresh extraction" for item in at.button)
    assert any("GLiNER medium" in item.value for item in at.caption)
    at.chat_input[0].set_value("Compare 1 and 2").run()
    assert not at.exception
    result = workspace.chat_history(at.session_state["portal_session_id"])[-1]["answer"]
    assert result["mode"] == "comparison" and result["comparison"]
    assert len(result["context"]["selected_candidates"]) == 2


def test_extractive_fact_ui_displays_validated_span_and_context(workspace):
    from candidate_screener.app.portal_extractive import PortalExtractiveService
    class Qa:
        def answer(self, question, passages):
            passage = passages[0]
            start = passage["quote"].index("Python")
            return {"status":"supported", "model_id":"deepset/roberta-base-squad2",
                    "spans":[{"passage_id":passage["passage_id"],"candidate_id":passage["candidate_id"],
                              "start":start,"end":start+6,"text":"Python"}]}
    workspace.qa_service = PortalExtractiveService(qa_client=Qa())
    at = app()
    at.chat_input[0].set_value("What programming language does candidate 1 use?").run()
    assert not at.exception
    result = workspace.chat_history(at.session_state["portal_session_id"])[-1]["answer"]
    assert result["mode"] == "extractive_qa"
    assert any(claim["text"] == "Python" for claim in result["claims"])
    assert any("RoBERTa-SQuAD2" in item.value for item in at.caption)
    assert not any("JD comparison not requested." in str(item.value) for item in at.dataframe)


@pytest.mark.parametrize("context", ["AWS certification. Valid until 2021.",
                                     "Bachelor of Science. Status: in progress."])
def test_comparison_table_keeps_qualifying_context_visible(context):
    testing = pytest.importorskip("streamlit.testing.v1")
    result = {"comparison": [{"category": "DEGREE", "requirement": "Qualification is not required.",
                              "answers": {"candidate-a": {"text": "Qualification", "context": context}}}]}
    at = testing.AppTest.from_string(
        "from candidate_screener.app.portal_ui import _comparison_table\n"
        f"_comparison_table({result!r})\n").run()
    assert not at.exception
    assert at.dataframe[0].value.loc[0, "Category"] == "Degree"
    assert at.dataframe[0].value.loc[0, "Requirement"] == "Qualification is not required."
    assert context in at.dataframe[0].value.loc[0, "candidate-a"]


def test_switching_csv_jobs_preserves_separate_review_histories(workspace):
    at = app()
    first_session = at.session_state["portal_session_id"]
    at.chat_input[0].set_value("What does this role require?").run()
    next(item for item in at.button if item.label == "Next JD").click().run()
    assert not at.exception
    assert at.session_state["portal_session_id"] != first_session
    assert not workspace.chat_history(at.session_state["portal_session_id"])
    next(item for item in at.button if item.label == "Previous JD").click().run()
    assert at.session_state["portal_session_id"] == first_session
    assert len(workspace.chat_history(first_session)) == 1


def test_guided_jd_publish_and_csv_review_work_without_live_model(workspace):
    at = app()
    next(item for item in at.radio if item.label == "Navigation").set_value("Create new JD").run()
    next(item for item in at.button if item.label == "Start a new JD").click().run()
    assert not at.exception
    for value in ("Platform engineer", "Tech Department", "Build reliable APIs"):
        at.chat_input[0].set_value(value).run()
        assert not at.exception
    job_id = at.session_state["portal_authoring_job"]
    next(item for item in at.button if item.label == "Confirm JD preview").click().run()
    next(item for item in at.button if item.label == "Publish JD").click().run()
    assert not at.exception
    assert workspace.job_service.get(job_id)["status"] == "published"
    assert any("not connected" in item.value for item in at.info)
    assert len(workspace.job_service.events()) == 1
    # Starting another draft must not reopen the selected previous JD on rerun.
    next(item for item in at.button if item.label == "Start a new JD").click().run()
    assert not at.exception
    assert at.session_state["portal_authoring_job"] != job_id
    next(item for item in at.radio if item.label == "Navigation").set_value("Review jobs").run()
    assert not at.exception and at.chat_input
    assert at.session_state["portal_selected_job"] in {job["jd_id"] for job in workspace.dataset.exports.jobs()}


def test_top_ten_email_requires_preview_then_creates_an_alert(workspace):
    at = app()
    next(item for item in at.selectbox if item.label == "Review action").select("Send to hiring department").run()
    assert not any(item.label == "Send to demo outbox" for item in at.button)
    next(item for item in at.button if item.label == "Preview email").click().run()
    assert not at.exception
    next(item for item in at.button if item.label == "Send to demo outbox").click().run()
    assert not at.exception
    assert any("captured" in item.value for item in at.success)
    next(item for item in at.radio if item.label == "Navigation").set_value("Alerts").run()
    assert any("Email Captured" in item.value for item in at.markdown)
    next(item for item in at.button if item.label == "Mark read").click().run()
    assert not at.exception
    assert next(item for item in at.button if item.label == "Mark unread")


def test_review_another_job_returns_to_cards_without_widget_state_error(workspace):
    at = app()
    next(item for item in at.selectbox if item.label == "Review action").select("Review another job").run()
    assert not at.exception
    assert not at.chat_input
    assert any("Choose a JD card" in item.value for item in at.info)
    next(item for item in at.button if item.label == "Review this JD").click().run()
    assert not at.exception and at.chat_input


def test_new_jd_live_completion_opens_same_review_chat_flow(workspace):
    from candidate_screener.app.portal_ranking import RankingService
    from candidate_screener.app.portal_workspace import attach_services
    from test_portal_workspace import ready_adapter
    attach_services(workspace)
    job = workspace.job_service.create({"title": "New platform role", "department": "Tech", "responsibilities": ["Build services"]})
    workspace.job_service.review(job["job_id"], job["revision"])
    workspace.job_service.publish(job["job_id"], job["revision"])
    ranking = RankingService(workspace.store.path, workspace.principal, guard=workspace._guard, adapter=ready_adapter, pool_ref="fixture-pool")
    workspace.ranking_service = workspace.dataset.ranking_service = ranking
    at = app()
    next(item for item in at.button if item.key == "portal_card_" + job["job_id"]).click().run()
    assert not at.chat_input
    next(item for item in at.button if item.label == "Rank candidates").click().run()
    assert not at.exception and at.chat_input
    assert workspace.snapshot(at.session_state["portal_session_id"]).jd_id == job["job_id"]
    assert "Compare candidates" not in next(item for item in at.selectbox if item.label == "Review action").options


def test_saved_review_selection_survives_chat_and_can_return_to_browser(workspace):
    at = app()
    first = at.session_state["portal_session_id"]
    next(item for item in at.button if item.label == "Next JD").click().run()
    next(item for item in at.radio if item.label == "Navigation").set_value("Saved reviews").run()
    next(item for item in at.selectbox if item.label == "Which saved review?").set_value(first).run()
    assert not at.exception
    at.chat_input[0].set_value("What does this role require?").run()
    assert not at.exception
    assert next(item for item in at.selectbox if item.label == "Which saved review?").value == first
    assert len(workspace.chat_history(first)) == 1
    next(item for item in at.selectbox if item.label == "Review action").select("Review another job").run()
    assert not at.exception
    assert next(item for item in at.radio if item.label == "Navigation").value == "Review jobs"


def test_refreshing_stale_saved_review_opens_one_current_session(workspace, monkeypatch):
    import json
    at = app()
    first = at.session_state["portal_session_id"]
    exports = workspace.dataset.exports
    snapshots = json.loads(exports._snapshots_json)
    selected_job = workspace.load_session(first)["jd_id"]
    snapshots[selected_job]["digest"] = "b" * 64
    workspace.dataset.exports = replace(exports, _snapshots_json=json.dumps(snapshots))
    next(item for item in at.radio if item.label == "Navigation").set_value("Saved reviews").run()
    next(item for item in at.selectbox if item.label == "Which saved review?").set_value(first).run()
    calls = []
    original = workspace.open_job
    def bounded_open(jd_id):
        calls.append(jd_id)
        assert len(calls) == 1, "Recovery must create one current review, without a rerun loop"
        return original(jd_id)
    monkeypatch.setattr(workspace, "open_job", bounded_open)
    next(item for item in at.selectbox if item.label == "Recover review").select("Refresh rankings").run()
    assert not at.exception
    assert len(calls) == 1
    assert at.session_state["portal_session_id"] != first
    assert next(item for item in at.radio if item.label == "Navigation").value == "Review jobs"
    at.run()
    assert not at.exception and len(calls) == 1
