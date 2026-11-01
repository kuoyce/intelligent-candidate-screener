"""Production shell tests use private synthetic CSV fixtures, never real sources."""
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/candidate_screener/app/streamlit_app.py"
TOKEN = "synthetic-login-token-0000000000"


@pytest.fixture(autouse=True)
def local_workspace(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from portal_model_stubs import stub_portal_models
    stub_portal_models(monkeypatch)
    from dataclasses import replace
    from candidate_screener.app import portal_ui as ui
    from candidate_screener.app.portal_auth import create_account
    from candidate_screener.app.portal_data import PortalDataset, synthetic_demo
    accounts = tmp_path / "accounts.json"
    create_account(accounts, "reviewer", TOKEN, ["benchmark"])
    monkeypatch.setenv("HR_PORTAL_ACCOUNTS", str(accounts))
    fixture = replace(synthetic_demo(), is_demo=False)
    monkeypatch.setattr(PortalDataset, "from_folder", lambda *a, **k: fixture)
    monkeypatch.setattr(ui, "PROCESSED", tmp_path)
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    # The app script defaults to the live adapter; empty values keep UI tests offline.
    monkeypatch.setenv("HR_PORTAL_RANKING_ADAPTER", "")
    monkeypatch.setenv("HR_PORTAL_CANDIDATE_POOL", "")


def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(SCRIPT), default_timeout=30).run()
    assert not at.exception
    return at


def select(at, label, value):
    next(s for s in at.selectbox if s.label == label).select(value).run()
    assert not at.exception
    return at


def open_demo(at):
    # Synthetic fixtures are injected internally; no public demo switch exists.
    next(field for field in at.text_input if field.label == "Login ID").set_value("reviewer")
    next(field for field in at.text_input if field.label == "Access token").set_value(TOKEN)
    next(button for button in at.button if button.label == "Sign in").click().run()
    assert not at.exception
    return at


def test_primary_demo_has_chat_after_job_selection_and_no_uploads():
    at = app()
    assert not at.exception and not at.chat_input
    assert not any(s.label == "Workspace" for s in at.selectbox)
    assert not at.get("file_uploader") and not at.get("camera_input")
    open_demo(at)
    assert at.dataframe
    assert len(at.chat_input) == 1
    assert at.session_state["portal_principal"].auth_mode == "local_token"


def test_chat_and_dropdown_share_saved_review_context(monkeypatch):
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    at = open_demo(app())
    at.chat_input[0].set_value("Compare candidates 1 and 2 on Python").run()
    assert not at.exception
    assert len(at.chat_input) == 1
    assert any(s.label == "Review action" for s in at.selectbox)
    assert any("Compare candidates 1 and 2" in message.value for message in at.markdown)
    # The dropdown context changes through Open CV, which records the selected applicant.
    select(at, "Review action", "Open CV")
    preview = next(s for s in at.selectbox if s.label == "Which CV would you like to open?")
    preview.select_index(2).run()
    assert not at.exception
    at.chat_input[0].set_value("What about candidate 2?").run()
    assert not at.exception
    assert any("What about candidate 2?" in message.value for message in at.markdown)


def test_report_and_session_resume_survive_reruns():
    at = open_demo(app())
    select(at, "Review action", "Generate session report")
    assert any("saved" in message.value.lower() for message in at.success)
    next(r for r in at.radio if r.label == "Navigation").set_value("Saved reviews").run()
    menu = next(s for s in at.selectbox if s.label == "Which saved review?")
    assert len(menu.options) >= 2
    menu.select(menu.options[1]).run()
    assert not at.exception


def test_company_workspace_does_not_load_real_data_before_login(monkeypatch, tmp_path):
    monkeypatch.setenv("HR_PORTAL_ACCOUNTS", str(tmp_path / "no-accounts.json"))
    from candidate_screener.app.portal_data import PortalDataset
    original = PortalDataset.from_folder
    def guarded(*args, **kwargs):
        if "_frames" in kwargs:
            return original(*args, **kwargs)
        raise AssertionError("Unauthorized real load")
    monkeypatch.setattr(PortalDataset, "from_folder", guarded)
    at = app()
    assert not any(s.label == "Workspace" for s in at.selectbox)
    assert not at.exception
    assert not any(s.label == "Select role" for s in at.selectbox)
    assert any("not configured" in message.value for message in at.info)


def test_company_mail_and_notifications_are_not_fabricated():
    at = open_demo(app())
    select(at, "Review action", "Send to hiring department")
    assert any("captured locally" in message.value for message in at.info)
    assert not any(button.label == "Send to demo outbox" for button in at.button)
    next(r for r in at.radio if r.label == "Navigation").set_value("Alerts").run()
    assert not at.exception


def test_explicit_local_token_login_opens_only_authorized_fixture(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from dataclasses import replace
    from candidate_screener.app.portal_auth import create_account
    from candidate_screener.app.portal_data import PortalDataset, synthetic_demo
    accounts = tmp_path / "accounts.json"
    token = "synthetic-login-token-0000000000"
    create_account(accounts, "other-reviewer", token, ["benchmark"])
    monkeypatch.setenv("HR_PORTAL_ACCOUNTS", str(accounts))
    fixture = replace(synthetic_demo(), is_demo=False)
    original = PortalDataset.from_folder
    monkeypatch.setattr(PortalDataset, "from_folder", lambda *a, **k: original(*a, **k) if "_frames" in k else fixture)
    # No real source data or durable production workspace is touched by this test.
    import candidate_screener.app.portal_ui as ui
    monkeypatch.setattr(ui, "PROCESSED", tmp_path)
    at = app()
    next(field for field in at.text_input if field.label == "Login ID").set_value("other-reviewer")
    next(field for field in at.text_input if field.label == "Access token").set_value(token)
    next(button for button in at.button if button.label == "Sign in").click().run()
    assert not at.exception
    assert any(menu.label == "Review action" for menu in at.selectbox)
    assert at.session_state["portal_principal"].auth_mode == "local_token"
    assert not any(field.label == "Access token" for field in at.text_input)


def test_sign_out_clears_active_review_without_rerun_loop():
    at = open_demo(app())
    assert "Sign out" not in next(s for s in at.radio if s.label == "Navigation").options
    next(b for b in at.sidebar.button if b.label == "Log out").click().run()
    assert not at.exception
    assert "portal_principal" not in at.session_state
    assert "portal_session_id" not in at.session_state


@pytest.mark.parametrize("failure", ["storage", "revision", "sqlite"])
def test_failed_chat_requires_explicit_retry_and_reuses_submission(monkeypatch, failure):
    from candidate_screener.app.portal_controller import PortalController
    from candidate_screener.app.portal_store import RevisionConflict
    import sqlite3
    original = PortalController.chat
    calls = []
    def flaky(self, session_id, question, turn_id, provider=None):
        calls.append((question, turn_id))
        if len(calls) == 1:
            raise {"storage": OSError, "revision": RevisionConflict, "sqlite": sqlite3.OperationalError}[failure]("temporary storage outage")
        return original(self, session_id, question, turn_id, provider=provider)
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    monkeypatch.setattr(PortalController, "chat", flaky)
    at = open_demo(app())
    at.chat_input[0].set_value("What does this role require?").run()
    assert not at.exception
    assert len(calls) == 1
    assert any(b.label == "Retry question" for b in at.button)
    assert any(s.label == "Review action" for s in at.selectbox)
    at.run()
    assert len(calls) == 1
    next(b for b in at.button if b.label == "Retry question").click().run()
    assert not at.exception
    assert len(calls) == 2 and calls[0] == calls[1]
    assert not any(b.label == "Retry question" for b in at.button)


def test_chat_provenance_reports_model_fallback(monkeypatch):
    import candidate_screener.app.portal_ui as ui
    captured = []
    monkeypatch.setattr(ui.st, "write", lambda text: captured.append(text))
    monkeypatch.setattr(ui.st, "caption", lambda text: captured.append(text))
    ui._chat_answer({"text": "Source evidence", "mode": "deterministic_evidence", "status": "model_fallback"})
    assert any("failed" in value.lower() for value in captured)


def test_primary_extractive_chat_does_not_instantiate_configured_qwen(monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v1")  # v1 behaviour; v3 is the default
    from candidate_screener.llm.base import ProviderUnavailable
    from candidate_screener.llm import qwen, qwen_assets
    at = open_demo(app())
    settings = {"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
                "SCREENER_QWEN_MODEL": "synthetic-qwen", "SCREENER_QWEN_REVISION": "fixture-revision",
                "SCREENER_QWEN_TOKENIZER": "service", "SCREENER_QWEN_RUNTIME": "fixture",
                "SCREENER_QWEN_QUANTIZATION": "none"}
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "false")
    monkeypatch.setattr(qwen_assets, "local_settings", lambda: settings)
    class Offline:
        def __init__(self, config):
            raise AssertionError("Primary extractive chat must not create Qwen clients")
        def count_tokens(self, messages):
            raise ProviderUnavailable("Synthetic offline service")
    monkeypatch.setattr(qwen, "QwenProvider", Offline)
    at.run()
    assert not at.exception
    assert any("RoBERTa-SQuAD2" in item.value for item in at.caption)


def test_empty_evidence_does_not_claim_quoted_extractive_answer(monkeypatch):
    import candidate_screener.app.portal_ui as ui
    captions = []
    monkeypatch.setattr(ui.st, "write", lambda text: None)
    monkeypatch.setattr(ui.st, "caption", captions.append)
    ui._chat_answer({"text": "No evidence", "mode": "deterministic_evidence", "status": "insufficient_evidence", "citations": []})
    assert any("no matching source evidence" in value.lower() for value in captions)


def test_local_qwen_settings_do_not_control_extractive_chat(monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v1")  # v1 behaviour; v3 is the default
    from candidate_screener.llm import qwen, qwen_assets
    at = open_demo(app())
    settings = {"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
                "SCREENER_QWEN_MODEL": "synthetic-qwen", "SCREENER_QWEN_REVISION": "fixture-revision",
                "SCREENER_QWEN_TOKENIZER": "/tmp/fixture-local-tokenizer", "SCREENER_QWEN_RUNTIME": "fixture",
                "SCREENER_QWEN_QUANTIZATION": "none"}
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "false")
    monkeypatch.setattr(qwen_assets, "local_settings", lambda: settings)
    class LocalOnly:
        def __init__(self, config):
            self.config = config
        def count_tokens(self, messages):
            raise AssertionError("Local tokenization cannot prove service availability")
    monkeypatch.setattr(qwen, "QwenProvider", LocalOnly)
    at.run()
    assert not at.exception
    assert any("RoBERTa-SQuAD2" in item.value for item in at.caption)
    assert not any("Qwen connected" in item.value for item in at.caption)


def test_dropdown_selection_is_used_by_chat_followup(monkeypatch):
    from candidate_screener.app.portal_controller import PortalController
    captured = []
    original = PortalController.chat
    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    monkeypatch.setattr(PortalController, "chat", capture)
    at = open_demo(app())
    select(at, "Review action", "Open CV")
    preview = next(s for s in at.selectbox if s.label == "Which CV would you like to open?")
    preview.select_index(3).run()
    selected = next(s for s in at.selectbox if s.label == "Which CV would you like to open?").value
    assert selected is not None
    at.chat_input[0].set_value("What are their skills?").run()
    assert not at.exception
    assert captured[-1]["context"]["selected_candidates"] == [selected]


def test_pending_followup_is_not_rebound_after_dropdown_change(monkeypatch):
    from candidate_screener.app.portal_controller import PortalController
    calls = []
    def unavailable(self, *args, **kwargs):
        calls.append(args)
        raise OSError("temporary outage")
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    monkeypatch.setattr(PortalController, "chat", unavailable)
    at = open_demo(app())
    at.chat_input[0].set_value("What about their skills?").run()
    assert any(b.label == "Retry question" for b in at.button)
    select(at, "Review action", "Open CV")
    preview = next(s for s in at.selectbox if s.label == "Which CV would you like to open?")
    preview.select_index(2).run()
    assert not at.exception
    assert len(calls) == 1
    assert not any(b.label == "Retry question" for b in at.button)
    assert any("context changed" in warning.value for warning in at.warning)


def test_saved_turn_after_lost_confirmation_is_displayed_once(monkeypatch):
    from candidate_screener.app.portal_controller import PortalController
    original = PortalController.chat
    calls = []
    def lost_confirmation(self, *args, **kwargs):
        calls.append(args)
        original(self, *args, **kwargs)
        raise OSError("confirmation lost")
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    monkeypatch.setattr(PortalController, "chat", lost_confirmation)
    at = open_demo(app())
    question = "What does this role require?"
    at.chat_input[0].set_value(question).run()
    assert not at.exception and len(calls) == 1
    assert not at.warning
    assert not any(b.label == "Retry question" for b in at.button)
    assert sum(message.value == question for message in at.markdown) == 1


def test_chat_source_identifiers_survive_display_redaction(monkeypatch):
    from contextlib import nullcontext
    import candidate_screener.app.portal_ui as ui
    labels = []
    monkeypatch.setattr(ui.st, "write", lambda *args: None)
    monkeypatch.setattr(ui.st, "caption", lambda *args: None)
    monkeypatch.setattr(ui.st, "text", lambda *args: None)
    monkeypatch.setattr(ui.st, "expander", lambda label, **kwargs: labels.append(label) or nullcontext())
    owner = "a2-cv-12345678901234567890"
    evidence_id = "qa:" + "1" * 24
    ui._chat_answer({"text": "Evidence", "citations": [{"owner": owner, "evidence_id": evidence_id,
                      "quote": "Python", "start": 0, "end": 6}]})
    assert labels == [f"Supporting source: {owner} · {evidence_id}"]


def test_chat_answers_question_before_collapsed_supporting_sources(monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v1")  # v1 behaviour; v3 is the default
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    at = open_demo(app())
    at.chat_input[0].set_value("What Python skills does candidate 1 have?").run()
    assert not at.exception
    replies = [message for message in at.chat_message if message.name == "assistant"]
    answer = next(message for message in replies if message.expander)
    visible_text = " ".join(item.value for item in answer.markdown)
    assert "Python" in visible_text and "software testing" in visible_text
    assert all(not source.proto.expanded for source in answer.expander)
    assert any(menu.label == "Review action" for menu in at.selectbox)


def test_supporting_source_is_collapsed_by_default(monkeypatch):
    from contextlib import nullcontext
    import candidate_screener.app.portal_ui as ui
    expanded = []
    monkeypatch.setattr(ui.st, "write", lambda *args: None)
    monkeypatch.setattr(ui.st, "caption", lambda *args: None)
    monkeypatch.setattr(ui.st, "text", lambda *args: None)
    monkeypatch.setattr(ui.st, "expander", lambda label, **kwargs: expanded.append(kwargs["expanded"]) or nullcontext())
    ui._chat_answer({"text": "Python development is stated.", "citations": [{"owner": "synthetic-cv-1-1",
                    "evidence_id": "qa:" + "a" * 24, "quote": "Python development", "start": 0, "end": 18}]})
    assert expanded == [False]
