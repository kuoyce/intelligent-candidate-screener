"""S10: with no accounts the Hiring Desk offers to create the first one; otherwise sign-in."""
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/candidate_screener/app/streamlit_app.py"
TOKEN = "synthetic-first-account-token-000"


@pytest.fixture
def accounts(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from portal_model_stubs import stub_portal_models
    stub_portal_models(monkeypatch)
    from dataclasses import replace
    from candidate_screener.app import portal_ui as ui
    from candidate_screener.app.portal_data import PortalDataset, synthetic_demo
    path = tmp_path / "hr-portal/accounts.json"
    monkeypatch.setenv("HR_PORTAL_ACCOUNTS", str(path))
    fixture = replace(synthetic_demo(), is_demo=False)
    monkeypatch.setattr(PortalDataset, "from_folder", lambda *a, **k: fixture)
    monkeypatch.setattr(ui, "PROCESSED", tmp_path)
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    # Empty values keep the UI tests offline (the app script defaults to the live adapter).
    monkeypatch.setenv("HR_PORTAL_RANKING_ADAPTER", "")
    monkeypatch.setenv("HR_PORTAL_CANDIDATE_POOL", "")
    return path


def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(SCRIPT), default_timeout=30).run()
    assert not at.exception
    return at


def field(at, label):
    return next(f for f in at.text_input if f.label == label)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def fill_create_form(at, login="first", departments="benchmark", token=TOKEN, confirmation=TOKEN):
    field(at, "Login ID").set_value(login)
    field(at, "Departments (comma-separated)").set_value(departments)
    field(at, "Access token").set_value(token)
    field(at, "Confirm access token").set_value(confirmation)
    button(at, "Create the first account").click().run()
    assert not at.exception
    return at


def test_create_screen_shows_only_when_no_accounts_exist(accounts):
    at = app()
    assert any(b.label == "Create the first account" for b in at.button)
    assert not any(b.label == "Sign in" for b in at.button)
    from streamlit.proto.TextInput_pb2 import TextInput
    assert all(field(at, label).proto.type == TextInput.PASSWORD
               for label in ("Access token", "Confirm access token"))
    assert field(at, "Login ID").proto.type != TextInput.PASSWORD
    days = next(n for n in at.number_input if n.label == "Account expiry (days)")
    assert days.value == 40 and days.max == 40
    from candidate_screener.app.portal_auth import create_account
    create_account(accounts, "existing", TOKEN, ["benchmark"])
    at = app()
    assert any(b.label == "Sign in" for b in at.button)
    assert not any(b.label == "Create the first account" for b in at.button)
    assert not any(f.label == "Confirm access token" for f in at.text_input)


def test_created_first_account_signs_in(accounts):
    at = fill_create_form(app())
    assert list(json.loads(accounts.read_text())["users"]) == ["first"]
    assert not any(b.label == "Create the first account" for b in at.button)
    assert any("created" in message.value for message in at.success)
    field(at, "Login ID").set_value("first")
    field(at, "Access token").set_value(TOKEN)
    button(at, "Sign in").click().run()
    assert not at.exception
    assert any(menu.label == "Review action" for menu in at.selectbox)
    assert at.session_state["portal_principal"].subject == "first"


@pytest.mark.parametrize("token,confirmation", [(TOKEN, TOKEN + "x"), ("short-token", "short-token")])
def test_mismatched_or_short_token_creates_nothing(accounts, token, confirmation):
    at = fill_create_form(app(), token=token, confirmation=confirmation)
    assert at.error
    assert not accounts.exists()
    assert any(b.label == "Create the first account" for b in at.button)
