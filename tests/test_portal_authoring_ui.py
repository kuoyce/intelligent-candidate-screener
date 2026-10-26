"""Guided answers preserve the requirement priority the reviewer entered."""
import pytest

pytest.importorskip("streamlit")
from candidate_screener.app.portal_authoring_ui import _answer


def test_unprefixed_requirement_after_preference_remains_required():
    assert _answer("skills", "Preferred: Kubernetes\nPython\nRequired: SQL") == {
        "required": ["Python", "SQL"], "preferred": ["Kubernetes"]}


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "101"])
def test_guided_experience_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        _answer("min_years", value)


def test_onnx_ranking_control_disables_d64_and_ignores_stale_selection(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("HR_PORTAL_EMBEDDING_BACKEND", "onnx")
    script = '''
import streamlit as st
from types import SimpleNamespace
from candidate_screener.app.portal_authoring_ui import _ranking_controls
if "calls" not in st.session_state:
    st.session_state.calls=[]
    st.session_state.portal_rank_d64_job=True
def rank(*args, **kwargs):
    st.session_state.calls.append(kwargs)
    st.stop()
service=SimpleNamespace(availability={"available":True},rank=rank)
controller=SimpleNamespace(ranking_service=service)
job={"job_id":"job","revision":1,"ranking_status":"service_unavailable","active_request_id":None}
_ranking_controls(controller,job)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.checkbox[0].disabled
    assert any("D64 reranker is unavailable" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["calls"] == [{"rerank":False,"system":"D63"}]


POLL_SCRIPT = '''
import streamlit as st
from types import SimpleNamespace
from candidate_screener.app import portal_authoring_ui as ui
from candidate_screener.app.portal_authoring_ui import _ranking_controls
ui.POLL_LIMIT_SECONDS = st.session_state.get("limit", 900)
st.session_state.setdefault("calls", 0)
st.session_state.setdefault("status", "running")
def refresh(request_id):
    st.session_state.calls += 1
    st.session_state.status = st.session_state.get("finish_with", "running")
    return {"status": st.session_state.status}
service = SimpleNamespace(availability={"available": True}, refresh=refresh,
                          get=lambda request_id: {"status": st.session_state.status})
controller = SimpleNamespace(ranking_service=service)
job = {"job_id": "job", "revision": 1, "ranking_status": "running", "active_request_id": "r1"}
_ranking_controls(controller, job)
'''


def poll_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string(POLL_SCRIPT, default_timeout=30)


def test_running_ranking_checks_itself_while_the_page_is_shown():
    pytest.importorskip("streamlit")
    app = poll_app().run()
    assert not app.exception
    assert app.session_state["calls"] == 1
    assert any("every 10 seconds" in item.value for item in app.caption)


def test_polling_stops_after_the_limit_and_restarts_on_the_next_render():
    pytest.importorskip("streamlit")
    app = poll_app()
    app.session_state["limit"] = 0
    app.run()
    assert not app.exception
    assert app.session_state["calls"] == 0, "an expired window must not keep calling the service"
    assert any("Automatic checking stopped" in item.value for item in app.caption)
    assert any(item.label == "Refresh ranking status" for item in app.button)
    app.session_state["limit"] = 900
    app.run()                  # the user returns: a fresh window
    assert not app.exception
    assert app.session_state["calls"] == 1
    assert any("every 10 seconds" in item.value for item in app.caption)
    assert not any("Automatic checking stopped" in item.value for item in app.caption)


def test_finished_ranking_ends_the_polling():
    pytest.importorskip("streamlit")
    app = poll_app()
    app.session_state["finish_with"] = "ready"
    app.run()
    assert not app.exception
    assert app.session_state["calls"] == 1
    assert not any("every 10 seconds" in item.value for item in app.caption)
    assert not any(item.label == "Refresh ranking status" for item in app.button)
