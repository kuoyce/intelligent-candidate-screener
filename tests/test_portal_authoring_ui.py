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
