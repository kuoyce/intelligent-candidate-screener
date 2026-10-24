"""Session changes and Streamlit integration use synthetic text/model doubles."""
from pathlib import Path
import json

import pytest

from candidate_screener.app.session import SCREENING_KEYS, input_key, invalidate


def test_input_change_clears_snapshot_results_and_followups_but_not_other_session():
    old = input_key("Python", "Engineer", "CV", ["Python"], {"shortlist": 50})
    first = {"screening_input_key": old, **{key: "old" for key in SCREENING_KEYS}}
    second = dict(first)
    new = input_key("SQL", "Engineer", "CV", ["Python"], {"shortlist": 50})
    assert invalidate(first, new)
    assert not set(SCREENING_KEYS) & set(first)
    assert second["snapshot"] == "old"
    assert not invalidate(second, old)
    assert second["chat_state"] == "old"


@pytest.mark.parametrize("field", ["jd", "title", "source", "documents", "configuration"])
def test_every_scoring_input_has_an_invalidation_key(field):
    values = {"jd": "Python", "title": "Engineer", "source": "CV",
              "documents": [b"Python"], "configuration": {"shortlist": 50}}
    def key(v):
        return input_key(v["jd"], v["title"], v["source"], v["documents"], v["configuration"])
    previous = key(values)
    values[field] = [b"SQL"] if field == "documents" else ({"shortlist": 1} if field == "configuration" else "SQL")
    assert key(values) != previous


def app(monkeypatch):
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    testing = pytest.importorskip("streamlit.testing.v1")
    from candidate_screener.extraction.live import GlinerEntityExtractor
    monkeypatch.setattr(GlinerEntityExtractor, "availability", lambda self: {"status": "unavailable"})
    monkeypatch.setattr(GlinerEntityExtractor, "__call__", lambda self, text, side: {
        "status": "unavailable", "entities": [], "provenance": {}})
    for key in ("SCREENER_CE_MANIFEST", "SCREENER_QWEN_BASE_URL", "SCREENER_QWEN_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] /
        "src/candidate_screener/app/legacy_streamlit_app.py"), default_timeout=30).run()


def demo(at):
    next(b for b in at.button if b.label == "Screen demonstration").click().run()
    assert not at.exception
    return at


def test_ui_input_change_invalidates_chat_and_hides_old_rankings(monkeypatch):
    at = demo(app(monkeypatch))
    at.chat_input[0].set_value("explain 1").run()
    assert at.session_state["chat"]
    next(t for t in at.text_area if t.label == "Job description").set_value("New SQL role").run()
    assert not at.exception
    assert "snapshot" not in at.session_state
    assert "chat" not in at.session_state
    assert at.chat_input[0].disabled


def test_unconfigured_ce_cannot_screen_with_silent_tfidf_replacement(monkeypatch):
    at = app(monkeypatch)
    next(s for s in at.selectbox if s.label == "Screening backend").select("Configured cross-encoder").run()
    next(b for b in at.button if b.label == "Screen demonstration").click().run()
    assert not at.exception
    assert "snapshot" not in at.session_state
    assert any("unavailable" in error.value for error in at.error)


def test_clear_session_and_separate_app_sessions_have_no_documents_in_common(monkeypatch):
    first = demo(app(monkeypatch))
    second = app(monkeypatch)
    assert "snapshot" not in second.session_state
    first.chat_input[0].set_value("compare 1 and 2").run()
    next(b for b in first.button if b.label == "Clear session").click().run()
    assert not first.exception
    assert "snapshot" not in first.session_state and "chat" not in first.session_state
    assert "chat_state" not in second.session_state


def test_ui_freeform_question_uses_qwen_and_validates_source_references(monkeypatch):
    # Uses the actual service orchestration, with provisioned model inference doubled.
    from candidate_screener.llm import qwen
    at = app(monkeypatch)
    configuration = {"BASE_URL": "http://127.0.0.1:8000/v1", "MODEL": "synthetic-Qwen",
                     "REVISION": "test", "TOKENIZER": "/not-downloaded", "RUNTIME": "test",
                     "QUANTIZATION": "none"}
    for name, value in configuration.items():
        monkeypatch.setenv("SCREENER_QWEN_" + name, value)
    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "false")

    class Provider:
        max_context_tokens = 10000
        max_output_tokens = 512

        def __init__(self, config):
            self.config = config
            self.calls = []

        def count_tokens(self, messages):
            return sum(len(m["content"]) for m in messages) // 4

        def complete(self, messages):
            payload = json.loads(messages[-1]["content"])
            self.calls.append(payload)
            if "passages" not in payload:
                return json.dumps({"kind": "explain", "args": ["1"]})
            return json.dumps({"snapshot_id": payload["snapshot_id"],
                               "passage_ids": [p["passage_id"] for p in payload["passages"]]})

    monkeypatch.setattr(qwen, "QwenProvider", Provider)
    at.run()
    demo(at)
    at.chat_input[0].set_value("Describe the strongest applicant's evidence in this pool").run()
    assert not at.exception
    assert not at.session_state["last_answer"].fallback, at.session_state["qwen"].calls
    assert at.session_state["last_answer"].citations
    assert "passage:" in at.session_state["chat"][-1][1]


def test_evidence_filter_requires_apply_and_preserves_candidate_identity(monkeypatch):
    at = demo(app(monkeypatch))
    old = at.session_state["snapshot"].snapshot_id
    at.chat_input[0].set_value("Only consider candidates with evidence of SQL").run()
    assert not at.exception
    assert at.session_state["snapshot"].snapshot_id == old
    answer = at.session_state["last_answer"]
    assert answer.status == "filter_preview"
    expected = set(answer.candidate_ids)
    assert expected
    next(b for b in at.button if b.label == "Apply evidence filter and screen again").click().run()
    assert not at.exception
    assert at.session_state["snapshot"].snapshot_id != old
    assert {str(r.cv_id) for r in at.session_state["snapshot"].results} == expected
    assert at.session_state["chat_state"].candidate_ids == ()
