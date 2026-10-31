"""Plan 2026-10-31-v3-default: v3 is the default chat; the 4B is the default v3 model."""
import json
from pathlib import Path

import pytest

from test_portal_v1_ui import app, live_job, workspace  # noqa: F401 (fixture)


@pytest.mark.parametrize("value,expected", [(None, "v3"), ("v1", "v1"), ("v2", "v2"), ("v3", "v3"),
                                            ("V3", "v3"), ("V2", "v2"), ("V1", "v1"),
                                            ("v4", "v1"), ("", "v1"), ("extractive", "v1")])
def test_chat_version_defaults_to_v3_and_unknown_values_behave_as_v1(monkeypatch, value, expected):
    from candidate_screener.app.portal_workspace import chat_version
    if value is None:
        monkeypatch.delenv("HR_PORTAL_CHAT_VERSION", raising=False)
    else:
        monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", value)
    assert chat_version() == expected


def test_call_sites_read_the_version_only_through_chat_version():
    src = Path(__file__).resolve().parents[1] / "src/candidate_screener"
    readers = {path.relative_to(src).as_posix() for path in src.rglob("*.py")
               if "HR_PORTAL_CHAT_VERSION" in path.read_text()}
    assert readers == {"app/portal_workspace.py"}, readers
    for name in ("portal_ui.py", "portal_workspace.py"):
        assert "chat_version()" in (src / "app" / name).read_text(), name


@pytest.mark.parametrize("value,service", [(None, "ConversationV3Service"), ("v1", "PortalExtractiveService"),
                                           ("bogus", "PortalExtractiveService"), ("v2", "ConversationService")])
def test_workspace_builds_the_service_of_the_selected_version(workspace, monkeypatch, value, service):
    from candidate_screener.app.portal_workspace import attach_services
    if value is None:
        monkeypatch.delenv("HR_PORTAL_CHAT_VERSION", raising=False)
    else:
        monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", value)
    attach_services(workspace)
    assert type(workspace.qa_service).__name__ == service


def test_unset_version_renders_v3_chat_caption_and_job_requirement_search(workspace, monkeypatch):
    from candidate_screener.app.conversation_v3 import ConversationV3Service
    monkeypatch.delenv("HR_PORTAL_CHAT_VERSION", raising=False)
    at = app()
    assert isinstance(workspace.qa_service, ConversationV3Service)
    assert any(item.value.startswith("Hiring Desk v3") for item in at.caption)
    assert any(item.label == "Find jobs by requirement" for item in at.expander)


def test_unset_version_offers_jd_only_questions_on_an_unranked_operational_job(workspace, monkeypatch):
    monkeypatch.delenv("HR_PORTAL_CHAT_VERSION", raising=False)
    from test_portal_workspace import ready_adapter
    job_id = live_job(workspace, ready_adapter)
    at = app()
    next(item for item in at.button if item.key == "portal_card_" + job_id).click().run()
    assert not at.exception
    assert any(item.label == "Ask about this job description" for item in at.expander)


def test_v1_rollback_hides_the_v3_only_surfaces(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v1")
    at = app()
    assert any("Extractive QA: RoBERTa" in item.value for item in at.caption)
    assert not any(item.label == "Find jobs by requirement" for item in at.expander)


# The composer's default model (Design §2).

def _manifest(path, model="Qwen/Qwen3.5-4B"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"model": model, "revision": "b" * 40,
                                                    "sha256": "a" * 64, "quantization": "Q4_K_M"}))
    return path / "manifest.json"


SMALL = {"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1", "SCREENER_QWEN_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
         "SCREENER_QWEN_REVISION": "c" * 40, "SCREENER_QWEN_SHA256": "d" * 64, "SCREENER_QWEN_TOKENIZER": "service",
         "SCREENER_QWEN_RUNTIME": "local Transformers service", "SCREENER_QWEN_QUANTIZATION": "none",
         "SCREENER_QWEN_CONTEXT_TOKENS": "4096", "SCREENER_QWEN_OUTPUT_TOKENS": "256",
         "SCREENER_QWEN_TIMEOUT": "90", "SCREENER_QWEN_TEMPERATURE": "0", "SCREENER_QWEN_JSON_MODE": "false"}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network and no process: the composer only reads configuration."""
    import socket
    import subprocess
    import urllib.request
    from candidate_screener.llm import gguf_assets, qwen_assets

    def forbidden(*args, **kwargs):
        raise AssertionError("configured_prose_composer must not fetch or start anything")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(gguf_assets, "provision", forbidden)
    monkeypatch.setattr(gguf_assets, "fetch", forbidden)
    monkeypatch.setattr(qwen_assets, "local_settings", lambda: dict(SMALL))
    destination = tmp_path / "models/chatbot/qwen35"
    monkeypatch.setattr(gguf_assets, "DESTINATION", destination)
    return destination


def test_no_setting_uses_the_provisioned_four_billion_manifest(offline):
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    _manifest(offline)
    composer = configured_prose_composer({})
    assert composer.config.model == "Qwen/Qwen3.5-4B"
    assert composer.config.base_url == "http://127.0.0.1:8001/v1"
    assert composer.config.enable_thinking is False
    assert composer.deadline_seconds == 60.


def test_no_setting_without_the_four_billion_manifest_keeps_the_small_model(offline):
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    assert not (offline / "manifest.json").exists()
    composer = configured_prose_composer({})
    assert composer.config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert composer.config.base_url == "http://127.0.0.1:8000/v1"
    assert composer.deadline_seconds == 30.


def test_explicit_manifest_and_explicit_service_still_win_over_the_default(offline, tmp_path):
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    _manifest(offline)
    other = _manifest(tmp_path / "other", model="Qwen/Qwen2.5-0.5B-Instruct")
    assert configured_prose_composer({"SCREENER_QWEN_MANIFEST": str(other)}).config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    explicit = configured_prose_composer(dict(SMALL, SCREENER_QWEN_MODEL="explicit-model"))
    assert explicit.config.model == "explicit-model"


def test_disabled_wins_over_a_provisioned_manifest(offline, monkeypatch):
    from candidate_screener.llm import gguf_server
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    _manifest(offline)
    monkeypatch.setattr(gguf_server, "local_settings", lambda path: pytest.fail("disabled must not read the manifest"))
    assert configured_prose_composer({"SCREENER_QWEN_DISABLED": "true"}) is None


def test_environment_default_reads_the_manifest_from_the_process_environment(offline, monkeypatch):
    from candidate_screener.llm.conversation_prose import configured_prose_composer
    for name in ("SCREENER_QWEN_BASE_URL", "SCREENER_QWEN_MODEL", "SCREENER_QWEN_MANIFEST", "SCREENER_QWEN_DISABLED"):
        monkeypatch.delenv(name, raising=False)
    _manifest(offline)
    assert configured_prose_composer().config.model == "Qwen/Qwen3.5-4B"
