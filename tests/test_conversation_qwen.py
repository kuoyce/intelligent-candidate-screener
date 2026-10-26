"""Qwen can order supported comparisons but cannot author their facts."""
from copy import deepcopy
import json
import subprocess
import sys
from time import monotonic

import pytest

from candidate_screener.llm.base import ContextOverflow, ProviderUnavailable
from candidate_screener.llm.conversation_qwen import ConversationComposer


class Provider:
    max_context_tokens = 4096
    max_output_tokens = 256
    model_id = "synthetic-qwen"

    def __init__(self, result=None, tokens=100):
        self.result = result if result is not None else {"relation_ids": ["contrast", "common"]}
        self.tokens = tokens
        self.calls = 0
        self.messages = None

    def count_tokens(self, messages):
        self.messages = deepcopy(messages)
        return self.tokens

    def complete(self, messages):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result if isinstance(self.result, str) else json.dumps(self.result)


@pytest.fixture
def frame():
    return {"snapshot_id": "snapshot", "jd_id": "job", "selected_candidates": ["a", "b"],
            "intent": "compare", "criteria": ["Python"]}


@pytest.fixture
def facts():
    return {"cells": [
        {"id": "a-python", "owner": "a", "context": "Python deployments.", "support_status": "explicit_positive"},
        {"id": "b-python", "owner": "b", "context": "No commercial Python experience.", "support_status": "explicit_negative"}],
        "relations": [
            {"id": "contrast", "kind": "explicit_contrast", "participants": ["a", "b"],
             "cell_ids": ["a-python", "b-python"], "text": "A has documented deployment; B denies commercial experience."},
            {"id": "common", "kind": "both_documented", "participants": ["a", "b"],
             "cell_ids": ["a-python", "b-python"], "text": "Both documents mention Python."}]}


def test_accepted_plan_contains_only_ids_and_full_qualifiers_reach_prompt(frame, facts):
    provider = Provider()
    result = ConversationComposer(provider).compose("Compare their Python evidence", frame, facts)
    assert result["relation_ids"] == ["contrast", "common"]
    assert result["mode"] == "qwen_answer_plan"
    assert result["model"] == "synthetic-qwen"
    assert "text" not in result
    payload = json.loads(provider.messages[1]["content"])
    assert payload["evidence"][1]["context"] == "No commercial Python experience."
    assert payload["evidence"][1]["support_status"] == "explicit_negative"
    assert provider.calls == 1


def test_model_cannot_omit_a_critical_qualification_even_when_both_owners_remain(frame, facts):
    with pytest.raises(ValueError, match="critical"):
        ConversationComposer(Provider({"relation_ids": ["common"]})).compose("Compare", frame, facts)


@pytest.mark.parametrize("result", [
    {"relation_ids": ["foreign"]}, {"relation_ids": ["contrast", "contrast"]},
    {"relation_ids": ["contrast"], "text": "Candidate B is best"},
    {"relation_ids": []}, {"relation_ids": "contrast"},
    "```json\n{\"relation_ids\":[\"contrast\"]}\n```", "{\"relation_ids\":[",
    {"relation_ids": [False]}, ["contrast"],
])
def test_forged_duplicate_prose_and_incomplete_outputs_are_rejected(frame, facts, result):
    with pytest.raises(ValueError):
        ConversationComposer(Provider(result)).compose("Compare", frame, facts)


def test_single_owner_output_cannot_erase_other_candidate(frame, facts):
    facts["relations"] = [{"id": "only-a", "kind": "both_documented", "participants": ["a"]}]
    with pytest.raises(ValueError, match="participant"):
        ConversationComposer(Provider({"relation_ids": ["only-a"]})).compose("Compare", frame, facts)


def test_budget_does_not_silently_trim_an_owner_or_start_generation(frame, facts):
    provider = Provider(tokens=3900)
    with pytest.raises(ContextOverflow):
        ConversationComposer(provider).compose("Compare", frame, facts)
    assert provider.calls == 0
    assert len(json.loads(provider.messages[1]["content"])["evidence"]) == 2


def test_missing_owner_and_no_relations_decline_generation(frame, facts):
    for changed in ({**facts, "cells": facts["cells"][:1]}, {**facts, "relations": []}):
        provider = Provider()
        with pytest.raises(ValueError):
            ConversationComposer(provider).compose("Compare", frame, changed)
        assert provider.calls == 0


def test_provider_unavailable_is_not_reported_as_model_success(frame, facts):
    with pytest.raises(ProviderUnavailable):
        ConversationComposer(Provider(ProviderUnavailable("Offline"))).compose("Compare", frame, facts)


def test_directed_explanation_prompt_keeps_requested_direction_and_counteradvantages(frame, facts):
    frame.update(intent="comparison_explanation", explanation_subject="b", explanation_reference="a", explanation_relation="better")
    facts["relations"].append({"id": "counter", "kind": "comparison_advantage", "participants": ["a", "b"],
                               "stronger_owner": "a", "weaker_owner": "b", "basis": "explicit_contrast",
                               "criterion": "Python", "cell_ids": ["a-python", "b-python"],
                               "text": "A has stronger documented Python evidence than B."})
    provider = Provider({"relation_ids": ["contrast", "common", "counter"]})
    ConversationComposer(provider).compose("Why is B better than A?", frame, facts)
    payload = json.loads(provider.messages[1]["content"])
    assert payload["intent"] == "comparison_explanation"
    assert payload["explanation"] == {"subject": "c2", "reference": "c1", "relation": "better"}
    counter_alias = next(row["id"] for row in payload["relations"] if row["kind"] == "comparison_advantage")
    assert counter_alias in payload["required_relation_ids"]


def test_directed_model_cannot_omit_counteradvantage_to_agree_with_question(frame, facts):
    frame.update(intent="comparison_explanation", explanation_subject="b", explanation_reference="a", explanation_relation="better")
    facts["relations"].append({"id": "counter", "kind": "comparison_advantage", "participants": ["a", "b"],
                               "stronger_owner": "a", "weaker_owner": "b", "cell_ids": ["a-python", "b-python"],
                               "text": "A has stronger documented Python evidence."})
    with pytest.raises(ValueError, match="critical"):
        ConversationComposer(Provider({"relation_ids": ["contrast", "common"]})).compose("Why is B better?", frame, facts)


def test_foreign_relation_participant_rejected_before_generation(frame, facts):
    facts["relations"][0]["participants"].append("foreign")
    provider = Provider()
    with pytest.raises(ValueError):
        ConversationComposer(provider).compose("Compare", frame, facts)
    assert provider.calls == 0


def test_local_factory_preserves_service_json_capability_and_manifest_output_budget(monkeypatch):
    from candidate_screener.llm import conversation_qwen
    result = conversation_qwen.configured_composer({
        "SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
        "SCREENER_QWEN_MODEL": "Qwen/local", "SCREENER_QWEN_REVISION": "revision",
        "SCREENER_QWEN_TOKENIZER": "service", "SCREENER_QWEN_RUNTIME": "local service",
        "SCREENER_QWEN_QUANTIZATION": "none", "SCREENER_QWEN_OUTPUT_TOKENS": "256",
        "SCREENER_QWEN_TIMEOUT": "90", "SCREENER_QWEN_JSON_MODE": "false"})
    assert isinstance(result, conversation_qwen.ConversationComposer)
    assert result.config.json_mode is False
    assert result.config.max_output_tokens == 256
    assert result.config.timeout_seconds == 30
    assert result.config.temperature == 0


def test_factory_disabled_flag_never_constructs_provider(monkeypatch):
    from candidate_screener.llm import conversation_qwen, qwen

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled factory instantiated a model provider")

    monkeypatch.setattr(qwen, "QwenProvider", forbidden)
    assert conversation_qwen.configured_composer({"SCREENER_QWEN_DISABLED": "true"}) is None


def test_factory_obeys_environment_disabled_before_loading_defaults(monkeypatch):
    from candidate_screener.llm import conversation_qwen, qwen_assets

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled factory loaded local defaults")

    monkeypatch.setenv("SCREENER_QWEN_DISABLED", "true")
    monkeypatch.setattr(qwen_assets, "local_settings", forbidden)
    assert conversation_qwen.configured_composer() is None


def test_environment_endpoint_uses_its_identity_without_local_defaults(monkeypatch):
    from candidate_screener.llm import conversation_qwen, qwen, qwen_assets

    for key in list(conversation_qwen.os.environ):
        if key.startswith("SCREENER_QWEN_"):
            monkeypatch.delenv(key)
    values = {
        "SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8123/v1",
        "SCREENER_QWEN_MODEL": "explicit/model", "SCREENER_QWEN_REVISION": "explicit-revision",
        "SCREENER_QWEN_TOKENIZER": "service", "SCREENER_QWEN_RUNTIME": "explicit service",
        "SCREENER_QWEN_QUANTIZATION": "none", "SCREENER_QWEN_JSON_MODE": "false"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit deployment loaded local checkpoint defaults")

    monkeypatch.setattr(qwen_assets, "local_settings", forbidden)
    composer = conversation_qwen.configured_composer()
    assert composer is not None
    assert composer.config.base_url == values["SCREENER_QWEN_BASE_URL"]
    assert composer.config.model == "explicit/model"
    assert composer.config.revision == "explicit-revision"


def test_environment_can_override_defaults_without_claiming_different_model(monkeypatch):
    from candidate_screener.llm import conversation_qwen, qwen, qwen_assets

    for key in list(conversation_qwen.os.environ):
        if key.startswith("SCREENER_QWEN_"):
            monkeypatch.delenv(key)
    defaults = {
        "SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
        "SCREENER_QWEN_MODEL": "local/model", "SCREENER_QWEN_REVISION": "local-revision",
        "SCREENER_QWEN_TOKENIZER": "service", "SCREENER_QWEN_RUNTIME": "local service",
        "SCREENER_QWEN_QUANTIZATION": "none", "SCREENER_QWEN_JSON_MODE": "false"}
    monkeypatch.setattr(qwen_assets, "local_settings", lambda: defaults)
    monkeypatch.setenv("SCREENER_QWEN_OUTPUT_TOKENS", "128")
    composer = conversation_qwen.configured_composer()
    assert composer is not None
    assert composer.config.max_output_tokens == 128
    assert composer.config.model == "local/model"


def test_incomplete_explicit_deployment_is_unavailable_not_local_fallback(monkeypatch):
    from candidate_screener.llm import conversation_qwen, qwen_assets

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete explicit deployment read local defaults")

    monkeypatch.setattr(qwen_assets, "local_settings", forbidden)
    assert conversation_qwen.configured_composer({"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8123/v1"}) is None


def test_short_request_local_relation_names_bind_back_to_canonical_facts(frame, facts):
    provider = Provider({"relation_ids": ["r2", "r1"]})
    result = ConversationComposer(provider).compose("Compare", frame, facts)
    assert result["relation_ids"] == ["common", "contrast"]
    payload = json.loads(provider.messages[1]["content"])
    assert [item["id"] for item in payload["relations"]] == ["r1", "r2"]
    assert payload["required_relation_ids"] == ["r1"]
    assert payload["relations"][0]["participants"] == ["c1", "c2"]
    assert [item["id"] for item in payload["evidence"]] == ["e1", "e2"]


def test_alias_and_canonical_duplicate_cannot_repeat_a_relation(frame, facts):
    with pytest.raises(ValueError, match="invalid relations"):
        ConversationComposer(Provider({"relation_ids": ["r1", "contrast"]})).compose("Compare", frame, facts)


def _config():
    from candidate_screener.llm.qwen import QwenConfig
    return QwenConfig(base_url="http://127.0.0.1:8000/v1", model="synthetic-qwen",
                      revision="revision", tokenizer_path="service", runtime="local test",
                      quantization="none", json_mode=False, max_output_tokens=256)


def test_shared_deadline_kills_and_reaps_blocked_model_worker(monkeypatch, frame, facts):
    from candidate_screener.llm import conversation_qwen

    monkeypatch.setattr(conversation_qwen, "_worker_command", lambda: [
        sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(0.6)"])
    processes = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(conversation_qwen.subprocess, "Popen", capture)
    composer = conversation_qwen.IsolatedConversationComposer(_config())
    started = monotonic()
    with pytest.raises(conversation_qwen.CompositionUnavailable) as failed:
        composer.compose_before("Compare", frame, facts, started + .1)
    assert failed.value.reason == "deadline_exceeded"
    assert monotonic() - started < .5
    assert len(processes) == 1 and processes[0].poll() is not None


def test_expired_shared_deadline_never_starts_worker(monkeypatch, frame, facts):
    from candidate_screener.llm import conversation_qwen

    def forbidden(*args, **kwargs):
        pytest.fail("Expired turn started a worker")

    monkeypatch.setattr(conversation_qwen.subprocess, "Popen", forbidden)
    with pytest.raises(conversation_qwen.CompositionUnavailable):
        conversation_qwen.IsolatedConversationComposer(_config()).compose_before(
            "Compare", frame, facts, monotonic() - 1)


@pytest.mark.parametrize("payload,exception", [
    ({"error": "provider_unavailable"}, ProviderUnavailable),
    ({"error": "context_overflow"}, ContextOverflow),
    ({"error": "invalid_plan"}, ValueError),
    ({"error": "foreign text with private data"}, ProviderUnavailable),
    (["not-an-envelope"], ProviderUnavailable),
])
def test_worker_failures_preserve_safe_fallback_category(monkeypatch, frame, facts, payload, exception):
    from candidate_screener.llm import conversation_qwen

    code = "import json; print(" + repr(json.dumps(payload)) + ")"
    monkeypatch.setattr(conversation_qwen, "_worker_command", lambda: [sys.executable, "-c", code])
    with pytest.raises(exception) as failed:
        conversation_qwen.IsolatedConversationComposer(_config()).compose_before(
            "Compare", frame, facts, monotonic() + 2)
    assert "private data" not in str(failed.value)


def test_rejected_model_plan_has_finite_document_free_reason(frame, facts):
    from candidate_screener.llm.conversation_qwen import CompositionPlanError

    with pytest.raises(CompositionPlanError) as failed:
        ConversationComposer(Provider({"relation_ids": ["common"]})).compose("Compare", frame, facts)
    assert failed.value.reason == "qualification_omission"


def test_real_worker_expired_input_emits_no_documents_or_configuration():
    from candidate_screener.llm.conversation_qwen import _worker_command

    request = {"deadline": monotonic() - 1, "question": "PRIVATE QUESTION",
               "config": {"api_key": "PRIVATE TOKEN"}}
    process = subprocess.run(_worker_command(), input=json.dumps(request).encode(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3, check=True)
    assert json.loads(process.stdout) == {"error": "deadline_exceeded"}
    assert b"PRIVATE" not in process.stdout and b"PRIVATE" not in process.stderr
