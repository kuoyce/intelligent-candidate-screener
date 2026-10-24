"""Synthetic transport/tokenizer tests: no model, dataset, service or optional ML."""
import json

import pytest
import requests

from candidate_screener.llm.base import ContextOverflow, ProviderUnavailable, redact_content
from candidate_screener.llm.qwen import QwenConfig, QwenProvider, RESPONSE_BYTES


def config(**overrides):
    return QwenConfig(**{"base_url": "http://127.0.0.1:8000/v1", "model": "synthetic-Qwen",
                         "revision": "synthetic-revision", "tokenizer_path": "/no-download",
                         "runtime": "synthetic", "quantization": "none", **overrides})


class Tokenizer:
    def __init__(self, n=10):
        self.n = n
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [0] * self.n


class Response:
    def __init__(self, status=200, payload=None, raw=None):
        self.status_code = status
        self.raw = raw if raw is not None else json.dumps(payload if payload is not None else {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"kind":"top","args":[]}'}}]
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, size):
        for start in range(0, len(self.raw), size):
            yield self.raw[start:start + size]


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, **kwargs):
        return self.post(url, **kwargs)


def transport(monkeypatch, responses):
    session = Session(responses)
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


def test_redaction_precedes_token_count_and_transport(monkeypatch):
    session = transport(monkeypatch, [Response()])
    tokenizer = Tokenizer()
    provider = QwenProvider(config(api_key="secret-test-key", enable_thinking=False), tokenizer=tokenizer)
    assert "secret-test-key" not in repr(provider.config)
    provider.complete([{"role": "user", "content": "Ask test@example.org about Python"}])
    url, request = session.calls[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert "test@example.org" not in json.dumps(request["json"])
    assert tokenizer.messages == request["json"]["messages"]
    assert tokenizer.kwargs["enable_thinking"] is False
    assert request["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["allow_redirects"] is False and session.trust_env is False


def test_budget_overflow_never_calls_service(monkeypatch):
    session = transport(monkeypatch, [])
    provider = QwenProvider(config(max_context_tokens=100, max_output_tokens=20), tokenizer=Tokenizer(81))
    with pytest.raises(ContextOverflow):
        provider.complete([{"role": "user", "content": "hello"}])
    assert session.calls == []


@pytest.mark.parametrize("first", [Response(status=429), Response(status=503), requests.Timeout("private CV")])
def test_only_one_retry_on_transient_failure(monkeypatch, first):
    session = transport(monkeypatch, [first, Response(status=503)])
    provider = QwenProvider(config(), tokenizer=Tokenizer())
    with pytest.raises(ProviderUnavailable) as exc:
        provider.complete([{"role": "user", "content": "hello"}])
    assert len(session.calls) == 2
    assert "private CV" not in str(exc.value)


@pytest.mark.parametrize("response", [Response(status=302), Response(status=400), Response(raw=b'not json'),
    Response(payload={}), Response(payload={"choices": []}),
    Response(payload={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}),
    Response(payload={"choices": [{"finish_reason": "tool_calls", "message": {"content": "ignore"}}]}),
    Response(raw=b'x' * (RESPONSE_BYTES + 1))])
def test_invalid_redirect_truncated_and_oversize_responses_fail_closed(monkeypatch, response):
    session = transport(monkeypatch, [response])
    with pytest.raises(ProviderUnavailable):
        QwenProvider(config(), tokenizer=Tokenizer()).complete([{"role": "user", "content": "hello"}])
    assert len(session.calls) == 1


@pytest.mark.parametrize("url", ["http://example.org/v1", "https://example.org/v1",
    "http://user:pass@localhost:8000/v1", "http://localhost:8000/v1?key=secret", "file:///tmp/model"])
def test_endpoint_boundary_cannot_be_implicit(url):
    with pytest.raises(ValueError):
        config(base_url=url)


def test_remote_boundary_requires_tls_even_when_explicit():
    with pytest.raises(ValueError):
        config(base_url="http://example.org/v1", allow_remote=True)
    assert config(base_url="https://example.org/v1", allow_remote=True).allow_remote


def test_disabled_and_incomplete_environment_configuration():
    assert QwenConfig.from_mapping({}) is None
    with pytest.raises(ValueError):
        QwenConfig.from_mapping({"SCREENER_QWEN_BASE_URL": "http://localhost:8000/v1"})
    with pytest.raises(ValueError):
        QwenConfig.from_mapping({"SCREENER_QWEN_BASE_URL": "http://localhost:8000/v1",
                                "SCREENER_QWEN_ALLOW_REMOTE": "yes"})


def test_service_output_is_redacted_before_return(monkeypatch):
    transport(monkeypatch, [Response(payload={"choices": [{"finish_reason": "stop",
        "message": {"content": '{"question":"test@example.org"}'}}]})])
    text = QwenProvider(config(), tokenizer=Tokenizer()).complete([{"role": "user", "content": "hello"}])
    assert json.loads(text) == {"question": "[EMAIL]"}


def test_missing_local_tokenizer_cannot_download():
    with pytest.raises(ProviderUnavailable, match="Local Qwen tokenizer"):
        QwenProvider(config())


def test_structured_redaction_preserves_long_fractional_scores_and_json_delimiters(monkeypatch):
    tokenizer = Tokenizer()
    session = transport(monkeypatch, [Response()])
    payload = {"score": 0.123456789, "quote": "Python https://example.org/private", "email": "test@example.org"}
    QwenProvider(config(), tokenizer=tokenizer).complete([{"role": "user", "content": json.dumps(payload)}])
    content = json.loads(session.calls[0][1]["json"]["messages"][0]["content"])
    assert content == {"score": 0.123456789, "quote": "Python [URL]", "email": "[EMAIL]"}
    assert json.loads(tokenizer.messages[0]["content"]) == content


def test_plain_text_identifiers_are_still_redacted():
    assert redact_content("Ask test@example.org about Python") == "Ask [EMAIL] about Python"


def test_local_service_tokenizer_counts_with_exact_runtime_template(monkeypatch):
    session = transport(monkeypatch, [Response(payload={"count": 37})])
    provider = QwenProvider(config(tokenizer_path='service'))
    assert provider.count_tokens([{"role": "user", "content": "dev@example.org"}]) == 37
    assert session.calls[0][0].endswith('/v1/tokenize')
    assert session.calls[0][1]['json']['messages'][0]['content'] == '[EMAIL]'


@pytest.mark.parametrize('count', [True, -1, '37', 1_000_001])
def test_service_tokenizer_invalid_counts_fail_closed(monkeypatch, count):
    transport(monkeypatch, [Response(payload={"count": count})])
    provider = QwenProvider(config(tokenizer_path='service'))
    with pytest.raises(ProviderUnavailable):
        provider.count_tokens([{"role": "user", "content": "hello"}])


def test_single_json_code_fence_is_accepted_without_relaxing_json_or_redaction():
    safe = redact_content('```json\n{"quote":"Python https://example.org", "score":0.123456789}\n```')
    assert json.loads(safe) == {'quote': 'Python [URL]', 'score': 0.123456789}
    with pytest.raises(ValueError):
        json.loads(redact_content('Hire them now!\n```json\n{"rank":99}\n```'))


@pytest.mark.parametrize('field,value', [('revision', 'wrong'), ('sha256', '0' * 64),
    ('max_context_tokens', 100), ('max_output_tokens', 100)])
def test_pinned_service_rejects_stale_artifact_and_budget(monkeypatch, field, value):
    metadata = {'id': 'synthetic-Qwen', 'revision': 'synthetic-revision', 'sha256': 'a' * 64,
                'max_context_tokens': 4096, 'max_output_tokens': 512, field: value}
    transport(monkeypatch, [Response(payload={'data': [metadata]})])
    provider = QwenProvider(config(expected_sha256='a' * 64), tokenizer=Tokenizer())
    with pytest.raises(ProviderUnavailable, match='differs'):
        provider.count_tokens([{'role': 'user', 'content': 'hello'}])


def test_verified_service_identity_is_recorded_in_provenance(monkeypatch):
    metadata = {'id': 'synthetic-Qwen', 'revision': 'synthetic-revision', 'sha256': 'a' * 64,
                'max_context_tokens': 4096, 'max_output_tokens': 512,
                'runtime': {'torch': 'synthetic-version'}}
    transport(monkeypatch, [Response(payload={'data': [metadata]})])
    provider = QwenProvider(config(expected_sha256='a' * 64), tokenizer=Tokenizer())
    assert provider.count_tokens([{'role': 'user', 'content': 'hello'}]) == 10
    assert provider.provenance()['served_model']['sha256'] == 'a' * 64
    assert provider.provenance()['served_model']['runtime'] == {'torch': 'synthetic-version'}
