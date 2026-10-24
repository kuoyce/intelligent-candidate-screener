"""Frozen local generation boundary, tested without torch or model downloads."""
from contextlib import nullcontext
from http.server import HTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from candidate_screener.llm.local_server import LocalQwenRuntime, load_runtime, make_handler, validate_messages


MESSAGES = [{'role': 'user', 'content': 'Only new generated text please'}]


class Tensor:
    shape = (1, 3)


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['add_generation_prompt'] is True
        if kwargs.get('return_dict'):
            return {'input_ids': Tensor()}
        return [10, 11, 12]

    def decode(self, tokens, **kwargs):
        assert kwargs['skip_special_tokens']
        return ','.join(str(i) for i in tokens if i != 99)


class Model:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[10, 11, 12, 42, 99]]


@pytest.fixture
def runtime():
    class Torch:
        inference_mode = staticmethod(nullcontext)
    return LocalQwenRuntime(Tokenizer(), Model(), Torch(), model_id='test-model', revision='a' * 40, sha256='b' * 64)


def test_generated_response_excludes_prompt_and_uses_inference_only(runtime):
    answer = runtime.complete(MESSAGES, 32)
    assert answer['content'] == '42'
    assert answer['prompt_tokens'] == 3
    assert answer['completion_tokens'] == 2
    assert answer['finish_reason'] == 'stop'
    call = runtime.model.calls[0]
    assert call['do_sample'] is False
    assert call['max_new_tokens'] == 32
    assert call['max_time'] == 60
    assert runtime.token_count(MESSAGES) == 3


def test_all_configured_eos_tokens_are_completed_answers(runtime):
    from types import SimpleNamespace
    runtime.model.generation_config = SimpleNamespace(eos_token_id=[99, 98])
    runtime.model.generate = lambda **kwargs: [[10, 11, 12, 42, 98]]
    assert runtime.complete(MESSAGES, 32)['finish_reason'] == 'stop'


@pytest.mark.parametrize('value', [0, 257, -1, True, 3.5, '32'])
def test_generation_output_bounds(runtime, value):
    with pytest.raises(ValueError):
        runtime.complete(MESSAGES, value)
    assert not runtime.model.calls


@pytest.mark.parametrize('messages', [[], [{'role': 'tool', 'content': 'execute'}], [{'role': 'user', 'content': 'x', 'tool_calls': []}],
                                     [{'role': 'user', 'content': 3}], [{'role': 'system', 'content': 'x' * 100001}]])
def test_message_contract_excludes_tools_and_unbounded_content(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_context_bound_before_generation(runtime):
    class HugeTensor:
        shape = (1, 4096)
    runtime.tokenizer.apply_chat_template = lambda *args, **kwargs: {'input_ids': HugeTensor()}
    with pytest.raises(ValueError, match='Context budget'):
        runtime.complete(MESSAGES, 1)
    assert not runtime.model.calls


@pytest.fixture
def server(runtime):
    try:
        server = HTTPServer(('127.0.0.1', 0), make_handler(runtime))
    except PermissionError:
        pytest.skip('Sandbox prohibits opening even a loopback listening socket')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    worker.join()


def post(server, path, body):
    request = Request(server + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_openai_compatible_response_and_tokenization(server):
    body = {'model': 'test-model', 'messages': MESSAGES, 'max_tokens': 20, 'temperature': 0, 'stream': False}
    response = post(server, '/v1/chat/completions', body)
    assert response['choices'][0]['message']['content'] == '42'
    assert response['usage']['total_tokens'] == 5
    assert post(server, '/v1/tokenize', {'model': 'test-model', 'messages': MESSAGES}) == {'count': 3}
    with urlopen(server + '/v1/models', timeout=5) as response:
        metadata = json.loads(response.read())['data'][0]
    assert metadata['revision'] == 'a' * 40
    assert metadata['device'] == 'cpu'


@pytest.mark.parametrize('extra', [{'model': 'wrong'}, {'tools': [{}]}, {'stream': True}, {'temperature': 0.5}, {'max_tokens': 300}, {'top_p': 0.9}])
def test_http_rejects_unapproved_request_fields(server, extra):
    body = {'model': 'test-model', 'messages': MESSAGES, **extra}
    with pytest.raises(HTTPError) as exc:
        post(server, '/v1/chat/completions', body)
    assert exc.value.code == 400
    assert b'Only new generated text' not in exc.value.read()


def test_loading_digest_mismatch_never_imports_model_runtime(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'model': 'test-model', 'revision': 'a' * 40, 'checkpoint': 'checkpoint', 'sha256': 'bad'}))
    with pytest.raises(ValueError, match='digest mismatch'):
        load_runtime(manifest)


def test_unprovisioned_reranker_is_explicitly_unavailable(server):
    with pytest.raises(HTTPError) as exc:
        post(server, '/v1/rerank', {'checkpoint_sha256': 'a' * 64, 'jd_text': ['Python'], 'cv_text': ['Python']})
    assert exc.value.code == 503
    assert b'not provisioned' in exc.value.read()
