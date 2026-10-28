"""GGUF facade tests use invented files and a fake backend, no model or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from candidate_screener.llm.gguf_server import BackendUnavailable, GGUFBackend, load_manifest, local_settings


@pytest.fixture
def manifest(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic model bytes")
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"synthetic binary")
    value = {"model": "Qwen/synthetic", "revision": "a" * 40, "gguf": str(model),
             "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "llama_server": str(binary),
             "quantization": "Q4_K_M", "max_context_tokens": 4096, "max_output_tokens": 256}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


class FakeBackend(GGUFBackend):
    def __init__(self, manifest):
        super().__init__(manifest)
        self.calls = []
        self.prompt_count = 3
        self.completion_count = 2
        self.reasoning = ""
        self.loaded_path = manifest["gguf"]
        self.finish_reason = "stop"

    def _request(self, method, path, payload, deadline):
        self.calls.append((method, path, deepcopy(payload)))
        if path == "/props":
            return {"model_path": self.loaded_path, "total_slots": 1, "default_generation_settings": {"n_ctx": 4096}}
        if path == "/v1/models":
            return {"data": [{"id": self.model_id}]}
        if path == "/apply-template":
            return {"prompt": "<START>source<END><ASSISTANT><think></think>"}
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        if path == "/v1/chat/completions":
            return {"choices": [{"message": {"role": "assistant", "content": "The degree is in progress.",
                                             "reasoning_content": self.reasoning}, "finish_reason": self.finish_reason}],
                    "usage": {"prompt_tokens": self.prompt_count, "completion_tokens": self.completion_count}}
        pytest.fail("Unexpected backend path")


def test_manifest_verifies_bytes_before_advertising_identity(manifest):
    value = load_manifest(manifest)
    assert Path(value["gguf"]).is_absolute()
    Path(value["gguf"]).write_bytes(b"changed model")
    with pytest.raises(ValueError, match="digest"):
        load_manifest(manifest)


def test_count_and_generation_share_thinking_template_and_special_tokens(manifest):
    runtime = FakeBackend(load_manifest(manifest))
    messages = [{"role": "user", "content": "What degree?"}]
    assert runtime.token_count(messages) == 3
    result = runtime.complete(messages)
    assert result["choices"][0]["message"]["content"] == "The degree is in progress."
    template = next(body for _, path, body in runtime.calls if path == "/apply-template")
    tokenizer = next(body for _, path, body in runtime.calls if path == "/tokenize")
    generation = next(body for _, path, body in runtime.calls if path == "/v1/chat/completions")
    assert template["chat_template_kwargs"] == generation["chat_template_kwargs"] == {"enable_thinking": False}
    assert template["messages"] == generation["messages"] == messages
    assert tokenizer["add_special"] is True and tokenizer["parse_special"] is True
    assert tokenizer["content"].endswith("<think></think>")
    assert generation["cache_prompt"] is False and generation["stream"] is False
    assert "response_format" not in generation


def test_template_count_mismatch_is_unavailable_not_silent_budget_drift(manifest):
    runtime = FakeBackend(load_manifest(manifest))
    runtime.prompt_count = 4
    with pytest.raises(BackendUnavailable, match="contract"):
        runtime.complete([{"role": "user", "content": "Question"}])


def test_reasoning_is_rejected_instead_of_exposed_as_answer(manifest):
    runtime = FakeBackend(load_manifest(manifest))
    runtime.reasoning = "private reasoning"
    with pytest.raises(BackendUnavailable) as error:
        runtime.complete([{"role": "user", "content": "Question"}])
    assert "private reasoning" not in str(error.value)


def test_wrong_loaded_model_cannot_inherit_manifest_attestation(manifest):
    runtime = FakeBackend(load_manifest(manifest))
    runtime.loaded_path += ".different"
    with pytest.raises(BackendUnavailable):
        runtime.metadata()


def test_truncation_reason_remains_visible_to_client(manifest):
    runtime = FakeBackend(load_manifest(manifest))
    runtime.finish_reason = "length"
    assert runtime.complete([{"role": "user", "content": "Question"}])["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("url", ["http://example.com:8002", "https://127.0.0.1", "http://localhost:8002",
                                "http://127.0.0.1:8002/path", "http://user:secret@127.0.0.1:8002"])
def test_backend_is_literal_loopback_only(manifest, url):
    with pytest.raises(ValueError, match="loopback"):
        GGUFBackend(load_manifest(manifest), url)


@pytest.mark.parametrize("budget", [True, 0, 257, -1])
def test_output_budget_rejected_before_backend(manifest, budget):
    runtime = FakeBackend(load_manifest(manifest))
    with pytest.raises(ValueError):
        runtime.complete([{"role": "user", "content": "Question"}], budget)
    assert not runtime.calls


def test_explicit_settings_preserve_old_service_and_disable_thinking(manifest):
    settings = local_settings(manifest)
    assert settings["SCREENER_QWEN_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert settings["SCREENER_QWEN_JSON_MODE"] == settings["SCREENER_QWEN_ENABLE_THINKING"] == "false"
    assert settings["SCREENER_QWEN_TOKENIZER"] == "service"


def test_no_torch_import_or_asset_download_in_facade():
    import candidate_screener.llm.gguf_server as module
    source = Path(module.__file__).read_text()
    assert "import torch" not in source and "import transformers" not in source
    assert "snapshot_download(" not in source


def test_facade_default_does_not_start_backend(manifest, monkeypatch):
    import candidate_screener.llm.gguf_server as module
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected automatic startup"))
    monkeypatch.setattr(module.GGUFBackend, "verify_backend", lambda self: None)

    class Server:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def serve_forever(self):
            pass

    monkeypatch.setattr(module, "ThreadingHTTPServer", Server)
    module.main(["--manifest", str(manifest)])


def test_explicit_launcher_bounds_cpu_and_disables_thinking_and_prompt_caches(manifest):
    from candidate_screener.llm.gguf_server import backend_command
    value = load_manifest(manifest)
    command = backend_command(value)
    assert command[0] == value["llama_server"]
    for key, expected in {"-m": value["gguf"], "--host": "127.0.0.1", "--port": "8002",
                          "--ctx-size": "4096", "--parallel": "1", "--n-gpu-layers": "0",
                          "--reasoning-budget": "0", "--cache-ram": "0"}.items():
        assert command[command.index(key) + 1] == expected
    assert json.loads(command[command.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert {"--no-cache-idle-slots", "--no-slots", "--no-webui", "--log-disable"} <= set(command)


# Plan 2026-10-28-gguf-cross-platform (G1): manifests now name the runtime's platform.

def test_manifest_with_and_without_the_runtime_platform_fields_loads(manifest):
    old = json.loads(manifest.read_text())
    assert load_manifest(manifest)["llama_server"] == old["llama_server"]
    new = {**old, "llama_release": "b11896", "runtime_platform": "linux-x64",
           "runtime_archive": "llama-b11896-bin-ubuntu-x64.tar.gz",
           "runtime_archive_sha256": "b" * 64, "start_check": "passed"}
    manifest.write_text(json.dumps(new))
    value = load_manifest(manifest)
    assert value["runtime_platform"] == "linux-x64"
    assert value["runtime_archive"] == "llama-b11896-bin-ubuntu-x64.tar.gz"
    assert value["start_check"] == "passed"
    assert value["llama_server"] == old["llama_server"]


def test_launch_backend_help_is_platform_neutral(capsys):
    import candidate_screener.llm.gguf_server as module
    with pytest.raises(SystemExit):
        module.main(["--help"])
    text = capsys.readouterr().out
    assert "--launch-backend" in text
    assert "Intel" not in text


# Plan 2026-10-28-gguf-cross-platform (G2b): `docker compose stop` sends SIGTERM, which must clean
# up like Ctrl+C. Before the fix the facade died at once and left llama-server running (G2a).

FAKE_LLAMA_SERVER = '''\
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
model, port, alias = value("-m"), int(value("--port")), value("--alias")
open(os.environ["FAKE_BACKEND_PID"], "w").write(str(os.getpid()))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        body = {"/props": {"model_path": model, "total_slots": 1,
                           "default_generation_settings": {"n_ctx": 4096}},
                "/v1/models": {"data": [{"id": alias}]}}.get(self.path)
        payload = json.dumps(body).encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def _free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(") ")[-1].startswith("Z"))


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs POSIX signals and /proc")
def test_sigterm_stops_the_launched_backend_like_ctrl_c(manifest, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    binary = Path(json.loads(manifest.read_text())["llama_server"])
    binary.write_text(f"#!{sys.executable}\n" + FAKE_LLAMA_SERVER)
    binary.chmod(0o755)
    pid_file = tmp_path / "backend.pid"
    facade_port, backend_port = _free_port(), _free_port()
    env = {**os.environ, "FAKE_BACKEND_PID": str(pid_file), "NO_PROXY": "*"}
    facade = subprocess.Popen(
        [sys.executable, "-m", "candidate_screener.llm.gguf_server", "--manifest", str(manifest),
         "--port", str(facade_port), "--backend-url", f"http://127.0.0.1:{backend_port}", "--launch-backend"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    backend = None
    try:
        ready = facade.stdout.readline()
        assert json.loads(ready)["status"] == "ready", facade.stderr.read()
        backend = int(pid_file.read_text())
        assert _alive(backend)
        started = time.monotonic()
        facade.send_signal(signal.SIGTERM)
        facade.wait(timeout=15)
        for _ in range(50):
            if not _alive(backend):
                break
            time.sleep(.1)
        assert not _alive(backend), "SIGTERM left the llama-server child running"
        assert time.monotonic() - started < 10
    finally:
        if facade.poll() is None:
            facade.kill()
            facade.wait()
        if backend is not None and _alive(backend):
            os.kill(backend, signal.SIGKILL)
