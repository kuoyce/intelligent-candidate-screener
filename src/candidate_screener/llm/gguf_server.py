"""Loopback facade for a separately provisioned llama.cpp GGUF server.

No Torch/Transformers imports, downloads, training, server startup or prompt logs.
Backend must use one slot, the manifest model alias and global template kwargs
{"enable_thinking":false}. See llama.cpp tools/server/README.md for the
/apply-template, /tokenize and /v1/chat/completions contracts.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import time
from time import monotonic
from urllib.parse import urlsplit

import requests

MAX_BODY_BYTES = 128_000
MAX_RESPONSE_BYTES = 256_000
MAX_CONTEXT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 256
TEMPLATE_KWARGS = {"enable_thinking": False}


class BackendUnavailable(RuntimeError):
    """Document-free runtime failure."""


def local_settings(manifest_path):
    """Explicit selection of an already provisioned facade; never starts it."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8001/v1",
            "SCREENER_QWEN_MODEL": manifest["model"], "SCREENER_QWEN_REVISION": manifest["revision"],
            "SCREENER_QWEN_SHA256": manifest["sha256"], "SCREENER_QWEN_TOKENIZER": "service",
            "SCREENER_QWEN_RUNTIME": "local llama.cpp GGUF service",
            "SCREENER_QWEN_QUANTIZATION": manifest["quantization"],
            "SCREENER_QWEN_CONTEXT_TOKENS": "4096", "SCREENER_QWEN_OUTPUT_TOKENS": "256",
            "SCREENER_QWEN_TIMEOUT": "60" if manifest["model"] == "Qwen/Qwen3.5-4B" else "30",
            "SCREENER_QWEN_TEMPERATURE": "0",
            "SCREENER_QWEN_JSON_MODE": "false", "SCREENER_QWEN_ENABLE_THINKING": "false"}


def validate_messages(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise ValueError("Invalid messages")
    for item in value:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] not in {"system", "user", "assistant"}
                or not isinstance(item["content"], str) or len(item["content"]) > 100_000):
            raise ValueError("Invalid message")
    return value


def load_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (not isinstance(value, dict)
            or any(not isinstance(value.get(key), str) or not value[key]
                   for key in ("model", "revision", "gguf", "sha256", "llama_server", "quantization"))
            or not re.fullmatch(r"[0-9a-f]{40}", value["revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or value.get("max_context_tokens", 4096) != 4096
            or value.get("max_output_tokens", 256) != 256):
        raise ValueError("Invalid pinned GGUF manifest")
    result = dict(value)
    for key in ("gguf", "llama_server"):
        asset = Path(value[key]).expanduser()
        asset = (path.parent / asset).resolve() if not asset.is_absolute() else asset.resolve()
        if not asset.is_file():
            raise ValueError("Manifest asset unavailable")
        result[key] = str(asset)
    digest = hashlib.sha256()
    with Path(result["gguf"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != value["sha256"]:
        raise ValueError("GGUF digest mismatch")
    return result


class GGUFBackend:
    def __init__(self, manifest, backend_url="http://127.0.0.1:8002", *, timeout_seconds=60.):
        parsed = urlsplit(backend_url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Backend must be a loopback HTTP origin")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
            raise ValueError("Deadline must be within 60 seconds")
        self.manifest, self.base_url = dict(manifest), backend_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.model_id = manifest["model"]

    def _request(self, method, path, payload, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise BackendUnavailable("Backend deadline exceeded")
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.request(method, self.base_url + path, json=payload,
                                     timeout=(min(2., remaining), remaining), allow_redirects=False,
                                     stream=True) as response:
                    if response.status_code != 200:
                        raise BackendUnavailable("Backend unavailable")
                    raw = bytearray()
                    for chunk in response.iter_content(4096):
                        if monotonic() >= deadline:
                            raise BackendUnavailable("Backend deadline exceeded")
                        raw.extend(chunk)
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raise BackendUnavailable("Backend response too large")
                    result = json.loads(raw)
                    if not isinstance(result, dict):
                        raise BackendUnavailable("Invalid backend response")
                    return result
        except (requests.RequestException, ValueError, TypeError):
            raise BackendUnavailable("Backend unavailable") from None

    def verify_backend(self, deadline=None):
        deadline = deadline or monotonic() + self.timeout_seconds
        props = self._request("GET", "/props", None, deadline)
        path = props.get("model_path")
        settings = props.get("default_generation_settings", {})
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).resolve() != Path(self.manifest["gguf"])
                or props.get("total_slots") != 1
                or not isinstance(settings, dict)
                or type(settings.get("n_ctx")) is not int or settings["n_ctx"] < MAX_CONTEXT_TOKENS):
            raise BackendUnavailable("Backend differs from pinned manifest")
        models = self._request("GET", "/v1/models", None, deadline).get("data")
        if not isinstance(models, list) or not any(isinstance(row, dict) and row.get("id") == self.model_id for row in models):
            raise BackendUnavailable("Backend model alias mismatch")

    def metadata(self):
        self.verify_backend()
        return {"id": self.model_id, "object": "model", "revision": self.manifest["revision"],
                "sha256": self.manifest["sha256"], "runtime": "local llama.cpp GGUF service",
                "quantization": self.manifest["quantization"], "device": "cpu",
                "max_context_tokens": MAX_CONTEXT_TOKENS, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "generation": {"temperature": 0, "enable_thinking": False,
                               "max_time_seconds": self.timeout_seconds}}

    def token_count(self, messages, *, deadline=None):
        deadline = deadline or monotonic() + self.timeout_seconds
        messages = validate_messages(messages)
        result = self._request("POST", "/apply-template",
                               {"messages": messages, "chat_template_kwargs": TEMPLATE_KWARGS}, deadline)
        prompt = result.get("prompt")
        if not isinstance(prompt, str) or len(prompt.encode()) > MAX_RESPONSE_BYTES:
            raise BackendUnavailable("Invalid applied template")
        result = self._request("POST", "/tokenize", {"content": prompt, "add_special": True,
                                                     "parse_special": True, "with_pieces": False}, deadline)
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or len(tokens) > 131072 or any(type(token) is not int for token in tokens):
            raise BackendUnavailable("Invalid tokenizer result")
        return len(tokens)

    def complete(self, messages, max_tokens=MAX_OUTPUT_TOKENS):
        messages = validate_messages(messages)
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError("Invalid output budget")
        deadline = monotonic() + self.timeout_seconds
        self.verify_backend(deadline)
        count = self.token_count(messages, deadline=deadline)
        if count + max_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError("Context budget exceeded")
        result = self._request("POST", "/v1/chat/completions",
                               {"model": self.model_id, "messages": messages, "max_tokens": max_tokens,
                                "temperature": 0, "stream": False, "chat_template_kwargs": TEMPLATE_KWARGS,
                                "cache_prompt": False}, deadline)
        try:
            choices, usage = result["choices"], result["usage"]
            choice = choices[0]
            message = choice["message"]
            if (len(choices) != 1 or choice["finish_reason"] not in {"stop", "length"}
                    or message.get("role") != "assistant" or not isinstance(message.get("content"), str)
                    or not message["content"].strip() or message.get("reasoning_content") or message.get("tool_calls")
                    or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != count
                    or type(usage.get("completion_tokens")) is not int
                    or not 0 <= usage["completion_tokens"] <= max_tokens):
                raise BackendUnavailable("Backend completion contract mismatch")
            return {"object": "chat.completion", "model": self.model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": message["content"]},
                                 "finish_reason": choice["finish_reason"]}],
                    "usage": {"prompt_tokens": count, "completion_tokens": usage["completion_tokens"],
                              "total_tokens": count + usage["completion_tokens"]}}
        except (KeyError, IndexError, TypeError):
            raise BackendUnavailable("Invalid backend completion") from None


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})
                return
            try:
                self.send_json(200, {"object": "list", "data": [runtime.metadata()]})
            except BackendUnavailable:
                self.send_json(503, {"error": {"message": "Pinned local backend unavailable"}})

        def do_POST(self):
            if self.path not in {"/v1/tokenize", "/v1/chat/completions"}:
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})
                return
            try:
                length = self.headers.get("Content-Length", "")
                if not length.isdigit() or not 0 < int(length) <= MAX_BODY_BYTES:
                    raise ValueError("Invalid length")
                self.connection.settimeout(5)
                body = json.loads(self.rfile.read(int(length)))
                allowed = {"model", "messages"} if self.path == "/v1/tokenize" else {
                    "model", "messages", "max_tokens", "temperature", "stream", "chat_template_kwargs"}
                if not isinstance(body, dict) or body.get("model") != runtime.model_id or not set(body) <= allowed:
                    raise ValueError("Invalid request")
                messages = validate_messages(body.get("messages"))
                if self.path == "/v1/tokenize":
                    self.send_json(200, {"count": runtime.token_count(messages)})
                else:
                    if (body.get("stream", False) is not False or body.get("temperature", 0) != 0
                            or body.get("chat_template_kwargs", TEMPLATE_KWARGS) != TEMPLATE_KWARGS):
                        raise ValueError("Only nonthinking greedy text generation supported")
                    self.send_json(200, runtime.complete(messages, body.get("max_tokens", MAX_OUTPUT_TOKENS)))
            except (ValueError, TypeError, KeyError, TimeoutError):
                self.send_json(400, {"error": {"message": "Invalid inference request"}})
            except BackendUnavailable:
                self.send_json(503, {"error": {"message": "Local inference unavailable"}})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8002")
    parser.add_argument("--launch-backend", action="store_true", help="Start the pinned local Intel runtime too")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    runtime = GGUFBackend(manifest, args.backend_url)
    child = None
    try:
        if args.launch_backend:
            parsed = urlsplit(args.backend_url)
            child = subprocess.Popen(backend_command(manifest, parsed.port or 80),
                                     stdin=subprocess.DEVNULL)
            deadline = monotonic() + 60
            while True:
                if child.poll() is not None:
                    raise BackendUnavailable("Pinned backend exited during startup")
                try:
                    runtime.verify_backend(min(deadline, monotonic() + 1))
                    break
                except BackendUnavailable:
                    if monotonic() >= deadline:
                        raise BackendUnavailable("Pinned backend startup timed out") from None
                    time.sleep(.25)
        else:
            runtime.verify_backend()
        with ThreadingHTTPServer((args.host, args.port), make_handler(runtime)) as server:
            print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "model": runtime.model_id}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def backend_command(manifest, port=8002):
    """Explicit CPU budgets and disabled thinking/caching for every request."""
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Use an unprivileged loopback port")
    return [manifest["llama_server"], "-m", manifest["gguf"], "--host", "127.0.0.1",
            "--port", str(port), "--alias", manifest["model"], "--ctx-size", "4096",
            "--parallel", "1", "--n-gpu-layers", "0", "--threads", "4", "--threads-batch", "4",
            "--chat-template-kwargs", json.dumps(TEMPLATE_KWARGS), "--reasoning-budget", "0",
            "--cache-ram", "0", "--no-cache-idle-slots", "--no-slots", "--no-webui", "--log-disable"]


if __name__ == "__main__":
    main()
