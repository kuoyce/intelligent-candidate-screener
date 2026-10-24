"""Bounded OpenAI-compatible Qwen client; no SDK, downloads or tool execution.

The model service is provisioned separately. A local matching tokenizer is required
so context limits include the runtime's chat template, rather than a word estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import math
import re
from pathlib import Path
from time import monotonic
from typing import Mapping, Sequence
from urllib.parse import urlsplit

import requests

from candidate_screener.llm.base import ContextOverflow, ProviderUnavailable, redact_content

RESPONSE_BYTES = 128 * 1024
TRANSIENT_STATUS = {429, 502, 503, 504}


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class QwenConfig:
    base_url: str
    model: str
    revision: str
    tokenizer_path: str
    runtime: str
    quantization: str
    max_context_tokens: int = 4096
    max_output_tokens: int = 512
    timeout_seconds: float = 30.0
    temperature: float = 0.2
    json_mode: bool = True
    enable_thinking: bool | None = None
    allow_remote: bool = False
    expected_sha256: str = ""
    api_key: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Use a model-service HTTP(S) base URL without credentials or query parameters.")
        if not _loopback(parsed.hostname) and not self.allow_remote:
            raise ValueError("Remote model services require an explicit data-boundary configuration.")
        if not _loopback(parsed.hostname) and parsed.scheme != "https":
            raise ValueError("Remote model services require HTTPS.")
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.model, self.revision, self.tokenizer_path, self.runtime, self.quantization)):
            raise ValueError("Model, revision, local tokenizer, runtime and quantization are required.")
        if (type(self.max_context_tokens) is not int or type(self.max_output_tokens) is not int
                or not 1 <= self.max_output_tokens < self.max_context_tokens <= 131072):
            raise ValueError("Use bounded positive context/output token limits.")
        if (isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 120):
            raise ValueError("Provider timeout must be within 120 seconds.")
        if (isinstance(self.temperature, bool) or not math.isfinite(self.temperature)
                or not 0 <= self.temperature <= 2):
            raise ValueError("Temperature must be finite and within [0, 2].")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be true, false or omitted.")
        if self.expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.expected_sha256):
            raise ValueError("Expected model checkpoint SHA-256 must be hexadecimal.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> QwenConfig | None:
        """Missing endpoint/model means disabled. Incomplete configured setup is an error."""
        endpoint = values.get("SCREENER_QWEN_BASE_URL", "")
        model = values.get("SCREENER_QWEN_MODEL", "")
        if not endpoint and not model:
            return None

        def flag(key, default=False):
            value = str(values.get(key, str(default))).lower()
            if value not in {"true", "false"}:
                raise ValueError(f"{key} must be true or false.")
            return value == "true"

        thinking = values.get("SCREENER_QWEN_ENABLE_THINKING")
        return cls(
            base_url=endpoint, model=model,
            revision=values.get("SCREENER_QWEN_REVISION", ""),
            tokenizer_path=values.get("SCREENER_QWEN_TOKENIZER", ""),
            runtime=values.get("SCREENER_QWEN_RUNTIME", ""),
            quantization=values.get("SCREENER_QWEN_QUANTIZATION", ""),
            api_key=values.get("SCREENER_QWEN_API_KEY", ""),
            max_context_tokens=int(values.get("SCREENER_QWEN_CONTEXT_TOKENS", 4096)),
            max_output_tokens=int(values.get("SCREENER_QWEN_OUTPUT_TOKENS", 512)),
            timeout_seconds=float(values.get("SCREENER_QWEN_TIMEOUT", 30)),
            temperature=float(values.get("SCREENER_QWEN_TEMPERATURE", 0.2)),
            json_mode=flag("SCREENER_QWEN_JSON_MODE", True),
            enable_thinking=flag("SCREENER_QWEN_ENABLE_THINKING") if thinking is not None else None,
            allow_remote=flag("SCREENER_QWEN_ALLOW_REMOTE"),
            expected_sha256=values.get("SCREENER_QWEN_SHA256", ""),
        )


def load_tokenizer(config: QwenConfig):
    """Only a provisioned local tokenizer; no remote code or model fetching."""
    if config.tokenizer_path == "service":
        return ServiceTokenizer(config)
    path = Path(config.tokenizer_path).expanduser()
    if not path.is_dir():
        raise ProviderUnavailable("Local Qwen tokenizer is unavailable.")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                             trust_remote_code=False)
    except (ImportError, OSError, ValueError, RuntimeError):
        raise ProviderUnavailable("Local Qwen tokenizer could not be loaded.") from None


class ServiceTokenizer:
    """Count with the provisioned service's exact tokenizer/chat template.

    The local workshop runtime supplies this endpoint so the Streamlit environment
    needs neither Torch nor Transformers. Messages follow the same data boundary.
    """
    def __init__(self, config):
        self.config = config

    def apply_chat_template(self, messages, **kwargs):
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.post(self.config.base_url.rstrip("/") + "/tokenize",
                                  json={"model": self.config.model, "messages": messages}, headers=headers,
                                  timeout=min(10, self.config.timeout_seconds), allow_redirects=False,
                                  stream=True) as response:
                    if response.status_code != 200:
                        raise ProviderUnavailable("Qwen tokenizer service is unavailable.")
                    content = bytearray()
                    for chunk in response.iter_content(1024):
                        content.extend(chunk)
                        if len(content) > 4096:
                            raise ProviderUnavailable("Qwen tokenizer response is invalid.")
                    count = json.loads(content)["count"]
                    if type(count) is not int or not 0 <= count <= 1_000_000:
                        raise ProviderUnavailable("Qwen tokenizer count is invalid.")
                    return range(count)
        except (requests.RequestException, KeyError, ValueError, TypeError):
            raise ProviderUnavailable("Qwen tokenizer service is unavailable.") from None


class QwenProvider:
    """Request resources are per call; cached clients carry no session documents."""

    def __init__(self, config: QwenConfig, *, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config)
        self.max_context_tokens = config.max_context_tokens
        self.max_output_tokens = config.max_output_tokens
        self.last_input_tokens = None
        self._service_metadata = None

    def verify_service(self):
        """Attest the provisioned loopback service against its expected artifact."""
        if not self.config.expected_sha256:
            return
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(self.config.base_url.rstrip("/") + "/models", headers=headers,
                                 timeout=min(10, self.config.timeout_seconds), allow_redirects=False,
                                 stream=True) as response:
                    if response.status_code != 200:
                        raise ProviderUnavailable("Pinned Qwen service metadata is unavailable.")
                    content = bytearray()
                    for chunk in response.iter_content(4096):
                        content.extend(chunk)
                        if len(content) > 16384:
                            raise ProviderUnavailable("Qwen service metadata is invalid.")
                    models = json.loads(content)["data"]
                    matches = [m for m in models if isinstance(m, dict) and m.get("id") == self.config.model]
                    if len(matches) != 1:
                        raise ProviderUnavailable("Pinned Qwen service model is unavailable.")
                    model = matches[0]
                    if (model.get("revision") != self.config.revision
                            or model.get("sha256") != self.config.expected_sha256
                            or type(model.get("max_context_tokens")) is not int
                            or type(model.get("max_output_tokens")) is not int
                            or model["max_context_tokens"] < self.max_context_tokens
                            or model["max_output_tokens"] < self.max_output_tokens):
                        raise ProviderUnavailable("Qwen service differs from the configured model artifact or budget.")
                    self._service_metadata = {k: model.get(k) for k in
                        ("id", "revision", "sha256", "runtime", "device", "dtype", "generation", "peak_rss_bytes")}
        except (requests.RequestException, KeyError, ValueError, TypeError):
            raise ProviderUnavailable("Pinned Qwen service metadata is unavailable.") from None

    def provenance(self):
        return {"model": self.config.model, "revision": self.config.revision,
                "runtime": self.config.runtime, "quantization": self.config.quantization,
                "context_tokens": self.max_context_tokens, "output_tokens": self.max_output_tokens,
                "temperature": self.config.temperature, "enable_thinking": self.config.enable_thinking,
                "served_model": self._service_metadata}

    def _messages(self, messages):
        if not messages or len(messages) > 32:
            raise ContextOverflow("A bounded nonempty conversation is required.")
        safe = []
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError("Only text chat messages are supported.")
            safe.append({"role": message["role"], "content": redact_content(message["content"])})
        return safe

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        self.verify_service()
        kwargs = {"enable_thinking": self.config.enable_thinking} if self.config.enable_thinking is not None else {}
        try:
            tokens = self.tokenizer.apply_chat_template(self._messages(messages), tokenize=True,
                                                       add_generation_prompt=True, **kwargs)
            return len(tokens)
        except (TypeError, ValueError, RuntimeError):
            raise ProviderUnavailable("Qwen chat template/tokenizer is unavailable.") from None

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        safe = self._messages(messages)
        self.last_input_tokens = self.count_tokens(safe)
        if self.last_input_tokens + self.max_output_tokens > self.max_context_tokens:
            raise ContextOverflow("Request exceeds the configured context budget.")
        body = {"model": self.config.model, "messages": safe, "stream": False,
                "max_tokens": self.max_output_tokens, "temperature": self.config.temperature}
        if self.config.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.config.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        # No proxy inherited from environment and no cross-host redirect carrying documents/secrets.
        with requests.Session() as session:
            session.trust_env = False
            deadline = monotonic() + self.config.timeout_seconds
            for attempt in range(2):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ProviderUnavailable("Qwen request exceeded its time budget.")
                try:
                    with session.post(self.config.base_url.rstrip("/") + "/chat/completions",
                                      json=body, headers=headers,
                                      timeout=(min(5.0, remaining), remaining),
                                      allow_redirects=False, stream=True) as response:
                        if response.status_code in TRANSIENT_STATUS and attempt == 0:
                            continue
                        if response.status_code != 200:
                            raise ProviderUnavailable("Qwen service rejected the request.")
                        content = bytearray()
                        for chunk in response.iter_content(8192):
                            if monotonic() > deadline:
                                raise ProviderUnavailable("Qwen request exceeded its time budget.")
                            content.extend(chunk)
                            if len(content) > RESPONSE_BYTES:
                                raise ProviderUnavailable("Qwen response exceeds the configured limit.")
                        payload = json.loads(content)
                        choice = payload["choices"][0]
                        if choice.get("finish_reason") != "stop":
                            raise ProviderUnavailable("Qwen did not return a complete answer.")
                        text = choice["message"]["content"]
                        if not isinstance(text, str) or not text.strip():
                            raise ProviderUnavailable("Qwen returned no text answer.")
                        return redact_content(text)
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 0:
                        continue
                    raise ProviderUnavailable("Qwen service is unavailable or timed out.") from None
                except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
                    raise ProviderUnavailable("Qwen returned an invalid response.") from None
        raise ProviderUnavailable("Qwen service is unavailable.")
