"""Minimal provider contract, independent of Streamlit and optional ML packages."""
from __future__ import annotations

from typing import Protocol, Sequence
import json
import re

from candidate_screener.annotation.redact import redact_text


def redact_content(content: str) -> str:
    """Redact textual JSON leaves without corrupting numeric facts or delimiters.

    Running regexes over JSON can consume a URL's closing quote or replace a
    float's fractional digits with a nonnumeric marker. Plain text still uses
    the standard project redactor; schema validation is the caller's responsibility.
    """
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", content.strip(), re.I)
    if fenced is not None:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return redact_text(content)
    if not isinstance(value, (dict, list)):
        return redact_text(content)

    def visit(node):
        if isinstance(node, str):
            return redact_text(node)
        if isinstance(node, list):
            return [visit(item) for item in node]
        if isinstance(node, dict):
            return {redact_text(key): visit(item) for key, item in node.items()}
        return node
    return json.dumps(visit(value), ensure_ascii=False, allow_nan=False)


class ProviderUnavailable(RuntimeError):
    """Safe, document/secret-free provider failure for template fallback."""


class ContextOverflow(ValueError):
    """A request cannot fit the configured tokenizer/context/output budget."""


class Provider(Protocol):
    max_context_tokens: int
    max_output_tokens: int

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int: ...

    def complete(self, messages: Sequence[dict[str, str]]) -> str: ...
