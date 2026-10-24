"""Frozen redaction with exact display-to-source boundary provenance."""
from __future__ import annotations
import hashlib
import inspect

from candidate_screener.annotation import redact as redaction


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


REDACTION_VERSION = digest(inspect.getsource(redaction))


def prepare_text(original: str) -> dict:
    if not isinstance(original, str):
        raise ValueError("Original document must be a string")
    text, boundaries, masked = original, list(range(len(original) + 1)), [False] * len(original)
    for _, pattern, replacement in redaction.PATTERNS:
        parts, mapping, flags, cursor = [], [boundaries[0]], [], 0
        for match in pattern.finditer(text):
            a, b = match.span()
            parts.extend((text[cursor:a], replacement))
            mapping.extend(boundaries[cursor + 1:a + 1])
            mapping.extend([None] * (len(replacement) - 1) + [boundaries[b]])
            flags.extend(masked[cursor:a] + [True] * len(replacement))
            cursor = b
        parts.append(text[cursor:])
        mapping.extend(boundaries[cursor + 1:])
        flags.extend(masked[cursor:])
        text, boundaries, masked = "".join(parts), mapping, flags
    if text != redaction.redact_text(original):
        raise ValueError("Frozen redaction mapping differs from display")
    intervals, start = [], None
    for index, is_masked in enumerate(masked + [False]):
        if is_masked and start is None:
            start = index
        elif not is_masked and start is not None:
            intervals.append([start, index])
            start = None
    return {"display_text": text, "source_sha256": digest(original),
            "display_sha256": digest(text), "redaction_version": REDACTION_VERSION,
            "boundary_map": boundaries, "masked_intervals": intervals}


def utf16_to_python(text: str, offset: int) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("UTF-16 offset must be a nonnegative integer")
    cursor = 0
    for index, character in enumerate(text):
        if cursor == offset:
            return index
        cursor += 2 if ord(character) > 0xFFFF else 1
        if cursor > offset:
            raise ValueError("UTF-16 boundary splits a surrogate pair")
    if cursor == offset:
        return len(text)
    raise ValueError("UTF-16 offset exceeds document")


def map_selection(prepared: dict, start_utf16: int, end_utf16: int) -> tuple[int, int]:
    text = prepared["display_text"]
    if digest(text) != prepared["display_sha256"]:
        raise ValueError("Display digest mismatch")
    start, end = utf16_to_python(text, start_utf16), utf16_to_python(text, end_utf16)
    if start >= end:
        raise ValueError("Selection must be nonempty and ordered")
    if any(start < b and a < end for a, b in prepared["masked_intervals"]):
        raise ValueError("Selection intersects a masked identifier")
    source_start, source_end = prepared["boundary_map"][start], prepared["boundary_map"][end]
    if source_start is None or source_end is None:
        raise ValueError("Selection boundary cannot map to original source")
    return source_start, source_end


def python_to_utf16(text: str, index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(text):
        raise ValueError("Python boundary outside document")
    return len(text[:index].encode("utf-16-le")) // 2
