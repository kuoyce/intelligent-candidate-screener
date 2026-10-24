"""Source-only context windows for multi-answer qualification questions.

This module identifies context units, never certifications or valid/current
qualifications. The extractor must still abstain when a window lacks an answer.
"""
from __future__ import annotations
import re

VERSION = "source-list-context-v1"
_QUALIFICATION = re.compile(r"\b(?:certifications?|certificates?|degrees?|qualifications?|licen[cs]es?|credentials?)\b", re.I)
_LIST_INTENT = re.compile(r"\b(?:what|which|list|all|enumerate|show|name)\b", re.I)
_CONTINUATION = re.compile(
    r"^(?:(?:it|this|that|these|they)\b|the\s+(?:qualification|certificate|certification|degree|licen[cs]e|credential)\b|"
    r"(?:expired?|expires|valid|issued|renewed|status|not|no|never|currently|ongoing|pending|pursuing|studying|planned|in\s+progress)\b|\d{4}\b)", re.I)


class WindowList(list):
    """List-compatible windows with explicit bounded retrieval coverage."""
    def __init__(self, windows, *, total_windows, is_list):
        super().__init__(windows)
        self.total_windows = total_windows
        self.incomplete = len(self) < total_windows
        self.is_list = is_list
        # Even a complete source partition does not prove an exhaustive answer.
        self.exhaustive = False


def is_list_question(question: str) -> bool:
    if not isinstance(question, str):
        raise ValueError("List QA question must be a string")
    return bool(_QUALIFICATION.search(question) and _LIST_INTENT.search(question))


def _units(text):
    boundaries = sorted({0, len(text), *(m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text))})
    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]


def _heading(text):
    text = text.strip()
    return bool(text and len(text) <= 80 and not re.match(r"^(?:[-*•]|\d+[.)])\s*", text) and
                (text.endswith(":") or (text.isupper() and any(c.isalpha() for c in text))))


def list_windows(question: str, passage_dict: dict, *, limit: int = 32) -> WindowList:
    """Return exact quote slices plus quote-local Python character offsets.

    A qualification section is retained as one context, and its individual lines/
    sentences are also queried so a one-span extractor can recover multiple items.
    All contexts are contiguous source slices: no headings, qualifiers or dates
    are fabricated or joined from distant locations. Repeated model answers are
    deduplicated by their remapped original passage offsets by the caller.
    """
    listed = is_list_question(question)
    if not isinstance(passage_dict, dict) or not isinstance(passage_dict.get("quote"), str):
        raise ValueError("List QA needs an immutable passage quote")
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("List QA window limit must be between one and32")
    text = passage_dict["quote"]
    if not text.strip():
        return WindowList([], total_windows=0, is_list=listed)
    if not listed:
        return WindowList([(text, 0)], total_windows=1, is_list=False)
    units = _units(text)
    spans = []
    # Individual contexts come first so a long section cannot monopolize a
    # bounded multi-answer call. Qualifier continuations stay with their item.
    for index, (start, end) in enumerate(units):
        stop = index + 1
        while stop < len(units) and _CONTINUATION.search(text[units[stop][0]:units[stop][1]].strip()):
            end = units[stop][1]
            stop += 1
        spans.append((start, end))
    # Preserve complete source sections including punctuation-free bullet lists.
    headings = [i for i, (a, b) in enumerate(units) if _heading(text[a:b])]
    for position, index in enumerate(headings):
        start, heading_end = units[index]
        heading_text = text[start:heading_end]
        if not (_QUALIFICATION.search(heading_text) or re.search(r"\beducation\b", heading_text, re.I)):
            continue
        next_heading = headings[position + 1] if position + 1 < len(headings) else len(units)
        end = units[next_heading - 1][1] if next_heading > index else heading_end
        spans.append((start, end))
    unique = list(dict.fromkeys(spans))
    windows = [(text[start:end], start) for start, end in unique[:limit]]
    return WindowList(windows, total_windows=len(unique), is_list=True)
