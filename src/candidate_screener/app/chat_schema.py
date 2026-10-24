"""Immutable session state and validated citation contracts for grounded recruiter QA."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    snapshot_id: str
    candidate_id: str
    evidence_id: str
    quote: str
    original_start: int | None
    original_end: int | None
    display_start: int
    display_end: int


@dataclass(frozen=True)
class ChatState:
    snapshot_id: str = ""
    candidate_ids: tuple[str, ...] = ()
    # Only bounded redacted questions are retained; generated prose is never evidence.
    questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatAnswer:
    snapshot_id: str
    text: str
    status: str
    candidate_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    fallback: bool = False
    state: ChatState = ChatState()
