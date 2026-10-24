"""Session-owned, source-mapped passages and deterministic lexical evidence search."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

VERSION = "question-evidence-lexical-v1"
_WORDS = re.compile(r"(?:\.?[\w]+(?:[.+#-][\w]+)*(?:\+\+|#)?|\S)", re.UNICODE)
_STOP = frozenset("a an and are as at be by can candidate did do does for from has have how i in is it of on or show that the their this to was were what which who with years".split())
_EXPANSIONS = {"certifications": {"certification", "certified", "certificate", "certificates"},
               "certification": {"certifications", "certified", "certificate", "certificates"}}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _terms(text: str) -> list[str]:
    return [word.casefold() for word in _WORDS.findall(text)
            if any(char.isalnum() for char in word) and word.casefold() not in _STOP]


@dataclass(frozen=True)
class EvidencePassage:
    snapshot_id: str
    document_id: str
    candidate_id: str | None
    document_digest: str
    passage_id: str
    original_start: int
    original_end: int
    display_start: int
    display_end: int
    quote: str
    section: str | None = None
    page: int | None = None
    score: float = 0.0

    @property
    def evidence_id(self):
        return self.passage_id

    @property
    def text(self):
        return self.quote


def _chunks(text: str, maximum: int, overlap: int) -> list[tuple[int, int]]:
    """Prefer complete lines/sentences; oversized units retain every character.

    This lexical index has a character budget, not a neural tokenizer budget.
    Downstream neural consumers must independently prepare bounded token inputs.
    """
    if not text:
        return []
    breaks = sorted({0, len(text), *(m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text))})
    spans, start = [], 0
    while start < len(text):
        ceiling = min(len(text), start + maximum)
        preferred = [point for point in breaks if start < point <= ceiling]
        end = max(preferred) if preferred else ceiling
        if end < len(text) and not preferred:
            whitespace = [match.end() for match in re.finditer(r"\s+", text[start:ceiling])]
            if whitespace:
                end = start + whitespace[-1]
        spans.append((start, end))
        if end == len(text):
            break
        target = max(start + 1, end - overlap)
        # Prefer a prior whole sentence/line as overlap, with guaranteed progress.
        previous = [point for point in breaks if target <= point < end and point > start]
        if previous:
            start = min(previous)
        else:
            # Avoid cutting a normal word solely to add context. An individually
            # overlong unit still advances via the hard character bound above.
            boundary = re.search(r"\s+", text[target:end])
            start = target + boundary.end() if boundary else end
    return spans


@dataclass(frozen=True)
class EvidenceIndex:
    snapshot_id: str
    passages: tuple[EvidencePassage, ...]
    index_digest: str
    _by_id: Mapping[str, EvidencePassage] = field(repr=False, compare=False)
    _candidate_ids: frozenset[str] = field(repr=False)

    @classmethod
    def from_sources(cls, snapshot_id: str, jd_text: str,
                     candidate_texts: Mapping[str, str], *, max_chars=1200,
                     overlap_chars=200) -> "EvidenceIndex":
        from candidate_screener.entity_annotation.text import prepare_text
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("Evidence requires an active snapshot identity")
        if (type(max_chars) is not int or max_chars < 32 or type(overlap_chars) is not int or
                not 0 <= overlap_chars < max_chars):
            raise ValueError("Invalid passage budget/overlap")
        if not isinstance(jd_text, str) or not isinstance(candidate_texts, Mapping):
            raise ValueError("JD and candidate sources must be strings and an ID-to-text mapping")
        if any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in candidate_texts.items()):
            raise ValueError("Candidate sources require nonempty stable string IDs and source text")
        passages, source_identities = [], []
        sources = [(None, "jd", jd_text)] + [(identity, "cv:" + identity, text) for identity, text in sorted(candidate_texts.items())]
        for candidate_id, document_id, original in sources:
            prepared = prepare_text(original)
            display = prepared["display_text"]
            source_identities.append((document_id, prepared["source_sha256"], prepared["display_sha256"],
                                      prepared["redaction_version"]))
            # Chunk redacted display, then map boundaries to original. Never cut
            # inside a redaction replacement with unavailable source boundaries.
            boundaries = prepared["boundary_map"]
            for start, end in _chunks(display, max_chars, overlap_chars):
                while start > 0 and boundaries[start] is None:
                    start -= 1
                while end < len(display) and boundaries[end] is None:
                    end += 1
                if start == end or not display[start:end].strip():
                    continue
                source_start, source_end = boundaries[start], boundaries[end]
                identity = json.dumps((VERSION, snapshot_id, document_id, prepared["source_sha256"],
                                       prepared["redaction_version"], source_start, source_end), separators=(",", ":"))
                passage_id = "passage:" + _digest(identity)[:32]
                # Headings remain redacted source excerpts rather than invented
                # metadata. Plain-text sources provide no reliable page numbers.
                preceding = display[:start].splitlines()
                heading = next((line.strip() for line in reversed(preceding) if line.strip()), "")
                section = heading if heading and len(heading) <= 80 and (heading.endswith(":") or heading.isupper()) else None
                passages.append(EvidencePassage(snapshot_id, document_id, candidate_id,
                    prepared["source_sha256"], passage_id, source_start, source_end,
                    start, end, display[start:end], section))
        unique = {passage.passage_id: passage for passage in passages}
        identity = json.dumps((VERSION, snapshot_id, source_identities, max_chars, overlap_chars), separators=(",", ":"))
        return cls(snapshot_id, tuple(unique.values()), _digest(identity),
                   MappingProxyType(unique), frozenset(candidate_texts))

    def get(self, passage_id: str, *, candidate_id: str | None) -> EvidencePassage:
        passage = self._by_id.get(passage_id)
        if passage is None or passage.candidate_id != candidate_id or passage.snapshot_id != self.snapshot_id:
            raise ValueError("Evidence is unavailable for this snapshot/document owner")
        return passage

    def search(self, question: str, candidate_ids, *, limit=8, per_candidate=2,
               include_jd=True) -> tuple[EvidencePassage, ...]:
        if not isinstance(question, str):
            raise ValueError("Evidence question must be a string")
        if (type(limit) is not int or limit < 0 or type(per_candidate) is not int or per_candidate < 1):
            raise ValueError("Evidence search needs nonnegative limit and positive per-candidate cap")
        if isinstance(candidate_ids, (str, bytes)) or candidate_ids is None or type(include_jd) is not bool:
            raise ValueError("Candidate IDs must be an explicit collection and JD inclusion must be boolean")
        selected = tuple(dict.fromkeys(candidate_ids))
        if any(not isinstance(identity, str) for identity in selected):
            raise ValueError("Requested candidate IDs must be strings")
        if any(identity not in self._candidate_ids for identity in selected):
            raise ValueError("Requested candidate is outside this evidence index")
        query = set(_terms(question))
        for term in tuple(query):
            query.update(_EXPANSIONS.get(term, ()))
        if not query or limit == 0:
            return ()
        permitted = [p for p in self.passages if p.candidate_id in selected or (include_jd and p.candidate_id is None)]
        counts = {p.passage_id: Counter(_terms(p.quote)) for p in permitted}
        frequencies = {term: sum(term in counts[p.passage_id] for p in permitted) for term in query}
        ranked = []
        for passage in permitted:
            words = counts[passage.passage_id]
            score = sum((1 + math.log(words[term])) * math.log(1 + (len(permitted) + 1) / (frequencies[term] + 1))
                        for term in query if words[term])
            if score > 0:
                ranked.append(replace(passage, score=score / (1 + .001 * sum(words.values()))))
        ranked.sort(key=lambda p: (-p.score, p.document_id, p.original_start, p.passage_id))
        # Give each requested candidate a relevant passage before extra passages.
        output, chosen, owner_counts = [], set(), Counter()
        for owner in selected:
            best = next((p for p in ranked if p.candidate_id == owner), None)
            if best and len(output) < limit:
                output.append(best); chosen.add(best.passage_id); owner_counts[owner] += 1
        for passage in ranked:
            if len(output) >= limit:
                break
            if passage.passage_id in chosen or owner_counts[passage.candidate_id] >= per_candidate:
                continue
            output.append(passage); chosen.add(passage.passage_id); owner_counts[passage.candidate_id] += 1
        return tuple(output)
