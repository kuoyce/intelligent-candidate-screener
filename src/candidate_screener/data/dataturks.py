"""DataTurks (B1) resume entities, with the offset defect repaired (task 5.1, was 3.1).

**The defect's cause is HTML escaping, not inclusive/exclusive `end`.** `content` holds
`&amp;` (177), `&lt;` and `&gt;` (2 each), while the offsets index the *unescaped* text:
every entity before a span shifts it by 4 (or 3) characters, which is why all 222 broken
spans are off by a positive multiple of 4 (+4 .. +36). Repair, in order:

1. `intact`: the stored text equals its slice of the raw content.
2. `unescape`: it equals its slice of the unescaped content (the root-cause repair).
3. `search`: the nearest occurrence of the stored text within `SEARCH_WINDOW` characters
   of the stored start (a fallback; on the shipped files it repairs nothing).
4. Otherwise the span is **dropped**, and counted.

Every surviving span is half-open, whitespace-trimmed and satisfies
`doc.text[span.start:span.end] == span.text`. `Name` and `Email Address` spans are
dropped entirely, and by default their characters, plus anything `annotation.redact`'s
patterns detect, are overwritten in place with `X` so offsets survive the masking.
Never print a document: they are real resumes.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from candidate_screener.config import RAW

DATATURKS = RAW / "dataturks"
FILES = {"train": "traindata.json", "test": "testdata.json"}
PII_LABELS = frozenset({"Name", "Email Address"})
SKILLS = "Skills"
SEARCH_WINDOW = 64
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str


@dataclass
class Document:
    doc_id: str
    split: str
    text: str
    spans: list[Span] = field(default_factory=list)

    def label_spans(self, label: str = SKILLS) -> list[tuple[int, int]]:
        return sorted({(s.start, s.end) for s in self.spans if s.label == label})


def unescape(content: str) -> str:
    """Decode HTML character references (only complete `&…;` ones)."""
    return _ENTITY.sub(lambda m: html.unescape(m.group()), content)


def repair(raw: str, text: str, start: int, end_inclusive: int, stored: str,
           window: int = SEARCH_WINDOW) -> tuple[int, int, str] | None:
    """`(start, end, method)` on the unescaped `text`, trimmed of whitespace, or None."""
    core = stored.strip()
    if not core:
        return None
    lead = len(stored) - len(stored.lstrip())
    if raw[start:end_inclusive + 1] == stored and text[start:end_inclusive + 1] == stored:
        method, s = "intact", start + lead
    elif text[start:end_inclusive + 1] == stored:
        method, s = "unescape", start + lead
    else:
        target, hits, i = start + lead, [], text.find(core)
        while i != -1:
            hits.append(i)
            i = text.find(core, i + 1)
        near = [h for h in hits if abs(h - target) <= window]
        if not near:
            return None
        method, s = "search", min(near, key=lambda h: (abs(h - target), h))
    return s, s + len(core), method


def mask(text: str, regions: list[tuple[int, int]]) -> str:
    """Overwrite non-space characters of `regions` and of redact's patterns with `X`."""
    from candidate_screener.annotation.redact import PATTERNS

    chars = list(text)
    spans = list(regions) + [m.span() for _, p, _ in PATTERNS for m in p.finditer(text)]
    for s, e in spans:
        for i in range(s, e):
            if not chars[i].isspace():
                chars[i] = "X"
    return "".join(chars)


def parse(lines: list[str], split: str, *, masked: bool = True) -> tuple[list[Document], Counter]:
    """Documents and the repair ledger from JSON lines (tests pass synthetic lines)."""
    docs, ledger = [], Counter()
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            ledger["malformed_lines"] += 1
            continue
        raw = rec["content"]
        text = unescape(raw)
        spans, pii = [], []
        for ann in rec.get("annotation") or []:
            label = (ann or {}).get("label")
            if not label:
                ledger["dropped_unlabelled"] += 1
                continue
            label = label[0] if isinstance(label, list) else label
            for p in ann.get("points", []):
                ledger["spans_total"] += 1
                fixed = repair(raw, text, p["start"], p["end"], p.get("text") or "")
                if fixed is None:
                    ledger["dropped_unrepairable"] += 1
                    continue
                s, e, method = fixed
                ledger[method] += 1
                if label in PII_LABELS:
                    ledger["dropped_pii"] += 1
                    pii.append((s, e))
                    continue
                spans.append((s, e, label))
        if masked:
            text = mask(text, pii)
        doc = Document(f"{split}-{len(docs):03d}", split, text,
                       [Span(s, e, lab, text[s:e]) for s, e, lab in sorted(spans)])
        docs.append(doc)
    return docs, ledger


def load(splits: tuple[str, ...] = ("train", "test"), *, masked: bool = True,
         root: Path = DATATURKS) -> tuple[list[Document], dict]:
    """Every document of `splits`, and the repair ledger summed over them."""
    docs, ledger = [], Counter()
    for split in splits:
        d, led = parse((root / FILES[split]).read_text(encoding="utf-8").splitlines(), split,
                       masked=masked)
        docs += d
        ledger += led
    return docs, dict(sorted(ledger.items()))


def to_bio(doc: Document, labels: tuple[str, ...] = (SKILLS,)) -> list[tuple[str, int, int, str]]:
    """Whitespace tokens as `(token, start, end, tag)`, BIO over the spans of `labels`."""
    spans = sorted({(s.start, s.end) for s in doc.spans if s.label in labels})
    out, prev_span = [], None
    for m in re.finditer(r"\S+", doc.text):
        hit = next((sp for sp in spans if sp[0] < m.end() and m.start() < sp[1]), None)
        tag = "O" if hit is None else ("I" if hit == prev_span else "B")
        out.append((m.group(), m.start(), m.end(), tag))
        prev_span = hit
    return out
