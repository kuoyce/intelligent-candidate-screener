"""Longest-match, case-insensitive phrase matcher over the skill vocabulary (task 2.2).

Text is split into tokens: maximal runs of letters/digits, and every other non-space
character on its own. An alias is the same token sequence plus, between each pair of
tokens, whether they touch (`C++`, `Node.js`, `CI/CD`) or are separated by spaces
(`machine learning`). A match must reproduce that pattern exactly, so `C++` never
matches `C + +` and `java` never matches inside `javascript` (a token is a maximal run,
which is the word boundary). A whitespace gap containing a line break never continues a
phrase: in a resume a line break separates list items.

Matching is leftmost-longest and non-overlapping. Offsets index the caller's string.

The separate ``extract_document`` API retains deterministic seed-vocabulary mentions
and clause-bounded context flags for immutable document fields.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from candidate_screener.config import VOCAB

from .schema import EXTRACTOR_VERSION, validate_entity
from .vocabulary import VOCABULARY

VOCAB_CSV = VOCAB / "skills.csv"

#: Letters/digits as one token; any other non-space character is a token by itself.
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]|_")

_TOUCH, _SPACE, _BREAK = 0, 1, 2  # gap kinds between consecutive tokens


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    text: str
    canonical: str


def tokenize(text: str) -> list[tuple[int, int, str]]:
    """`(start, end, lowercased token)` for every token of `text`."""
    return [(m.start(), m.end(), m.group().lower()) for m in _TOKEN.finditer(text)]


def _gap(text: str, prev_end: int, start: int) -> int:
    if prev_end == start:
        return _TOUCH
    return _BREAK if ("\n" in text[prev_end:start] or "\r" in text[prev_end:start]) else _SPACE


def alias_key(alias: str) -> tuple:
    """The token/gap pattern an alias must reproduce; `()` for an alias with no token."""
    toks = tokenize(alias)
    key: list = []
    for i, (s, _e, t) in enumerate(toks):
        key.append((None if i == 0 else _gap(alias, toks[i - 1][1], s), t))
    return tuple(key)


class SkillMatcher:
    """Phrase matcher over a vocabulary frame with columns `canonical` and `alias`.

    When two rows share an alias pattern (case-insensitively), the first row wins, so
    the result is a deterministic function of the frame.
    """

    def __init__(self, vocab: pd.DataFrame):
        self._trie: dict = {}
        self.n_aliases = 0
        for alias, canonical in zip(vocab["alias"].astype(str), vocab["canonical"].astype(str)):
            key = alias_key(alias)
            if not key or any(g == _BREAK for g, _ in key):
                continue
            node = self._trie
            for step in key:
                node = node.setdefault(step[1] if step[0] is None else step, {})
            if None not in node:  # first row wins
                node[None] = canonical
                self.n_aliases += 1

    def find(self, text: str) -> list[Match]:
        """Leftmost-longest, non-overlapping matches, sorted by start."""
        toks = tokenize(text)
        out: list[Match] = []
        i, n = 0, len(toks)
        while i < n:
            node = self._trie.get(toks[i][2])
            best = None
            j = i
            while node is not None:
                if None in node:
                    best = (j, node[None])
                if j + 1 >= n:
                    break
                node = node.get((_gap(text, toks[j][1], toks[j + 1][0]), toks[j + 1][2]))
                j += 1
            if best is None:
                i += 1
                continue
            last, canonical = best
            start, end = toks[i][0], toks[last][1]
            out.append(Match(start, end, text[start:end], canonical))
            i = last + 1
        return out

    def canonicals(self, text: str) -> set[str]:
        return {m.canonical for m in self.find(text)}


def load_vocab(path: Path = VOCAB_CSV) -> pd.DataFrame:
    """Read the vocabulary with every cell a string (`NA`, `null` are skills, not NaN)."""
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; build it with "
            "`uv run python -m candidate_screener.extraction.vocab --build`")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


@lru_cache(maxsize=1)
def default_matcher() -> SkillMatcher:
    """The matcher over the committed `data/vocab/skills.csv`."""
    return SkillMatcher(load_vocab())


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    # Punctuation and contrast delimit bounded contexts; the 100-character bound
    # prevents a distant qualification from silently applying to a later mention.
    before = re.split(r"[;.!?\n]|\b(?:but|however)\b", text[:start], flags=re.I)[-1][-100:]
    after = re.split(r"[;.!?\n]|\b(?:but|however)\b", text[end:], flags=re.I)[0][:100]
    return before, after


def _context(text: str, start: int, end: int, field: str) -> str:
    before, after = _clause(text, start, end)
    if re.search(r"\b(?:no|without|lack(?:ing|s)?(?:\s+of)?|not)\s*$|\b(?:no|without|lack(?:ing|s)?(?:\s+of)?)\s+(?:experience|knowledge|skills?|proficiency|background)(?:\s+(?:in|with|of))?\s*$|\bnever\s+(?:used|worked\s+with)\s*$", before, re.I) or re.match(
        r"\s+(?:experience\s+)?(?:not\s+(?:required|used)|is\s+not\s+required)\b", after, re.I
    ):
        return "negated"
    if field.casefold() == "looking for" or re.search(
        r"\b(?:learn(?:ing)?|studying|aspir(?:e|ing)|want(?:ing)?\s+to|interested\s+in)\b", before, re.I
    ):
        return "learning"
    return "unqualified"


def _ambiguous_allowed(surface: str, text: str, start: int, end: int, field: str) -> bool:
    if surface.casefold() not in {"go", "r"}:
        return True
    if surface not in {"Go", "R"}:  # Case is meaningful for these two aliases.
        return False
    before, after = _clause(text, start, end)
    context = before + surface + after
    return bool(re.search(r"\b(?:programming|language|developer|development|code|coding|golang|statistics|statistical)\b", context, re.I))


def extract_document(doc: dict) -> list[dict]:
    """Extract exact mentions independently per immutable source field."""
    required = ("doc_id", "source", "doc_type", "fields")
    if any(key not in doc for key in required):
        raise ValueError("Document requires doc_id, source, doc_type and fields")
    result = []
    for field, text in sorted(doc["fields"].items()):
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError("Document fields must contain strings or null")
        candidates = []
        for label, entries in VOCABULARY.items():
            for canonical, aliases in entries.items():
                for alias in aliases:
                    pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                    for match in re.finditer(pattern, text, re.I):
                        if _ambiguous_allowed(match.group(), text, *match.span(), field):
                            candidates.append((match.start(), match.end(), label, canonical))
        # Prefer longest spans globally, then restore source order.
        accepted = []
        for candidate in sorted(candidates, key=lambda c: (-(c[1]-c[0]), c[0], c[2], c[3])):
            if not any(candidate[0] < other[1] and other[0] < candidate[1] for other in accepted):
                accepted.append(candidate)
        for start, end, label, canonical in sorted(accepted):
            record = {"doc_id": str(doc["doc_id"]), "source": doc["source"],
                      "doc_type": doc["doc_type"], "field": field,
                      "text_version": doc.get("text_version", "raw-v1"),
                      "start": start, "end": end, "label": label, "surface": text[start:end],
                      "canonical_id": f"seed:{canonical}", "method": "seed-rule",
                      "extractor_version": EXTRACTOR_VERSION, "confidence": None,
                      "context_status": _context(text, start, end, field)}
            record["entity_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            validate_entity(record, text)
            result.append(record)
    return result
