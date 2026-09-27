"""Longest-match, case-insensitive phrase matcher over the skill vocabulary (task 2.2).

Text is split into tokens: maximal runs of letters/digits, and every other non-space
character on its own. An alias is the same token sequence plus, between each pair of
tokens, whether they touch (`C++`, `Node.js`, `CI/CD`) or are separated by spaces
(`machine learning`). A match must reproduce that pattern exactly, so `C++` never
matches `C + +` and `java` never matches inside `javascript` (a token is a maximal run,
which is the word boundary). A whitespace gap containing a line break never continues a
phrase: in a resume a line break separates list items.

Matching is leftmost-longest and non-overlapping. Offsets index the caller's string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from candidate_screener.config import VOCAB

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
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; build it with "
            "`uv run python -m candidate_screener.extraction.vocab --build`")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


@lru_cache(maxsize=1)
def default_matcher() -> SkillMatcher:
    """The matcher over the committed `data/vocab/skills.csv`."""
    return SkillMatcher(load_vocab())
