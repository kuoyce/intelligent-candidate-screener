"""C2: the CV's **set** of role families (plan 2026-10-07-shortlist-recall, 01 §2 C2).

A person may have held several roles (decision 1b), so a CV gets a set:

    roles(cv) = {families of the CV's title spans} ∪ {the role classifier's top-k families}

- Spans: P3's `title_spans` against its alias table, **every** span, not just the first.
- Option `context` (chosen on dev-select): count only spans in a **role context**, to drop
  mentions such as "worked with the QA team". The rule, fixed before any dev figure:
  a span is in a role context when, on its own line,
    (a) nothing but bullets, numbering, punctuation or spaces precedes it (the line starts
        with the title), or
    (b) the text before it ends with a cue: `position`, `role`, `title`, `job title`,
        `occupation`, `worked as`, `working as`, `work as`, `as a(n)`, `i am a(n)`,
        `i'm a(n)`, optionally followed by `:` or a dash, or
    (c) the line is a work-history header: at most `HEADER_MAX_CHARS` long and holding a
        date range (P4's CV range pattern, e.g. `2019 - 2021`, `Mar 2020 – present`).
- Option `k` in {0, 1, 2, 3} (chosen on dev-select): the classifier's top-k families,
  from P3's CV-side model (text only).

Only CV **text** is read. `Position`, `Primary Keyword` and the experience fields never are.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from candidate_screener.extraction import role_family as rf
from candidate_screener.extraction.experience import _RANGE

KS = (0, 1, 2, 3)
HEADER_MAX_CHARS = 150
_LEAD = re.compile(r"^[\s\W\d_]*$")
_CUE = re.compile(r"(?:\bposition|\brole|\btitle|\bjob\s+title|\boccupation|\bworked\s+as"
                  r"|\bworking\s+as|\bwork\s+as|\bas\s+an?|\bi\s+am\s+an?|\bi'm\s+an?)"
                  r"\s*[:\-–—]?\s*$", re.I)


@dataclass(frozen=True)
class CvRoles:
    """What C2 needs from one CV, computed once: its spans and the classifier's ranking."""
    spans: tuple[tuple[str, str, bool], ...]   # (family, normalised alias, in role context)
    top: tuple[str, ...]                       # classifier families, best first (len max(KS))

    def families(self, k: int, context: bool) -> frozenset[str]:
        if k not in KS:
            raise ValueError(f"k must be one of {KS}")
        named = {f for f, _, ctx in self.spans if ctx or not context}
        return frozenset(named | set(self.top[:k]))

    def titles(self, context: bool) -> tuple[str, ...]:
        return tuple(t for _, t, ctx in self.spans if ctx or not context)


def in_role_context(text: str, start: int) -> bool:
    """Rule (a)-(c) of the module docstring for a span starting at `start`."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    line_end = len(text) if line_end < 0 else line_end
    prefix = text[line_start:start]
    if _LEAD.match(prefix):
        return True
    if _CUE.search(prefix):
        return True
    line = text[line_start:line_end]
    return len(line.strip()) <= HEADER_MAX_CHARS and bool(_RANGE.search(line))


def cv_roles(texts: Sequence[str], model: rf.RoleModel | None = None) -> list[CvRoles]:
    """One `CvRoles` per text. Text only: no metadata field is read."""
    model = model or rf.load_role_model()
    texts = ["" if rf._missing(t) else str(t) for t in texts]
    proba = rf.predict_proba(texts, "cv", model) if texts else np.zeros((0, len(model.labels)))
    order = np.argsort(-proba, axis=1, kind="stable")[:, :max(KS)]
    cache: dict[str, tuple] = {}
    out = []
    for text, row in zip(texts, order):
        if text not in cache:
            cache[text] = tuple((s.family, s.title, in_role_context(text, s.start))
                                for s in rf.title_spans(text, model.aliases))
        out.append(CvRoles(cache[text], tuple(model.labels[j] for j in row)))
    return out
