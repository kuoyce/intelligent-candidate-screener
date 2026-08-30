"""Keyword boosting — the hand-written skill list, and its digest *(Phase 4, D19)*.

`SKILL_PATTERNS` is a **fixed, hand-written keyword list**. It is not learned, not
derived from a vocabulary, and not exhaustive: it covers eight families of technology
term that happened to be salient when the baseline was written. Naming that here is
the point — a reader who assumes it is vocabulary-derived will misread every score
it produces.

`data/vocab/` (task 3.5) will make a real ESCO-backed skill vocabulary available.
Consuming it is a **modelling change**, deliberately out of scope for the phase that
made this baseline repeatable *(assumption A9)*, and it is the obvious next step.

The list is digested into `baseline-metrics.json` (`digest()`), so an edit to it can
never slide past `run --check` as an unexplained metric move.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

#: Eight families of technology keyword, matched case-insensitively against the
#: lowercased document. Every pattern has exactly one capture group, so `findall`
#: returns the matched term itself rather than a tuple.
SKILL_PATTERNS = [
    r'\b(python|java|c\+\+|sql|javascript|typescript|go|rust|scala|perl)\b',
    r'\b(tensorflow|keras|pytorch|scikit-learn|pandas|numpy)\b',
    r'\b(aws|gcp|azure|kubernetes|docker|jenkins|gitlab|github)\b',
    r'\b(react|angular|vue|node\.?js)\b',
    r'\b(postgres|mongodb|redis|cassandra|mysql|elasticsearch)\b',
    r'\b(rest|graphql|rpc|soap)\b',
    r'\b(agile|scrum|kanban)\b',
    r'\b(git|svn|mercurial)\b',
]

#: Each match is emitted twice. This is the whole of the "boost": appending the
#: doubled terms to the document raises their term frequency without touching the
#: TF-IDF configuration. It is a crude lever and is recorded as one.
REPEAT = 2


def extract_skills(text: object) -> str:
    """Return the matched skill terms, each repeated `REPEAT` times, space-joined.

    Returns `""` for empty, `None` or no-match input — never `None`, so the result
    is always safe to concatenate onto a document.
    """
    if not text:
        return ""
    skills: list[str] = []
    lower_text = str(text).lower()
    for pattern in SKILL_PATTERNS:
        matches = re.findall(pattern, lower_text, re.IGNORECASE)
        skills.extend(matches * REPEAT)
    return " ".join(skills)


def augment(text: pd.Series) -> pd.Series:
    """`text + " " + extract_skills(text)`, the representation both scorers consume.

    One definition instead of the four inline expressions the notebook carried: a
    resume augmented one way and a JD augmented another would put the two sides of
    a pair in subtly different representations and nothing would raise.
    """
    return text + " " + text.map(extract_skills)


def digest() -> str:
    """sha256 over the pattern list — a silent edit becomes a visible check failure."""
    joined = "\n".join(SKILL_PATTERNS) + f"\nrepeat={REPEAT}\n"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
