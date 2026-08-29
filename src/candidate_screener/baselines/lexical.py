"""One tokeniser, shared by both scorers — the bug this module exists to prevent.

BM25 originally tokenised with `.lower().split()` while TF-IDF used sklearn's
word-boundary regex. `"python,"` and `"python"` were therefore different terms to
one scorer and the same term to the other, which silently suppressed most of the
real resume/JD lexical overlap on the BM25 side. It did not raise, did not change a
row count, and produced a plausible number.

So the pattern has exactly **one** definition here, and `test_lexical.py` asserts it
still equals `TfidfVectorizer().token_pattern`. If sklearn ever changes its default,
that is a test failure to be read and decided on, not a drift to be discovered later
from a metric.
"""
from __future__ import annotations

import re

#: sklearn's `TfidfVectorizer` default, restated once and asserted in tests.
#: Accepted consequence: `\w\w+` drops one-character tokens and `+`, so `c++`
#: survives as neither `c++` nor `c`. It is dropped for *both* scorers, which is
#: what keeps them comparable — do not "fix" it on one side alone.
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: object) -> list[str]:
    """Lowercase, then split on the shared pattern. `""`/`None` give `[]`."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(str(text).lower())
