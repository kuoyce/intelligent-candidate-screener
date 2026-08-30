"""`baselines.lexical` — the shared tokeniser.

`test_token_pattern_matches_sklearn_default` is one of the three named bug
regressions (design spec §5.2). BM25 originally tokenised with `.lower().split()`
while TF-IDF used sklearn's word-boundary regex, so `"python,"` and `"python"` were
different terms to one scorer and the same term to the other.
"""
from __future__ import annotations

from candidate_screener.baselines import tfidf
from candidate_screener.baselines.lexical import TOKEN_PATTERN, tokenize


def test_token_pattern_matches_sklearn_default():
    """REGRESSION: the tokeniser divergence bug, reduced to one assertion.

    If sklearn ever changes its default this fails, which is the point: that is a
    decision to be read and made, not a drift to be discovered later from a metric.
    """
    assert TOKEN_PATTERN.pattern == tfidf.sklearn_token_pattern()


def test_punctuation_is_stripped_not_glued():
    """The concrete shape of the bug: `.lower().split()` gives `['python,']`."""
    assert tokenize("Python, java. SQL!") == ["python", "java", "sql"]


def test_one_character_tokens_and_plus_are_dropped():
    """An accepted property, written down so it is not 'fixed' asymmetrically.

    `\\w\\w+` drops `c++` — and drops it for TF-IDF too. Losing it on both sides is
    what keeps the two scorers comparable; losing it on one side is a bug.
    """
    assert tokenize("Python, Java. C++") == ["python", "java"]


def test_tokenize_lowercases():
    assert tokenize("AWS Docker") == ["aws", "docker"]


def test_empty_input_gives_no_tokens():
    assert tokenize("") == [] and tokenize(None) == []
