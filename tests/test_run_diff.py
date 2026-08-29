"""`run._differences` — the comparison rule the golden check is built on.

Runs without `data/`: the tolerance policy is worth testing on its own, because it
is the part of `--check` that decides what counts as a regression *(Q23)*.
"""
from __future__ import annotations

from candidate_screener.baselines.run import FLOAT_TOLERANCE, _differences


def test_identical_documents_have_no_differences():
    doc = {"a": 1, "b": [1.0, 2.0], "c": {"d": "x"}}
    assert _differences(doc, doc) == []


def test_floats_within_tolerance_are_equal():
    assert _differences({"a": 1.0}, {"a": 1.0 + FLOAT_TOLERANCE / 2}) == []


def test_floats_outside_tolerance_differ():
    assert len(_differences({"a": 1.0}, {"a": 1.0 + FLOAT_TOLERANCE * 100})) == 1


def test_integers_are_compared_exactly():
    """A confusion count off by one is a row that changed class — never a rounding."""
    assert len(_differences({"n": 344}, {"n": 345})) == 1


def test_a_missing_field_is_reported_not_ignored():
    assert "missing" in _differences({"a": 1, "b": 2}, {"a": 1})[0]


def test_an_unexpected_field_is_reported():
    assert "unexpected" in _differences({"a": 1}, {"a": 1, "b": 2})[0]


def test_list_length_change_is_reported():
    assert "length" in _differences({"a": [1, 2]}, {"a": [1, 2, 3]})[0]


def test_nested_paths_name_the_field():
    diffs = _differences({"models": {"tfidf": {"accuracy": 0.5}}},
                         {"models": {"tfidf": {"accuracy": 0.9}}})
    assert diffs[0].startswith("models.tfidf.accuracy")


def test_booleans_are_not_treated_as_integers():
    assert len(_differences({"a": True}, {"a": 1})) == 1
