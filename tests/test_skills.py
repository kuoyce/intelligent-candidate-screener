"""`baselines.skills` — the hand-written keyword list and its digest."""
from __future__ import annotations

import pytest

from candidate_screener.baselines import skills


def test_each_match_is_repeated():
    """The whole of the 'boost' is emitting each matched term `REPEAT` times."""
    out = skills.extract_skills("we use sql").split()
    assert out == ["sql"] * skills.REPEAT


def test_match_is_case_insensitive():
    assert skills.extract_skills("Python") == skills.extract_skills("python") != ""


@pytest.mark.parametrize("text", ["", None, 0])
def test_empty_input_returns_empty_string(text):
    """Never None — the result is concatenated straight onto a document."""
    assert skills.extract_skills(text) == ""


def test_no_match_returns_empty_string():
    assert skills.extract_skills("a document about horticulture and bees") == ""


def test_multiple_families_are_all_scanned():
    out = skills.extract_skills("python on aws with docker, agile team, git")
    assert {"python", "aws", "docker", "agile", "git"} <= set(out.split())


def test_augment_appends_skills_to_the_text():
    import pandas as pd
    out = skills.augment(pd.Series(["we need sql"]))
    assert out.iloc[0] == "we need sql " + skills.extract_skills("we need sql")


def test_skill_pattern_digest_is_pinned():
    """A silent edit to SKILL_PATTERNS must not slide past `run --check`.

    The digest is recorded in `baseline-metrics.json`. If this fails you changed the
    keyword list, which is a **modelling change** (assumption A9): re-freeze the
    baseline deliberately and update this constant in the same commit.
    """
    assert skills.digest() == (
        "aeee52261870360baa956934bcc8f1094464cb2e182467c6ffa508d2459c54c0")
