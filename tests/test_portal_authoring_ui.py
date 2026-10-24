"""Guided answers preserve the requirement priority the reviewer entered."""
import pytest

pytest.importorskip("streamlit")
from candidate_screener.app.portal_authoring_ui import _answer


def test_unprefixed_requirement_after_preference_remains_required():
    assert _answer("skills", "Preferred: Kubernetes\nPython\nRequired: SQL") == {
        "required": ["Python", "SQL"], "preferred": ["Kubernetes"]}


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "101"])
def test_guided_experience_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        _answer("min_years", value)
