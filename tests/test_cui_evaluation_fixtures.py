"""CUI fixture guards use only authored synthetic materials."""
from copy import deepcopy
import json

import pytest

from candidate_screener.evaluation.cui.fixtures import load_fixtures, validate_fixtures


@pytest.fixture
def fixtures():
    return load_fixtures()


def test_draft_study_counts_are_explicit(fixtures):
    summary = validate_fixtures(fixtures)
    assert summary["review_status"] == "draft"
    assert summary["splits"]["eval"] == {
        "families": 10, "questions": 30, "dialogues": 8,
        "dialogue_turns": 40, "eligible_turns": 68,
    }
    assert summary["splits"]["dev"]["families"] == 2


def test_fact_must_match_its_owner_exact_source_span(fixtures):
    fact = fixtures["families"][0]["questions"][0]["expected"]["required_facts"][0]
    fact["quote"] = "Invented unsupported completed qualification."
    with pytest.raises(ValueError, match="quote does not match"):
        validate_fixtures(fixtures)


def test_valid_quote_from_another_owner_is_rejected(fixtures):
    family = fixtures["families"][0]
    fact = family["questions"][0]["expected"]["required_facts"][0]
    fact["owner"] = family["candidates"][1]["id"]
    with pytest.raises(ValueError, match="source span|outside source"):
        validate_fixtures(fixtures)


def test_outside_family_fact_cannot_be_cited(fixtures):
    fact = fixtures["families"][0]["questions"][0]["expected"]["required_facts"][0]
    fact["owner"] = fixtures["families"][1]["candidates"][0]["id"]
    with pytest.raises(ValueError, match="outside active family"):
        validate_fixtures(fixtures)


@pytest.mark.parametrize("offset", [-1, True, 1.5, "0"])
def test_span_offsets_are_exact_integer_coordinates(fixtures, offset):
    fixtures["families"][0]["questions"][0]["expected"]["required_facts"][0]["start"] = offset
    with pytest.raises(ValueError, match="offsets"):
        validate_fixtures(fixtures)


def test_duplicate_turn_ids_are_rejected(fixtures):
    family = fixtures["families"][0]
    family["questions"][1]["id"] = family["questions"][0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_fixtures(fixtures)


def test_source_lineage_cannot_cross_development_evaluation(fixtures):
    fixtures["families"][2]["lineage"] = fixtures["families"][0]["lineage"]
    with pytest.raises(ValueError, match="lineage crosses"):
        validate_fixtures(fixtures)


def test_renaming_lineage_cannot_hide_copied_source(fixtures):
    fixtures["families"][2]["jd"]["text"] = fixtures["families"][0]["jd"]["text"]
    with pytest.raises(ValueError, match="identical source text crosses"):
        validate_fixtures(fixtures)


def test_answerable_fact_requires_source_evidence(fixtures):
    fixtures["families"][0]["questions"][0]["expected"]["required_facts"] = []
    with pytest.raises(ValueError, match="supporting evidence"):
        validate_fixtures(fixtures)


def test_scope_switch_rejects_previous_candidate_ids(fixtures):
    family = next(f for f in fixtures["families"] if f["id"] == "eval-08")
    family["dialogues"][0]["turns"][3]["expected"]["participants"] = [family["candidates"][0]["id"]]
    with pytest.raises(ValueError, match="active-family candidate"):
        validate_fixtures(fixtures)


def test_scope_switch_cannot_target_another_family(fixtures):
    family = next(f for f in fixtures["families"] if f["id"] == "eval-08")
    family["dialogues"][0]["turns"][2]["scope_event"]["scope_id"] = "eval-01"
    with pytest.raises(ValueError, match="same family"):
        validate_fixtures(fixtures)


def test_completion_criteria_are_required(fixtures):
    fixtures["families"][0]["questions"][0]["expected"]["completion_criteria"] = []
    with pytest.raises(ValueError, match="completion_criteria"):
        validate_fixtures(fixtures)


def test_expected_boolean_cannot_be_truthy_string(fixtures):
    fixtures["families"][0]["questions"][0]["expected"]["uncertainty"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        validate_fixtures(fixtures)


def test_load_uses_explicit_path_and_does_not_mutate(fixtures, tmp_path):
    original = deepcopy(fixtures)
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(fixtures))
    assert load_fixtures(path) == original
    validate_fixtures(fixtures)
    assert fixtures == original


@pytest.mark.parametrize("value", [None, [], {"schema_version": 99}])
def test_malformed_top_level_fails_as_value_error(value):
    with pytest.raises(ValueError):
        validate_fixtures(value)
