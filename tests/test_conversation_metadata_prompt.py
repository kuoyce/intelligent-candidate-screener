"""Metadata-first prompt ordering and final-answer transport invariants."""
import pytest

from candidate_screener.llm.conversation_prose import ProseComposer, ProseError, checked_prose
from test_conversation_prose import Provider


def context():
    frame = {"jd_id": "job", "selected_candidates": ["a", "b"],
             "intent": "comparison", "criteria": ["Python and deployment"]}
    facts = {"labels": {"a": "Candidate 1", "b": "Candidate 9"}, "history": [],
             "evidence": [{"owner": "a", "context": "Python deployments."},
                          {"owner": "b", "context": "Master degree in progress."}],
             "cells": [], "evidence_matrix": [
                 {"criterion": "Cloud", "scope": "Broad evidence",
                  "statuses": {"a": "Documented", "b": "Not Found"}},
                 {"criterion": "Degree", "scope": "Broad evidence",
                  "statuses": {"a": "Not Found", "b": "Qualified"}}]}
    return frame, facts


def test_matrix_precedes_selected_passages_with_exact_owned_flags():
    frame, facts = context()
    provider = Provider()
    ProseComposer(provider).compose("Compare candidate 1 and 9.", frame, facts)
    payload = provider.messages[-1]["content"]
    matrix = payload.split("[Evidence Matrix]:\n", 1)[1].split("[Selected Passages]:", 1)[0]
    assert 'Candidate 1 (Documented) | Candidate 9 (Not Found)' in matrix
    assert 'Candidate 1 (Not Found) | Candidate 9 (Qualified)' in matrix
    assert "Python deployments." not in matrix
    assert payload.index("[Job Priority]:") < payload.index("[Evidence Matrix]:") < payload.index("[Selected Passages]:")
    assert payload.endswith("Current user question:\nCompare candidate 1 and 9.\n\n[Reasoning Steps]:")
    assert "Master degree in progress." in payload


def test_matrix_requires_current_requested_owners():
    frame, facts = context()
    facts["evidence_matrix"][0]["statuses"]["foreign"] = "Documented"
    with pytest.raises(ProseError):
        ProseComposer(Provider()).compose("Compare", frame, facts)


@pytest.mark.parametrize("raw", [
    "[Reasoning Steps]:\n1. Candidate 2's degree is ongoing.\n[Conclusion]: Candidate 2 is pursuing a Master degree.",
    "1. Candidate 2's degree is ongoing.\n[Conclusion]: Candidate 2 is pursuing a Master degree.",
    "[Conclusion]: Candidate 2 is pursuing a Master degree.",
])
def test_only_final_conclusion_is_returned_for_validation_and_persistence(raw):
    assert checked_prose(raw) == "Candidate 2 is pursuing a Master degree."


@pytest.mark.parametrize("raw", [
    "[Reasoning Steps]:\n1. Candidate 1 looks stronger.",
    "[Conclusion]: First answer.\n[Conclusion]: Conflicting answer.",
    "[Reasoning Steps]:\n1. Review.\n[Conclusion]:",
    "[Conclusion]: Answer.\n[Reasoning Steps]:\n1. More commentary.",
    "1. Candidate 1 is better.",
])
def test_incomplete_duplicate_or_reversed_output_sections_are_rejected(raw):
    with pytest.raises(ProseError):
        checked_prose(raw)


def test_examples_do_not_teach_missing_cloud_evidence_as_absence():
    from candidate_screener.llm.conversation_prose import SYSTEM
    assert "### EXAMPLES" in SYSTEM and "### EXAMPLE 1" in SYSTEM and "### EXAMPLE 2" in SYSTEM
    assert "[Evidence Matrix]:" in SYSTEM and "[Conclusion]:" in SYSTEM
    assert "Candidate 2 lacks" not in SYSTEM
    assert "which Candidate 2 lacks" not in SYSTEM
    assert "not established" in SYSTEM
    assert "{{INSERT_" not in SYSTEM


def test_single_summary_closing_is_conditional_on_checked_jd_overlap():
    from candidate_screener.llm.conversation_prose import PROMPT_VERSION, SYSTEM
    assert PROMPT_VERSION == "hiring-desk-v3-summary-conclusion-7"
    assert "source-checked overlap" in SYSTEM
    assert "promising candidate for further review" in SYSTEM
    assert "subject to confirming the remaining requirements" in SYSTEM
    assert "A MySQL mention does not establish MySQL development experience" in SYSTEM
    assert "unrelated, unavailable, partial-only or negative-only" in SYSTEM
    assert "do not invent a second candidate or author a Fit verdict" in SYSTEM
    assert "one concise sentence on a single line" in SYSTEM
    comparison_examples = SYSTEM.split("### EXAMPLES", 1)[1].split("### YOUR TASK", 1)[0]
    assert "promising candidate" not in comparison_examples
