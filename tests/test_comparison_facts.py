"""Comparison semantics preserve qualifiers independently of exact QA spans."""
import hashlib
import os
from types import SimpleNamespace

import pytest

from candidate_screener.app.comparison_facts import build_facts, support_state, jd_criteria
from candidate_screener.app.conversation_render import render_comparison, render_explanation
from candidate_screener.app.portal_extractive import PortalExtractiveService


@pytest.mark.parametrize("text, category, terms", [
    ("Required skills: Python, SQL and Docker.", "SKILL", ["Python", "SQL", "Docker"]),
    ("Knowledge required: SQL and Python.", "KNOWLEDGE", ["SQL", "Python"]),
    ("Knowledge of SQL and Python is required.", "KNOWLEDGE", ["SQL", "Python"]),
])
def test_named_required_skill_and_knowledge_lists_retain_literal_criteria(text, category, terms):
    rows = jd_criteria(SimpleNamespace(jd_text=text))
    assert [row["criterion"] for row in rows] == terms
    assert all(row["category"] == category and row["criterion"] in text for row in rows)


def fixture(first, second, *, category="SKILL", criterion="Python", qa_text=None):
    current = SimpleNamespace(snapshot_id="synth-snapshot", jd_id="JD-SYNTH",
        jd_text="Python skills required.", candidates=tuple(
            {"cv_id": owner, "rank": rank, "preview_text": text,
             "preview_status": "available_redacted_text"}
            for owner, rank, text in (("CV-A", 1, first), ("CV-B", 2, second))))
    citations, answers = [], {}
    for index, (owner, text) in enumerate((("CV-A", first), ("CV-B", second))):
        ids = []
        if text:
            ids = [f"e{index}"]
            citations.append({"evidence_id": ids[0], "owner": owner, "owner_kind": "cv",
                              "snapshot_id": current.snapshot_id, "document_digest": hashlib.sha256(text.encode()).hexdigest(),
                              "coordinate_space": "redacted_preview", "quote": text, "start": 0, "end": len(text)})
        answers[owner] = {"text": qa_text or text, "context": text, "status": "model_span" if text else "not_found",
                          "evidence_ids": ids, "qa_status": "supported" if text else "not_found"}
    return current, {"context": {"selected_candidates": ["CV-A", "CV-B"], "criteria": [criterion]},
                     "citations": citations, "comparison": [{"category": category, "answers": answers}]}


def rendered(current, result):
    # Explicit v1 control demonstrates the behavior guarded by the new assertions.
    if os.environ.get("SCREENER_TEST_V1_COMPARISON") == "1":
        return PortalExtractiveService().compare(current, ["CV-A", "CV-B"])["text"]
    return render_comparison(current, build_facts(current, result))


def test_comparison_conclusion_explains_completed_versus_ongoing_qualification():
    current, result = fixture("Completed BSc degree.", "MSc degree in progress.",
                              category="DEGREE", criterion="qualifications")
    text = rendered(current, result)
    assert "Candidate 1 states a completed qualification" in text
    assert "Candidate 2 lists a qualification with an incomplete or ongoing status" in text


def test_comparison_conclusion_orders_only_explicit_same_skill_tenure():
    current, result = fixture("Three years of Python experience.", "Five years of Python experience.",
                              category="EXPERIENCE_YEARS", criterion="Python tenure")
    text = rendered(current, result)
    assert "Candidate 2 reports more python tenure than Candidate 1" in text
    assert "60 versus 36 months" in text


def test_naked_positive_span_cannot_reverse_negated_source():
    current, result = fixture("No commercial Python experience.", "Python experience.", qa_text="Python")
    facts = build_facts(current, result)
    assert [cell["support_status"] for cell in facts["cells"]] == ["explicit_negative", "explicit_positive"]
    assert facts["relations"][0]["kind"] == "explicit_contrast"
    assert "No commercial Python experience" in render_comparison(current, facts)


def test_unrelated_negation_is_not_a_negative_python_claim():
    assert support_state("No driving license; Python skills.", "SKILL", {"python"}) == "explicit_positive"
    assert support_state("No driving license. Python skills.", "SKILL", {"python"}) == "explicit_positive"
    assert support_state("Python experience.", "SKILL", {"commercial", "python"}) == "unclear"
    assert support_state("Production support experience. Deployment experience.", "SKILL", {"production", "deployment"}) == "unclear"


def test_conflicting_source_claims_are_never_composed_as_positive():
    current, result = fixture("No Python experience. Python experience.", "Python skills.")
    facts = build_facts(current, result)
    assert facts["cells"][0]["support_status"] == "conflicting"
    assert any(item["kind"] == "source_conflict" for item in facts["relations"])
    assert not any(item["kind"] == "both_documented" for item in facts["relations"])
    assert support_state("No Python experience; Python skills.", "SKILL", {"python"}) == "conflicting"


@pytest.mark.parametrize("first,second", [
    ("Ten years total experience. Python skills.", "Five years of Python experience."),
    ("Python developer. Ten years of work experience.", "Five years of Python experience."),
    ("Three to five years of Python experience.", "Four years of Python experience."),
    ("Three years of Python experience. Five years of Python experience.", "Four years of Python experience."),
])
def test_ambiguous_total_range_or_conflicting_duration_never_orders_tenure(first, second):
    current, result = fixture(first, second, category="EXPERIENCE_YEARS", criterion="Python tenure")
    assert not any(item["kind"] == "duration_greater" for item in build_facts(current, result)["relations"])


def test_incomplete_and_expired_credentials_stay_qualified():
    assert support_state("MSc in progress.", "DEGREE") == "qualified"
    assert support_state("AWS certification. Expired in 2021.", "CERTIFICATION", {"aws"}) == "qualified"


@pytest.mark.parametrize("source", ["MSc completed. Status: in progress.", "MSc was never completed.", "MSc was not awarded."])
def test_late_or_negated_completion_status_never_becomes_completed_degree(source):
    assert support_state(source, "DEGREE") != "explicit_positive"


def test_distinct_qualified_degrees_do_not_collapse_into_a_single_negative():
    assert support_state("No MSc but completed BSc.", "DEGREE") == "conflicting"


@pytest.mark.parametrize("source", ["Experienced Java developer learning Python.", "I hope to become proficient in Python."])
def test_unrelated_experience_and_aspiration_do_not_establish_python_experience(source):
    assert support_state(source, "SKILL", {"python"}) != "explicit_positive"


def test_explicit_skill_duration_survives_unrelated_negative_source_sentence():
    current, result = fixture("Three years of Python experience. Production deployment experience.",
                              "Five years of Python experience. No production deployment experience.",
                              category="EXPERIENCE_YEARS", criterion="Python tenure")
    facts = build_facts(current, result)
    assert any(item["kind"] == "duration_greater" for item in facts["relations"])
    assert "No production deployment experience" in render_comparison(current, facts)


def test_production_deployment_is_linked_not_generic_neighboring_experience():
    assert support_state("Production deployment experience.", "SKILL", {"production", "deployment"}) == "explicit_positive"
    assert support_state("No production deployment experience.", "SKILL", {"production", "deployment"}) == "explicit_negative"
    assert support_state("Production support. Deployment experience.", "SKILL", {"production", "deployment"}) == "unclear"
    assert support_state("Used production databases and deployment training is planned.", "SKILL", {"production", "deployment"}) != "explicit_positive"
    assert support_state("Production deployment experience is planned.", "SKILL", {"production", "deployment"}) != "explicit_positive"
    assert support_state("Production, deployment experience.", "SKILL", {"production", "deployment"}) != "explicit_positive"
    assert support_state("Holds MSc. Status: never completed.", "DEGREE") != "explicit_positive"


def test_missing_participant_is_present_and_not_inferred_weaker():
    current, result = fixture("Python experience.", "")
    facts = build_facts(current, result)
    assert len(facts["cells"]) == 2
    assert facts["cells"][1]["support_status"] == "not_found_in_retrieved_evidence"
    text = render_comparison(current, facts)
    assert "Candidate 1" in text and "Candidate 2" in text
    assert "does not establish absence" in text
    assert "weaker" not in text


def test_foreign_or_stale_citation_cannot_provide_a_fact():
    current, result = fixture("Python experience.", "Python skills.")
    result["citations"][0]["snapshot_id"] = "stale"
    assert build_facts(current, result)["cells"][0]["support_status"] == "not_found_in_retrieved_evidence"


def test_relation_selection_cannot_drop_critical_qualification():
    current, result = fixture("Completed BSc degree.", "MSc in progress.", category="DEGREE", criterion="qualifications")
    facts = build_facts(current, result)
    assert "incomplete or ongoing" in render_comparison(current, facts, selected_relation_ids=[])
    with pytest.raises(ValueError):
        render_comparison(current, facts, selected_relation_ids=["forged"])


def test_three_participant_comparison_retains_friendly_labels_and_unknowns():
    current, result = fixture("Python experience.", "Python skills.")
    current.candidates += ({"cv_id": "CV-C", "rank": 3, "preview_status": "unavailable"},)
    result["context"]["selected_candidates"].append("CV-C")
    result["comparison"][0]["answers"]["CV-C"] = {"status": "source_unavailable"}
    facts = build_facts(current, result)
    text = render_comparison(current, facts)
    assert all(f"Candidate {rank}" in text for rank in (1, 2, 3))
    assert "CV-A" not in text and "CV-C" not in text
    assert facts["cells"][2]["support_status"] == "unavailable"


def test_composer_order_is_visible_and_cannot_hide_critical_relations():
    current, result = fixture("Three years of Python experience.", "Five years of Python experience.",
                              category="EXPERIENCE_YEARS", criterion="Python tenure")
    facts = build_facts(current, result)
    ordered = list(reversed([item["id"] for item in facts["relations"]]))
    text = render_comparison(current, facts, selected_relation_ids=ordered)
    assert text.startswith("Candidate 2 reports more python tenure")
    for index in range(8):
        facts["relations"].append({"id": f"critical-{index}", "kind": "source_conflict",
                                   "participants": ["CV-A"], "cell_ids": [],
                                   "text": f"Critical qualifier {index}."})
    text = render_comparison(current, facts, selected_relation_ids=ordered)
    assert all(f"Critical qualifier {index}." in text for index in range(8))


def test_broad_render_context_follows_qualification_conclusion_not_first_categories():
    facts = {"participants": ["CV-A"], "labels": {"CV-A": "Candidate 1"},
             "cells": [{"id": f"cell-{index}", "owner": "CV-A", "context": f"Topic {index}.", "support_status": "unclear"}
                       for index in range(3)] + [{"id": "degree", "owner": "CV-A", "context": "MSc in progress.", "support_status": "qualified"}],
             "relations": [{"id": "degree-relation", "kind": "completion_status_differs",
                            "participants": ["CV-A"], "cell_ids": ["degree"], "text": "Qualification remains in progress."}]}
    assert "MSc in progress." in render_comparison(None, facts)


def test_jd_criteria_preserve_literal_required_phrases_and_reject_partial_parsing():
    current, _ = fixture("Python skills.", "Python skills.")
    current.jd_text = "Python experience and production deployment experience required. Bachelor degree preferred."
    assert jd_criteria(current) == [
        {"criterion": "Python experience", "topic": "Python", "category": "SKILL"},
        {"criterion": "production deployment experience", "topic": "production deployment", "category": "SKILL"},
        {"criterion": "Bachelor degree", "topic": "qualifications", "category": "DEGREE"}]
    current.jd_text = "No AWS certification required. We offer cloud experience. Research and development experience required."
    assert jd_criteria(current) == []


def test_missing_certification_does_not_read_as_missing_degree_or_all_evidence():
    current, result = fixture("Completed BSc degree.", "MSc in progress.", category="DEGREE", criterion="qualifications")
    result["comparison"].append({"category": "CERTIFICATION", "answers": {}})
    text = render_comparison(current, build_facts(current, result))
    assert "Candidate 1 — Certification:" in text
    assert "Candidate 2 — Certification:" in text
    assert "Candidate 1: I could not establish this" not in text
    assert text.count("does not establish absence") == 1
    assert "completed qualification" in text and "incomplete or ongoing" in text


def test_three_candidates_get_supported_pairwise_qualification_conclusions():
    current, result = fixture("Completed BSc degree.", "MSc in progress.", category="DEGREE", criterion="qualifications")
    source = "Completed Bachelor degree."
    current.candidates += ({"cv_id": "CV-C", "rank": 3, "preview_text": source, "preview_status": "available_redacted_text"},)
    result["context"]["selected_candidates"].append("CV-C")
    result["citations"].append({"evidence_id": "third", "owner": "CV-C", "owner_kind": "cv",
                                "snapshot_id": current.snapshot_id, "document_digest": hashlib.sha256(source.encode()).hexdigest(),
                                "coordinate_space": "redacted_preview", "quote": source, "start": 0, "end": len(source)})
    result["comparison"][0]["answers"]["CV-C"] = {"status": "source_only", "evidence_ids": ["third"], "context": source}
    facts = build_facts(current, result)
    contrasts = [item for item in facts["relations"] if item["kind"] == "completion_status_differs"]
    assert {frozenset(item["participants"]) for item in contrasts} == {frozenset(("CV-A", "CV-B")), frozenset(("CV-B", "CV-C"))}
    text = render_comparison(current, facts)
    assert "Candidate 3 states a completed qualification" in text
    assert "same qualification or level" in text
    assert all(f"Candidate {rank}" in text for rank in (1, 2, 3))


def test_named_jd_criterion_supports_deployment_contrast_without_ordinal_rank():
    current, result = fixture("Production deployment experience.", "No production deployment experience.",
                              category="SKILL", criterion="production deployment")
    result["comparison"][0].update(criterion="production deployment experience", topic="production deployment")
    facts = build_facts(current, result)
    assert any(item["kind"] == "explicit_contrast" for item in facts["relations"])
    text = render_comparison(current, facts)
    assert "Candidate 1 reports production deployment evidence" in text
    assert "Candidate 2's CV explicitly negates it" in text
    assert "ranked" not in text and "better" not in text


def test_concise_render_avoids_duplicate_sources_but_preserves_exact_qualifiers():
    current, result = fixture("Completed Bachelor degree.", "Master degree. Status: in progress.",
                              category="DEGREE", criterion="qualifications")
    facts = build_facts(current, result)
    text = render_comparison(current, facts, include_source_context=False)
    assert "Bachelor" in text and "Master" in text and "Status: in progress." in text
    assert text.count("Master degree.") == 1
    assert "\n\nCandidate 1:" not in text
    assert "\n\nCandidate 2:" not in text


def test_concise_render_keeps_source_for_unclear_claim():
    current, result = fixture("Experienced Java developer learning Python.", "Python skills.")
    text = render_comparison(current, build_facts(current, result), include_source_context=False)
    assert "Experienced Java developer learning Python" in text


@pytest.mark.parametrize("source", [
    "Will be awarded MSc in 2027.", "Awarded MSc pending thesis defense.",
    "MSc awarded conditionally.", "About to be awarded MSc.", "MSc completion planned."])
def test_future_or_conditional_award_never_becomes_completed_degree(source):
    assert support_state(source, "DEGREE") != "explicit_positive"


def test_focused_missing_evidence_names_requested_frame_criterion():
    current, result = fixture("", "", category="GENERAL", criterion="cloud experience")
    facts = build_facts(current, result)
    assert facts["cells"][0]["criterion"] == "cloud experience"
    assert "Candidate 1 — cloud experience:" in render_comparison(current, facts, include_source_context=False)


def directed_fixture(first, second, *, criterion="Python experience", category="SKILL", topic="Python"):
    current, result = fixture(first, second, category=category, criterion=criterion)
    current.jd_text = criterion + " required."
    result["context"]["intent"] = "comparison_explanation"
    result["citations"].append({"evidence_id": "jd-source", "owner": current.jd_id, "owner_kind": "jd",
                                "snapshot_id": current.snapshot_id, "document_digest": hashlib.sha256(current.jd_text.encode()).hexdigest(),
                                "coordinate_space": "redacted_preview", "quote": current.jd_text, "start": 0, "end": len(current.jd_text)})
    result["comparison"][0].update(criterion=criterion, topic=topic, requirement_evidence_ids=["jd-source"])
    return current, result


def test_directed_reason_has_jd_and_both_owned_sources_not_an_overall_verdict():
    current, result = directed_fixture("Python experience.", "No Python experience.")
    facts = build_facts(current, result)
    advantage = facts["advantages"][0]
    assert advantage["stronger_owner"] == "CV-A" and advantage["weaker_owner"] == "CV-B"
    assert advantage["basis"] == "explicit_contrast"
    assert set(advantage["evidence_ids"]) == {"jd-source", "e0", "e1"}
    text = render_explanation(current, facts, "CV-A", "CV-B")
    assert "stronger documented evidence for Candidate 1" in text
    assert "overall better-candidate verdict" in text
    assert "[S3]" in text and "[S1]" in text and "[S2]" in text
    assert facts["citation_labels"] == {"e0": "S1", "e1": "S2", "jd-source": "S3"}


def test_directed_missing_evidence_is_source_coverage_not_absent_skill():
    current, result = directed_fixture("Python experience.", "")
    facts = build_facts(current, result)
    assert facts["advantages"][0]["basis"] == "coverage"
    text = render_explanation(current, facts, "CV-A", "CV-B")
    assert "does not establish this criterion" in text
    assert "not proof that the candidate lacks" in text
    assert "Candidate 2 lacks" not in text


def test_reverse_directed_reason_corrects_requested_premise_and_preserves_counterevidence():
    current, result = directed_fixture("No Python experience.", "Python experience.")
    facts = build_facts(current, result)
    text = render_explanation(current, facts, "CV-A", "CV-B", selected_relation_ids=[])
    assert "cannot support the premise" in text
    assert "instead favor Candidate 2" in text
    assert "Counterevidence:" in text
    assert "No Python experience" in text


def test_mixed_directional_reasons_never_make_requested_candidate_globally_better():
    current, result = directed_fixture("Three years of Python experience. Production deployment experience.",
                                       "Five years of Python experience. No production deployment experience.")
    current.jd_text = "Python experience and production deployment experience required."
    result["citations"][-1].update(quote=current.jd_text, end=len(current.jd_text),
                                  document_digest=hashlib.sha256(current.jd_text.encode()).hexdigest())
    result["comparison"].append({**result["comparison"][0], "criterion": "production deployment experience", "topic": "production deployment"})
    facts = build_facts(current, result)
    assert {item["stronger_owner"] for item in facts["advantages"]} == {"CV-A", "CV-B"}
    assert {item["basis"] for item in facts["advantages"]} == {"duration", "explicit_contrast"}
    text = render_explanation(current, facts, "CV-A", "CV-B", selected_relation_ids=[])
    assert "trade-off" in text and "Counterevidence:" in text and "Supporting evidence:" in text
    assert "Candidate 2 reports more explicitly linked tenure" in text


def test_directional_advantage_requires_literal_registered_jd_criterion_and_valid_jd_source():
    current, result = directed_fixture("Python experience.", "No Python experience.")
    result["comparison"][0]["requirement_evidence_ids"] = ["e0"]
    assert build_facts(current, result)["advantages"] == []
    result["comparison"][0]["requirement_evidence_ids"] = ["jd-source"]
    result["comparison"][0]["criterion"] = "Invented aptitude"
    facts = build_facts(current, result)
    assert facts["advantages"] == []
    assert "does not establish a directed advantage" in render_explanation(current, facts, "CV-A", "CV-B")


def test_directional_completion_preserves_degree_context_and_does_not_order_degree_levels():
    current, result = directed_fixture("Completed Bachelor degree.", "Master degree in progress.",
                                       criterion="Bachelor degree", category="DEGREE", topic="qualifications")
    facts = build_facts(current, result)
    assert facts["advantages"][0]["basis"] == "completion"
    text = render_explanation(current, facts, "CV-A", "CV-B")
    assert "Bachelor" in text and "Master degree in progress" in text
    assert "higher degree title or enrollment does not establish" in text
    current, result = directed_fixture("Completed Master degree.", "Bachelor degree in progress.",
                                       criterion="Bachelor degree", category="DEGREE", topic="qualifications")
    assert build_facts(current, result)["advantages"] == []


def test_qualified_negation_and_raw_skill_mentions_do_not_create_directional_advantage():
    current, result = directed_fixture("Python experience.", "No commercial Python experience.")
    assert build_facts(current, result)["advantages"] == []
    current, result = directed_fixture("Python", "No Python experience.")
    assert build_facts(current, result)["advantages"] == []
    current, result = directed_fixture("SQL knowledge.", "", criterion="SQL knowledge", category="KNOWLEDGE", topic="SQL")
    facts = build_facts(current, result)
    assert facts["cells"][0]["source_terms"] == ["SQL"]
    assert facts["advantages"][0]["basis"] == "coverage"


def test_rank_explanation_reads_saved_order_without_inventing_model_causality():
    current, result = directed_fixture("No Python experience.", "Python experience.")
    text = render_explanation(current, build_facts(current, result), "CV-A", "CV-B", ranked_above=True)
    assert "saved ranking places Candidate 1 at 1 and Candidate 2 at 2" in text
    assert "do not prove why the ranking model ordered them" in text
    assert "instead favor Candidate 2" in text
    reverse = render_explanation(current, build_facts(current, result), "CV-B", "CV-A", ranked_above=True)
    assert "Candidate 2 is not ranked above Candidate 1" in reverse


def test_actual_service_directed_fixture_uses_the_shared_table_and_corrects_reverse_premise():
    from candidate_screener.app.conversation_service import ConversationService
    class NoQa:
        def answer(self, question, passages):
            return {"status": "not_found", "spans": []}
    current, _ = directed_fixture("No Python experience.", "Python experience.")
    engine = ConversationService(PortalExtractiveService(qa_client=NoQa()))
    result = engine.answer(current, "Why is candidate 1 better than candidate 2 for this role?")
    assert result["context"]["intent"] == "comparison_explanation"
    assert result["show_comparison_table"] is True
    assert "cannot support the premise" in result["text"]
    assert "instead favor Candidate 2" in result["text"]
    assert {citation["owner"] for citation in result["citations"]} == {current.jd_id, "CV-A", "CV-B"}
