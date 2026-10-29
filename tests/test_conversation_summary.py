"""Single-candidate summaries retain strengths beyond the parsed JD subset."""
import pytest

from test_conversation_v3 import Composer, service, snapshot


@pytest.mark.parametrize("output", [
    "Candidate 1 reports ruby development and Python programming skills.", "Candidate 1 is the best fit.", None,
])
@pytest.mark.parametrize("question", ["Summarise why candidate 1 is a good fit to this JD",
                                       "Any relevant job experiences for candidate 1?"])
def test_summary_retains_broad_strengths_and_unresolved_jd_gap(output, question):
    current = snapshot()
    current.jd_text = "Ruby developer wanted for education projects. MySQL skills required."
    current.candidates[0]["preview_text"] = "Ruby development and Python programming skills."
    model = Composer(output) if output else None
    engine = service(model)
    calls = []
    compare = engine.reader.base.compare

    def counted(*args, **kwargs):
        calls.append(True)
        return compare(*args, **kwargs)

    engine.reader.base.compare = counted
    answer = engine.answer(current, question)
    assert len(calls) == 1
    assert answer["show_comparison_table"] is False
    assert answer["summary_scope"] == "broad_candidate_evidence"
    if output and output.startswith("Candidate 1 reports"):
        assert answer["generation_mode"] == "qwen_prose"
    if model:
        bundle = model.calls[-1][2]
        assert len(bundle["evidence_matrix"]) >= 7
        assert any("Ruby" in item["context"] for item in bundle["evidence"])
        assert any("Python" in item["context"] for item in bundle["evidence"])
    if answer["generation_mode"] == "qwen_fallback":
        assert "Ruby" in answer["text"] and "Python" in answer["text"]
        assert "MySQL" in answer["text"]
        assert "does not establish absence" in answer["text"] or "doesn't mean" in answer["text"]
    assert "best fit" not in answer["text"]


def test_summary_fallback_keeps_incomplete_qualification_and_source_owner():
    current = snapshot()
    current.jd_text = "MySQL skills required."
    current.candidates[0]["preview_text"] = "Ruby development skills. Master degree in progress."
    current.candidates[1]["preview_text"] = "Completed Master degree."
    answer = service(None).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "Ruby" in answer["text"]
    assert "in progress" in answer["text"] or "not confirmed as completed" in answer["text"]
    assert "Completed Master" not in answer["text"]


def test_long_summary_uses_source_owned_entity_mentions_without_upgrading_tenure():
    import hashlib
    current = snapshot()
    current.jd_text = "MySQL skills required."
    source = "Ruby development skills " + "building reliable applications " * 10 + "."
    current.candidates[0]["preview_text"] = source

    class Entities:
        def prepare(self, snapshot, selected):
            return {"status": "complete", "documents": {"CV-A": {
                "status": "complete", "source_digest": hashlib.sha256(source.encode()).hexdigest(),
                "entities": [{"start": 0, "end": 4, "text": "Ruby", "label": "SKILL"}]}}}

    engine = service(None)
    engine.reader.base.entity_service = Entities()
    answer = engine.answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "CV mentions Ruby" in answer["text"]
    assert "do not establish proficiency or skill-specific tenure" in answer["text"]
    assert "building reliable applications building" not in answer["text"]


def conclusion_bundle(source="Ruby development skills.", jd="Ruby developer required. MySQL skills required.",
                      state="explicit_positive", mentions=("Ruby",)):
    return {"summary_scope": "broad_candidate_evidence", "summary_alignment": True,
            "selected_candidates": ["CV-A"], "jd_id": "JD-A", "labels": {"CV-A": "Candidate 1"},
            "evidence": [{"id": "cv-source", "owner": "CV-A", "owner_kind": "cv", "context": source},
                         {"id": "jd-source", "owner": "JD-A", "owner_kind": "jd", "context": jd}],
            "summary_cells": [{"id": "matrix:skill", "owner": "CV-A", "category": "SKILL",
                               "criterion": "Skill", "support_status": state, "context": source,
                               "evidence_ids": ["cv-source"], "mentions": list(mentions)}],
            "cells": [], "checked_paraphrases": [], "matrix_assertions": []}


def test_documented_overlap_adds_conditional_review_conclusion_last():
    from candidate_screener.app.conversation_summary import summary_conclusion, summary_fallback
    bundle = conclusion_bundle()
    conclusion = summary_conclusion(bundle)
    assert conclusion["evidence_ids"] == ["cv-source", "jd-source"]
    assert conclusion["owners"] == ["CV-A"]
    assert "promising candidate for further review" in conclusion["text"]
    assert summary_fallback(bundle).endswith(conclusion["text"])


@pytest.mark.parametrize("changes", [
    {"jd": "Database administrator required."},
    {"source": "No Ruby development skills.", "state": "explicit_negative"},
    {"source": "Learning Ruby development skills.", "state": "qualified"},
    {"source": "No Ruby experience.", "state": "unclear"},
    {"source": "Learning Ruby development skills.", "state": "unclear"},
    {"state": "partial_source"}, {"state": "unavailable"}, {"state": "conflicting"},
    {"source": "Developer experience.", "jd": "Developer experience required.", "mentions": ("Developer", "experience")},
])
def test_conditional_review_conclusion_requires_nontrivial_positive_owned_overlap(changes):
    from candidate_screener.app.conversation_summary import summary_conclusion
    assert summary_conclusion(conclusion_bundle(**changes)) is None


def test_summary_conclusion_is_registered_and_generated_claim_keeps_source_mapping():
    from candidate_screener.app.conversation_context import validate_prose
    current = snapshot()
    current.jd_text = "Python skills required. MySQL skills required."
    current.candidates[0]["preview_text"] = "Python skills."
    model = Composer("unsupported")
    service(model).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    bundle = model.calls[-1][2]
    conclusion = bundle["summary_conclusion"]
    mappings, reason = validate_prose(conclusion["text"], bundle)
    assert reason is None
    assert set(mappings[0]["evidence_ids"]) == set(conclusion["evidence_ids"])


def test_rails_overlap_conclusion_follows_unconfirmed_mysql_requirement():
    import hashlib
    current = snapshot()
    current.jd_text = "Ruby on Rails developer required. MySQL development experience."
    source = "Ruby on Rails development skills. MySQL mentioned as a technology."
    current.candidates[0]["preview_text"] = source

    class Entities:
        def prepare(self, snapshot, selected):
            return {"status": "complete", "documents": {"CV-A": {
                "status": "complete", "source_digest": hashlib.sha256(source.encode()).hexdigest(),
                "entities": [{"start": 0, "end": 13, "text": "Ruby on Rails", "label": "SKILL"}]}}}

    engine = service(None)
    engine.reader.base.entity_service = Entities()
    result = engine.answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "Ruby on Rails" in result["text"]
    assert "I can't confirm MySQL development experience" in result["text"]
    assert result["text"].endswith("a promising candidate for further review, subject to confirming the remaining requirements.")


def test_source_quotations_share_one_attribution_and_preserve_late_qualifiers():
    from candidate_screener.app.conversation_summary import summary_fallback, summary_conclusion
    bundle = conclusion_bundle()
    for index, (source, state, category) in enumerate([
        ("Master degree in progress.", "qualified", "DEGREE"),
        ("AWS certification expired in 2021.", "qualified", "CERTIFICATION"),
    ]):
        key = f"extra-{index}"
        bundle["evidence"].append({"id": key, "owner": "CV-A", "owner_kind": "cv", "context": source})
        bundle["summary_cells"].append({"id": key, "owner": "CV-A", "category": category,
                                        "criterion": category, "support_status": state,
                                        "context": source, "evidence_ids": [key]})
    answer = summary_fallback(bundle)
    assert answer.count("Candidate 1's CV states:") == 1
    for source in ["Ruby development skills.", "Master degree in progress.", "AWS certification expired in 2021."]:
        assert f"“{source}”" in answer
    assert answer.endswith("\n\n" + summary_conclusion(bundle)["text"])


@pytest.mark.parametrize("include_conclusion", [False, True])
def test_accepted_summary_conclusion_is_a_separate_final_paragraph(include_conclusion):
    current = snapshot()
    current.jd_text = "Python skills required. MySQL skills required."
    current.candidates[0]["preview_text"] = "Python skills."
    conclusion = ("Hence, Candidate 1 is a promising candidate for further review, "
                  "subject to confirming the remaining requirements.")
    model_text = "Candidate 1 describes Python skills relevant to this role."
    if include_conclusion:
        model_text += " " + conclusion
    result = service(Composer(model_text)).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert result["generation_mode"] == "qwen_prose"
    assert result["text"].endswith("\n\n" + conclusion)
    assert result["text"].count(conclusion) == 1
    assert result["validated_claim_sources"]


def test_accepted_summary_gets_registered_closing_with_explicit_application_provenance():
    current = snapshot()
    current.jd_text = "Python skills required. MySQL skills required."
    current.candidates[0]["preview_text"] = "Python skills."
    model = Composer("Candidate 1 describes Python skills relevant to this role. "
                     "I can't confirm the requested MySQL skills for Candidate 1 from what we have.")
    result = service(model).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    conclusion = model.calls[-1][2]["summary_conclusion"]
    assert result["generation_mode"] == "qwen_prose"
    assert result["text"].startswith(model.text)
    assert result["text"].endswith(conclusion["text"])
    assert result["generation_provenance"]["summary_conclusion_composer"] == "application"
    assert set(result["validated_claim_sources"][-1]["evidence_ids"]) == set(conclusion["evidence_ids"])


def test_manual_qa_summary_without_entity_mentions_has_grounded_closing():
    current = snapshot()
    current.jd_text = "QA Manual tester required. Test case design and bug reports using Jira."
    current.candidates[0]["preview_text"] = (
        "Created bug reports using Jira. Test case design and checklist preparation. "
        "Tested an invented LMS portal (QA Manual)."
    )
    result = service(None).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "Jira" in result["text"]
    assert result["text"].endswith(
        "a promising candidate for further review, subject to confirming the remaining requirements.")
    for category in ("Knowledge", "Skill", "Title", "Degree", "Experience Years", "Certification"):
        assert f"I can't confirm {category} for" not in result["text"]


@pytest.mark.parametrize("source,jd", [
    ("Created bug reports using Jira.", "Bug reports using Jira are required."),
    ("Performed manual testing for an LMS portal.", "Manual testing is required."),
    ("Test case design and checklist preparation.", "Test case design is required."),
])
def test_literal_owned_overlap_does_not_require_entity_mentions(source, jd):
    from candidate_screener.app.conversation_summary import summary_conclusion
    conclusion = summary_conclusion(conclusion_bundle(source=source, jd=jd, mentions=()))
    assert conclusion is not None
    assert conclusion["evidence_ids"] == ["cv-source", "jd-source"]


@pytest.mark.parametrize("source,jd", [
    ("Worked on software development projects.", "Software development projects require experience."),
    ("Seven years of professional experience.", "Seven years of professional experience required."),
    ("Created bug reports using Jira.", "No Jira experience is required."),
    ("Learning manual testing.", "Manual testing required."),
    ("No commercial Jira experience.", "Jira experience required."),
    ("Created Jira integrations.", "JiraCloud administration required."),
])
def test_literal_overlap_cannot_use_generic_qualified_or_substring_matches(source, jd):
    from candidate_screener.app.conversation_summary import summary_conclusion
    assert summary_conclusion(conclusion_bundle(source=source, jd=jd, mentions=())) is None


def test_summary_suppresses_category_gaps_but_keeps_concrete_jd_gap():
    from candidate_screener.app.conversation_summary import summary_fallback
    bundle = conclusion_bundle()
    bundle["cells"] = [
        {"id": "generic", "owner": "CV-A", "criterion": "Degree", "category": "DEGREE",
         "support_status": "not_found_in_retrieved_evidence", "evidence_ids": [],
         "requirement_evidence_ids": []},
        {"id": "mysql", "owner": "CV-A", "criterion": "MySQL skills", "category": "SKILL",
         "support_status": "not_found_in_retrieved_evidence", "evidence_ids": [],
         "requirement_evidence_ids": ["jd-source"]},
    ]
    text = summary_fallback(bundle)
    assert "I can't confirm Degree" not in text
    assert "I can't confirm MySQL skills" in text
    assert "Missing evidence does not establish absence." in text
