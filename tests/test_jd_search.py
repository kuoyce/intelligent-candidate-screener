"""Authorized JD-only lookup must not turn mentions or preferences into requirements."""
import pytest

from candidate_screener.app.jd_search import is_jd_search, search_jobs


def doc(text, identity="mobile", title="Mobile engineer"):
    return {"jd_id": identity, "title": title, "text": text, "source_version": "revision-1"}


@pytest.mark.parametrize("question", ["Which JD has this specific skills -Flutter?",
                                     "Which JD has Flutter skills?",
                                     "Which JD has the skill Leadership?"])
def test_owner_skill_search_wording(question):
    result = search_jobs([doc("Flutter required. Leadership preferred.")], question)
    assert is_jd_search(question)
    assert result["matches"]
    assert result["skill"] in {"Flutter", "Leadership"}


def test_requirement_search_keeps_preferred_and_negated_distinct():
    result = search_jobs([
        doc("Skills\nRequired:\n- Flutter\nPreferred:\n- Leadership"),
        doc("Skills\nRequired:\n- Python\nPreferred:\n- Flutter", "web", "Web engineer"),
        doc("Flutter is not required.", "other", "Other role"),
    ], "Which JD requires Flutter?")
    assert [(row["jd_id"], row["level"]) for row in result["matches"]] == [
        ("mobile", "required"), ("web", "preferred"), ("other", "negated")]
    assert "Mobile engineer requires Flutter" in result["text"]
    assert "Web engineer" in result["text"] and "preferred" in result["text"]
    assert "Other role requires" not in result["text"]


def test_preference_and_plain_mention_never_answer_mandatory():
    result = search_jobs([doc("Flutter would be a plus."), doc("We use Flutter.", "other")],
                         "Which JD requires Flutter?")
    assert not any(row["level"] == "required" for row in result["matches"])
    assert "haven't found a JD that makes Flutter mandatory" in result["text"]


def test_source_quotes_offsets_versions_are_owned_and_redacted():
    source = doc("Contact dev@example.com.\nRequired: Leadership.")
    result = search_jobs([source], "Which JD requires Leadership?")
    from candidate_screener.annotation.redact import redact_text
    redacted = redact_text(source["text"])
    for item in result["citations"]:
        assert item["quote"] == redacted[item["start"]:item["end"]]
        assert item["owner"] == source["jd_id"]
        assert item["source_version"] == source["source_version"]
    assert "@" not in str(result)


def test_partial_coverage_does_not_claim_exhaustive_absence():
    result = search_jobs([doc("Python required"), doc(None, "unavailable"),
                          doc("Flutter required", "beyond-cap")], "Which JD requires Flutter?", max_documents=2)
    assert result["coverage"] == {"available": 3, "searched": 1, "unavailable": 1,
                                  "not_searched": 1, "omitted_matches": 0, "complete": False}
    assert result["status"] == "partial"
    assert "Search coverage is partial" in result["text"]


def test_literal_skill_search_does_not_match_substrings_or_expand_synonyms():
    result = search_jobs([doc("JavaScript required; team management preferred.")],
                         "Which JD requires Java?")
    assert not result["matches"]
    assert not search_jobs([doc("team management required")], "Which JD requires Leadership?")["matches"]


def test_sections_reset_and_local_negation_overrides_required_header():
    result = search_jobs([doc("Required:\nFlutter is not required.\nResponsibilities\nUse Flutter.")],
                         "Which JD requires Flutter?")
    assert result["matches"][0]["level"] == "mixed"
    assert "requires Flutter" not in result["text"]


@pytest.mark.parametrize("question", ["Which JD requires Flutter?", "Which jobs mention Leadership?",
                                     "Find JDs with Python", "Which job descriptions require C++?"])
def test_routes_only_explicit_jd_search(question):
    assert is_jd_search(question)


@pytest.mark.parametrize("question", ["Compare candidates 1 and 2", "Does candidate 1 know Flutter?",
                                     "Which nationality does this JD prefer?", "Hi", "Which JD?"])
def test_other_queries_are_not_skill_search(question):
    assert not is_jd_search(question)


def test_empty_or_sensitive_query_asks_for_job_related_skill():
    result = search_jobs([], "Which JD requires nationality?")
    assert result["status"] == "clarification"
    assert not result["matches"]


def test_conflicting_mentions_are_not_collapsed_into_a_requirement():
    result = search_jobs([doc("Flutter is required. Flutter is optional.")], "Which JD requires Flutter?")
    assert result["matches"][0]["level"] == "mixed"
    assert "conflicting" in result["text"]


@pytest.mark.parametrize("skill", ["Node.js", ".NET", "C++", "C#"])
def test_punctuation_in_technology_name_is_preserved(skill):
    result = search_jobs([doc(f"{skill} is required. Leadership is preferred.")], f"Which JD requires {skill}?")
    assert result["matches"][0]["level"] == "required"
    assert result["skill"] == skill


def test_no_skill_required_is_not_a_positive_requirement():
    result = search_jobs([doc("No Flutter experience is required.")], "Which JD requires Flutter?")
    assert result["matches"][0]["level"] == "negated"


def test_mixed_requirement_levels_in_one_clause_are_not_guessed():
    result = search_jobs([doc("Python is required and Flutter is optional.")], "Which JD requires Python?")
    assert result["matches"][0]["level"] == "mixed"


def test_storage_limit_removes_whole_matches_with_honest_coverage():
    import json
    documents = [doc("Required: Flutter " + "application development " * 55, str(index), "工程师" * 50)
                 for index in range(20)]
    result = search_jobs(documents, "Which JD requires Flutter?", max_matches=20)
    assert len(json.dumps(result, ensure_ascii=False).encode()) < 24_000
    assert result["coverage"]["omitted_matches"] > 0
    assert result["status"] == "partial"
    owners = {row["jd_id"] for row in result["matches"]}
    assert {row["owner"] for row in result["citations"]} == owners


def test_duplicates_and_invalid_limits_fail_closed():
    with pytest.raises(ValueError):
        search_jobs([doc("Flutter required"), doc("Flutter preferred")], "Which JD requires Flutter?")
    with pytest.raises(ValueError):
        search_jobs([], "Which JD requires Flutter?", max_documents=0)
