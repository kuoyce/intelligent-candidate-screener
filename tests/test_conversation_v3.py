"""V3 factual prose, provenance and history invariants with invented sources."""
import os
from types import SimpleNamespace

import pytest

from candidate_screener.app.portal_extractive import PortalExtractiveService
from candidate_screener.app.conversation_service import ConversationService


class NoQa:
    def answer(self, question, passages):
        return {"status": "not_found", "spans": []}


class Composer:
    def __init__(self, text):
        self.text, self.calls = text, []

    def compose_before(self, question, frame, facts, deadline):
        self.calls.append((question, frame, facts))
        return {"text": self.text, "mode": "qwen_prose", "model": "synthetic-qwen", "provenance": {}}


def snapshot():
    return SimpleNamespace(snapshot_id="s-one", jd_id="JD-A", jd_text="Bachelor degree required.",
        candidates=tuple({"cv_id": owner, "rank": rank, "preview_text": text,
                          "preview_status": "available_redacted_text"}
                         for owner, rank, text in [("CV-A", 1, "Completed Bachelor degree."),
                                                   ("CV-B", 2, "Master degree in progress.")]))


def service(composer):
    base = PortalExtractiveService(qa_client=NoQa())
    if os.getenv("V3_CONTEXT_BASELINE"):
        return ConversationService(base, composer=None)
    from candidate_screener.app.conversation_v3 import ConversationV3Service
    return ConversationV3Service(base, composer)


def test_model_authored_single_candidate_prose_replaces_raw_span():
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    result = service(model).answer(snapshot(), "What degree does candidate 1 have?")
    assert result["text"] == model.text
    assert result["generation_mode"] == "qwen_prose"
    assert result["mode"] == "conversation_v3"
    assert result["generation_provenance"]["support_check"] == "bounded_rule_checked"
    assert result["validated_claim_sources"]


def test_comparison_followup_accepts_qualified_contrast_without_source_labels():
    model = Composer("Candidate 1 has completed a Bachelor degree, while Candidate 2 is pursuing a Master degree.")
    result = service(model).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert result["generation_mode"] == "qwen_prose"
    assert result["show_comparison_table"] is True
    assert "Source excerpt" not in result["text"]
    assert "[S" not in result["text"]


@pytest.mark.parametrize("text", [
    "Candidate 2 has completed a Master degree.",
    "Candidate 2 is pursuing a Master degree and has five years of Python experience.",
    "Candidate 1 has completed a Master degree.",
    '{"answer": "Candidate 2 has a degree"}',
    "Candidate 2 is the best candidate and should be hired.",
    "Candidate 2 is pursuing a Master degree. I have emailed them.",
])
def test_unsupported_or_action_claims_are_not_displayed(text):
    model = Composer(text)
    result = service(model).answer(snapshot(), "What degree does candidate 2 have?")
    assert result["generation_mode"] == "qwen_fallback"
    assert result["text"] != text
    assert result["validated_claim_sources"] == []


def test_history_is_explicit_and_never_promoted_to_evidence():
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    engine, current = service(model), snapshot()
    first = engine.answer(current, "What degree does candidate 1 have?")
    history = [{"question": "What degree does candidate 1 have?", "answer": first}]
    engine.answer(current, "What about their qualifications?", first["context"], history=history)
    bundle = model.calls[-1][2]
    assert bundle["history"][0]["answer"] == model.text
    assert all("question" not in item for item in bundle["evidence"])
    first["generation_provenance"]["scope_digest"] = "another-scope"
    engine.answer(current, "What degree does candidate 1 have?", history=history)
    assert model.calls[-1][2]["history"] == []


def test_changed_source_discards_history_even_when_snapshot_id_is_reused():
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    engine, current = service(model), snapshot()
    first = engine.answer(current, "What degree does candidate 1 have?")
    current.candidates[0]["preview_text"] = "Bachelor degree in progress."
    engine.answer(current, "What degree does candidate 1 have?", history=[{"question": "Earlier?", "answer": first}])
    assert model.calls[-1][2]["history"] == []


def test_greeting_does_not_call_model_or_extract():
    model = Composer("unused")
    result = service(model).answer(snapshot(), "Hi")
    assert result["text"] == "Hi! What would you like to review?"
    assert not model.calls


def test_unavailable_model_keeps_sources_without_raw_dump():
    result = service(None).answer(snapshot(), "What degree does candidate 1 have?")
    assert result["generation_mode"] == "qwen_fallback"
    assert "unavailable" in result["text"]
    assert result["citations"]
    assert "Source excerpt" not in result["text"]


def test_why_followup_keeps_expandable_application_table():
    model = Composer("Candidate 1 has completed a Bachelor degree, while Candidate 2 is pursuing a Master degree.")
    engine, current = service(model), snapshot()
    first = engine.answer(current, "Compare candidates 1 and 2 on qualifications")
    result = engine.answer(current, "Why is 1 better than 2?", first["context"])
    assert result["show_comparison_table"] is True
    assert result["comparison_table_expandable"] is True
    assert first["comparison_table_expandable"] is False


def test_history_drops_whole_old_exchanges_after_four():
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    engine, current = service(model), snapshot()
    first = engine.answer(current, "What degree does candidate 1 have?")
    history = [{"question": f"Question {index}", "answer": first} for index in range(8)]
    engine.answer(current, "What degree does candidate 1 have?", history=history)
    assert [turn["question"] for turn in model.calls[-1][2]["history"]] == [f"Question {index}" for index in range(4, 8)]


def test_matching_words_do_not_establish_semantic_support():
    model = Composer("Candidate 2 has made progress and completed a Master degree.")
    result = service(model).answer(snapshot(), "What degree does candidate 2 have?")
    assert result["generation_mode"] == "qwen_fallback"
    assert result["generation_provenance"]["fallback_reason"] == "unsupported_paraphrase"


def test_tampered_citation_does_not_authorize_a_claim():
    from candidate_screener.app.conversation_context import build_context, validate_prose
    current = snapshot()
    original = ConversationService(PortalExtractiveService(qa_client=NoQa())).answer(current, "What degree does candidate 1 have?")
    original["citations"][0]["document_digest"] = "invalid"
    bundle = build_context(current, original)
    assert bundle["evidence"] == []
    assert validate_prose("Candidate 1 has completed a Bachelor degree.", bundle)[1] == "unsupported_paraphrase"


def test_storage_guard_drops_whole_records_not_qualification_suffix():
    import json
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    engine, current = service(model), snapshot()
    original = engine.reader.answer

    def oversized(*args, **kwargs):
        result = original(*args, **kwargs)
        result["comparison"][0]["answers"]["CV-A"]["text"] = "x" * 30_000
        return result

    engine.reader.answer = oversized
    result = engine.answer(current, "What degree does candidate 1 have?")
    assert len(json.dumps(result, ensure_ascii=False).encode()) < 24_000
    assert result["generation_provenance"]["fallback_reason"] == "answer_storage_budget"
    assert result["comparison"] == []
    assert result["context"]["last_evidence"] == []


@pytest.mark.parametrize("answer", [
    {"context": None, "generation_provenance": {}, "mode": "jd_search", "text": "A role lists Flutter."},
    {"context": {}, "generation_provenance": None, "mode": "conversation_v3", "text": "Earlier."},
])
def test_non_conversation_or_invalid_history_metadata_is_skipped(answer):
    model = Composer("Candidate 1 has completed a Bachelor degree.")
    result = service(model).answer(snapshot(), "What degree does candidate 1 have?",
                                  history=[{"question": "Which JD requires Flutter?", "answer": answer}])
    assert result["generation_mode"] == "qwen_prose"
    assert model.calls[-1][2]["history"] == []


def test_qualification_confirmation_retrieves_status_not_user_assertion():
    model = Composer("Candidate 2 is pursuing a Master degree.")
    engine, current = service(model), snapshot()
    question = "So candidate 2 has completed a Master degree?"
    result = engine.answer(current, question)
    assert result["generation_mode"] == "qwen_prose"
    assert model.calls[-1][0] == question
    assert any("in progress" in item["context"] for item in model.calls[-1][2]["evidence"])
    source = engine.answer(current, "Where does it say the Master degree is in progress?", result["context"])
    assert source["generation_mode"] == "citation_reopen"
    assert "in progress" in source["text"]


@pytest.mark.parametrize("reason", ["non_prose_output", "invalid_context", "context_overflow"])
def test_rejected_model_output_is_not_misreported_as_model_unavailable(reason):
    from candidate_screener.llm.base import ContextOverflow

    class Rejected(Exception):
        pass

    class RejectingComposer:
        def compose_before(self, *args):
            if reason == "context_overflow":
                raise ContextOverflow("Too many tokens")
            error = Rejected("Rejected output")
            error.reason = reason
            raise error

    result = service(RejectingComposer()).answer(snapshot(), "What degree does candidate 1 have?")
    assert result["generation_provenance"]["fallback_reason"] == reason
    assert "unavailable" not in result["text"]


@pytest.mark.parametrize("text", ["Candidate 1 has a Bachelor degree.", "Candidate 1 holds a Bachelor degree."])
def test_completed_qualification_allows_narrow_has_and_holds_aliases(text):
    result = service(Composer(text)).answer(snapshot(), "What degree does candidate 1 have?")
    assert result["generation_mode"] == "qwen_prose"
    assert result["text"] == text


def test_has_qualification_in_progress_keeps_the_completion_qualifier():
    text = "Candidate 1 has completed a Bachelor degree, while Candidate 2 has a Master degree in progress."
    result = service(Composer(text)).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert result["generation_mode"] == "qwen_prose"
    assert result["text"] == text
    rejected = service(Composer("Candidate 2 has a Master degree.")).answer(snapshot(), "What degree does candidate 2 have?")
    assert rejected["generation_mode"] == "qwen_fallback"


def test_degree_possessive_wording_preserves_supported_completion():
    result = service(Composer("Candidate 1 has completed a Bachelor's degree.")).answer(
        snapshot(), "What degree does candidate 1 have?")
    assert result["generation_mode"] == "qwen_prose"
    rejected = service(Composer("Candidate 2 has completed a Master's degree.")).answer(
        snapshot(), "What degree does candidate 2 have?")
    assert rejected["generation_mode"] == "qwen_fallback"


def test_ongoing_degree_explanation_preserves_owner_and_noncompletion():
    text = ("Candidate 2 is currently pursuing a Master's degree, which does not confirm that they have completed it. "
            "The available evidence only states that the degree is in progress.")
    assert service(Composer(text)).answer(snapshot(), "So candidate 2 has completed a Master degree?")["generation_mode"] == "qwen_prose"
    assert service(Composer(text.replace("Candidate 2", "Candidate 1"))).answer(
        snapshot(), "What degree does candidate 1 have?")["generation_mode"] == "qwen_fallback"


def test_why_fallback_preserves_both_qualification_states_without_repeated_advantage():
    engine = service(Composer("Candidate 1 is fully verified and better overall."))
    first = engine.answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    answer = engine.answer(snapshot(), "Why is 1 better than 2?", context=first["context"])
    assert "Candidate 2 is pursuing" in answer["text"]
    assert answer["text"].count("stronger documented evidence") <= 1


def test_good_fit_summary_uses_job_criteria_without_authoring_fit_call():
    current = snapshot()
    current.jd_text = "Python skills required. Bachelor degree required."
    current.candidates[0]["preview_text"] = "Python programming skills. Completed Bachelor degree."
    result = service(Composer("unsupported")).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert result.get("comparison_scope") == "recognized_jd_criteria"
    assert "Python" in result["text"]
    assert any(item["owner"] == current.jd_id for item in result["citations"])
    assert "good fit" not in result["text"].lower()


def test_relevant_job_experience_is_not_restricted_to_tenure():
    current = snapshot()
    current.jd_text = "Python experience required."
    current.candidates[0]["preview_text"] = "Python development experience building production services."
    result = service(Composer("unsupported")).answer(current, "Any relevant job experiences for candidate 1?")
    assert any("production services" in item["quote"] for item in result["citations"])
    assert "Python" in result["text"]


def test_long_source_fallback_summarizes_evidence_state_without_raw_dump():
    current = snapshot()
    current.jd_text = "Python skills required."
    current.candidates[0]["preview_text"] = "Python programming skills include " + "building reliable services " * 10 + "."
    result = service(Composer("unsupported")).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "Python" in result["text"]
    assert "Candidate 1" in result["text"]
    assert "building reliable services building" not in result["text"]


def test_fallback_explains_multiple_role_strengths_in_plain_language():
    current = snapshot()
    current.jd_text = "Python skills required. Bachelor degree required."
    current.candidates[0]["preview_text"] = "Python programming skills. Completed Bachelor degree."
    result = service(Composer("unsupported")).answer(current, "Summarise why candidate 1 is a good fit to this JD")
    assert "Python" in result["text"] and "Bachelor" in result["text"]
    assert "relevant to this role" in result["text"]
    assert "affirmative CV evidence" not in result["text"]
    assert len(result["text"].split(". ")) <= 3


def test_long_qualified_fallback_explains_noncompletion_without_technical_language():
    current = snapshot()
    current.candidates[1]["preview_text"] = "Master degree in progress with " + "advanced research modules " * 10 + "."
    result = service(Composer("unsupported")).answer(current, "What degree does candidate 2 have?")
    assert "not confirmed as completed" in result["text"]
    assert "status restriction" not in result["text"]
    assert "advanced research modules advanced" not in result["text"]


def test_reasoning_fallback_retains_reverse_advantage_and_missing_evidence_caveat():
    current = snapshot()
    current.jd_text = "Python skills required. Cloud skills required."
    current.candidates[0]["preview_text"] = "Python programming skills."
    current.candidates[1]["preview_text"] = "Cloud engineering skills."
    result = service(Composer("unsupported")).answer(current, "Why is 1 better than 2?")
    assert "Python" in result["text"] and "Cloud" in result["text"]
    assert "tradeoff" in result["text"].lower()
    assert "doesn't mean" in result["text"]
    assert len(result["text"].split(". ")) <= 3
