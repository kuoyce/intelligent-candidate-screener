"""Whole-turn v2 continuity and exact source reopening, without model weights."""
import os
import json
import time
from types import SimpleNamespace

from candidate_screener.app.portal_extractive import PortalExtractiveService
from candidate_screener.app.conversation_service import ConversationService


class NoQa:
    def answer(self, question, passages):
        return {"status": "not_found", "spans": []}


def snapshot(identity="snap-one", **changes):
    values = dict(snapshot_id=identity, jd_id="JD-A", jd_text="Python programming skills. Bachelor degree required.",
                  candidates=tuple({"cv_id": owner, "rank": rank, "preview_text": text,
                                    "preview_status": "available_redacted_text"}
                                   for owner, rank, text in (
                                       ("CV-A", 1, "Python programming skills. Completed Bachelor degree."),
                                       ("CV-B", 2, "SQL skills. Master degree in progress."),
                                       ("CV-C", 3, "Cloud engineering skills. AWS certification expired in 2021."))))
    values.update(changes)
    return SimpleNamespace(**values)


def service():
    base = PortalExtractiveService(qa_client=NoQa())
    return base if os.getenv("V2_CONVERSATION_BASELINE") else ConversationService(base)


def test_plural_qualification_followup_keeps_comparison_pair():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare candidates 1 and 2")
    second = engine.answer(current, "What about their qualifications?", first["context"])
    assert second["status"] != "clarification"
    assert second["context"]["selected_candidates"] == ["CV-A", "CV-B"]
    assert second["context"]["category"] == "DEGREE"
    assert "progress" in second["text"]


def test_pending_criterion_resumes_when_hr_supplies_candidate_numbers():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare their cloud experience")
    assert first["status"] == "clarification"
    second = engine.answer(current, "1 and 3", first["context"])
    assert second["status"] != "clarification"
    assert second["context"]["selected_candidates"] == ["CV-A", "CV-C"]
    assert "cloud" in second["context"]["last_question"].lower()


def test_ordinal_refers_to_active_order_not_saved_rank():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare candidates 3 and 1")
    second = engine.answer(current, "What degree does the second one have?", first["context"])
    assert second["context"]["selected_candidates"] == ["CV-A"]
    assert "Bachelor" in second["text"]


def test_add_and_remove_candidates_keep_criterion_and_order():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare candidates 1 and 2 on qualifications")
    added = engine.answer(current, "Now include candidate 3", first["context"])
    assert added["context"]["selected_candidates"] == ["CV-A", "CV-B", "CV-C"]
    removed = engine.answer(current, "Remove candidate 2", added["context"])
    assert removed["context"]["selected_candidates"] == ["CV-A", "CV-C"]
    assert removed["context"]["category"] == "DEGREE"


def test_citation_challenge_reopens_same_evidence_without_new_qa():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare candidates 1 and 2 on qualifications")
    second = engine.answer(current, "Where does it say the Master degree is in progress?", first["context"])
    assert second.get("generation_mode") == "citation_reopen"
    assert second["citations"]
    assert all(citation in first["citations"] for citation in second["citations"])
    assert {citation["owner"] for citation in second["citations"]} == {"CV-B"}


def test_citation_reopen_rejects_changed_source_even_with_same_snapshot_id():
    current = snapshot()
    engine = service()
    first = engine.answer(current, "Compare candidates 1 and 2 on qualifications")
    rows = [dict(row) for row in current.candidates]
    rows[1]["preview_text"] = "Master degree completed."
    changed = snapshot(candidates=tuple(rows))
    second = engine.answer(changed, "Where does it say that?", first["context"])
    assert second["status"] == "source_unavailable"
    assert second["citations"] == []


def test_new_snapshot_clears_previous_candidate_references():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    second = engine.answer(snapshot("snap-two"), "What about their degrees?", first["context"])
    assert second["status"] == "clarification"
    assert second["context"]["selected_candidates"] == []


def test_stack_up_preserves_pair_and_singular_ambiguous_reference_clarifies():
    engine = service()
    current = snapshot()
    first = engine.answer(current, "Compare candidates 1 and 2")
    follow = engine.answer(current, "How do these two stack up on qualifications?", first["context"])
    assert follow["status"] != "clarification"
    assert follow["context"]["selected_candidates"] == ["CV-A", "CV-B"]
    ambiguous = engine.answer(current, "Does he have a degree?", follow["context"])
    assert ambiguous["status"] == "clarification"


def test_actions_still_use_existing_guidance_without_execution():
    result = service().answer(snapshot(), "Email candidate 1")
    assert result["status"] == "action_guidance"
    assert result["citations"] == []


def test_invalid_composer_plan_falls_back_without_losing_candidate():
    class BadComposer:
        def compose(self, question, frame, facts):
            return {"relation_ids": ["foreign-claim"]}
    engine = ConversationService(PortalExtractiveService(qa_client=NoQa()), BadComposer())
    result = engine.answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert result["generation_mode"] == "qwen_fallback"
    assert set(result["comparison"][0]["answers"]) == {"CV-A", "CV-B"}
    assert "foreign-claim" not in result["text"]


def test_large_comparison_remains_within_saved_turn_budget():
    source = " ".join(["Python programming skills include " + "engineering " * 35 + ".",
                       "Bachelor degree completed in " + "engineering " * 35 + ".",
                       "Experience includes five years of Python and " + "engineering " * 35 + "."])
    current = snapshot()
    current.candidates = tuple({**row, "preview_text": source} for row in current.candidates)
    result = ConversationService(PortalExtractiveService(qa_client=NoQa())).compare(current, ["CV-A", "CV-B", "CV-C"])
    assert len(json.dumps(result).encode()) <= 23_900
    assert result["context"]["selected_candidates"] == ["CV-A", "CV-B", "CV-C"]


def test_missing_candidate_cannot_be_silently_ignored_during_add():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    result = engine.answer(snapshot(), "Add candidate 9", first["context"])
    assert result["status"] == "clarification"


def test_nonexistent_ordinal_clarifies_instead_of_reusing_only_candidate():
    engine = service()
    first = engine.answer(snapshot(), "What degree does candidate 1 have?")
    result = engine.answer(snapshot(), "What about the second one?", first["context"])
    assert result["status"] == "clarification"


def test_source_challenge_cannot_substitute_another_candidates_old_source():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    result = engine.answer(snapshot(), "Where does candidate 3 say that?", first["context"])
    assert result["status"] == "clarification"
    assert result["citations"] == []


def test_single_candidate_reply_resumes_pending_qualification_question():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    unclear = engine.answer(snapshot(), "Does he have a degree?", first["context"])
    result = engine.answer(snapshot(), "2", unclear["context"])
    assert result["status"] != "clarification"
    assert result["context"]["selected_candidates"] == ["CV-B"]
    assert "progress" in result["text"]


def test_criterion_superlative_compares_active_pair_instead_of_returning_rank_one():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    result = engine.answer(snapshot(), "Who has the strongest evidence on Python?", first["context"])
    assert result["context"]["selected_candidates"] == ["CV-A", "CV-B"]


def test_role_comparison_boilerplate_preserves_available_cv_evidence():
    from candidate_screener.app.conversation_smoke import fixture
    result = service().answer(fixture(), "Compare candidates 1 and 2 for this role")
    assert {item["owner"] for item in result["citations"]} >= {"CV-A", "CV-B"}
    assert "completed" in result["text"].lower()
    assert "in progress" in result["text"].lower()


def test_generic_cloud_experience_does_not_require_skill_tenure():
    from candidate_screener.app.conversation_smoke import fixture
    result = service().answer(fixture(), "Compare candidates 1 and 3 on cloud experience")
    assert "CV-C" in {item["owner"] for item in result["citations"]}
    assert "Cloud engineering experience" in result["text"]


def test_explicit_candidate_id_overrides_plural_comparison_context():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    result = engine.answer(snapshot(), "What does CV-C say about their qualifications?", first["context"])
    assert result["context"]["selected_candidates"] == ["CV-C"]


def test_candidate_ids_can_add_and_remove_from_active_comparison():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2")
    added = engine.answer(snapshot(), "Include CV-C", first["context"])
    assert added["context"]["selected_candidates"] == ["CV-A", "CV-B", "CV-C"]
    removed = engine.answer(snapshot(), "Remove CV-B", added["context"])
    assert removed["context"]["selected_candidates"] == ["CV-A", "CV-C"]


def test_large_citation_reopen_requests_focused_scope_without_clipping():
    from candidate_screener.app.conversation_schema import evidence_descriptors
    from candidate_screener.app.portal_extractive import _documents, _citation
    source = "\n".join("Statement " + str(index) + ": " + "engineering " * 60 + "." for index in range(20))
    current = snapshot()
    current.candidates = tuple({**row, "preview_text": source} for row in current.candidates)
    document = _documents(current, ["CV-A"])["CV-A"]
    citations = [_citation(current, document, unit) for unit in document["units"]]
    context = {"snapshot_id": current.snapshot_id, "jd_id": current.jd_id,
               "selected_candidates": ["CV-A"], "last_evidence": evidence_descriptors(citations)}
    result = ConversationService(PortalExtractiveService(qa_client=NoQa())).answer(current, "Where does it say that?", context)
    assert result["status"] == "clarification"
    assert result["citations"] == []
    assert len(json.dumps(result).encode()) <= 23_900


def test_reader_calls_share_absolute_deadline_without_mutating_client_timeout():
    class DeadlineQa:
        timeout = 120
        def __init__(self):
            self.calls, self.legacy_calls = [], []
        def answer(self, question, passages):
            self.legacy_calls.append(question)
            return {"status": "unavailable", "spans": []}
        def answer_before(self, question, passages, absolute_deadline):
            self.calls.append(absolute_deadline)
            return {"status": "unavailable", "spans": []}
    reader = DeadlineQa()
    started = time.monotonic()
    ConversationService(PortalExtractiveService(qa_client=reader)).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert len(reader.calls) == 3
    assert len(set(reader.calls)) == 1
    assert started < reader.calls[0] <= started + 30.1
    assert reader.legacy_calls == []
    assert reader.timeout == 120


def test_composer_receives_same_deadline_as_native_reader():
    class DeadlineQa(NoQa):
        calls = []
        def answer_before(self, question, passages, absolute_deadline):
            self.calls.append(absolute_deadline)
            return super().answer(question, passages)
    class DeadlineComposer:
        def __init__(self):
            self.calls = []
        def compose(self, question, frame, facts):
            return {"relation_ids": []}
        def compose_before(self, question, frame, facts, absolute_deadline):
            self.calls.append(absolute_deadline)
            return {"relation_ids": []}
    reader, composer = DeadlineQa(), DeadlineComposer()
    ConversationService(PortalExtractiveService(qa_client=reader), composer).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert composer.calls == [reader.calls[0]]


def test_budget_rerender_does_not_claim_qwen_authored_the_replacement():
    engine = ConversationService(PortalExtractiveService(qa_client=NoQa()))
    current = snapshot()
    result = engine.compare(current, ["CV-A", "CV-B"])
    result.update(generation_mode="qwen_answer_plan", model="test-qwen",
                  generation_provenance={"model": "test-qwen", "accepted": True},
                  text="Long comparison text " * 2000)
    result = engine._bounded(current, result)
    assert result["generation_mode"] == "controlled"
    assert result["model"] is None
    assert result["generation_provenance"]["fallback_reason"] == "answer_storage_budget"
    assert not result["generation_provenance"].get("accepted")


def test_exhausted_shared_budget_skips_later_documents_and_composer(monkeypatch):
    from candidate_screener.app import conversation_service as module
    clock = [0.]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    class DeadlineQa(NoQa):
        def __init__(self):
            self.calls = []
        def answer_before(self, question, passages, absolute_deadline):
            self.calls.append(absolute_deadline)
            clock[0] = 31.
            return super().answer(question, passages)
    class Composer:
        def compose(self, *args):
            pytest.fail("Composition must not start after the shared inference deadline")
    import pytest
    reader = DeadlineQa()
    result = ConversationService(PortalExtractiveService(qa_client=reader), Composer()).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert reader.calls == [30.]
    assert result["generation_mode"] == "qwen_fallback"
    assert result["generation_provenance"]["fallback_reason"] == "TimeoutError"
    assert result["comparison"]


def test_explicit_candidate_id_in_source_challenge_cannot_select_other_owner():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    result = engine.answer(snapshot(), "Where does CV-C say that?", first["context"])
    assert result["status"] == "clarification"
    assert result["citations"] == []


def test_broad_role_comparison_explains_explicit_jd_deployment_tradeoff_without_qa():
    from candidate_screener.app.conversation_smoke import fixture
    class NeverQa:
        def answer(self, question, passages):
            pytest.fail("Broad criterion comparison should use source contexts without neural startup")
    import pytest
    result = ConversationService(PortalExtractiveService(qa_client=NeverQa())).answer(fixture(), "Compare candidates 1 and 2 for this role")
    contrasts = [relation for relation in result.get("comparison_relations", []) if relation["kind"] == "explicit_contrast"]
    assert contrasts
    assert any("deployment" in relation["text"].lower() for relation in contrasts)
    assert "Production deployment experience" in result["text"]
    assert all(row.get("criterion") for row in result["comparison"])
    assert "KNOWLEDGE" not in {row["category"] for row in result["comparison"]}


def test_explicit_role_comparison_resets_prior_qualification_focus():
    from candidate_screener.app.conversation_smoke import fixture
    engine = service()
    first = engine.answer(fixture(), "Compare candidates 1 and 2 on qualifications")
    result = engine.answer(fixture(), "Compare candidates 1 and 2 for this role", first["context"])
    assert any("deployment" in relation["text"].lower() for relation in result.get("comparison_relations", []))


def test_rank_explanation_reports_saved_rank_without_invented_causality():
    from candidate_screener.app.conversation_smoke import fixture
    current = fixture()
    original = [(row["cv_id"], row["rank"], row.get("score")) for row in current.candidates]
    result = service().answer(current, "Why is candidate 1 ranked first?")
    assert "Candidate 1 is ranked 1 in the saved ranking" in result["text"]
    assert "causal" in result["text"]
    assert "Production deployment experience" in result["text"]
    assert result["context"]["selected_candidates"] == ["CV-A"]
    assert original == [(row["cv_id"], row["rank"], row.get("score")) for row in current.candidates]


def test_unbalanced_relations_skip_composition_without_model_failure():
    from candidate_screener.app.conversation_smoke import fixture
    class NeverComposer:
        def compose(self, *args):
            raise AssertionError("Unbalanced relations must not reach the composer")
    result = ConversationService(PortalExtractiveService(qa_client=NoQa()), NeverComposer()).answer(fixture(), "Compare candidates 1 and 3 on cloud experience")
    assert result["generation_mode"] == "controlled"
    assert result["generation_provenance"]["composition_skipped"] == "no_balanced_relations"
    assert "fallback_reason" not in result["generation_provenance"]
    assert "Cloud engineering experience" in result["text"]


def test_no_relations_skip_composition_but_keep_complete_source_context():
    class NeverComposer:
        def compose(self, *args):
            raise AssertionError("No relations must not reach the composer")
    current = snapshot()
    current.candidates = tuple({**row, "preview_text": "Python."} for row in current.candidates)
    result = ConversationService(PortalExtractiveService(qa_client=NoQa()), NeverComposer()).answer(current, "Compare candidates 1 and 2 on Python")
    assert result["generation_mode"] == "controlled"
    assert result["generation_provenance"]["composition_skipped"] == "no_balanced_relations"
    assert {item["owner"] for item in result["citations"]} >= {"CV-A", "CV-B"}


def test_recognized_jd_criteria_do_not_claim_exhaustive_requirement_parsing():
    from candidate_screener.app.conversation_smoke import fixture
    result = service().answer(fixture(), "Compare candidates 1 and 2 for this role")
    assert result["comparison_scope"] == "recognized_jd_criteria"
    assert result["criteria_exhaustive"] is False


def test_model_plan_rejection_uses_safe_specific_reason_code():
    from candidate_screener.llm.conversation_qwen import CompositionPlanError
    class InvalidJsonComposer:
        def compose(self, *args):
            raise CompositionPlanError("invalid_json")
    result = ConversationService(PortalExtractiveService(qa_client=NoQa()), InvalidJsonComposer()).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    assert result["generation_mode"] == "qwen_fallback"
    assert result["generation_provenance"]["fallback_reason"] == "invalid_json"


def test_bare_directed_why_resolves_pair_and_keeps_comparison_table():
    from candidate_screener.app.conversation_smoke import fixture
    result = service().answer(fixture(), "Why is 1 better than 2?")
    assert result["status"] != "clarification"
    assert result["context"]["selected_candidates"] == ["CV-A", "CV-B"]
    assert result["context"]["intent"] == "comparison_explanation"
    assert result["show_comparison_table"] is True
    assert result["comparison"]
    assert "tradeoff" in result["text"].casefold() or "trade-off" in result["text"].casefold()
    assert "deployment" in result["text"].casefold()


def test_directed_why_after_comparison_reuses_qualification_criterion():
    engine = service()
    first = engine.answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    result = engine.answer(snapshot(), "Why is 1 better than 2?", first["context"])
    assert result["context"]["selected_candidates"] == ["CV-A", "CV-B"]
    assert result["context"]["category"] == "DEGREE"
    assert result["show_comparison_table"] is True
    assert "stronger documented evidence" in result["text"].casefold()
    assert "qualification" in result["text"].casefold() or "degree" in result["text"].casefold()


def test_directed_why_can_favor_lower_saved_rank_on_named_knowledge_criterion():
    current = snapshot(jd_text="Python skills required. Database knowledge required.")
    current.candidates = tuple({**row, "preview_text": "Python skills." if row["rank"] == 1 else "Python skills. Database knowledge."}
                               for row in current.candidates)
    before = [(row["cv_id"], row["rank"]) for row in current.candidates]
    result = service().answer(current, "Why is 2 better than 1?")
    assert result["context"]["selected_candidates"] == ["CV-B", "CV-A"]
    assert "Candidate 2 has stronger documented evidence" in result["text"]
    assert "database" in result["text"].casefold()
    assert ("does not establish absence" in result["text"].casefold()
            or "not proof that the candidate lacks" in result["text"].casefold())
    assert before == [(row["cv_id"], row["rank"]) for row in current.candidates]


def test_ranked_above_question_corrects_false_rank_premise():
    result = service().answer(snapshot(), "Why is 2 ranked above 1?")
    assert result["context"]["selected_candidates"] == ["CV-B", "CV-A"]
    assert "Candidate 2 at 2" in result["text"] and "Candidate 1 at 1" in result["text"]
    assert "not ranked above" in result["text"]
    assert "do not prove why" in result["text"]


def test_directed_why_does_not_parse_years_as_second_candidate():
    result = service().answer(snapshot(), "Why does candidate 1 have 2 years of Python experience?")
    assert result["context"]["selected_candidates"] == ["CV-A"]
    assert result["context"]["intent"] != "comparison_explanation"


def test_compact_compare_then_directed_why_reuses_exact_table_evidence():
    from candidate_screener.app.conversation_smoke import fixture
    engine = service()
    current = fixture()
    first = engine.answer(current, "compare1and2")
    second = engine.answer(current, "why is1betterthan2", first["context"])
    assert second["context"]["selected_candidates"] == ["CV-A", "CV-B"]
    assert second["show_comparison_table"] is True
    assert second["comparison"] == first["comparison"]
    assert {item["evidence_id"] for item in first["citations"]} == {item["evidence_id"] for item in second["citations"]}


def test_directed_why_rejects_unknown_or_same_candidates():
    engine = service()
    for question in ("Why is 1 better than 9?", "Why is 1 better than 1?"):
        result = engine.answer(snapshot(), question)
        assert result["status"] == "clarification"
        assert result["citations"] == []


def test_partial_cv_cannot_establish_directed_coverage_advantage():
    current = snapshot(jd_text="Python skills required.")
    current.candidates = tuple({**row, "preview_text": "Python skills. " + "unsearched history " * 120 + "."
                               if row["rank"] == 1 else "SQL skills."} for row in current.candidates)
    result = service().answer(current, "Why is 1 better than 2?")
    assert not any(item["kind"] == "comparison_advantage" for item in result.get("comparison_relations", []))
    assert result["comparison"][0]["answers"]["CV-A"]["source_status"] == "partial"


def test_specific_python_why_corrects_premise_and_reverse_respects_source_duration():
    from candidate_screener.app.conversation_smoke import fixture
    engine = service()
    wrong = engine.answer(fixture(), "Why is 1 better than 2 on Python experience?")
    assert "cannot support the premise" in wrong["text"]
    assert "Candidate 2" in wrong["text"]
    assert "60" in wrong["text"] and "36" in wrong["text"]
    right = engine.answer(fixture(), "Why is 2 better than 1?", wrong["context"])
    assert right["context"]["selected_candidates"] == ["CV-B", "CV-A"]
    assert "stronger documented evidence for Candidate 2" in right["text"]
    assert right["show_comparison_table"] is True


def test_broad_directed_why_preserves_separate_conflicting_passages():
    current = snapshot(jd_text="Python skills required.")
    current.candidates = tuple({**row, "preview_text": "Python skills. Maintained deployment pipelines. No Python skills."
                               if row["rank"] == 1 else "SQL skills."} for row in current.candidates)
    result = service().answer(current, "Why is 1 better than 2?")
    assert not any(item["kind"] == "comparison_advantage" for item in result.get("comparison_relations", []))
    assert any(item["kind"] == "source_conflict" and item["participants"] == ["CV-A"] for item in result["comparison_relations"])
    cell = result["comparison"][0]["answers"]["CV-A"]
    assert "Python skills." in cell["context"] and "No Python skills." in cell["context"]
    assert len(cell["evidence_ids"]) == 2
    assert "conflict" in result["text"].casefold() or "inconsistent" in result["text"].casefold()


def test_shared_broad_table_and_directed_why_keep_both_conflict_citations():
    current = snapshot(jd_text="Python skills required.")
    current.candidates = tuple({**row, "preview_text": "Python skills. Other project history. No Python skills."
                               if row["rank"] == 1 else "SQL skills."} for row in current.candidates)
    engine = service()
    first = engine.answer(current, "Compare candidates 1 and 2")
    why = engine.answer(current, "Why is 1 better than 2?", first["context"])
    assert first["comparison"] == why["comparison"]
    assert {item["evidence_id"] for item in first["citations"]} == {item["evidence_id"] for item in why["citations"]}
    assert len(why["comparison"][0]["answers"]["CV-A"]["evidence_ids"]) == 2


def test_overlapping_retained_contexts_use_source_union_without_losing_negation():
    from candidate_screener.app.portal_extractive import _documents
    source = "Python experience. No commercial Python experience."
    current = snapshot()
    current.candidates = tuple({**row, "preview_text": source} for row in current.candidates)
    document = _documents(current, ["CV-A"])["CV-A"]
    second = source.index("No commercial")
    relevant = [(2, 0, (0, len(source), source), []),
                (1, second, (second, len(source), source[second:]), [])]
    citations, claims = {}, []
    cell = ConversationService._criterion_cell(current, document, relevant, citations, claims)
    assert cell["context"] == source
    assert len(cell["evidence_ids"]) == len(citations) == len(claims) == 1
    citation = next(iter(citations.values()))
    assert citation["quote"] == source[citation["start"]:citation["end"]] == source
