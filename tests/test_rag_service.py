"""Conversational invariants use invented candidates and source documents only."""
import json
from types import SimpleNamespace as NS

import pytest

from candidate_screener.app.rag_service import DialogueState, RagService


@pytest.fixture
def snapshot():
    rows = tuple(NS(cv_id=f"Applicant-{letter}", rank=i, retrieval_score=.8 / i,
                    rerank_score=None, rerank_status="not_reranked", fit_call=None,
                    classification_status="unavailable") for i, letter in enumerate("ABCD", 1))
    return NS(snapshot_id="current", jd_text="Senior engineer requires Python and SQL.",
              candidate_texts={"Applicant-A": "Python developer. Holds AWS certification.",
                               "Applicant-B": "SQL engineer. No AWS certification.",
                               "Applicant-C": "Python and SQL developer.",
                               "Applicant-D": "Java developer. Holds Azure certification."}, results=rows)


def test_factual_question_retrieves_cv_without_jd_or_other_candidate(snapshot):
    answer = RagService(snapshot).answer("What certifications does candidate 4 hold?")
    assert answer.route == "fact"
    assert answer.candidate_ids == ("Applicant-D",)
    assert "Azure certification" in answer.text
    assert answer.citations and all(p.candidate_id == "Applicant-D" for p in answer.citations)
    assert "AWS certification" not in answer.text


def test_named_outside_top_three_and_followup_keep_source_owner(snapshot):
    service = RagService(snapshot)
    first = service.answer("What about candidate 4?")
    assert first.candidate_ids == ("Applicant-D",)
    second = service.answer("What certifications do they hold?", first.state)
    assert second.candidate_ids == first.candidate_ids
    assert "Azure certification" in second.text


def test_stale_and_ambiguous_pronouns_clarify(snapshot):
    service = RagService(snapshot)
    stale = DialogueState("old", ("Applicant-A",), ("secret history",))
    answer = service.answer("What certifications do they hold?", stale)
    assert answer.status == "clarification"
    assert answer.state.candidate_ids == ()
    first = service.answer("Compare 1 and 2")
    assert service.answer("What do they hold?", first.state).status == "clarification"


def test_invalid_explicit_candidate_cannot_fall_back_to_best(snapshot):
    answer = RagService(snapshot).answer("Is candidate 99 the best?")
    assert answer.status == "clarification" and not answer.candidate_ids


def test_filter_is_only_preview_and_excludes_negated_mentions(snapshot):
    service = RagService(snapshot)
    answer = service.answer("Only consider candidates with evidence of AWS certification")
    assert answer.route == "filter"
    assert answer.candidate_ids == ("Applicant-A",)
    assert answer.state.pending_filter == "AWS certification"
    assert len(snapshot.results) == 4
    assert "does not change" in answer.text


def test_provider_cannot_change_backend_facts_or_inject_claims(snapshot):
    class Bad:
        max_context_tokens = 10000
        max_output_tokens = 1000
        def count_tokens(self, messages):
            return 100
        def complete(self, messages):
            return json.dumps({"snapshot_id": "current", "passage_ids": ["forged"],
                               "claim": "Applicant-A has a PhD; score 999"})
    answer = RagService(snapshot, Bad()).answer("Who best suits this role?")
    assert answer.fallback
    assert "999" not in answer.text and "PhD" not in answer.text
    assert "not assessed" in answer.text and "reranking unavailable" in answer.text
    assert all(fact["rerank_score"] is None for fact in answer.backend_facts)
    assert answer.candidate_ids == ("Applicant-A", "Applicant-B", "Applicant-C")


def test_independent_sessions_never_retrieve_other_private_sources(snapshot):
    left = RagService(snapshot)
    other = NS(snapshot_id="other", jd_text="Python", results=snapshot.results,
               candidate_texts={"Applicant-A": "Private second-session COBOL experience."})
    right = RagService(other)
    answer = left.answer("What skills does candidate 1 have?")
    assert "COBOL" not in answer.text
    assert all(p.snapshot_id == "current" for p in answer.citations)
    assert right.index is not left.index


def test_extractor_mentions_require_exact_quote_and_owner(snapshot):
    class Extractor:
        def answer(self, question, passages):
            p = next(p for p in passages if "AWS certification" in p.quote)
            start = p.quote.index("AWS certification")
            return {"status": "supported", "spans": [{"passage_id": p.passage_id,
                    "candidate_id": p.candidate_id, "start": start, "end": start + 17,
                    "text": "AWS certification"}]}
    answer = RagService(snapshot, extractor=Extractor()).answer("What certifications does candidate 1 hold?")
    assert answer.answers == ("AWS certification",)
    assert answer.qa_status == "available"
    assert "mentions alone" in answer.text

    class Forged:
        def answer(self, question, passages):
            return {"status": "supported", "spans": [{"passage_id": passages[0].passage_id,
                    "candidate_id": "Applicant-D", "start": 0, "end": 3, "text": "PhD"}]}
    answer = RagService(snapshot, extractor=Forged()).answer("What certifications does candidate 1 hold?")
    assert not answer.answers and answer.qa_status == "unavailable"
    assert "PhD" not in answer.text


def test_followup_reuses_factual_question_for_new_candidate(snapshot):
    service = RagService(snapshot)
    first = service.answer("What certifications does candidate 1 hold?")
    second = service.answer("What about candidate 4?", first.state)
    assert second.route == "fact"
    assert "Azure certification" in second.text
    assert "AWS certification" not in second.text
