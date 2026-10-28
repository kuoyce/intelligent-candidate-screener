"""Directed fallback summaries retain checked direction without paragraph dumps."""
from types import SimpleNamespace

import pytest

from candidate_screener.app.conversation_service import ConversationService
from candidate_screener.app.portal_extractive import PortalExtractiveService


class NoQa:
    def answer(self, *args):
        return {"status": "not_found", "spans": []}


def answer(first, second, jd="Python skills required. Cloud skills required."):
    snapshot = SimpleNamespace(snapshot_id="brief", jd_id="JD", jd_text=jd,
        candidates=tuple({"cv_id": owner, "rank": rank, "preview_text": text,
                          "preview_status": "available_redacted_text"}
                         for owner, rank, text in (("A", 1, first), ("B", 9, second))))
    service = ConversationService(PortalExtractiveService(qa_client=NoQa()))
    result = service.answer(snapshot, "Why is candidate 1 better than candidate 9?")
    return snapshot, result, service._facts(snapshot, result)


def brief(*args):
    from candidate_screener.app.conversation_brief import brief_explanation
    return brief_explanation(*args)


def test_forward_only_is_source_strength_not_overall_verdict():
    text = brief(*answer("Python programming skills.", "SQL skills."))
    assert "Candidate 1" in text and "Python" in text
    assert "stronger documented" in text
    assert "overall" in text and "not" in text
    assert "lacks" not in text and "higher priority" not in text
    assert "\n" not in text


def test_reverse_only_rejects_users_superiority_premise():
    text = brief(*answer("SQL skills.", "Python programming skills."))
    assert "cannot support the premise" in text
    assert "Candidate 9" in text and "Python" in text


def test_tradeoff_preserves_both_sides_without_priority_invention():
    text = brief(*answer("Python programming skills.", "Cloud engineering skills."))
    assert "trade-off" in text
    assert "Python" in text and "Cloud" in text
    assert "Candidate 1" in text and "Candidate 9" in text
    assert "higher priority" not in text and "\n" not in text


def test_missing_basis_is_explicit_without_absence_claims():
    text = brief(*answer("SQL skills.", "SQL skills."))
    assert "cannot establish" in text
    assert "Candidate 1" in text and "Candidate 9" in text
    assert "lacks" not in text


def test_ongoing_degree_does_not_become_completed_or_higher():
    text = brief(*answer("Completed Bachelor degree.", "Master degree in progress.",
                        "Bachelor degree required."))
    assert "completion" in text
    assert "not confirmed" in text or "unresolved" in text
    assert "higher" not in text and "\n" not in text


def test_source_linked_duration_can_support_more_but_never_fabricated_numbers():
    text = brief(*answer("Five years of Python experience.", "Three years of Python experience.",
                        "Python experience required."))
    assert "Python" in text
    assert "more explicitly linked" in text
    assert "6" not in text and "double" not in text
    assert "overall" in text


def test_wrong_participants_are_rejected():
    snapshot, result, facts = answer("Python programming skills.", "Cloud engineering skills.")
    result["context"]["explanation_subject"] = "FOREIGN"
    with pytest.raises(ValueError):
        brief(snapshot, result, facts)


def test_saved_rank_does_not_become_causal_explanation():
    snapshot, result, facts = answer("Python programming skills.", "SQL skills.")
    result["context"]["explanation_relation"] = "ranked_above"
    text = brief(snapshot, result, facts)
    assert "saved rank" in text and "why" in text
