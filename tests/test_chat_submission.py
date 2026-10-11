"""Applicant confirmation is a scoring boundary, independent of Streamlit/models."""
import pytest
from candidate_screener.app.submissions import Submission, applicant_id
from candidate_screener.app import intents
from candidate_screener.pipeline import Result


def test_unreviewed_or_changed_translation_cannot_be_scored():
    submission = Submission(english_text="Python, four years.", state="translated")
    with pytest.raises(ValueError, match="Confirm"):
        submission.scoring_text()
    submission.confirm("Python, four years.")
    assert submission.scoring_text() == "Python, four years."
    submission.english_text = "Python, ten years."
    with pytest.raises(ValueError, match="Confirm"):
        submission.scoring_text()


def test_supplement_requires_written_cv_and_preserves_source_sections():
    submission = Submission()
    with pytest.raises(ValueError):
        submission.confirm(" ")
    submission.confirm("Interview skill: SQL.")
    with pytest.raises(ValueError, match="written CV"):
        submission.scoring_text("CV supplement")
    submission.written_text = "CV skill: Python."
    assert submission.scoring_text("CV supplement") == "Written CV:\nCV skill: Python.\n\nReviewed interview:\nInterview skill: SQL."
    assert applicant_id(1) == "Applicant-001"


def test_entity_chat_distinguishes_unavailability_and_translated_sources():
    result = Result("Applicant-001", 1, .5, None, 1, None, None, None,
                    entity_status="unavailable", source_kind="interview",
                    translation_provenance={"model": "MarianMT"})
    answer = intents.answer(intents.parse("entities 1"), [result])
    assert "unavailable" in answer
    assert all(label in answer for label in ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS"))
    assert "Missing spans do not establish" in answer
    assert "reviewed English translation" in intents.answer(intents.parse("source 1"), [result])
