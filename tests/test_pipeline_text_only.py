import pandas as pd
import pytest

from candidate_screener import pipeline
from candidate_screener.matching.base import BaseSystem


class Ranker(BaseSystem):
    name = "synthetic text ranker"
    def _score(self, pairs):
        assert set(pairs.columns) == {"cv_id", "cv_text", "jd_id", "jd_text", "pair_id"}
        return pairs.cv_text.str.len().to_numpy()


class Matcher:
    def find(self, text):
        return []


def extracted(text, side):
    rows = []
    for label, surface in [("KNOWLEDGE", "Python"), ("SKILL", "teamwork"),
                           ("TITLE", "Engineer"), ("DEGREE", "BSc"), ("EXPERIENCE_YEARS", "3 years")]:
        start = text.find(surface)
        if start >= 0:
            rows.append({"label": label, "surface": surface, "start": start, "end": start + len(surface),
                         "method": "gliner-zero-shot", "confidence": .8, "context_status": "unqualified"})
    return {"status": "complete" if rows else "completed_zero", "entities": rows,
            "provenance": {"model": "synthetic"}}


def test_metadata_cannot_reach_ranker_or_fit_call():
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": ["Python"], "Position": ["Oracle"],
                        "Primary Keyword": ["Oracle"], "cv_keyword": ["Oracle"], "cv_exp_band": ["7+"]})
    first = pipeline.screen("Python", cvs, {"keyword": "Oracle", "exp_band": "7+"}, Ranker(), Matcher())
    cvs["cv_keyword"] = "Changed"
    second = pipeline.screen("Python", cvs, {"keyword": "Changed", "exp_band": "0-1"}, Ranker(), Matcher())
    assert first[0].score == second[0].score
    assert first[0].fit_call is None and second[0].fit_call is None
    assert pipeline.default_ranker(True).text_variant == "v2"


def test_five_entities_and_exact_gliner_evidence_are_attached():
    text = "Engineer BSc Python teamwork 3 years of experience"
    result = pipeline.screen("Python teamwork", pd.DataFrame({"cv_id": ["a"], "cv_text": [text]}),
                             ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)[0]
    assert len(result.entities) == 5 and result.entity_status == "complete"
    assert result.years_stated == 3
    assert result.ranker_name == Ranker.name
    assert {e["status"] for e in result.evidence} == {"matched"}
    assert all(text[e["start"]:e["end"]] == e["quote"] for e in result.evidence)


def test_company_age_not_experience_and_failed_extraction_not_zero():
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": ["Our company has 25 years of history"]})
    result = pipeline.screen("Experience required", cvs, ranker=Ranker(), matcher=Matcher(),
                             entity_extractor=lambda text, side: {"status": "failed", "entities": []})[0]
    assert result.years_stated is None and result.entity_status == "failed"
    from candidate_screener.extraction.experience import jd_experience, cv_experience
    assert jd_experience("Our company has 25 years of history").min_years is None
    assert cv_experience("I have 5 years of experience at this company").years == 5


def test_empty_candidate_is_rejected_before_ranking():
    with pytest.raises(ValueError, match="nonempty"):
        pipeline.screen("Python", pd.DataFrame({"cv_id": ["a"], "cv_text": [""]}),
                        ranker=Ranker(), matcher=Matcher())


def test_gliner_negative_context_does_not_match():
    result = pipeline.screen("Python", pd.DataFrame({"cv_id": ["a"], "cv_text": ["No experience with Python"]}),
                             ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)[0]
    assert result.evidence[0]["status"] == "missing"


def test_interview_entities_and_evidence_link_only_reviewed_english_segments():
    from candidate_screener.multimodal.entity_bridge import reviewed_document
    document = reviewed_document([{"english": "Python", "french": "Python", "start": 1, "end": 2}], confirmed=True)
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": [document["text"]],
                        "source_kind": ["interview"], "translation_provenance": [document]})
    result = pipeline.screen("Python", cvs, ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)[0]
    for row in [result.entities[0], result.evidence[0]]:
        assert row["quote_origin"] == "reviewed_english_translation"
        assert row["audio_segments"][0]["audio_start"] == 1
    assert result.jd_entity_status == "complete"


def test_applicant_edits_disable_audio_links_and_combined_written_span_stays_written():
    from candidate_screener.multimodal.entity_bridge import reviewed_document
    document = reviewed_document([{"english": "Python", "french": "Python", "start": 1, "end": 2}],
                                 "Python teamwork", confirmed=True)
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": [document["text"]],
                        "source_kind": ["interview"], "translation_provenance": [document]})
    result = pipeline.screen("Python", cvs, ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)[0]
    assert all(not e["audio_segments"] for e in result.entities)
    combined = reviewed_document([{"english": "Python", "french": "Python", "start": 1, "end": 2}], confirmed=True)
    for segment in combined["segments"]:
        segment["english_start"] += 9
        segment["english_end"] += 9
    cvs.loc[0, "cv_text"] = "teamwork\nPython"
    cvs.loc[0, "source_kind"] = "combined"
    cvs.at[0, "translation_provenance"] = combined
    result = pipeline.screen("teamwork Python", cvs, ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)[0]
    origin = {e["surface"]: e["quote_origin"] for e in result.entities}
    assert origin == {"teamwork": "written_cv", "Python": "reviewed_english_translation"}


def test_shared_v2_builder_keeps_exact_bytes_and_never_reads_position():
    from candidate_screener.matching.corpus import build_cv_text
    from candidate_screener.extraction.role_family import cv_text_v2
    cv = pd.DataFrame({"Position": ["Secret"], "CV": ["  Body\n\n\n"], "Highlights": [None],
                       "Moreinfo": ["Details  "], "Looking For": [" More "]})
    assert cv_text_v2(cv).equals(build_cv_text(cv, "v2"))
    assert cv_text_v2(cv).iloc[0] == "Body\n\nDetails\n\nMore"


def test_translation_links_cannot_be_attached_to_different_text():
    from candidate_screener.multimodal.entity_bridge import reviewed_document
    import hashlib
    document = reviewed_document([{"english": "Python", "french": "Python", "start": 1, "end": 2}], confirmed=True)
    segment = document["segments"][0]
    segment.pop("english")
    segment["english_sha256"] = hashlib.sha256(b"Docker").hexdigest()
    cvs = pd.DataFrame({"cv_id": ["a"], "cv_text": ["Python"],
                        "source_kind": ["interview"], "translation_provenance": [document]})
    with pytest.raises(ValueError, match="alignment digest"):
        pipeline.screen("Python", cvs, ranker=Ranker(), matcher=Matcher(), entity_extractor=extracted)
