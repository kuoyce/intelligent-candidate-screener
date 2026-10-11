import pytest

from candidate_screener.multimodal.entity_bridge import (
    extract_reviewed_document, extract_translation_preview, reviewed_document, translated_segments,
)


def segments():
    return [{"english": "Python for 3 years", "french": "Python depuis 3 ans", "start": 0, "end": 2},
            {"english": "Python", "french": "Python", "start": 2, "end": 3}]


def span(start, end, text, label="KNOWLEDGE"):
    return {"start": start, "end": end, "surface": text[start:end], "label": label,
            "context_status": "unqualified", "confidence": .8, "method": "gliner-zero-shot"}


def test_confirmation_gate_rejects_implicit_and_truthy_values():
    for confirmation in (False, None, 1, "yes"):
        with pytest.raises(ValueError, match="confirmation"):
            reviewed_document(segments(), confirmed=confirmation)
    assert reviewed_document(segments(), confirmed=True)["status"] == "ready"


def test_segment_offsets_quote_translation_and_link_french_audio():
    document = reviewed_document(segments(), confirmed=True)
    for segment in document["segments"]:
        assert document["text"][segment["english_start"]:segment["english_end"]] == segment["english"]
    assert document["segments"][1]["audio_start"] == 2
    assert document["quote_origin"] == "reviewed_english_translation"
    assert document["source_language"] == "fr"


def test_edits_remove_automatic_offsets_but_keep_source_provenance():
    document = reviewed_document(segments(), "Four years of Python", confirmed=True)
    assert document["applicant_edited"] and not document["alignment_valid"]
    assert document["segments"] == []
    assert len(document["source_segments"]) == 2


def test_preview_never_ready_and_uses_cv_side():
    calls = []
    def extractor(text, side):
        calls.append(side)
        return [span(0, 6, text)]
    result = extract_translation_preview(segments(), extractor)
    assert calls == ["cv"]
    assert result["status"] == "review_required" and not result["screening_ready"]
    assert result["document"]["confirmed"] is False
    assert result["score"] is None and result["fit_call"] is None


def test_missing_zero_and_repetition_are_explicit():
    document = reviewed_document(segments(), "Python Python", confirmed=True)
    assert extract_reviewed_document(document, None)["status"] == "unavailable"
    zero = extract_reviewed_document(document, lambda text, side: [])
    assert zero["status"] == "completed_zero" and len(zero["counts"]) == 5
    result = extract_reviewed_document(document, lambda text, side: [span(0, 6, text), span(7, 13, text)])
    assert result["counts"]["KNOWLEDGE"] == 2
    assert result["counts"]["EXPERIENCE_YEARS"] == 0


def test_corrupt_spans_cannot_be_displayed():
    def extractor(text, side):
        entity = span(0, 6, text)
        entity["surface"] = "fabricated"
        return [entity]
    with pytest.raises(ValueError, match="slice"):
        extract_reviewed_document(reviewed_document(segments(), confirmed=True), extractor)


def test_direct_extraction_cannot_bypass_review():
    preview = extract_translation_preview(segments(), None)
    for document in (preview["document"], "Python", {"confirmed": True, "status": "review_required"}):
        with pytest.raises(ValueError, match="Confirmed ready"):
            extract_reviewed_document(document, None)


def test_cross_segment_span_has_separate_audio_links_and_edits_have_none():
    document = reviewed_document(segments(), confirmed=True)
    audit = extract_reviewed_document(document, lambda text, side: [span(0, len(text), text)])
    entity = audit["entities"][0]
    assert len(entity["audio_segments"]) == 2
    assert entity["quote_origin"] == "reviewed_english_translation"
    edited = reviewed_document(segments(), "Python", confirmed=True)
    assert extract_reviewed_document(edited, lambda text, side: [span(0, 6, text)])["entities"][0]["audio_segments"] == []


def test_chunk_count_mismatch_and_invalid_timestamps_fail():
    with pytest.raises(ValueError, match="counts"):
        translated_segments([{"french": "Bonjour", "start": 0, "end": 1}], [])
    invalid = segments()
    invalid[0]["end"] = float("nan")
    with pytest.raises(ValueError, match="timestamps"):
        reviewed_document(invalid, confirmed=True)
