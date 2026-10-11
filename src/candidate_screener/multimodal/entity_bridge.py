"""Preview translated interview entities without enabling applicant screening."""
from __future__ import annotations

import math

from candidate_screener.extraction.schema import LABELS, validate_entity


def translated_segments(linked_chunks: list[dict], translations: list[str]) -> list[dict]:
    """Connect the demo's translated chunks to their parent speech timestamps."""
    if len(linked_chunks) != len(translations):
        raise ValueError("Translation and source chunk counts differ")
    return [{"english": translated, "french": chunk["french"],
             "start": chunk["start"], "end": chunk["end"]}
            for chunk, translated in zip(linked_chunks, translations)]


def _document(segments: list[dict], reviewed_text: str | None, confirmed: bool) -> dict:
    if not segments:
        raise ValueError("Translated speech segments required")
    linked, parts, offset = [], [], 0
    for segment in segments:
        english = segment["english"]
        french = segment["french"]
        start, end = float(segment["start"]), float(segment["end"])
        if not isinstance(english, str) or not english.strip() or not isinstance(french, str) or not french.strip():
            raise ValueError("Nonempty English and French text required")
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("Invalid speech timestamps")
        linked.append({"english_start": offset, "english_end": offset + len(english),
                       "audio_start": start, "audio_end": end,
                       "french": french, "english": english})
        parts.append(english)
        offset += len(english) + 1
    original = "\n".join(parts)
    text = original if reviewed_text is None else reviewed_text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty reviewed English text required")
    aligned = text == original
    return {"status": "ready" if confirmed else "review_required", "confirmed": confirmed,
            "text": text, "source_kind": "translated_audio", "source_language": "fr",
            "text_language": "en", "quote_origin": "reviewed_english_translation" if confirmed
            else "unreviewed_english_translation", "alignment_valid": aligned,
            "segments": linked if aligned else [], "source_segments": linked,
            "applicant_edited": not aligned}


def reviewed_document(translated: list[dict], reviewed_text: str | None = None,
                      confirmed: bool = False) -> dict:
    """Require explicit applicant confirmation before returning a ready document.

    A caller must obtain confirmation; this function does not establish its identity.
    Edited text loses automatic timestamp alignment, retaining the source separately.
    """
    if confirmed is not True:
        raise ValueError("Applicant review confirmation required")
    return _document(translated, reviewed_text, True)


def _extract_entities(text: str, extractor) -> dict:
    """Extract validated five-label spans; extraction is neither ranking nor fit prediction."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty English text required")
    counts = {label: 0 for label in sorted(LABELS)}
    if extractor is None:
        return {"status": "unavailable", "counts": counts, "entities": [],
                "score": None, "fit_call": None}
    rows = extractor(text, "cv")
    if rows is None:
        raise ValueError("Extractor returned no completion result")
    entities = list(rows)
    for entity in entities:
        validate_entity(entity, text)
        counts[entity["label"]] += 1
    return {"status": "complete" if entities else "completed_zero", "counts": counts,
            "entities": entities, "score": None, "fit_call": None}


def _aligned_audit(document: dict, extractor) -> dict:
    audit = _extract_entities(document["text"], extractor)
    for entity_index, entity in enumerate(audit["entities"]):
        # Copy predictions: attaching provenance must not mutate an extractor cache.
        attached = dict(entity)
        attached["quote_origin"] = document["quote_origin"]
        attached["audio_segments"] = [
            {"audio_start": segment["audio_start"], "audio_end": segment["audio_end"],
             "english_start": segment["english_start"], "english_end": segment["english_end"]}
            for segment in document["segments"]
            if entity["start"] < segment["english_end"] and entity["end"] > segment["english_start"]
        ]
        audit["entities"][entity_index] = attached
    return audit


def extract_reviewed_document(document: dict, extractor) -> dict:
    """Extract only an explicitly reviewed ready document, never arbitrary text."""
    if not isinstance(document, dict) or document.get("confirmed") is not True or document.get("status") != "ready":
        raise ValueError("Confirmed ready applicant document required")
    return _aligned_audit(document, extractor)


def extract_translation_preview(translated: list[dict], extractor) -> dict:
    """Exploratory extraction before review; never emit a screening-ready document."""
    document = _document(translated, None, False)
    return {"document": document, "entity_audit": _aligned_audit(document, extractor),
            "status": "review_required", "screening_ready": False, "score": None, "fit_call": None}
