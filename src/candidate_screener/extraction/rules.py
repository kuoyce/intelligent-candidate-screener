"""Deterministic mention extraction, without proficiency or employment inference."""
import hashlib
import json
import re

from .schema import EXTRACTOR_VERSION, validate_entity
from .vocabulary import VOCABULARY


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    # Punctuation and contrast delimit bounded contexts; the 100-character bound
    # prevents a distant qualification from silently applying to a later mention.
    before = re.split(r"[;.!?\n]|\b(?:but|however)\b", text[:start], flags=re.I)[-1][-100:]
    after = re.split(r"[;.!?\n]|\b(?:but|however)\b", text[end:], flags=re.I)[0][:100]
    return before, after


def _context(text: str, start: int, end: int, field: str) -> str:
    before, after = _clause(text, start, end)
    if re.search(r"\b(?:no|without|lack(?:ing|s)?(?:\s+of)?|not)\s*$|\b(?:no|without|lack(?:ing|s)?(?:\s+of)?)\s+(?:experience|knowledge|skills?|proficiency|background)(?:\s+(?:in|with|of))?\s*$|\bnever\s+(?:used|worked\s+with)\s*$", before, re.I) or re.match(
        r"\s+(?:experience\s+)?(?:not\s+(?:required|used)|is\s+not\s+required)\b", after, re.I
    ):
        return "negated"
    if field.casefold() == "looking for" or re.search(
        r"\b(?:learn(?:ing)?|studying|aspir(?:e|ing)|want(?:ing)?\s+to|interested\s+in)\b", before, re.I
    ):
        return "learning"
    return "unqualified"


def _ambiguous_allowed(surface: str, text: str, start: int, end: int, field: str) -> bool:
    if surface.casefold() not in {"go", "r"}:
        return True
    if surface not in {"Go", "R"}:  # Case is meaningful for these two aliases.
        return False
    before, after = _clause(text, start, end)
    context = before + surface + after
    return bool(re.search(r"\b(?:programming|language|developer|development|code|coding|golang|statistics|statistical)\b", context, re.I))


def extract_document(doc: dict) -> list[dict]:
    """Extract exact mentions independently per immutable source field."""
    required = ("doc_id", "source", "doc_type", "fields")
    if any(key not in doc for key in required):
        raise ValueError("Document requires doc_id, source, doc_type and fields")
    result = []
    for field, text in sorted(doc["fields"].items()):
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError("Document fields must contain strings or null")
        candidates = []
        for label, entries in VOCABULARY.items():
            for canonical, aliases in entries.items():
                for alias in aliases:
                    pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                    for match in re.finditer(pattern, text, re.I):
                        if _ambiguous_allowed(match.group(), text, *match.span(), field):
                            candidates.append((match.start(), match.end(), label, canonical))
        # Prefer longest spans globally, then restore source order.
        accepted = []
        for candidate in sorted(candidates, key=lambda c: (-(c[1]-c[0]), c[0], c[2], c[3])):
            if not any(candidate[0] < other[1] and other[0] < candidate[1] for other in accepted):
                accepted.append(candidate)
        for start, end, label, canonical in sorted(accepted):
            record = {"doc_id": str(doc["doc_id"]), "source": doc["source"],
                      "doc_type": doc["doc_type"], "field": field,
                      "text_version": doc.get("text_version", "raw-v1"),
                      "start": start, "end": end, "label": label, "surface": text[start:end],
                      "canonical_id": f"seed:{canonical}", "method": "seed-rule",
                      "extractor_version": EXTRACTOR_VERSION, "confidence": None,
                      "context_status": _context(text, start, end, field)}
            record["entity_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            validate_entity(record, text)
            result.append(record)
    return result
