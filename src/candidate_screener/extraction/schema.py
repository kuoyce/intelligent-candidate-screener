"""Evidence validation shared by extraction and artifact consumers."""
EXTRACTOR_VERSION = "rules-v1"
LABELS = frozenset({"KNOWLEDGE", "SKILL", "TITLE", "DEGREE"})
CONTEXT_STATUSES = frozenset({"negated", "learning", "unqualified"})


def validate_entity(entity: dict, text: str) -> None:
    """Raise when evidence is not an exact half-open slice of original text."""
    start, end = entity["start"], entity["end"]
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError("Invalid field-relative entity offsets")
    if text[start:end] != entity["surface"]:
        raise ValueError("Entity surface differs from original field slice")
    if entity["label"] not in LABELS or entity["context_status"] not in CONTEXT_STATUSES:
        raise ValueError("Unsupported entity label or context")
    if entity["confidence"] is not None:
        raise ValueError("Rule confidence must be null")
