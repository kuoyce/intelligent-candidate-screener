"""Bounded mention previews: fixed rule aliases or explicitly synthetic documents.

Contact redaction cannot remove all personal names. Unreviewed neural spans from
real documents therefore never enter a saved notebook preview through this helper.
"""
from candidate_screener.annotation.redact import redact_text
from candidate_screener.extraction.vocabulary import VOCABULARY

PREVIEW_FIELDS = ("surface", "label", "field", "context_status", "doc_type")
_REVIEWED_ALIASES = {
    alias.casefold() for entries in VOCABULARY.values()
    for aliases in entries.values() for alias in aliases
}


def preview_entities(entities, *, limit=5):
    """Return contact-redacted, reviewed rows; suppress real unreviewed spans.

    Rule mentions must match the bounded seed vocabulary. Synthetic examples must
    carry source='synthetic' and a synthetic-prefixed doc_id. Other predictions
    remain in local parquet and need a separate deliberate local review.
    """
    if not isinstance(limit, int) or not 0 <= limit <= 5:
        raise ValueError("Saved entity previews are limited to five rows")
    rows = []
    for entity in entities:
        surface = entity.get("surface", "")
        synthetic = (entity.get("source") == "synthetic"
                     and str(entity.get("doc_id", "")).startswith("synthetic"))
        reviewed_rule = (entity.get("method") == "seed-rule"
                         and str(surface).casefold() in _REVIEWED_ALIASES)
        if not synthetic and not reviewed_rule:
            continue
        if limit == 0:
            break
        row = {key: entity.get(key) for key in PREVIEW_FIELDS}
        row["surface"] = redact_text(surface)[:100]
        rows.append(row)
        if len(rows) == limit:
            break
    return rows
