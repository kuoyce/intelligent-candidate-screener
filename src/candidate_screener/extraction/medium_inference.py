"""Sentence-aware medium extraction retaining exact original-source evidence."""
from __future__ import annotations

import hashlib
import json
import math
import time

from .rules import _context
from .schema import validate_entity
from .transformer import MODEL_LABELS

VERSION = "gliner-medium-sentence-v1"


def _covered(intervals, length):
    end = 0
    for start, stop in sorted(intervals):
        if start > end:
            return False
        end = max(end, stop)
    return end == length


def extract_document(doc, model, threshold=.5, overlap_tokens=32, deadline=None):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and bounded")
    if hasattr(model, "extract_document"):
        return model.extract_document(doc, threshold=threshold,
                                      overlap_tokens=overlap_tokens, deadline=deadline)
    from .sentence_chunking import plan_chunks
    records, coverage = [], {}
    for field, text in sorted(doc["fields"].items()):
        text = "" if text is None else text
        if not isinstance(text, str):
            raise ValueError("Document fields must contain strings or null")
        plan = plan_chunks(text, model, overlap_tokens=overlap_tokens, headroom_tokens=16)
        intervals, unique, invalid, interrupted = [], {}, 0, False
        for start, end in plan["windows"]:
            if deadline is not None and time.monotonic() >= deadline:
                interrupted = True
                break
            chunk = text[start:end]
            predictions = model.predict_entities(chunk, list(MODEL_LABELS), threshold=threshold,
                                                  flat_ner=False) if chunk.strip() else []
            intervals.append([start, end])
            for prediction in predictions:
                try:
                    local_start, local_end = prediction["start"], prediction["end"]
                    if (isinstance(local_start, bool) or isinstance(local_end, bool)
                            or not isinstance(local_start, int) or not isinstance(local_end, int)
                            or not 0 <= local_start < local_end <= len(chunk)):
                        raise ValueError("Invalid prediction offsets")
                    if prediction["text"] != chunk[local_start:local_end]:
                        raise ValueError("Prediction differs from source")
                    if isinstance(prediction["score"], bool):
                        raise ValueError("Boolean prediction score")
                    record = {"doc_id": str(doc["doc_id"]), "source": doc["source"],
                              "doc_type": doc["doc_type"], "field": field,
                              "text_version": doc.get("text_version", "raw-v1"),
                              "start": start + local_start, "end": start + local_end,
                              "surface": prediction["text"], "label": MODEL_LABELS[prediction["label"]],
                              "canonical_id": None, "method": "gliner-zero-shot",
                              "extractor_version": VERSION, "confidence": float(prediction["score"]),
                              "context_status": _context(text, start + local_start, start + local_end, field)}
                    validate_entity(record, text)
                    if record["confidence"] < threshold:
                        continue
                    key = (record["start"], record["end"], record["label"])
                    if key not in unique or record["confidence"] > unique[key]["confidence"]:
                        unique[key] = record
                except (KeyError, TypeError, ValueError, OverflowError):
                    invalid += 1
        for key in sorted(unique):
            record = unique[key]
            record["entity_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            records.append(record)
        quarantine = plan["quarantined_intervals"]
        coverage[field] = {**plan["diagnostics"], "intervals": intervals, "characters": len(text),
                           "chunks": len(intervals), "planned_chunks": len(plan["windows"]),
                           "complete": not quarantine and not interrupted and _covered(intervals, len(text)),
                           "quarantined_intervals": quarantine, "quarantined_predictions": invalid,
                           "interrupted": interrupted}
    return records, coverage
