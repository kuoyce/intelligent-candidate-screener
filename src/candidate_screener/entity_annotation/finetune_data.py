"""Exact human-gold alignment and nonoverlapping, gold-safe training windows."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

from .export import build_exports, write_exports


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

VERSION = "entity-gold-safe-training-v1"
LABEL_PROMPTS = {
    "KNOWLEDGE": "technology, software tool or programming language",
    "SKILL": "professional skill or competency",
    "TITLE": "job title or occupation",
    "DEGREE": "academic degree or qualification",
}


def freeze_snapshot(campaign_path: Path, destination: Path, *, annotator="annotator1",
                    expected_documents=10) -> dict:
    from .service import AnnotationService
    service = AnnotationService(campaign_path)
    with service._locked():
        campaign = service.campaign()
        records = service.records()
        rows = build_exports(campaign, records, annotator)
        if len(rows) != expected_documents or any(r["split"] != "train" for r in rows):
            raise ValueError("Snapshot requires exactly ten latest fully reviewed training documents")
        if any(set(r["categories"]) != set(LABEL_PROMPTS) for r in rows):
            raise ValueError("Training snapshot categories differ from the approved four labels")
        result = write_exports(rows, destination)
    return result


def prepare_training(rows: list[dict], model, *, sentence_detector=None,
                     tokenizer_manifest: dict | None = None) -> dict:
    """Return records only when every gold span aligns and is supervised once.

    Windows partition GLiNER words without overlap: no partially visible positive
    can accidentally become a negative. Whitespace remains in source coverage
    intervals, though GLiNER's word representation naturally omits it.
    """
    from candidate_screener.extraction.sentence_chunking import chunk_parameters, sentence_intervals
    if not rows or len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("Unique nonempty human training documents are required")
    processor = model.data_processor
    if tokenizer_manifest is not None:
        from .character_tokenization import validate_manifest
        validate_manifest(tokenizer_manifest, rows=rows)
        if getattr(processor.words_splitter, "tokenizer_sha256", None) != tokenizer_manifest["tokenizer_sha256"]:
            raise ValueError("Model does not use the frozen training tokenizer")
        if model.config.max_width < tokenizer_manifest["max_width_units"]:
            raise ValueError("Model span width cannot represent the frozen training units")
    parameters = chunk_parameters(model)
    max_width = int(model.config.max_width)
    if max_width <= 0:
        raise ValueError("Model maximum span width must be positive")
    detector = sentence_detector or sentence_intervals
    diagnostics = {"version": VERSION, "documents": len(rows), "gold_spans": 0,
                   "unalignable_spans": 0, "width_exceeded_spans": 0,
                   "budget_exceeded_spans": 0, "unrepresentable_words": 0,
                   "examples": [], "token_limit": parameters["token_limit"],
                   "word_limit": parameters["word_limit"], "max_width": max_width}
    if tokenizer_manifest is not None:
        diagnostics["tokenizer_sha256"] = tokenizer_manifest["tokenizer_sha256"]
    records, windows = [], []

    def fits(words):
        if len(words) > parameters["word_limit"]:
            return False
        prepared, _ = processor.prepare_inputs([words], {label: i for i, label in enumerate(LABEL_PROMPTS.values(), 1)})
        encoded = processor.transformer_tokenizer(prepared, is_split_into_words=True,
                                                  truncation=False, add_special_tokens=True)
        return len(encoded["input_ids"][0]) <= parameters["token_limit"]

    def issue(kind, row, span):
        diagnostics[kind] += 1
        if len(diagnostics["examples"]) < 5:
            # Originals remain only in local snapshot; diagnostics may be shared.
            try:
                from candidate_screener.annotation.redact import redact_text
                surface = redact_text(row["text"][span["start"]:span["end"]])[:200]
            except ImportError:
                # The isolated neural runtime deliberately excludes pandas.
                # Withhold text entirely rather than invent another redactor.
                surface = "[source text withheld; inspect local snapshot]"
            diagnostics["examples"].append({"kind": kind, "doc_id": row["doc_id"],
                "start": span["start"], "end": span["end"],
                "surface": surface})

    for row in rows:
        if row.get("split") != "train" or row.get("reviewed_all") is not True:
            raise ValueError("Only exhaustively reviewed entity train documents can train")
        if set(row["categories"]) != set(LABEL_PROMPTS):
            raise ValueError("Training categories differ from approved four labels")
        text = row["text"]
        if digest(text) != row["source_sha256"]:
            raise ValueError("Training source digest mismatch")
        words = list(processor.words_splitter(text))
        if tokenizer_manifest is not None:
            cursor = 0
            for _, start, end in words:
                if start != cursor or not start < end <= len(text):
                    raise ValueError("Character tokenizer must partition the unchanged source")
                cursor = end
            if cursor != len(text):
                raise ValueError("Character tokenizer omitted original source content")
        starts = {w[1]: i for i, w in enumerate(words)}
        ends = {w[2]: i + 1 for i, w in enumerate(words)}
        aligned = []
        for span in row["spans"]:
            diagnostics["gold_spans"] += 1
            a, b = span["start"], span["end"]
            if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text) or
                    span["surface"] != text[a:b] or span["label"] not in LABEL_PROMPTS):
                raise ValueError("Invalid original gold span")
            if a not in starts or b not in ends:
                issue("unalignable_spans", row, span)
                continue
            start, stop = starts[a], ends[b]
            if stop - start > max_width:
                issue("width_exceeded_spans", row, span)
                continue
            if not fits([w[0] for w in words[start:stop]]):
                issue("budget_exceeded_spans", row, span)
                continue
            aligned.append((start, stop, span["label"]))
        if any(right[0] < left[1] for left, right in zip(sorted(aligned), sorted(aligned)[1:])):
            raise ValueError("Overlapping human gold spans are unsupported")
        safe = {i for i in range(1, len(words) + 1) if not any(a < i < b for a, b, _ in aligned)}
        sentence_ends = {b for _, b in detector(text)}
        preferred = {i for i in safe if i == len(words) or any(words[i - 1][2] <= end <= words[i][1] for end in sentence_ends)}
        index = 0
        while index < len(words):
            lo, hi, stop = index + 1, min(len(words), index + parameters["word_limit"]), index
            while lo <= hi:
                middle = (lo + hi) // 2
                if fits([w[0] for w in words[index:middle]]):
                    stop, lo = middle, middle + 1
                else:
                    hi = middle - 1
            choices = [i for i in safe if index < i <= stop]
            if not choices:
                diagnostics["unrepresentable_words"] += 1
                # Do not emit later potentially mislabeled regions from this doc.
                break
            sentence_choices = [i for i in choices if i in preferred]
            stop = max(sentence_choices or choices)
            targets = [[a - index, b - index - 1, LABEL_PROMPTS[label]]
                       for a, b, label in aligned if index <= a and b <= stop]
            records.append({"tokenized_text": [w[0] for w in words[index:stop]], "ner": targets})
            windows.append({"doc_id": row["doc_id"], "start": 0 if index == 0 else words[index][1],
                            "end": len(text) if stop == len(words) else words[stop][1],
                            "word_start": index, "word_end": stop, "gold_spans": len(targets)})
            index = stop
        if not words:
            windows.append({"doc_id": row["doc_id"], "start": 0, "end": len(text),
                            "word_start": 0, "word_end": 0, "gold_spans": 0})
    diagnostics["blocked"] = any(diagnostics[k] for k in ("unalignable_spans", "width_exceeded_spans", "budget_exceeded_spans", "unrepresentable_words"))
    diagnostics["window_count"] = len(records)
    diagnostics["supervised_spans"] = sum(len(r["ner"]) for r in records)
    if not diagnostics["blocked"] and diagnostics["supervised_spans"] != diagnostics["gold_spans"]:
        raise ValueError("Every gold span must be supervised exactly once")
    return {"records": [] if diagnostics["blocked"] else records,
            "windows": windows, "diagnostics": diagnostics}
