"""Historical saved-span/source audit. Traceability never establishes NER accuracy."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from candidate_screener.annotation.redact import assert_clean, redact_text
from candidate_screener.config import OUTPUT, PROCESSED, RAW
from candidate_screener.extraction.inputs import PARTITION, audit_sources, select_training, sha256

RUNS = PROCESSED / "extraction/transformer-runs"


def _valid_intervals(field: dict) -> list[tuple[int, int]]:
    return [tuple(pair) for pair in field.get("intervals", []) if isinstance(pair, (list, tuple))
            and len(pair) == 2 and all(isinstance(value, int) and not isinstance(value, bool) for value in pair)]


def inventory_runs(root: Path = RUNS) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/metadata.json")):
        record = json.loads(path.read_text())
        rows.append({"run_id": record.get("run_id", path.parent.name),
                     "status": record.get("summary", {}).get("status", "unknown"),
                     "requested_documents": record.get("summary", {}).get("requested_documents"),
                     "completed_documents": record.get("summary", {}).get("completed_documents"),
                     "labels": ", ".join(sorted(record.get("provenance", {}).get("labels", {}).values())),
                     "threshold": record.get("provenance", {}).get("threshold")})
    return pd.DataFrame(rows)


def inventory_gold(raw_root: Path = RAW, output_root: Path = OUTPUT) -> dict:
    availability = audit_sources(raw_root)
    records = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*.json")
                     if path.name == "extraction-metrics.json" or any(term in path.name.lower() for term in ("ner", "skillspan", "dataturks", "gold")))
    return {"sources": {key: availability[key] for key in ("dataturks", "skillspan")},
            "existing_record_paths": records,
            "gliner_gold_accuracy": "unavailable: no matching frozen-GLiNER gold predictions scored here",
            "warning": "Other learned-tagger records do not measure this saved GLiNER configuration"}


def audit_entities(documents: list[dict], entities: pd.DataFrame, coverage: list[dict],
                   *, labels: set[str], threshold: float = .5) -> dict:
    failures = Counter()
    lookup = {(d["source"], d["doc_id"], field): (d["doc_type"], text)
              for d in documents for field, text in d["fields"].items()}
    if len({(d["source"], d["doc_id"]) for d in documents}) != len(documents):
        failures["duplicate_source_documents"] += 1
    key_columns = ["source", "doc_id", "field", "start", "end", "label"]
    required = set(key_columns + ["entity_id", "doc_type", "surface", "confidence"])
    if not required <= set(entities.columns):
        return {"passed": False, "failures": {"missing_entity_columns": 1}, "entities_checked": 0}
    failures["duplicate_entity_ids"] = int(entities.entity_id.duplicated().sum())
    failures["duplicate_span_label_keys"] = int(entities.duplicated(key_columns).sum())
    for e in entities.to_dict("records"):
        key = (e["source"], e["doc_id"], e["field"])
        if key not in lookup:
            failures["unknown_document_or_field"] += 1
            continue
        kind, text = lookup[key]
        if e["doc_type"] != kind:
            failures["wrong_document_type"] += 1
        start, end = e["start"], e["end"]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            failures["invalid_offsets"] += 1
        elif text[start:end] != e["surface"]:
            failures["surface_mismatch"] += 1
        if e["label"] not in labels:
            failures["label_outside_saved_prompt"] += 1
        score = e["confidence"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not threshold <= score <= 1:
            failures["invalid_model_score"] += 1
    seen, field_lookup = set(), {}
    for field in coverage:
        key = (field["source"], field["doc_id"], field["field"])
        if key in seen:
            failures["duplicate_field_coverage"] += 1
        seen.add(key)
        field_lookup[key] = field
        if key not in lookup:
            failures["unknown_field_coverage"] += 1
            continue
        end_covered = 0
        contiguous = True
        intervals = _valid_intervals(field)
        if len(intervals) != len(field.get("intervals", [])):
            failures["invalid_coverage_offsets"] += 1
            contiguous = False
        for start, end in intervals:
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(lookup[key][1]):
                failures["invalid_coverage_offsets"] += 1
                contiguous = False
                continue
            if start > end_covered:
                contiguous = False
            end_covered = max(end_covered, end)
        if not field.get("complete"):
            failures["incomplete_fields"] += 1
        if field.get("complete") and (not contiguous or end_covered != len(lookup[key][1])):
            failures["false_complete_coverage"] += 1
    failures["missing_field_coverage"] = len(set(lookup) - seen)
    for e in entities.to_dict("records"):
        field = field_lookup.get((e["source"], e["doc_id"], e["field"]))
        if not isinstance(e["start"], int) or not isinstance(e["end"], int):
            continue
        valid_intervals = _valid_intervals(field) if field else []
        if field and not any(start <= e["start"] < e["end"] <= end for start, end in valid_intervals):
            failures["entity_outside_submitted_coverage"] += 1
    failures = {name: count for name, count in sorted(failures.items()) if count}
    return {"passed": not failures, "failures": failures, "entities_checked": len(entities),
            "documents_checked": len(documents), "fields_checked": len(lookup),
            "scope": "Source traceability and operational integrity; not entity-label accuracy or missing-entity recall"}


def audit_saved_run(run_id: str | None = None, *, run_root: Path = RUNS,
                    raw_root: Path = RAW, partition_path: Path = PARTITION) -> dict:
    if run_id is None:
        run_id = json.loads((OUTPUT / "extraction/transformer-metrics.json").read_text())["run_id"]
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    folder = run_root / run_id
    metadata = json.loads((folder / "metadata.json").read_text())
    provenance = json.loads((folder / "sample-manifest.json").read_text())
    failures = Counter()
    if metadata.get("run_id") != run_id:
        failures["run_id_mismatch"] += 1
    required_artifacts = {"entities.parquet", "sample-manifest.json", "field-coverage.json"}
    if not required_artifacts <= set(metadata.get("artifact_sha256", {})):
        failures["missing_artifact_hashes"] += 1
    if metadata.get("provenance") != provenance:
        failures["manifest_metadata_mismatch"] += 1
    for name, digest in metadata.get("artifact_sha256", {}).items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or sha256(path) != digest:
            failures["artifact_hash_mismatch"] += 1
    if sha256(partition_path) != provenance["partition_sha256"]:
        failures["partition_hash_mismatch"] += 1
    partition = pd.read_csv(partition_path, dtype=str)
    documents = []
    for kind, fields in provenance["fields"].items():
        source = f"djinni-{kind}"
        frames = []
        names = [name for name in provenance["input_sha256"] if name.startswith(source + "/")]
        current_names = {f"{source}/{p.name}" for p in (raw_root / source).glob("*-train-*.parquet")}
        if set(names) != current_names:
            failures["source_file_set_mismatch"] += 1
        for name in names:
            path = raw_root / name
            if not path.resolve().is_relative_to(raw_root.resolve()) or not path.is_file():
                failures["missing_source_file"] += 1
                continue
            if sha256(path) != provenance["input_sha256"][name]:
                failures["source_hash_mismatch"] += 1
            frames.append(pq.read_table(path, columns=["id", *fields]).to_pandas())
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        ids = provenance["selected_ids"][kind]
        if len(ids) != len(set(ids)):
            failures["duplicate_selected_ids"] += 1
        try:
            sampled = select_training(frame, partition, kind, provenance["sample_size"], provenance["seed"])
            if sampled.id.tolist() != ids:
                failures["selected_ids_do_not_reproduce"] += 1
        except ValueError:
            failures["invalid_source_partition_membership"] += 1
        if frame.id.duplicated().any():
            failures["duplicate_raw_ids"] += 1
            continue
        by_id = frame.set_index("id")
        for doc_id in ids:
            if doc_id not in by_id.index:
                failures["missing_selected_document"] += 1
                continue
            row = by_id.loc[doc_id]
            documents.append({"doc_id": doc_id, "source": source, "doc_type": kind,
                              "fields": {field: row[field] if pd.notna(row[field]) else "" for field in fields}})
    entities = pd.read_parquet(folder / "entities.parquet")
    coverage = json.loads((folder / "field-coverage.json").read_text())
    labels = set(provenance["labels"].values())
    report = audit_entities(documents, entities, coverage, labels=labels, threshold=provenance["threshold"])
    failures.update(report["failures"])
    summary = metadata.get("summary", {})
    if summary.get("status") != "complete" or summary.get("completed_documents") != len(documents):
        failures["run_completion_mismatch"] += 1
    report.update({"run_id": run_id, "passed": not failures, "failures": dict(sorted(failures.items())),
                   "labels": sorted(labels), "threshold": provenance["threshold"],
                   "by_label": entities.label.value_counts().to_dict(),
                   "by_document_type": entities.doc_type.value_counts().to_dict(),
                   "historical_code_policy": "Recorded historical digests retained; current-code equality is not required"})
    return {"report": report, "documents": documents, "entities": entities,
            "field_coverage": coverage, "provenance": provenance}


def examples(audit: dict, label: str | None = None, doc_type: str | None = None,
             limit: int = 5, context_chars: int = 90) -> pd.DataFrame:
    if not 0 <= limit <= 5 or not 0 <= context_chars <= 120:
        raise ValueError("Examples are bounded to five rows and short source context")
    frame = audit["entities"]
    if label:
        frame = frame.loc[frame.label == label]
    if doc_type:
        frame = frame.loc[frame.doc_type == doc_type]
    lookup = {(d["source"], d["doc_id"], f): text for d in audit["documents"] for f, text in d["fields"].items()}
    rows = []
    for e in frame.sort_values(["doc_type", "label", "doc_id", "field", "start"]).head(limit).to_dict("records"):
        text = lookup.get((e["source"], e["doc_id"], e["field"]))
        if text is None or not 0 <= e["start"] < e["end"] <= len(text):
            continue
        context = text[max(0, e["start"] - context_chars // 2):min(len(text), e["end"] + context_chars // 2)]
        rows.append({"surface": redact_text(e["surface"])[:80], "label": e["label"], "score": e["confidence"],
                     "doc_type": e["doc_type"], "field": e["field"], "start": e["start"], "end": e["end"],
                     "context": redact_text(context)[:150]})
    result = pd.DataFrame(rows, columns=["surface", "label", "score", "doc_type", "field", "start", "end", "context"])
    assert_clean(result.surface, "Source validation entity display")
    assert_clean(result.context, "Source validation bounded context")
    return result


def comparison_examples(audit: dict, category: str = "shared", limit: int = 5) -> pd.DataFrame:
    """Current rules on historical fields: disagreement is inspection, not gold."""
    from candidate_screener.extraction import extract_document
    from candidate_screener.extraction.eda import exact_comparison
    from candidate_screener.extraction.run import ENTITY_COLUMNS
    if category not in {"shared", "neural_only", "rule_only"}:
        raise ValueError("Unknown comparison category")
    rules = pd.DataFrame([e for doc in audit["documents"] for e in extract_document(doc)], columns=ENTITY_COLUMNS)
    rules = rules.loc[rules.label.isin(audit["report"]["labels"])]
    frames = exact_comparison(audit["entities"], rules)
    return examples({**audit, "entities": frames[category]}, limit=limit)


def suspicious_examples(audit: dict, limit: int = 5) -> pd.DataFrame:
    """Deterministic inspection queue; heuristic flags never assert wrong labels."""
    frame = audit["entities"]
    candidates = frame.loc[frame.label.isin(["TITLE", "DEGREE"])].copy()
    candidates["inspection_priority"] = candidates.surface.str.contains(
        r"\b(?:experience|knowledge|skills|ability|level|background|team|degree)\b", case=False, regex=True)
    candidates = candidates.sort_values(["inspection_priority", "confidence"], ascending=[False, True])
    # Keep the selected heuristic queue before the deterministic display sort.
    return examples({**audit, "entities": candidates.head(limit)}, limit=limit)
