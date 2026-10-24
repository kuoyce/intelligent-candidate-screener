"""Validated per-annotator entity gold exports; source text stays local."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from candidate_screener.config import PROJECT_ROOT


def build_exports(campaign: dict, records: list[dict], annotator: str) -> list[dict]:
    categories = set(campaign.get("labels", campaign.get("categories", [])))
    guideline = campaign.get("guideline_version", "entity-guide-v1")
    documents = {str(doc["doc_id"]): doc for doc in campaign["documents"]}
    if len(documents) != len(campaign["documents"]):
        raise ValueError("Duplicate campaign document identity")
    groups = {}
    for doc in documents.values():
        text = doc.get("source_text", doc.get("text"))
        normalized = " ".join(text.casefold().split())
        group = hashlib.sha256(normalized.encode()).hexdigest()
        if group in groups and groups[group] != doc["split"]:
            raise ValueError("Duplicate source text crosses entity splits")
        groups[group] = doc["split"]
    latest = {}
    for record in records:
        if record["annotator"] != annotator:
            continue
        if record.get("campaign_id") != campaign["campaign_id"]:
            raise ValueError("Annotation campaign mismatch")
        if record.get("guideline_version") != guideline:
            raise ValueError("Annotation guideline mismatch")
        if type(record.get("revision")) is not int or record["revision"] < 1:
            raise ValueError("Invalid annotation revision")
        if record.get("status") not in {"draft", "reviewed", "skipped", "uncertain", "in_progress"}:
            raise ValueError("Invalid annotation review status")
        doc_id = str(record["doc_id"])
        if doc_id not in documents:
            raise ValueError("Annotation document outside campaign")
        if "assignments" in campaign and doc_id not in campaign["assignments"].get(annotator, []):
            raise ValueError("Document not assigned to annotator")
        source = documents[doc_id].get("source_text", documents[doc_id].get("text"))
        if record.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("Annotation source digest mismatch")
        previous = latest.get(doc_id)
        if previous is None or record["revision"] > previous["revision"]:
            latest[doc_id] = record
        elif record["revision"] == previous["revision"] and record != previous:
            raise ValueError("Conflicting annotation revision")
    rows = []
    for doc_id, record in sorted(latest.items()):
        if record["status"] != "reviewed":
            continue
        if record.get("reviewed_all") is not True:
            raise ValueError("Reviewed gold requires whole-document review")
        doc = documents[doc_id]
        if doc["split"] not in {"train", "dev", "test"}:
            raise ValueError("Invalid entity split")
        text = doc.get("source_text", doc.get("text"))
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        if doc.get("source_sha256", source_hash) != source_hash or record.get("source_sha256") != source_hash:
            raise ValueError("Annotation source digest mismatch")
        if record.get("guideline_version", guideline) != guideline:
            raise ValueError("Annotation guideline mismatch")
        if type(record["revision"]) is not int or record["revision"] < 1:
            raise ValueError("Invalid annotation revision")
        if doc["split"] in {"train", "dev"} and doc.get("source_split") == "test" and doc.get("source", "").lower().startswith("a1"):
            raise ValueError("A1 test document cannot enter training/development export")
        assistance = record.get("assistance")
        mode = record.get("annotation_mode", "blind")
        if mode != ("model_assisted" if assistance else "blind"):
            raise ValueError("Annotation assistance provenance is inconsistent")
        if assistance:
            from .service import AnnotationService
            AnnotationService._validate_assistance_context(campaign, doc, annotator, assistance)
        spans, seen = [], set()
        for span in record["spans"]:
            start, end, label = span["start"], span["end"], span["label"]
            if (type(start) is not int or type(end) is not int or
                    not 0 <= start < end <= len(text) or label not in categories or
                    span["surface"] != text[start:end]):
                raise ValueError("Invalid original-source entity span")
            key = (start, end, label)
            if key in seen:
                raise ValueError("Duplicate entity span")
            seen.add(key)
            spans.append({key: span[key] for key in ("start", "end", "label", "surface")})
        ordered = sorted(spans, key=lambda span: span["start"])
        if any(left["end"] > right["start"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Gold ontology prohibits overlapping spans")
        rows.append({"campaign_id": campaign["campaign_id"], "doc_id": doc_id,
                     "doc_type": doc.get("doc_type", doc.get("side")), "split": doc["split"], "text": text,
                     "source_sha256": source_hash,
                     "annotator": annotator, "revision": record["revision"],
                     "guideline_version": guideline,
                     "categories": sorted(categories), "reviewed_all": True,
                     "annotation_mode": mode, "assistance": assistance,
                     "spans": sorted(spans, key=lambda s: (s["start"], s["end"], s["label"]))})
    return rows


def write_exports(rows: list[dict], destination: Path) -> dict:
    """Write a new immutable revision directory strictly under ignored data/."""
    destination = destination.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if not destination.is_relative_to(data_root) or destination == data_root:
        raise ValueError("Gold exports must remain under ignored data/")
    if not rows:
        raise ValueError("No fully reviewed documents to export; finish whole-document review first")
    if destination.exists():
        raise FileExistsError("Gold export revision already exists; choose a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gold-export-", dir=destination.parent))
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    try:
        (staging / "gold.jsonl").write_text(encoded, encoding="utf-8")
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), staging / "gold.parquet")
        manifest = {"documents": len(rows), "jsonl_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    "parquet_sha256": hashlib.sha256((staging / "gold.parquet").read_bytes()).hexdigest(),
                    "source_revisions": [{key: row[key] for key in ("campaign_id", "doc_id", "annotator", "revision", "source_sha256", "guideline_version")} for row in rows]}
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        # Serialize publication so cooperating exporters cannot replace a revision.
        import fcntl
        with (destination.parent / ".export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists():
                raise FileExistsError("Gold export revision already exists")
            os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    import argparse
    from .service import AnnotationService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    service = AnnotationService(args.campaign)
    rows = build_exports(service.campaign(), service.records(), args.annotator)
    try:
        result = write_exports(rows, args.destination)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"documents": result["documents"], "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
