"""Separate medium-v2.5 sentence-aware campaign over original fit documents."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import time
import pandas as pd
from candidate_screener.config import PROCESSED
from candidate_screener.extraction.inputs import sha256
from candidate_screener.extraction.run import ENTITY_COLUMNS, _json
from candidate_screener.extraction.transformer_run import _model_manifest, _runtime, _code
from candidate_screener.extraction.medium_assets import MODEL_DIR
from candidate_screener.extraction.validation import audit_entities

ROOT = PROCESSED / "extraction/medium-sentence-campaigns"
LABELS = ["KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS"]


def documents_from_pairs(pairs: pd.DataFrame) -> list[dict]:
    documents = []
    for kind, id_column, text_column in (("cv", "resume_id", "resume_text"), ("jd", "jd_id", "job_description_text")):
        values = {}
        for row in pairs[[id_column, text_column]].to_dict("records"):
            identity, text = row[id_column], row[text_column]
            if not isinstance(identity, str) or not identity or not isinstance(text, str):
                raise ValueError("Fit source IDs/text must be non-null strings")
            if identity in values and values[identity] != text:
                raise ValueError("Conflicting texts for a fit document identity")
            values[identity] = text
        for identity, text in sorted(values.items()):
            documents.append({"doc_id": identity, "source": f"fit-{kind}", "doc_type": kind, "fields": {"text": text}})
    return documents


def inventory(source_root: Path = PROCESSED / "fit") -> dict:
    frames, hashes = [], {}
    for split in ("train", "test"):
        path = source_root / f"{split}.parquet"
        frame = pd.read_parquet(path)
        required = {"resume_id", "jd_id", "resume_text", "job_description_text", "label"}
        if not required <= set(frame):
            raise ValueError("Fit parquet lacks required source columns")
        frame = frame.assign(split=split)
        if frame.duplicated(["resume_id", "jd_id"]).any():
            raise ValueError("Duplicate fit pair identities within split")
        frames.append(frame)
        hashes[split] = sha256(path)
    pairs = pd.concat(frames, ignore_index=True)
    documents = documents_from_pairs(pairs)
    summary = {split: {"pairs": int((pairs.split == split).sum()),
                       "cvs": pairs.loc[pairs.split == split, "resume_id"].nunique(),
                       "jds": pairs.loc[pairs.split == split, "jd_id"].nunique()} for split in ("train", "test")}
    summary["total"] = {"pairs": len(pairs), "unique_documents": len(documents),
                         "characters": sum(len(d["fields"]["text"]) for d in documents)}
    return {"pairs": pairs, "documents": documents, "summary": summary,
            "provenance": {"source_root": str(source_root.resolve()), "source_sha256": hashes,
                "document_ids": {kind: [d["doc_id"] for d in documents if d["doc_type"] == kind] for kind in ("cv", "jd")}}}


def _identity(payload: dict, model_dir: Path, budget: float) -> dict:
    from .transformer import MODEL_LABELS, CONTEXT_VERSION
    from .medium_inference import VERSION
    from .sentence_chunking import STRATEGY_VERSION
    if not (model_dir / "runtime-versions.json").is_file():
        raise FileNotFoundError("Medium runtime manifest is required before campaign initialization")
    runtime_record = _runtime(model_dir)
    required_runtime = {"torch", "transformers", "gliner", "spacy", "tokenizers", "numpy",
                        "huggingface-hub", "sentencepiece", "protobuf", "click"}
    versions = runtime_record.get("manifest", {})
    if not required_runtime <= set(versions) or any(not isinstance(value, str) or not value
                                                  for value in versions.values()):
        raise ValueError("Medium runtime manifest must contain all pinned package versions")
    from .medium_assets import verify_assets
    verify_assets(model_dir)
    folder = Path(__file__).parent
    code = {**_code(), **{name: sha256(folder / name) for name in
            ("medium_run.py", "medium_inference.py", "medium_model.py", "medium_runtime.py",
             "medium_worker.py", "sentence_chunking.py", "medium_assets.py")}}
    runtime = folder.parents[2] / "runtime/gliner-medium"
    for name in ("pyproject.toml", "uv.lock"):
        if (runtime / name).exists():
            code[f"runtime/gliner-medium/{name}"] = sha256(runtime / name)
    return {**payload["provenance"], "model": _model_manifest(model_dir), "model_dir": str(model_dir.resolve()),
            "runtime": runtime_record, "code_sha256": code,
            "labels": MODEL_LABELS, "threshold": .5, "overlap_tokens": 32, "device": "cpu", "threads": 4,
            "per_document_seconds": budget, "transformer_version": VERSION, "context_version": CONTEXT_VERSION,
            "chunk_strategy": STRATEGY_VERSION, "headroom_tokens": 16,
            "profile": "gliner-medium-v2.5-sentence"}


def _atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value))
    temporary.replace(path)


def _checkpoint_name(doc: dict) -> str:
    return hashlib.sha256((doc["source"] + "\0" + doc["doc_id"]).encode()).hexdigest() + ".json"


def _load_checkpoint(path: Path, doc: dict, identity: str) -> dict:
    checkpoint = json.loads(path.read_text())
    if checkpoint["campaign_id"] != identity or checkpoint["source"] != doc["source"] or checkpoint["doc_id"] != doc["doc_id"]:
        raise ValueError("Checkpoint identity mismatch")
    digest = hashlib.sha256(_json({k: v for k, v in checkpoint.items() if k != "sha256"}).encode()).hexdigest()
    if digest != checkpoint["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    status, entities = checkpoint["status"], checkpoint["entities"]
    if status not in {"complete", "complete_zero", "incomplete", "failed"}:
        raise ValueError("Unknown checkpoint state")
    if (status == "complete_zero" and entities) or (status == "complete" and not entities) or (status == "failed" and entities):
        raise ValueError("Checkpoint state and entity count disagree")
    if checkpoint["status"] in {"complete", "complete_zero", "incomplete"}:
        report = audit_entities([doc], pd.DataFrame(checkpoint["entities"], columns=ENTITY_COLUMNS),
            [{"source": doc["source"], "doc_id": doc["doc_id"], "field": f, **info} for f, info in checkpoint["coverage"].items()],
            labels=set(LABELS))
        failures = set(report["failures"]) - ({"incomplete_fields", "missing_field_coverage"} if checkpoint["status"] == "incomplete" else set())
        if failures:
            raise ValueError("Checkpoint evidence integrity failure: " + ",".join(sorted(failures)))
    return checkpoint


def run_campaign(limit: int | None = None, *, per_document_seconds: float = 120,
                 source_root: Path = PROCESSED / "fit", campaign_root: Path = ROOT,
                 model_dir: Path = MODEL_DIR) -> dict:
    import math
    if not math.isfinite(per_document_seconds) or per_document_seconds <= 0 or (limit is not None and limit < 0):
        raise ValueError("Invalid document budget or pilot limit")
    payload = inventory(source_root)
    provenance = _identity(payload, model_dir, per_document_seconds)
    campaign_id = hashlib.sha256(_json(provenance).encode()).hexdigest()[:24]
    folder = campaign_root / campaign_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign_id": campaign_id, "provenance": provenance, "inventory": payload["summary"]}
    if (folder / "manifest.json").exists() and json.loads((folder / "manifest.json").read_text()) != manifest:
        raise ValueError("Incompatible existing campaign")
    _atomic(folder / "manifest.json", manifest)
    checkpoints = folder / "documents"
    checkpoints.mkdir(exist_ok=True)
    pending = []
    for doc in payload["documents"]:
        path = checkpoints / _checkpoint_name(doc)
        if path.exists():
            _load_checkpoint(path, doc, campaign_id)
        else:
            pending.append(doc)
    if limit is not None:
        # Pilot covers both sides rather than only the lexically first CVs.
        cvs = [d for d in pending if d["doc_type"] == "cv"]
        jds = [d for d in pending if d["doc_type"] == "jd"]
        interleaved = [d for i in range(max(len(cvs), len(jds))) for group in (cvs, jds) for d in group[i:i+1]]
        pending = interleaved[:limit]
    if pending:
        from .medium_runtime import MediumRuntimeProxy
        from .medium_inference import extract_document
        model = MediumRuntimeProxy(model_dir, device="cpu")
        actual = getattr(model, "runtime_versions", {})
        if not actual:
            import torch
            from importlib.metadata import version
            torch.set_num_threads(4)
            actual = {name: version(name) for name in provenance["runtime"].get("manifest", {})}
        expected = provenance["runtime"].get("manifest", {})
        if any(actual.get(name) != value for name, value in expected.items()):
            if hasattr(model, "close"):
                model.close()
            raise ValueError("Live runtime versions mismatch")
        try:
            for index, doc in enumerate(pending):
                started = time.monotonic()
                record = {"campaign_id": campaign_id, "source": doc["source"], "doc_id": doc["doc_id"],
                          "doc_type": doc["doc_type"], "entities": [], "coverage": {}, "error": None}
                try:
                    entities, coverage = extract_document(doc, model, threshold=.5, overlap_tokens=32,
                                                                       deadline=started + per_document_seconds)
                    complete = set(coverage) == set(doc["fields"]) and all(info["complete"] for info in coverage.values())
                    record.update(entities=entities, coverage=coverage,
                                  status=("complete" if entities else "complete_zero") if complete else "incomplete")
                except Exception as exc:
                    record.update(status="failed", error=type(exc).__name__)
                record.update(runtime_seconds=time.monotonic() - started, actual_runtime_versions=actual,
                              chunk_parameters=model.chunk_parameters())
                record["sha256"] = hashlib.sha256(_json(record).encode()).hexdigest()
                # Validate before publishing a checkpoint, never leave invalid evidence resumable.
                staging = checkpoints / (_checkpoint_name(doc) + ".verify")
                _atomic(staging, record)
                _load_checkpoint(staging, doc, campaign_id)
                staging.unlink()
                _atomic(checkpoints / _checkpoint_name(doc), record)
                print(json.dumps({"processed_this_invocation": index + 1, "status": record["status"],
                                  "doc_type": doc["doc_type"], "entities": len(record["entities"])}), flush=True)
        finally:
            if hasattr(model, "close"):
                model.close()
    result = load_campaign(campaign_id, campaign_root=campaign_root)
    if not result["summary"]["states"].get("pending", 0):
        _atomic(campaign_root / "latest.json", {"campaign_id": campaign_id})
    return {"campaign_id": campaign_id, "campaign_dir": str(folder), "summary": result["summary"]}


def load_campaign(campaign_id: str | None = None, *, campaign_root: Path = ROOT) -> dict:
    if not campaign_id:
        pointer = campaign_root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError("No finished medium campaign yet. Pass an explicit campaign ID to inspect partial results.")
        campaign_id = json.loads(pointer.read_text())["campaign_id"]
    if not re.fullmatch(r"[a-f0-9]{24}", campaign_id):
        raise ValueError("Invalid campaign identifier")
    folder = campaign_root / campaign_id
    metadata = json.loads((folder / "manifest.json").read_text())
    if metadata["campaign_id"] != campaign_id or hashlib.sha256(_json(metadata["provenance"]).encode()).hexdigest()[:24] != campaign_id:
        raise ValueError("Campaign manifest identity mismatch")
    payload = inventory(Path(metadata["provenance"]["source_root"]))
    for key, value in payload["provenance"].items():
        if metadata["provenance"][key] != value:
            raise ValueError("Campaign sources changed")
    entities, statuses = [], []
    for doc in payload["documents"]:
        path = folder / "documents" / _checkpoint_name(doc)
        record = _load_checkpoint(path, doc, campaign_id) if path.exists() else {"status": "pending", "entities": [], "runtime_seconds": None}
        entities.extend(record["entities"])
        statuses.append({"source": doc["source"], "doc_id": doc["doc_id"], "doc_type": doc["doc_type"],
                         "status": record["status"], "entities": len(record["entities"]) if record["status"] not in {"pending", "failed"} else None,
                         "runtime_seconds": record["runtime_seconds"],
                         **{key: sum(info.get(key, 0) for info in record.get("coverage", {}).values())
                            if record["status"] not in {"pending", "failed"} else None for key in
                            ("chunks", "fallback_windows", "sentence_count")},
                         "quarantined_spans": sum(len(info.get("quarantined_intervals", []))
                                                   for info in record.get("coverage", {}).values())
                            if record["status"] not in {"pending", "failed"} else None})
    return {**payload, "metadata": metadata, "entities": pd.DataFrame(entities, columns=ENTITY_COLUMNS),
            "statuses": pd.DataFrame(statuses), "summary": {"requested_documents": len(statuses),
                "states": dict(Counter(s["status"] for s in statuses)), "entities": len(entities),
                "runtime_seconds": sum(s["runtime_seconds"] or 0 for s in statuses)}}


def report_tables(campaign: dict) -> dict[str, pd.DataFrame]:
    entities, statuses, pairs = campaign["entities"], campaign["statuses"], campaign["pairs"]
    counts, coverage, docs, scores, pair_rows, fit_rows, states = [], [], [], [], [], [], []
    for split in ("train", "test"):
        split_pairs = pairs.loc[pairs.split == split]
        cv_completed = set(statuses.loc[(statuses.doc_type == "cv") & statuses.status.isin(["complete", "complete_zero"]), "doc_id"])
        jd_completed = set(statuses.loc[(statuses.doc_type == "jd") & statuses.status.isin(["complete", "complete_zero"]), "doc_id"])
        for kind, id_column in (("cv", "resume_id"), ("jd", "jd_id")):
            ids = set(split_pairs[id_column])
            state = statuses.loc[(statuses.doc_type == kind) & statuses.doc_id.isin(ids)].copy()
            completed_ids = set(state.loc[state.status.isin(["complete", "complete_zero"]), "doc_id"])
            subset = entities.loc[(entities.doc_type == kind) & entities.doc_id.isin(completed_ids)]
            state["split"] = split
            docs.append(state)
            for status in ("complete", "complete_zero", "incomplete", "failed", "pending"):
                states.append({"split": split, "doc_type": kind, "status": status, "documents": int((state.status == status).sum())})
            for label in LABELS:
                rows = subset.loc[subset.label == label]
                covered = rows.doc_id.nunique()
                counts.append({"split": split, "doc_type": kind, "label": label, "entities": len(rows),
                               "unique_mentions": rows.surface.str.casefold().nunique()})
                coverage.append({"split": split, "doc_type": kind, "label": label, "covered_documents": covered,
                                 "requested_documents": len(ids), "completed_documents": len(completed_ids),
                                 "operational_coverage": covered / len(ids) if ids else None})
                scores.append({"split": split, "doc_type": kind, "label": label, "entities": len(rows),
                               "min": rows.confidence.min() if len(rows) else None, "max": rows.confidence.max() if len(rows) else None,
                               "mean": rows.confidence.mean() if len(rows) else None})
            by_doc = subset.groupby("doc_id").size().to_dict()
            for label, group in split_pairs.groupby("label"):
                eligible = group.loc[group.resume_id.isin(cv_completed) & group.jd_id.isin(jd_completed)]
                fit_rows.append({"split": split, "doc_type": kind, "fit_label": label, "pairs": len(group),
                                 "completed_document_pairs": len(eligible),
                                 "mean_entities_per_pair_document": sum(by_doc.get(d, 0) for d in eligible[id_column]) / len(eligible) if len(eligible) else None,
                                 "limitation": "A1 labels do not reproduce (A13); descriptive only"})
        cv_completed = set(statuses.loc[(statuses.doc_type == "cv") & statuses.status.isin(["complete", "complete_zero"]), "doc_id"])
        jd_completed = set(statuses.loc[(statuses.doc_type == "jd") & statuses.status.isin(["complete", "complete_zero"]), "doc_id"])
        pair_rows.append({"split": split, "pairs": len(split_pairs),
                          "both_documents_complete": int((split_pairs.resume_id.isin(cv_completed) & split_pairs.jd_id.isin(jd_completed)).sum())})
    return {"counts": pd.DataFrame(counts), "label_coverage": pd.DataFrame(coverage), "document_counts": pd.concat(docs, ignore_index=True),
            "score_summary": pd.DataFrame(scores), "pair_counts": pd.DataFrame(pair_rows),
            "fit_conditioned": pd.DataFrame(fit_rows), "states": pd.DataFrame(states)}


def examples(campaign: dict, side: str, doc_id: str, limit: int = 5) -> pd.DataFrame:
    from .validation import examples as source_examples
    frame = campaign["entities"].loc[(campaign["entities"].doc_type == side) & (campaign["entities"].doc_id == doc_id)]
    return source_examples({"entities": frame, "documents": campaign["documents"]}, limit=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--campaign-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=PROCESSED / "fit")
    args = parser.parse_args()
    print(_json(run_campaign(args.limit, model_dir=args.model_dir,
                             campaign_root=args.campaign_root, source_root=args.source_root)))


if __name__ == "__main__":
    main()
