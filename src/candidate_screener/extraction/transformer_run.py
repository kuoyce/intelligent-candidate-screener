"""Bounded local transformer exploration, cached artifacts and provenance checks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import resource
from pathlib import Path
import time

import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.config import PROJECT_ROOT
from candidate_screener.extraction import extract_document
from candidate_screener.extraction.inputs import prepare_inputs, sha256
from candidate_screener.extraction.run import ENTITY_COLUMNS, _json, _validate

MODEL_DIR = PROCESSED / "extraction/models/gliner-small-v2.1"
ARTIFACT_ROOT = PROCESSED / "extraction/transformer-runs"


def _model_manifest(model_dir: Path) -> dict:
    path = model_dir / "model-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Pinned local transformer manifest missing; download the approved checkpoint first")
    manifest = json.loads(path.read_text())
    if not manifest.get("revision") or not manifest.get("artifact_sha256"):
        raise ValueError("Model manifest must record revision and artifact hashes")
    for name, digest in manifest["artifact_sha256"].items():
        target = model_dir / name
        if not target.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("Model manifest contains unsafe artifact path")
        if not target.is_file() or sha256(target) != digest:
            raise ValueError("Pinned model artifact digest mismatch")
    return {k: v for k, v in manifest.items() if k != "downloaded_at"}


def _dependencies() -> dict:
    result = {}
    for name in ("torch", "transformers", "gliner", "tokenizers", "intelligent-candidate-screener"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


def _runtime(model_dir: Path) -> dict:
    path = model_dir / "runtime-versions.json"
    return {"manifest": json.loads(path.read_text()), "manifest_sha256": sha256(path)} if path.exists() else {"versions": _dependencies()}


def _code() -> dict:
    folder = Path(__file__).parent
    result = {name: sha256(folder / name) for name in
              ("transformer_run.py", "transformer.py", "inputs.py", "schema.py", "rules.py", "vocabulary.py")}
    for name in ("transformer_worker.py", "runtime_proxy.py"):
        if (folder / name).exists():
            result[name] = sha256(folder / name)
    for name in ("pyproject.toml", "uv.lock"):
        path = PROJECT_ROOT / "runtime/transformer" / name
        if path.exists():
            result[f"runtime/transformer/{name}"] = sha256(path)
    return result


def _coverage_summary(entities: list[dict], documents: list[dict], field_records: list[dict]) -> dict:
    result = {}
    for kind in ("cv", "jd"):
        subset = [e for e in entities if e["doc_type"] == kind]
        docs = [d for d in documents if d["doc_type"] == kind]
        fields = [f for f in field_records if f["doc_type"] == kind]
        scores = sorted(e["confidence"] for e in subset)
        result[kind] = {"requested_documents": len(docs),
                       "completed_documents": sum(all(f["complete"] for f in fields if f["doc_id"] == d["doc_id"])
                                                  and any(f["doc_id"] == d["doc_id"] for f in fields) for d in docs),
                       "documents_with_entities": len({e["doc_id"] for e in subset}),
                       "entities": len(subset), "by_label": dict(sorted(Counter(e["label"] for e in subset).items())),
                       "documents_with_label": {label: len({e["doc_id"] for e in subset if e["label"] == label})
                                                for label in ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE")},
                       "fields_requested": sum(len(d["fields"]) for d in docs),
                       "fields_attempted": len(fields), "fields_complete": sum(f["complete"] for f in fields),
                       "fields_incomplete": sum(not f["complete"] for f in fields),
                       "quarantined_predictions": sum(f.get("quarantined_predictions", 0) for f in fields),
                       "raw_uncalibrated_scores": {"min": scores[0] if scores else None,
                           "max": scores[-1] if scores else None,
                           "mean": sum(scores) / len(scores) if scores else None}}
    return result


def _comparison(entities: list[dict], documents: list[dict], completed_ids: set[tuple]) -> dict:
    def keys(records):
        return {(e["source"], e["doc_id"], e["field"], e["start"], e["end"], e["label"]) for e in records}
    docs = [d for d in documents if (d["source"], d["doc_id"]) in completed_ids]
    rules = [e for doc in docs for e in extract_document(doc)]
    neural = [e for e in entities if (e["source"], e["doc_id"]) in completed_ids]
    result = {"interpretation": "Exact span-label disagreement, not accuracy", "completed_documents_compared": len(docs)}
    for kind in ("cv", "jd"):
        r = keys([e for e in rules if e["doc_type"] == kind])
        n = keys([e for e in neural if e["doc_type"] == kind])
        result[kind] = {"rule_entities": len(r), "transformer_entities": len(n),
                        "exact_shared": len(r & n), "rule_only": len(r - n), "transformer_only": len(n - r)}
    return result


def run_transformer(sample_size: int = 100, seed: int = 0, threshold: float = .5,
                    max_seconds: float = 1800, *, model_dir: Path = MODEL_DIR,
                    artifact_root: Path = ARTIFACT_ROOT, device: str = "cpu",
                    threads: int = 4, overlap_tokens: int = 32) -> dict:
    from candidate_screener.extraction.transformer import (CONTEXT_VERSION, MODEL_LABELS,
        TRANSFORMER_VERSION, chunk_parameters, extract_transformer_document, load_model)
    if not math.isfinite(threshold) or not 0 < threshold < 1 or not math.isfinite(max_seconds) or max_seconds <= 0 or threads < 1 or overlap_tokens < 0:
        raise ValueError("Invalid threshold, time budget, threads or overlap")
    payload = prepare_inputs(sample_size, seed)
    manifest = _model_manifest(model_dir)
    provenance = {**payload["provenance"], "model": manifest,
                  "model_directory": str(model_dir.resolve()), "dependencies": _runtime(model_dir),
                  "code_sha256": _code(), "transformer_version": TRANSFORMER_VERSION,
                  "context_version": CONTEXT_VERSION, "labels": MODEL_LABELS,
                  "threshold": threshold, "overlap_tokens": overlap_tokens, "device": device,
                  "threads": threads, "max_seconds": max_seconds, "text_version": "raw-v1",
                  "score_policy": "Raw uncalibrated model score; no probability-of-correctness claim"}
    identity = hashlib.sha256(_json(provenance).encode()).hexdigest()[:20]
    for path in sorted(artifact_root.glob(f"{identity}-*/metadata.json"), reverse=True):
        result = {**json.loads(path.read_text()), "run_dir": str(path.parent)}
        if result["provenance"] != provenance:
            raise ValueError("Cached transformer run provenance differs")
        validate_transformer_run(result, payload=payload)
        return result
    if threads != 4:
        raise ValueError("The transformer runtime currently uses exactly four CPU threads")
    started = time.monotonic()
    deadline = started + max_seconds
    model = load_model(model_dir, device=device)
    actual_runtime_versions = getattr(model, "runtime_versions", _dependencies())
    expected_runtime_versions = provenance["dependencies"].get("manifest", provenance["dependencies"].get("versions", {}))
    if any(actual_runtime_versions.get(name) != expected for name, expected in expected_runtime_versions.items()):
        if callable(getattr(model, "close", None)):
            model.close()
        raise ValueError("Live transformer runtime differs from recorded dependency versions")
    actual_chunk_parameters = chunk_parameters(model)
    entities, fields, completed = [], [], set()
    for doc in payload["documents"]:
        if time.monotonic() >= deadline:
            break
        records, coverage = extract_transformer_document(doc, model, threshold=threshold,
                                  overlap_tokens=overlap_tokens, deadline=deadline)
        entities.extend(records)
        for field, info in coverage.items():
            fields.append({"doc_id": doc["doc_id"], "source": doc["source"], "doc_type": doc["doc_type"],
                           "field": field, **info})
        if set(coverage) == set(doc["fields"]) and all(info["complete"] for info in coverage.values()):
            completed.add((doc["source"], doc["doc_id"]))
    if callable(getattr(model, "close", None)):
        model.close()
    runtime = time.monotonic() - started
    integrity = _validate(entities, payload["documents"])
    status = "complete" if len(completed) == len(payload["documents"]) else "partial"
    summary = {"status": status, "runtime_seconds": runtime, "time_budget_seconds": max_seconds,
               "actual_chunk_parameters": actual_chunk_parameters,
               "actual_runtime_versions": actual_runtime_versions,
               "memory_observation": {"process_peak_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                      "units": "bytes on macOS; KiB on Linux", "scope": "parent process only; excludes isolated inference worker"},
               "requested_documents": len(payload["documents"]), "completed_documents": len(completed),
               "coverage": _coverage_summary(entities, payload["documents"], fields),
               "rule_comparison": _comparison(entities, payload["documents"], completed),
               "integrity": integrity, "gold_accuracy": {"status": "blocked", "reason": "No validated local NER gold adapters"}}
    run_id = identity + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(entities, columns=ENTITY_COLUMNS).to_parquet(run_dir / "entities.parquet", index=False)
    (run_dir / "field-coverage.json").write_text(_json(fields))
    (run_dir / "sample-manifest.json").write_text(_json(provenance))
    metadata = {"run_id": run_id, "provenance": provenance, "summary": summary,
                "availability": payload["availability"],
                "artifact_sha256": {name: sha256(run_dir / name) for name in
                    ("entities.parquet", "sample-manifest.json", "field-coverage.json")}}
    (run_dir / "metadata.json").write_text(_json(metadata))
    return {**metadata, "run_dir": str(run_dir)}


def validate_transformer_run(result: dict, *, payload: dict | None = None) -> dict:
    run_dir = Path(result["run_dir"])
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata != {k: v for k, v in result.items() if k != "run_dir"}:
        raise ValueError("Transformer metadata changed")
    provenance = result["provenance"]
    if json.loads((run_dir / "sample-manifest.json").read_text()) != provenance:
        raise ValueError("Transformer sample manifest changed")
    if any(sha256(run_dir / name) != digest for name, digest in result["artifact_sha256"].items()):
        raise ValueError("Transformer artifact digest mismatch")
    if provenance["code_sha256"] != _code() or provenance["dependencies"] != _runtime(Path(provenance["model_directory"])):
        raise ValueError("Transformer implementation/dependencies changed")
    if _model_manifest(Path(provenance["model_directory"])) != provenance["model"]:
        raise ValueError("Transformer checkpoint provenance changed")
    payload = payload or prepare_inputs(provenance["sample_size"], provenance["seed"])
    if any(payload["provenance"][key] != provenance[key] for key in payload["provenance"]):
        raise ValueError("Transformer source provenance changed")
    entities = pd.read_parquet(run_dir / "entities.parquet").to_dict("records")
    fields = json.loads((run_dir / "field-coverage.json").read_text())
    lookup = {(d["source"], d["doc_id"], f): text for d in payload["documents"] for f, text in d["fields"].items()}
    field_keys = [(f["source"], f["doc_id"], f["field"]) for f in fields]
    if len(set(field_keys)) != len(field_keys):
        raise ValueError("Duplicate transformer field coverage")
    for record in fields:
        key = (record["source"], record["doc_id"], record["field"])
        if key not in lookup:
            raise ValueError("Unknown transformer field coverage")
        intervals = record["intervals"]
        previous_end = 0
        for start, end in intervals:
            if not 0 <= start <= end <= len(lookup[key]):
                raise ValueError("Invalid transformer coverage intervals")
            if record["complete"] and start > previous_end:
                raise ValueError("Complete transformer coverage contains a gap")
            previous_end = max(previous_end, end)
        if record["complete"] and previous_end != len(lookup[key]):
            raise ValueError("Transformer field marked complete with uncovered tail")
    if _coverage_summary(entities, payload["documents"], fields) != result["summary"]["coverage"]:
        raise ValueError("Transformer aggregate coverage differs from saved evidence")
    field_lookup = {key: field for key, field in zip(field_keys, fields)}
    for entity in entities:
        key = (entity["source"], entity["doc_id"], entity["field"])
        if key not in field_lookup or not any(start <= entity["start"] < entity["end"] <= end
                                             for start, end in field_lookup[key]["intervals"]):
            raise ValueError("Transformer evidence lies outside submitted coverage")
    return {**_validate(entities, payload["documents"]), "artifact_hashes": True,
            "checkpoint_hashes": True, "field_coverage_valid": True,
            "status": result["summary"]["status"], "score_replay": "Not required; raw scores can vary across devices"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run_transformer(args.sample_size, args.seed, max_seconds=args.max_seconds, threads=args.threads)
    print(_json(validate_transformer_run(result) if args.check else {"run_id": result["run_id"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
