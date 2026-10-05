"""Reproducible local extraction artifacts and aggregate-only reporting CLI."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.extraction import EXTRACTOR_VERSION, VOCABULARY_VERSION, extract_document, validate_entity, vocabulary_digest
from candidate_screener.extraction.inputs import prepare_inputs, sha256

ENTITY_COLUMNS = ["entity_id", "doc_id", "source", "doc_type", "field", "text_version",
                  "start", "end", "label", "surface", "canonical_id", "method",
                  "extractor_version", "confidence", "context_status"]


def code_digests() -> dict:
    folder = Path(__file__).parent
    return {name: sha256(folder / name) for name in
            ("inputs.py", "run.py", "rules.py", "schema.py", "vocabulary.py")}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _validate(entities: list[dict], documents: list[dict]) -> dict:
    lookup = {(doc["source"], doc["doc_id"], field): text
              for doc in documents for field, text in doc["fields"].items()}
    ids = [entity["entity_id"] for entity in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entity IDs")
    for entity in entities:
        key = (entity["source"], entity["doc_id"], entity["field"])
        if key not in lookup:
            raise ValueError("Entity refers to an unknown input field")
        validate_entity(entity, lookup[key])
    return {"entities_checked": len(entities), "unique_entity_ids": True,
            "exact_field_offsets": True, "training_only": True}


def run_extraction(sample_size: int = 1000, seed: int = 0, *,
                   artifact_root: Path = PROCESSED / "extraction") -> dict:
    payload = prepare_inputs(sample_size, seed)
    provenance = {**payload["provenance"], "extractor_version": EXTRACTOR_VERSION,
                  "vocabulary_version": VOCABULARY_VERSION,
                  "vocabulary_sha256": vocabulary_digest(),
                  "package_version": version("intelligent-candidate-screener"),
                  "code_sha256": code_digests(),
                  "text_version": "raw-v1"}
    run_id = hashlib.sha256(_json(provenance).encode()).hexdigest()[:20]
    run_dir = artifact_root / run_id
    entities = [entity for doc in payload["documents"] for entity in extract_document(doc)]
    integrity = _validate(entities, payload["documents"])
    document_counts = {kind: sum(doc["doc_type"] == kind for doc in payload["documents"]) for kind in ("cv", "jd")}
    coverage = {}
    for kind in ("cv", "jd"):
        subset = [entity for entity in entities if entity["doc_type"] == kind]
        covered = len({entity["doc_id"] for entity in subset})
        coverage[kind] = {"documents": document_counts[kind], "documents_with_entities": covered,
                          "descriptive_coverage": covered / document_counts[kind] if document_counts[kind] else None,
                          "entities": len(subset), "by_label": dict(sorted(Counter(e["label"] for e in subset).items())),
                          "documents_with_label": {label: len({e["doc_id"] for e in subset if e["label"] == label})
                                                   for label in ("DEGREE", "KNOWLEDGE", "SKILL", "TITLE")},
                          "by_context_status": dict(sorted(Counter(e["context_status"] for e in subset).items()))}
    summary = {"input_counts": payload["summary"], "coverage": coverage,
               "integrity": integrity, "gold_accuracy": {"status": "blocked",
                "reason": "Gold adapters pending; unlabelled A2 measures descriptive coverage only"}}
    metadata = {"run_id": run_id, "summary": summary, "provenance": provenance,
                "availability": payload["availability"]}
    if run_dir.exists():
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Existing extraction run is incomplete; refusing overwrite")
        saved = json.loads(metadata_path.read_text())
        if {k: v for k, v in saved.items() if k != "artifact_sha256"} != metadata:
            raise ValueError("Existing extraction run is incompatible; refusing overwrite")
        if saved["artifact_sha256"] != {name: sha256(run_dir / name) for name in ("entities.parquet", "sample-manifest.json")}:
            raise ValueError("Stored extraction artifact digest mismatch")
        stored = pd.read_parquet(run_dir / "entities.parquet").to_dict("records")
        if stored != entities:
            raise ValueError("Stored extraction records differ; refusing overwrite")
        if json.loads((run_dir / "sample-manifest.json").read_text()) != provenance:
            raise ValueError("Stored sample manifest differs; refusing overwrite")
        metadata = saved
    else:
        run_dir.mkdir(parents=True)
        pd.DataFrame(entities, columns=ENTITY_COLUMNS).to_parquet(run_dir / "entities.parquet", index=False)
        (run_dir / "sample-manifest.json").write_text(_json(provenance))
        metadata["artifact_sha256"] = {name: sha256(run_dir / name) for name in ("entities.parquet", "sample-manifest.json")}
        (run_dir / "metadata.json").write_text(_json(metadata))
    return {**metadata, "run_dir": str(run_dir)}


def validate_run(result: dict | None = None, sample_size: int = 1000, seed: int = 0) -> dict:
    result = result or run_extraction(sample_size, seed)
    provenance = result["provenance"]
    run_dir = Path(result["run_dir"])
    disk_metadata = json.loads((run_dir / "metadata.json").read_text())
    if disk_metadata != {k: v for k, v in result.items() if k != "run_dir"}:
        raise ValueError("Stored metadata differs from run record")
    if json.loads((run_dir / "sample-manifest.json").read_text()) != provenance:
        raise ValueError("Stored sample manifest differs from provenance")
    if result["artifact_sha256"] != {name: sha256(run_dir / name) for name in ("entities.parquet", "sample-manifest.json")}:
        raise ValueError("Extraction artifact digest mismatch")
    if provenance["code_sha256"] != code_digests() or provenance["vocabulary_sha256"] != vocabulary_digest():
        raise ValueError("Extractor implementation changed since this run")
    payload = prepare_inputs(provenance["sample_size"], provenance["seed"])
    if any(payload["provenance"][key] != provenance[key] for key in payload["provenance"]):
        raise ValueError("Source provenance changed since extraction")
    entities = pd.read_parquet(Path(result["run_dir"]) / "entities.parquet").to_dict("records")
    integrity = _validate(entities, payload["documents"])
    expected = [entity for doc in payload["documents"] for entity in extract_document(doc)]
    if entities != expected:
        raise ValueError("Run does not reproduce under current extractor")
    return {**integrity, "reproduces": True,
            "entities_sha256": sha256(Path(result["run_dir"]) / "entities.parquet")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run_extraction(args.sample_size, args.seed)
    print(_json(validate_run(result) if args.check else {"run_id": result["run_id"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
