"""Strict, training-only A2 input selection; original field strings stay local."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from candidate_screener.config import DOCS_DATA, RAW
from candidate_screener.data.sources import SOURCES

FIELDS = {"cv": ("Position", "Moreinfo", "Looking For", "Highlights"),
          "jd": ("Position", "Long Description")}
PARTITION = DOCS_DATA / "manifests/a2-partition.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources(raw_root: Path = RAW) -> dict:
    result = {}
    for key in ("djinni-cv", "djinni-jd", "dataturks", "skillspan", "esco", "data-jobs"):
        source = SOURCES[key]
        folder = raw_root / source.directory
        missing = [a.glob for a in source.artifacts
                   if len(list(folder.glob(a.glob))) < a.min_files]
        result[key] = {"catalog_id": source.catalog_id,
                       "status": "missing" if missing else "present_unvalidated",
                       "missing_artifacts": missing}
    return result


def select_training(frame: pd.DataFrame, partition: pd.DataFrame, doc_type: str,
                    sample_size: int, seed: int) -> pd.DataFrame:
    """Validate full population membership before deterministic sampling."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{doc_type}: null or duplicate raw IDs")
    if not frame["id"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError(f"{doc_type}: IDs must be nonempty strings")
    if set(partition.columns) != {"doc_id", "doc_type", "region"}:
        raise ValueError("partition schema must be doc_id, doc_type, region")
    if partition.isna().any().any() or partition.duplicated(["doc_type", "doc_id"]).any():
        raise ValueError("partition contains missing or multiply assigned IDs")
    if not set(partition["doc_type"]) <= {"cv", "jd"} or not set(partition["region"]) <= {"train", "eval"}:
        raise ValueError("partition contains unknown type or region")
    part = partition.loc[partition.doc_type == doc_type]
    if set(frame["id"]) != set(part.doc_id):
        raise ValueError(f"{doc_type}: raw IDs and partition coverage differ")
    eligible = sorted(part.loc[part.region == "train", "doc_id"])
    selected = random.Random(seed).sample(eligible, min(sample_size, len(eligible)))
    return frame.set_index("id").loc[sorted(selected)].reset_index()


def prepare_inputs(sample_size: int = 1000, seed: int = 0, *,
                   raw_root: Path = RAW, partition_path: Path = PARTITION) -> dict:
    if not partition_path.is_file():
        raise FileNotFoundError("A2 partition missing; restore the committed manifest")
    partition = pd.read_csv(partition_path, dtype=str)
    documents, counts, input_digests, selected_ids = [], {}, {}, {}
    for doc_type, fields in FIELDS.items():
        source = f"djinni-{doc_type}"
        paths = sorted((raw_root / source).glob("*-train-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"Mandatory local A2 source missing: {source}")
        tables = []
        for path in paths:
            schema = pq.read_schema(path)
            for name in ("id", *fields):
                if name not in schema.names or not (pa.types.is_string(schema.field(name).type)
                                                   or pa.types.is_large_string(schema.field(name).type)):
                    raise ValueError(f"{source}: required string column missing/invalid: {name}")
            tables.append(pq.read_table(path, columns=["id", *fields]).to_pandas())
            input_digests[f"{source}/{path.name}"] = sha256(path)
        frame = pd.concat(tables, ignore_index=True)
        sample = select_training(frame, partition, doc_type, sample_size, seed)
        selected_ids[doc_type] = sample["id"].tolist()
        counts[doc_type] = {"raw_documents": len(frame),
                            "eligible_documents": int(((partition.doc_type == doc_type) & (partition.region == "train")).sum()),
                            "sampled_documents": len(sample), "fields": list(fields)}
        for row in sample.to_dict("records"):
            documents.append({"doc_id": row["id"], "source": source, "doc_type": doc_type,
                              "fields": {field: row[field] if pd.notna(row[field]) else "" for field in fields}})
    return {"documents": documents, "summary": counts, "availability": audit_sources(raw_root),
            "provenance": {"seed": seed, "sample_size": sample_size, "region": "train",
                           "sampling_rule": "sort IDs; Python Random(seed).sample; sort selected IDs",
                           "input_sha256": input_digests, "partition_sha256": sha256(partition_path),
                           "selected_ids": selected_ids, "fields": {k: list(v) for k, v in FIELDS.items()}}}
