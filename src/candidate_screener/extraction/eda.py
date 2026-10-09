"""Descriptive analysis of saved spans, never neural inference or accuracy scoring."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from candidate_screener.annotation.redact import assert_clean, redact_text
from candidate_screener.config import OUTPUT, PROCESSED
from candidate_screener.extraction import extract_document
from candidate_screener.extraction.inputs import prepare_inputs, sha256
from candidate_screener.extraction.run import ENTITY_COLUMNS, _validate

KEY = ["source", "doc_id", "field", "start", "end", "label"]
DOC_KEY = ["source", "doc_id", "doc_type"]
# Rules support four labels; historical transformer runs used this same scope.
LABELS = ["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"]


def analysis_labels(entities: pd.DataFrame, provenance: dict | None = None) -> list[str]:
    """Use the saved run's prompt scope, preserving historical four-label runs."""
    saved = (provenance or {}).get("labels")
    labels = list(dict.fromkeys(saved.values())) if saved else list(LABELS)
    return labels + sorted(set(entities.label) - set(labels))


def validate_population(documents: pd.DataFrame, entities: pd.DataFrame) -> None:
    if documents.duplicated(["source", "doc_id"]).any():
        raise ValueError("Duplicate documents in analysis population")
    if entities.duplicated(KEY).any():
        raise ValueError("Duplicate entity span-label keys")
    population = set(documents[DOC_KEY].itertuples(index=False, name=None))
    if not set(entities[DOC_KEY].itertuples(index=False, name=None)) <= population:
        raise ValueError("Entities outside the analysis population")


def load_analysis(metrics_path: Path = OUTPUT / "extraction/transformer-metrics.json") -> dict:
    record = json.loads(metrics_path.read_text())
    run_dir = PROCESSED / "extraction/transformer-runs" / record["run_id"]
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata["summary"]["status"] != "complete":
        raise ValueError("EDA requires a completed saved transformer run")
    if record["artifact_sha256"] != metadata["artifact_sha256"]:
        raise ValueError("Committed and local transformer artifact hashes differ")
    for name, digest in metadata["artifact_sha256"].items():
        if sha256(run_dir / name) != digest:
            raise ValueError("Transformer artifact hash mismatch")
    provenance = metadata["provenance"]
    if json.loads((run_dir / "sample-manifest.json").read_text()) != provenance:
        raise ValueError("Transformer sample manifest mismatch")
    payload = prepare_inputs(provenance["sample_size"], provenance["seed"])
    if any(payload["provenance"][name] != provenance[name] for name in payload["provenance"]):
        raise ValueError("EDA source or sampled population changed")
    from candidate_screener.extraction.transformer_run import validate_transformer_run
    validate_transformer_run({**metadata, "run_dir": str(run_dir)}, payload=payload)
    if metadata["summary"]["completed_documents"] != len(payload["documents"]):
        raise ValueError("EDA requires every sampled document to be complete")
    fields = json.loads((run_dir / "field-coverage.json").read_text())
    expected_fields = {(d["source"], d["doc_id"], field) for d in payload["documents"] for field in d["fields"]}
    if {(f["source"], f["doc_id"], f["field"]) for f in fields} != expected_fields or not all(f["complete"] for f in fields):
        raise ValueError("EDA requires complete coverage of every sampled field")
    neural = pd.read_parquet(run_dir / "entities.parquet")
    rule_records = [e for doc in payload["documents"] for e in extract_document(doc)]
    rules = pd.DataFrame(rule_records, columns=ENTITY_COLUMNS)
    documents = pd.DataFrame(payload["documents"])
    for frame in (neural, rules):
        validate_population(documents, frame)
        _validate(frame.to_dict("records"), payload["documents"])
    return {"documents": documents, "neural": neural, "rules": rules,
            "provenance": provenance, "summary": metadata["summary"], "run_id": record["run_id"],
            "validation": {"artifact_hashes": True, "sample_identity": True, "exact_source_spans": True,
                           "rule_comparison": "Rules regenerated on identical sampled IDs; no neural reinference"}}


def per_document_counts(documents: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    validate_population(documents, entities)
    totals = entities.groupby(DOC_KEY).size().rename("entities").reset_index()
    result = documents[DOC_KEY].merge(totals, how="left", on=DOC_KEY)
    result["entities"] = result["entities"].fillna(0).astype(int)
    return result


def threshold_counts(documents: pd.DataFrame, entities: pd.DataFrame,
                     thresholds=(.5, .6, .7, .8, .9), labels=None) -> pd.DataFrame:
    validate_population(documents, entities)
    labels = analysis_labels(entities) if labels is None else labels
    rows = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or not .5 <= threshold <= 1:
            raise ValueError("Thresholds must be finite and within the saved score floor 0.5 to 1.0")
        retained = entities.loc[entities.confidence >= threshold]
        for kind in ("cv", "jd"):
            denominator = int((documents.doc_type == kind).sum())
            subset = retained.loc[retained.doc_type == kind]
            for label in labels:
                spans = subset.loc[subset.label == label]
                covered = spans.doc_id.nunique()
                rows.append({"threshold": threshold, "doc_type": kind, "label": label,
                             "retained_entities": len(spans), "covered_documents": covered,
                             "population_documents": denominator,
                             "descriptive_coverage": covered / denominator if denominator else None})
    return pd.DataFrame(rows)


def exact_comparison(neural: pd.DataFrame, rules: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = neural.set_index(KEY, drop=False)
    r = rules.set_index(KEY, drop=False)
    if not n.index.is_unique or not r.index.is_unique:
        raise ValueError("Duplicate entity span-label keys")
    return {"shared": n.loc[n.index.isin(r.index)].reset_index(drop=True),
            "neural_only": n.loc[~n.index.isin(r.index)].reset_index(drop=True),
            "rule_only": r.loc[~r.index.isin(n.index)].reset_index(drop=True)}


def analyze(analysis: dict) -> dict[str, pd.DataFrame]:
    documents = analysis["documents"]
    frames = {"GLiNER": analysis["neural"], "rules": analysis["rules"]}
    counts, per_docs, coverage, cooccurrence, mentions, contexts = [], [], [], [], [], []
    for method, frame in frames.items():
        validate_population(documents, frame)
        labels = analysis_labels(frame, analysis.get("provenance")) if method == "GLiNER" else LABELS
        per_docs.append(per_document_counts(documents, frame).assign(method=method))
        for kind in ("cv", "jd"):
            total = int((documents.doc_type == kind).sum())
            subset = frame.loc[frame.doc_type == kind]
            for label in labels:
                group = subset.loc[subset.label == label]
                counts.append({"method": method, "doc_type": kind, "label": label, "entities": len(group)})
                coverage.append({"method": method, "doc_type": kind, "label": label,
                    "covered_documents": group.doc_id.nunique(), "population_documents": total,
                    "descriptive_coverage": group.doc_id.nunique() / total if total else None})
                frequencies = group.surface.map(str.casefold).value_counts()
                for mention, count in frequencies.items():
                    mentions.append({"method": method, "doc_type": kind, "label": label,
                                     "mention_casefold": redact_text(mention), "entities": int(count)})
            doc_labels = subset.groupby(["source", "doc_id"]).label.agg(set)
            for left in labels:
                for right in labels:
                    cooccurrence.append({"method": method, "doc_type": kind, "label_a": left, "label_b": right,
                                         "documents": sum(left in labels and right in labels for labels in doc_labels)})
        contexts.append(frame.groupby(["doc_type", "field", "context_status", "label"]).size().rename("entities").reset_index().assign(method=method))
    differences = exact_comparison(analysis["neural"], analysis["rules"])
    overlap = [{"doc_type": kind, "category": category, "entities": int((frame.doc_type == kind).sum())}
               for category, frame in differences.items() for kind in ("cv", "jd")]
    return {"counts": pd.DataFrame(counts), "document_counts": pd.concat(per_docs, ignore_index=True),
            "label_coverage": pd.DataFrame(coverage), "cooccurrence": pd.DataFrame(cooccurrence),
            "mentions": pd.DataFrame(mentions), "thresholds": threshold_counts(documents, analysis["neural"],
                labels=analysis_labels(analysis["neural"], analysis.get("provenance"))),
            "overlap": pd.DataFrame(overlap), "field_context": pd.concat(contexts, ignore_index=True)}


def bounded_examples(analysis: dict, category: str = "shared", limit: int = 5,
                     context_chars: int = 90) -> pd.DataFrame:
    if category not in {"shared", "neural_only", "rule_only"} or not 0 <= limit <= 5 or not 0 <= context_chars <= 120:
        raise ValueError("Examples require known category, at most five rows and bounded context")
    differences = exact_comparison(analysis["neural"], analysis["rules"])
    documents = {(doc["source"], doc["doc_id"]): doc for doc in analysis["documents"].to_dict("records")}
    rows = []
    for entity in differences[category].sort_values(KEY).head(limit).to_dict("records"):
        text = documents[(entity["source"], entity["doc_id"])]["fields"][entity["field"]]
        start, end = entity["start"], entity["end"]
        context = text[max(0, start - context_chars // 2):min(len(text), end + context_chars // 2)]
        rows.append({"surface": redact_text(entity["surface"])[:100], "label": entity["label"],
                     "doc_type": entity["doc_type"], "field": entity["field"], "start": start, "end": end,
                     "context": redact_text(context)[:160]})
    result = pd.DataFrame(rows, columns=["surface", "label", "doc_type", "field", "start", "end", "context"])
    assert_clean(result.surface, "EDA entity preview")
    assert_clean(result.context, "EDA bounded source context")
    return result


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    """Commit aggregates and bounded mention vocabulary; keep document IDs local."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in tables.items():
        if name == "document_counts":
            frame = frame.groupby(["method", "doc_type", "entities"]).size().rename("documents").reset_index()
        elif name == "mentions":
            frame = frame.sort_values("entities", ascending=False).groupby(["method", "doc_type", "label"], sort=False).head(5)
            frame = frame.copy()
            frame["mention_casefold"] = frame["mention_casefold"].map(lambda text: redact_text(text)[:100])
            assert_clean(frame.mention_casefold, "EDA saved bounded vocabulary")
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    return paths
