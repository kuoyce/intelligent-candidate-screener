"""Aggregate saved medium/baseline campaigns without displaying source evidence.

Counts describe operational behaviour, never entity precision, recall or F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fit_plots import COMPLETE, LABELS, STATES, split_views

BASELINE_CAMPAIGN_ID = "901a26c9e785920223b07c9b"


def comparison_tables(baseline: dict, medium: dict) -> dict[str, pd.DataFrame]:
    """Compare identical requested populations; missing results remain unknown.

    Label metrics use completed documents only. Runtime includes all attempted
    documents. Optional chunk counters are null when the saved format lacks them.
    """
    source_hashes = baseline["metadata"].get("provenance", {}).get("source_sha256")
    medium_hashes = medium["metadata"].get("provenance", {}).get("source_sha256")
    if not source_hashes or source_hashes != medium_hashes:
        raise ValueError("Campaign source digests differ or are missing")
    keys = ["split", "resume_id", "jd_id"]
    left = baseline["pairs"][keys].sort_values(keys).reset_index(drop=True)
    right = medium["pairs"][keys].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Campaign pair populations differ")
    operations, labels = [], []
    for model, campaign in (("small-v2.1 / word windows", baseline), ("medium-v2.5 / sentence windows", medium)):
        for split in ("train", "test"):
            for kind in ("cv", "jd"):
                statuses, entities = split_views(campaign, split, kind)
                unknown = set(statuses.status) - set(STATES)
                if unknown:
                    raise ValueError(f"Unknown processing states: {sorted(unknown)}")
                requested = len(statuses)
                completed = int(statuses.status.isin(COMPLETE).sum())
                row = {"model": model, "split": split, "doc_type": kind,
                       "requested_documents": requested, "completed_documents": completed,
                       "completed_spans": len(entities)}
                for state in STATES:
                    n = int((statuses.status == state).sum())
                    row[state] = n
                    row[f"{state}_rate"] = n / requested if requested else None
                for metric in ("runtime_seconds", "chunks", "fallback_windows", "quarantined_spans", "sentence_count"):
                    values = pd.to_numeric(statuses[metric], errors="raise") if metric in statuses else pd.Series(dtype=float)
                    row[metric] = float(values.sum(min_count=1)) if values.notna().any() else None
                operations.append(row)
                for label in LABELS:
                    subset = entities.loc[entities.label == label]
                    support = subset.doc_id.nunique()
                    scores = subset.confidence.astype(float)
                    labels.append({"model": model, "split": split, "doc_type": kind, "label": label,
                                   "requested_documents": requested, "completed_documents": completed,
                                   "spans": len(subset), "covered_completed_documents": support,
                                   "operational_coverage": support / requested if requested else None,
                                   "score_min": float(scores.min()) if len(scores) else None,
                                   "score_median": float(scores.median()) if len(scores) else None,
                                   "score_max": float(scores.max()) if len(scores) else None})
    return {"operations": pd.DataFrame(operations), "labels": pd.DataFrame(labels)}


def save_comparison(baseline: dict, medium: dict, directory: Path) -> dict:
    """Write only aggregate records and identities, never documents or mentions."""
    tables = comparison_tables(baseline, medium)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    record = {"baseline_campaign_id": baseline["metadata"]["campaign_id"],
              "medium_campaign_id": medium["metadata"]["campaign_id"],
              "limitation": "Model and chunk strategy change together; no attribution to chunking alone. Counts and scores are not extraction accuracy. No entity gold evaluation.",
              "denominators": "Unique documents per split and side; a document shared across splits appears in each split, not an additive total.",
              "peak_rss": "Read campaign /usr/bin/time -l logs; document runtimes do not measure process peak RSS."}
    (directory / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
