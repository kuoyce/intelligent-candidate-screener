"""Inspect frozen GLiNER evidence without claiming replay by the current adapter.

Source membership, original spans and saved artifact integrity are checked. The
producing code/runtime fingerprints are reported as historical provenance, never
compared to today's implementation to manufacture a replay claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from candidate_screener.config import DOCS_DATA, OUTPUT, PROCESSED
from .eda import DOC_KEY, validate_population
from .inputs import prepare_inputs, sha256
from .run import _validate

LABELS = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS")
METRICS = OUTPUT / "extraction/transformer-experience-years-metrics.json"
ARTIFACT_ROOT = PROCESSED / "extraction/transformer-runs"
DESTINATION = OUTPUT / "extraction/entity-distributions"


def load_frozen(metrics_path: Path = METRICS, *, artifact_root: Path = ARTIFACT_ROOT,
                input_loader=None) -> dict:
    """Load the committed campaign, requiring full checked original-field coverage."""
    record = json.loads(Path(metrics_path).read_text())
    run_id = record["run_id"]
    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        raise ValueError("Unsafe frozen run identity")
    folder = Path(artifact_root) / run_id
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["run_id"] != run_id or metadata["summary"] != record["summary"]:
        raise ValueError("Frozen metadata differs from committed campaign")
    provenance = metadata["provenance"]
    if any(provenance.get(key) != value for key, value in record["provenance"].items()):
        raise ValueError("Frozen producing provenance differs from committed record")
    required = {"entities.parquet", "field-coverage.json", "sample-manifest.json"}
    if set(record["artifact_sha256"]) != required or metadata["artifact_sha256"] != record["artifact_sha256"]:
        raise ValueError("Frozen artifact manifest differs")
    for name in required:
        if sha256(folder / name) != record["artifact_sha256"][name]:
            raise ValueError("Frozen artifact digest mismatch")
    if json.loads((folder / "sample-manifest.json").read_text()) != provenance:
        raise ValueError("Frozen sample manifest differs from producing provenance")
    if metadata["summary"]["status"] != "complete" or provenance["region"] != "train":
        raise ValueError("Distribution EDA requires complete training-only evidence")
    if set(provenance["labels"].values()) != set(LABELS):
        raise ValueError("Frozen campaign must contain all five requested labels")
    payload = (input_loader or prepare_inputs)(provenance["sample_size"], provenance["seed"])
    if any(provenance.get(key) != value for key, value in payload["provenance"].items()):
        raise ValueError("Frozen source provenance or sampled membership changed")
    documents = pd.DataFrame(payload["documents"])
    entities = pd.read_parquet(folder / "entities.parquet")
    validate_population(documents, entities)
    _validate(entities.to_dict("records"), payload["documents"])
    lookup = {(doc["source"], doc["doc_id"], field): (doc["doc_type"], text)
              for doc in payload["documents"] for field, text in doc["fields"].items()}
    fields = json.loads((folder / "field-coverage.json").read_text())
    keys = [(row["source"], row["doc_id"], row["field"]) for row in fields]
    if len(set(keys)) != len(keys) or set(keys) != set(lookup):
        raise ValueError("Frozen field coverage must contain every sampled field exactly once")
    for key, row in zip(keys, fields):
        kind, text = lookup[key]
        if not row["complete"] or row["doc_type"] != kind or row["characters"] != len(text):
            raise ValueError("Frozen field coverage is incomplete or mismatched")
        if row.get("interrupted") or row.get("quarantined_intervals"):
            raise ValueError("Frozen field coverage contains unsubmitted text")
        end_so_far = 0
        for start, end in row["intervals"]:
            if not 0 <= start <= end <= len(text) or start > end_so_far:
                raise ValueError("Frozen coverage contains invalid intervals or gaps")
            end_so_far = max(end_so_far, end)
        if end_so_far != len(text):
            raise ValueError("Frozen field coverage has an uncovered tail")
    field_lookup = dict(zip(keys, fields))
    for entity in entities.to_dict("records"):
        key = (entity["source"], entity["doc_id"], entity["field"])
        if entity["label"] not in LABELS or not any(
                start <= entity["start"] < entity["end"] <= end
                for start, end in field_lookup[key]["intervals"]):
            raise ValueError("Frozen entity lies outside submitted field coverage")
    if metadata["summary"]["completed_documents"] != len(documents):
        raise ValueError("Frozen completion count differs from sampled population")
    for kind in ("cv", "jd"):
        subset = entities.loc[entities.doc_type == kind]
        saved = metadata["summary"]["coverage"][kind]
        counts = subset.label.value_counts()
        doc_counts = subset.groupby("label").doc_id.nunique()
        if saved["entities"] != len(subset) or saved["requested_documents"] != int((documents.doc_type == kind).sum()):
            raise ValueError("Frozen aggregate population differs from evidence")
        if any(saved["by_label"].get(label, 0) != int(counts.get(label, 0)) or
               saved["documents_with_label"].get(label, 0) != int(doc_counts.get(label, 0)) for label in LABELS):
            raise ValueError("Frozen aggregate label counts differ from evidence")
    return {"documents": documents, "entities": entities, "run_id": run_id,
            "provenance": provenance, "validation": {
                "artifact_hashes": True, "sample_membership": True, "source_provenance": True,
                "original_spans": True, "complete_field_coverage": True,
                "current_code_replay": False,
                "interpretation": "Historical saved predictions; no reinference or accuracy claim"}}


def distribution_tables(analysis: dict) -> dict:
    """Anonymous document counts including zeros; uniqueness is string identity.

    A unique mention is (label, casefolded whitespace-normalized source span).
    It is descriptive string uniqueness, not canonical ontology resolution.
    """
    documents, entities = analysis["documents"], analysis["entities"]
    validate_population(documents, entities)
    grouped = {key: frame for key, frame in entities.groupby(DOC_KEY)}
    rows, frequencies = [], []
    for kind in ("cv", "jd"):
        docs = documents.loc[documents.doc_type == kind].sort_values(["source", "doc_id"])
        for row_index, doc in enumerate(docs.to_dict("records"), 1):
            frame = grouped.get((doc["source"], doc["doc_id"], kind), entities.iloc[:0])
            characters = sum(len(text or "") for text in doc["fields"].values())
            for label in LABELS:
                spans = frame.loc[frame.label == label]
                normalized = spans.surface.map(lambda text: " ".join(text.casefold().split()))
                unique = normalized.nunique()
                rows.append({"doc_type": kind, "row_index": row_index, "label": label,
                             "characters": characters, "mentions": len(spans),
                             "unique_mentions": int(unique), "repeated_mentions": len(spans) - int(unique)})
                for mention in normalized.unique():
                    frequencies.append({"doc_type": kind, "label": label, "normalized": mention})
    per_doc = pd.DataFrame(rows)
    summary = []
    for (kind, label), frame in per_doc.groupby(["doc_type", "label"], sort=False):
        n = len(frame)
        summary.append({"doc_type": kind, "label": label, "documents": n,
                        "total_mentions": int(frame.mentions.sum()), "median_mentions": float(frame.mentions.median()),
                        "documents_with_mentions": int((frame.mentions > 0).sum()),
                        "zero_mention_documents": int((frame.mentions == 0).sum()),
                        "descriptive_coverage": float((frame.mentions > 0).mean()),
                        "total_unique_mentions": int(frame.unique_mentions.sum()),
                        "total_repeated_mentions": int(frame.repeated_mentions.sum())})
    # Export only frequency distributions, never strings which may contain PII.
    frequency_rows = []
    freq = pd.DataFrame(frequencies, columns=["doc_type", "label", "normalized"])
    for (kind, label), frame in freq.groupby(["doc_type", "label"]):
        histogram = frame.normalized.value_counts().value_counts().sort_index()
        for support, concepts in histogram.items():
            frequency_rows.append({"doc_type": kind, "label": label,
                                   "document_support": int(support), "unique_strings": int(concepts)})
    return {"per_document": per_doc, "summary": pd.DataFrame(summary),
            "string_support": pd.DataFrame(frequency_rows, columns=["doc_type", "label", "document_support", "unique_strings"])}


def labelled_overlap(analysis: dict, labels_path: Path = DOCS_DATA / "manifests/a2-final-labels.csv") -> dict:
    """Count independently labelled pairs actually present; never invent pairings."""
    if not Path(labels_path).is_file():
        return {"status": "unavailable", "reason": "Independent pair manifest missing"}
    labels = pd.read_csv(labels_path, dtype=str)
    documents = analysis["documents"]
    cvs = set(documents.loc[documents.doc_type == "cv", "doc_id"])
    jds = set(documents.loc[documents.doc_type == "jd", "doc_id"])
    matched = labels.loc[labels.cv_id.isin(cvs) & labels.jd_id.isin(jds)]
    return {"status": "present" if len(matched) else "no_overlap", "labelled_pairs_in_campaign": len(matched),
            "reference_pairs": len(labels),
            "interpretation": "No grade-based evidence plots when campaign has no independently labelled pairs"}


def save_distribution_charts(tables: dict, directory: Path) -> dict:
    """Shared-scale count heatmaps plus separate CV/JD distributions and length plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = tables["per_document"]
    paths = {}
    def finish(fig, name):
        fig.tight_layout()
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    maximum = max(1, int(table.mentions.max()))
    for ax, kind in zip(axes, ("cv", "jd")):
        matrix = table.loc[table.doc_type == kind].pivot(index="row_index", columns="label", values="mentions").reindex(columns=LABELS)
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Anonymous document row")
        ax.set_title(f"{kind.upper()} (n={len(matrix)}) — shared count scale")
    fig.colorbar(im, ax=axes.tolist(), label="Predicted mentions", shrink=.6)
    finish(fig, "document-label-heatmaps")
    fig, axes = plt.subplots(2, len(LABELS), figsize=(19, 7))
    for row, kind in enumerate(("cv", "jd")):
        for col, label in enumerate(LABELS):
            values = table.loc[(table.doc_type == kind) & (table.label == label), "mentions"]
            axes[row, col].hist(values, bins=min(20, max(1, int(values.max()) + 1)))
            axes[row, col].set_title(f"{kind.upper()} / {label}\nn={len(values)}, zeros={(values == 0).sum()}", fontsize=9)
            axes[row, col].set_xlabel("Predicted mentions")
    finish(fig, "label-count-histograms")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ("cv", "jd")):
        groups = [table.loc[(table.doc_type == kind) & (table.label == label), "mentions"] for label in LABELS]
        ax.boxplot(groups, tick_labels=LABELS, showfliers=True)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.set_title(f"{kind.upper()} — zeros retained")
        ax.set_ylabel("Predicted mentions")
    finish(fig, "label-count-boxplots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    totals = table.groupby(["doc_type", "row_index"]).agg(characters=("characters", "first"), mentions=("mentions", "sum"), unique_mentions=("unique_mentions", "sum"))
    for ax, kind in zip(axes, ("cv", "jd")):
        subset = totals.loc[kind]
        ax.scatter(subset.characters, subset.mentions, label="All mentions", alpha=.6)
        ax.scatter(subset.characters, subset.unique_mentions, label="Unique label/string mentions", marker="x", alpha=.6)
        ax.set_title(f"{kind.upper()} (n={len(subset)})")
        ax.set_xlabel("Submitted source characters")
        ax.set_ylabel("Predicted mention count")
        ax.legend(fontsize=8)
    finish(fig, "length-and-mention-counts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ("cv", "jd")):
        subset = tables["string_support"].loc[tables["string_support"].doc_type == kind]
        for label in LABELS:
            group = subset.loc[subset.label == label].sort_values("document_support")
            if len(group):
                ax.plot(group.document_support, group.unique_strings, marker="o", label=label)
        ax.set_title(f"{kind.upper()} — common versus rare strings")
        ax.set_xlabel("Documents containing the normalized label/string")
        ax.set_ylabel("Unique strings with this support")
        if len(subset):
            ax.legend(fontsize=8)
    finish(fig, "string-document-support")
    return paths


def save_distributions(analysis: dict, directory: Path = DESTINATION) -> dict:
    """Save anonymous numerical tables, charts and historical provenance only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = distribution_tables(analysis)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    charts = save_distribution_charts(tables, directory)
    provenance = analysis["provenance"]
    record = {"run_id": analysis["run_id"], "validation": analysis["validation"],
              "producing_code_sha256": provenance["code_sha256"],
              "producing_adapter_version": provenance["transformer_version"],
              "producing_labels": provenance["labels"],
              "uniqueness": "label plus casefolded whitespace-normalized source string; not ontology concepts",
              "interpretation": "Descriptive frozen prediction counts, not recall or extraction accuracy",
              "labelled_overlap": labelled_overlap(analysis),
              "summary": tables["summary"].to_dict("records"),
              "charts": {name: Path(path).name for name, path in charts.items()}}
    (directory / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
