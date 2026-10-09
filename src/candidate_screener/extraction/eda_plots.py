"""Standalone Matplotlib EDA figures from aggregate extraction analysis tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE")
COLORS = ("#087f8c", "#7356a4")
METHOD_COLORS = {"rules": COLORS[0], "GLiNER": COLORS[1]}


def _finish(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _paired_bars(table, value, title, ylabel, directory, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    methods = sorted(table["method"].unique())
    for ax, kind in zip(axes, ("cv", "jd")):
        subset = table[table["doc_type"] == kind]
        x = np.arange(len(LABELS))
        width = .36
        for i, method in enumerate(methods):
            rows = subset[subset["method"] == method].set_index("label")
            values = rows[value].reindex(LABELS, fill_value=0).to_numpy()
            bars = ax.bar(x + (i - .5) * width, values, width, label=method, color=METHOD_COLORS[method])
            ax.bar_label(bars, fmt="%.0f" if value != "descriptive_coverage" else "%.2f", fontsize=8)
        ax.set_xticks(x, LABELS, rotation=15)
        ax.set_title(kind.upper())
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        if value == "descriptive_coverage":
            ax.set_ylim(0, 1.12)
    fig.suptitle(title, fontsize=14)
    return _finish(fig, directory, name)


def counts_plot(table, directory):
    return _paired_bars(table, "entities", "Entity counts on the identical population", "Extracted spans", directory, "entity-counts")


def coverage_plot(table, directory):
    return _paired_bars(table, "descriptive_coverage", "Documents containing each label", "Fraction of sampled documents", directory, "document-label-coverage")


def document_distribution_plot(table, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    methods = sorted(table["method"].unique())
    maximum = int(table["entities"].max()) if len(table) else 0
    bins = np.arange(-.5, maximum + 1.5, 1)
    for ax, kind in zip(axes, ("cv", "jd")):
        for i, method in enumerate(methods):
            values = table.loc[(table["doc_type"] == kind) & (table["method"] == method), "entities"]
            ax.hist(values, bins=bins, histtype="step", linewidth=2, color=METHOD_COLORS[method],
                    label=f"{method} (zeros: {(values == 0).sum()})")
        ax.set_title(kind.upper())
        ax.set_xlabel("Entities per document (including zero)")
        ax.set_ylabel("Documents")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Entity count distributions — all sampled documents retained")
    return _finish(fig, directory, "entities-per-document")


def cooccurrence_plot(table, directory):
    methods = sorted(table["method"].unique())
    fig, axes = plt.subplots(2, len(methods), figsize=(11, 8), squeeze=False)
    for row, kind in enumerate(("cv", "jd")):
        for col, method in enumerate(methods):
            subset = table[(table["doc_type"] == kind) & (table["method"] == method)]
            matrix = subset.pivot(index="label_a", columns="label_b", values="documents").reindex(index=LABELS, columns=LABELS).fillna(0)
            ax = axes[row, col]
            maximum = max(1, int(table["documents"].max()))
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
            for i in range(4):
                for j in range(4):
                    value = int(matrix.iloc[i,j])
                    ax.text(j, i, str(value), ha="center", va="center", color="white" if value > maximum * .55 else "black")
            ax.set_xticks(range(4), LABELS, rotation=20)
            ax.set_yticks(range(4), LABELS)
            ax.set_title(f"{kind.upper()} / {method}")
    fig.suptitle("Label co-occurrence: document counts (diagonal is label support)")
    return _finish(fig, directory, "label-cooccurrence")


def mentions_plot(table, directory):
    figures = {}
    for method in sorted(table["method"].unique()):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
        for row, kind in enumerate(("cv", "jd")):
            for col, label in enumerate(LABELS):
                subset = table[(table["method"] == method) & (table["doc_type"] == kind) & (table["label"] == label)].nlargest(5, "entities").iloc[::-1]
                ax = axes[row, col]
                ax.barh(subset["mention_casefold"].str.slice(0, 45), subset["entities"], color=COLORS[1] if method != "rules" else COLORS[0])
                ax.set_title(f"{kind.upper()} / {label}")
                ax.set_xlabel("Occurrences")
                ax.tick_params(axis="y", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Most frequent actual {method} mentions — case-folded strings, not canonical IDs")
        figures[f"mentions-{method}"] = _finish(fig, directory, f"mentions-{method}")
    return figures


def score_plot(neural, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind in zip(axes, ("cv", "jd")):
        for label in LABELS:
            scores = neural.loc[(neural["doc_type"] == kind) & (neural["label"] == label), "confidence"]
            ax.hist(scores, bins=np.linspace(.5, 1, 21), histtype="step", linewidth=1.8, label=label)
        ax.set_title(kind.upper())
        ax.set_xlabel("Raw uncalibrated GLiNER score")
        ax.set_ylabel("Stored predicted spans")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Scores conditional on the original ≥0.5 extraction threshold")
    return _finish(fig, directory, "raw-score-distribution")


def thresholds_plot(table, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind in zip(axes, ("cv", "jd")):
        for label in LABELS:
            rows = table[(table["doc_type"] == kind) & (table["label"] == label)].sort_values("threshold")
            ax.plot(rows["threshold"], rows["retained_entities"], marker="o", label=label)
        ax.set_title(kind.upper())
        ax.set_xlabel("Post-hoc threshold on stored raw score")
        ax.set_ylabel("Retained spans")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Threshold sensitivity: retention only, not estimated precision")
    return _finish(fig, directory, "threshold-retention")


def overlap_plot(table, directory):
    table = table.pivot(index="doc_type", columns="category", values="entities").fillna(0).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for key, title, color in (("shared", "Exact shared span + label", "#78909c"), ("rule_only", "Rules only", COLORS[0]), ("neural_only", "GLiNER only", COLORS[1])):
        values = table[key].to_numpy()
        bars = ax.bar(x, values, bottom=bottom, label=title, color=color)
        ax.bar_label(bars, label_type="center", fontsize=10, color="white")
        bottom += values
    ax.set_xticks(x, table["doc_type"].str.upper())
    ax.set_ylabel("Distinct spans in the union")
    ax.set_title("Exact span-and-label overlap — disagreement, not correctness")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1,1))
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, directory, "exact-span-overlap")


def field_context_plot(table, directory):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, column, title in zip(axes, ("field", "context_status"), ("Source field", "Heuristic context")):
        grouped = table.groupby(["method", column], observed=True)["entities"].sum().unstack("method", fill_value=0)
        grouped.plot.bar(ax=ax, color=[METHOD_COLORS[method] for method in grouped.columns], rot=25)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Extracted spans")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Field and heuristic-context distributions (CV and JD combined)")
    return _finish(fig, directory, "field-context")


def save_charts(tables, neural, output_dir):
    """Save standalone PNGs; inputs contain aggregates plus stored neural scores."""
    paths = {
        "counts": counts_plot(tables["counts"], output_dir),
        "coverage": coverage_plot(tables["label_coverage"], output_dir),
        "document_counts": document_distribution_plot(tables["document_counts"], output_dir),
        "cooccurrence": cooccurrence_plot(tables["cooccurrence"], output_dir),
        "scores": score_plot(neural, output_dir),
        "thresholds": thresholds_plot(tables["thresholds"], output_dir),
        "overlap": overlap_plot(tables["overlap"], output_dir),
        "field_context": field_context_plot(tables["field_context"], output_dir),
    }
    paths.update(mentions_plot(tables["mentions"], output_dir))
    return paths
