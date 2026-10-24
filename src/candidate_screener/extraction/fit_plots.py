"""Descriptive figures for a saved fit campaign; never model inference."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS")
STATES = ("complete", "complete_zero", "incomplete", "failed", "pending")
COMPLETE = {"complete", "complete_zero"}
COLORS = {"cv": "#7356a4", "jd": "#087f8c"}


def split_views(campaign, split, kind):
    """Attach one saved document to its split without counting repeated pairs."""
    column = "resume_id" if kind == "cv" else "jd_id"
    ids = set(campaign["pairs"].loc[campaign["pairs"].split == split, column])
    statuses = campaign["statuses"].loc[(campaign["statuses"].doc_type == kind) & campaign["statuses"].doc_id.isin(ids)]
    entities = campaign["entities"].loc[(campaign["entities"].doc_type == kind) & campaign["entities"].doc_id.isin(ids)]
    if statuses.doc_id.duplicated().any() or set(statuses.doc_id) != ids:
        raise ValueError("Saved statuses do not match the requested split population")
    complete_ids = set(statuses.loc[statuses.status.isin(COMPLETE), "doc_id"])
    completed_entities = entities.loc[entities.doc_id.isin(complete_ids)]
    return statuses, completed_entities


def label_document_counts(campaign, split, kind):
    """Five columns for every completed document, retaining measured zeros only."""
    statuses, entities = split_views(campaign, split, kind)
    ids = statuses.loc[statuses.status.isin(COMPLETE), "doc_id"]
    counts = entities.groupby(["doc_id", "label"]).size().unstack(fill_value=0)
    return counts.reindex(index=ids, columns=LABELS, fill_value=0).fillna(0).astype(int)


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def counts_plot(campaign, directory):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(5)
    for row, split in enumerate(("train", "test")):
        for i, kind in enumerate(("cv", "jd")):
            _, entities = split_views(campaign, split, kind)
            span_counts = entities.label.value_counts().reindex(LABELS, fill_value=0)
            string_counts = entities.assign(mention=entities.surface.str.casefold()).groupby("label").mention.nunique().reindex(LABELS, fill_value=0)
            for col, values in enumerate((span_counts, string_counts)):
                ax = axes[row,col]
                bars = ax.bar(x + (i-.5)*.36, values, .36, color=COLORS[kind], label=kind.upper())
                ax.bar_label(bars, fontsize=7)
                ax.set_xticks(x, LABELS, rotation=20)
                ax.set_title(split.upper() + (" — mention spans" if col == 0 else " — unique case-folded strings"))
                ax.set_ylabel("Spans" if col == 0 else "Unique strings")
                ax.legend(frameon=False)
                ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Five frozen categories — completed documents only; strings are not canonical IDs")
    return _save(fig, directory, "five-label-counts")


def states_plot(campaign, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    state_colors = ("#7356a4", "#087f8c", "#e6a23c", "#b44d58", "#9a9a9a")
    for ax, split in zip(axes, ("train", "test")):
        bottom = np.zeros(2)
        for state, color in zip(STATES, state_colors):
            counts = [int((split_views(campaign, split, kind)[0].status == state).sum()) for kind in ("cv","jd")]
            bars = ax.bar([0,1], counts, bottom=bottom, label=state, color=color)
            ax.bar_label(bars, labels=[str(n) if n else "" for n in counts], label_type="center", color="white", fontsize=9)
            bottom += counts
        ax.set_xticks([0,1],["CV","JD"])
        ax.set_title(split.upper())
        ax.set_ylabel("Unique requested documents")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, bbox_to_anchor=(1,1), loc="upper left")
    fig.suptitle("Completed zeros and unavailable results remain separate states")
    return _save(fig, directory, "completion-states")


def coverage_plot(campaign, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x=np.arange(5)
    for ax, split in zip(axes, ("train","test")):
        for i, kind in enumerate(("cv","jd")):
            statuses, entities = split_views(campaign, split, kind)
            denominator = len(statuses)
            support = entities.groupby("label").doc_id.nunique().reindex(LABELS,fill_value=0)
            values = support / denominator if denominator else support * 0
            bars=ax.bar(x+(i-.5)*.36,values,.36,label=f"{kind.upper()} (requested n={denominator})",color=COLORS[kind])
            ax.bar_label(bars,fmt="%.2f",fontsize=8)
        ax.set_xticks(x,LABELS,rotation=20)
        ax.set_ylim(0,1.1)
        ax.set_title(split.upper())
        ax.set_ylabel("Completed evidence / all requested documents")
        ax.legend(frameon=False,fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Descriptive coverage — missing results cannot establish absence")
    return _save(fig,directory,"document-label-coverage")


def distribution_plot(campaign,directory):
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row,split in enumerate(("train","test")):
        for col,kind in enumerate(("cv","jd")):
            counts=label_document_counts(campaign,split,kind)
            ax=axes[row,col]
            if len(counts):
                ax.boxplot([counts[label].to_numpy() for label in LABELS],tick_labels=LABELS,showfliers=True)
            else:
                ax.text(.5,.5,"No completed documents",transform=ax.transAxes,ha="center")
                ax.set_xticks([])
            ax.tick_params(axis="x",rotation=20)
            ax.set_ylabel("Mention spans per completed document")
            ax.set_title(f"{split.upper()} / {kind.upper()} (completed n={len(counts)})")
            ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("All five labels per document — measured zero counts retained")
    return _save(fig,directory,"per-document-label-distributions")


def scores_plot(campaign,directory):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,split in enumerate(("train","test")):
        for col,kind in enumerate(("cv","jd")):
            _,entities=split_views(campaign,split,kind)
            ax=axes[row,col]
            for label in LABELS:
                scores=entities.loc[entities.label==label,"confidence"].astype(float)
                ax.hist(scores,bins=np.linspace(.5,1,21),histtype="step",linewidth=1.6,label=label)
            ax.set_title(f"{split.upper()} / {kind.upper()}")
            ax.set_xlabel("Raw uncalibrated GLiNER score")
            ax.set_ylabel("Stored spans in completed documents")
            ax.legend(frameon=False,fontsize=7)
            ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Score distributions conditional on frozen threshold ≥0.5 — not correctness")
    return _save(fig,directory,"raw-score-distributions")


def save_fit_charts(campaign, output_dir):
    return {"counts":counts_plot(campaign,output_dir),"states":states_plot(campaign,output_dir),
            "coverage":coverage_plot(campaign,output_dir),"distribution":distribution_plot(campaign,output_dir),
            "scores":scores_plot(campaign,output_dir)}


def example_counts(campaign, cv_id, jd_id):
    """Completed-only counts for the earlier pair, retaining five configured labels."""
    result = []
    for kind, identity in (("cv", cv_id), ("jd", jd_id)):
        statuses = campaign["statuses"].loc[(campaign["statuses"].doc_type == kind) & (campaign["statuses"].doc_id == identity)]
        if len(statuses) != 1:
            raise ValueError("Earlier example document is absent or ambiguous")
        status = statuses.iloc[0].status
        frame = campaign["entities"].loc[(campaign["entities"].doc_type == kind) & (campaign["entities"].doc_id == identity)]
        counts = frame.label.value_counts().reindex(LABELS,fill_value=0)
        result.append({"doc_type":kind,"status":status,"current_spans":len(frame) if status in COMPLETE else None,
                       "by_label":{label:int(counts[label]) if status in COMPLETE else None for label in LABELS},
                       "earlier_limited_example_spans":15 if kind == "cv" else 0})
    return result


def pair_mentions(campaign, cv_id, jd_id):
    """Case-folded string overlap per label; descriptive, not canonical matching."""
    import pandas as pd
    selected = {}
    states = {}
    for kind, identity in (("cv", cv_id), ("jd", jd_id)):
        status = campaign["statuses"].loc[(campaign["statuses"].doc_type == kind) & (campaign["statuses"].doc_id == identity)]
        if len(status) != 1:
            raise ValueError("Example document is absent or ambiguous")
        states[kind] = status.iloc[0].status
        selected[kind] = campaign["entities"].loc[(campaign["entities"].doc_type == kind) & (campaign["entities"].doc_id == identity)]
    complete = all(state in COMPLETE for state in states.values())
    rows = []
    for label in LABELS:
        sets = {kind: set(frame.loc[frame.label == label, "surface"].str.casefold()) for kind,frame in selected.items()}
        rows.append({"label":label,"cv_state":states["cv"],"jd_state":states["jd"],
                     "cv_unique_strings":len(sets["cv"]) if complete else None,
                     "jd_unique_strings":len(sets["jd"]) if complete else None,
                     "shared_strings":len(sets["cv"] & sets["jd"]) if complete else None,
                     "cv_only_strings":len(sets["cv"] - sets["jd"]) if complete else None,
                     "jd_only_strings":len(sets["jd"] - sets["cv"]) if complete else None})
    return pd.DataFrame(rows)
