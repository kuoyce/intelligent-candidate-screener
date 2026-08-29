"""Classification figures for the A1 label task — with the floor attached *(D19)*.

`metrics.py` enforces "no retrieval figure without its *n*". This module enforces the
same discipline for classification, against the number that is by far the most
misreadable thing this baseline produces: **its accuracy is below the majority-class
floor.**

A1's test split is 52.2% `No Fit`, so "always predict No Fit" scores 0.522 accuracy
and ~0.229 macro-F1. Both baselines beat that on macro-F1 and neither beats it on
accuracy. That is the expected behaviour of one linear threshold over three
overlapping, imbalanced classes — not a defect — but a reader who meets `0.4917`
without the floor beside it will reopen it as a bug every time.

So the floor is not something a caller computes if they remember to. It is a field
of `Report`, constructed with it, printed with it, and written into
`baseline-metrics.json` beside it. And, as in `metrics.Figure`, a report over zero
predictions cannot be constructed at all.

Scoring lives here rather than in `baselines/` because scoring a model and *being* a
model are different lifetimes *(Phase 3, deviation W11)*.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

#: A1's three classes, in the reading order used everywhere in this project.
LABELS = ("Good Fit", "Potential Fit", "No Fit")


@dataclass(frozen=True)
class Floor:
    """What "always predict the commonest class" scores on the same truth."""
    majority_class: str
    accuracy: float
    macro_f1: float

    def __str__(self) -> str:
        return (f'floor ("always {self.majority_class}")  '
                f"accuracy {self.accuracy:.4f}  macro-F1 {self.macro_f1:.4f}")


@dataclass(frozen=True)
class Report:
    """One model's figures on one split. Cannot exist without its *n* or its floor."""
    model: str
    labels: tuple[str, ...]
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: tuple[tuple[int, ...], ...]
    floor: Floor

    def __post_init__(self) -> None:
        if not self.n:
            raise ValueError(
                f"{self.model} has no predictions — a classification figure without "
                "its n is not reportable. Check that the split loaded.")

    @property
    def beats_floor_on_accuracy(self) -> bool:
        return self.accuracy > self.floor.accuracy

    @property
    def beats_floor_on_macro_f1(self) -> bool:
        return self.macro_f1 > self.floor.macro_f1

    def __str__(self) -> str:
        verdict = (
            "above the floor on both" if self.beats_floor_on_accuracy and self.beats_floor_on_macro_f1
            else "above the floor on macro-F1, below it on accuracy — expected of one "
                 "linear threshold over overlapping imbalanced classes"
            if self.beats_floor_on_macro_f1
            else "at or below the floor on macro-F1 — this model is not learning")
        return (f"{self.model:<8} accuracy {self.accuracy:.4f}  macro-F1 {self.macro_f1:.4f}  "
                f"n={self.n}\n         {self.floor}\n         {verdict}")

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n": self.n,
            "per_class": self.per_class,
            "confusion": [list(row) for row in self.confusion],
        }


def majority_floor(y_true: Sequence[str], labels: Sequence[str] = LABELS) -> Floor:
    """The trivial baseline: predict the commonest true class for every row.

    Ties break on `labels` order, which is fixed, so the floor is deterministic.
    """
    y_true = np.asarray(y_true)
    counts = {label: int((y_true == label).sum()) for label in labels}
    majority = max(labels, key=lambda label: counts[label])
    y_pred = np.full(len(y_true), majority)
    return Floor(
        majority_class=majority,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro",
                                labels=list(labels), zero_division=0)),
    )


def report(model: str, y_true: Sequence[str], y_pred: Sequence[str],
           labels: Sequence[str] = LABELS) -> Report:
    """Score one model's predictions, floor included.

    `labels` is passed explicitly to every sklearn call so the confusion matrix and
    the per-class table keep a fixed row order even when a degenerate model never
    predicts one of the three classes — otherwise the committed matrix would silently
    change shape and the golden diff would report a shape error instead of the
    collapse that caused it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{model}: {len(y_true)} true labels against "
                         f"{len(y_pred)} predictions")
    labels = tuple(labels)
    # A label outside `labels` would be counted by accuracy but excluded from the
    # confusion matrix, the per-class table and the floor — three figures quietly
    # describing a different set of rows than the fourth.
    unknown = sorted((set(y_true.tolist()) | set(y_pred.tolist())) - set(labels))
    if unknown:
        raise ValueError(
            f"{model}: label(s) {unknown} are not in {list(labels)}. Accuracy would "
            "count them while the confusion matrix and the floor would not.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    return Report(
        model=model,
        labels=labels,
        n=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro",
                                labels=list(labels), zero_division=0)),
        per_class={label: {"precision": float(p), "recall": float(r),
                           "f1": float(f), "support": int(s)}
                   for label, p, r, f, s in zip(labels, precision, recall, f1, support)},
        confusion=tuple(tuple(int(v) for v in row) for row in
                        confusion_matrix(y_true, y_pred, labels=list(labels))),
        floor=majority_floor(y_true, labels),
    )
