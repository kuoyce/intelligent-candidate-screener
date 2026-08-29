"""`evaluation.classification` — the report, and the floor it always carries."""
from __future__ import annotations

import pytest

from candidate_screener.evaluation import classification as cx

LABELS = cx.LABELS


def test_report_cannot_be_constructed_with_no_predictions():
    """Mirrors `metrics.Figure`: a figure without its n is not reportable."""
    with pytest.raises(ValueError, match="no predictions"):
        cx.Report("x", LABELS, 0, 0.0, 0.0, {}, ((0,),), cx.Floor("No Fit", 0.0, 0.0))


def test_every_report_carries_its_floor():
    """The floor travels *with* the report — it is not a caller's responsibility.

    Accuracy below the majority floor is the single most misreadable number this
    baseline produces, so it cannot be reported without the floor beside it.
    """
    rep = cx.report("m", ["No Fit"] * 6 + ["Good Fit"] * 4, ["No Fit"] * 10)
    assert rep.floor.majority_class == "No Fit"
    assert rep.floor.accuracy == pytest.approx(0.6)


def test_macro_f1_matches_a_hand_computed_example():
    """Three classes, two rows each. Both Potential Fit rows are predicted No Fit.

    Good Fit:      TP=2 FP=0 FN=0 -> P=1,   R=1   -> F1 = 1.0
    Potential Fit: TP=0, never predicted -> P=0,   R=0   -> F1 = 0.0
    No Fit:        TP=2 FP=2 FN=0 -> P=0.5, R=1   -> F1 = 2/3
    macro-F1 = (1 + 0 + 2/3) / 3 = 5/9 = 0.5555…

    The zero-division convention is the part worth pinning: a class that is never
    predicted scores 0.0 rather than raising, so macro-F1 is *penalised* by the
    collapse instead of silently averaging over two classes.
    """
    y_true = ["Good Fit", "Good Fit", "Potential Fit", "Potential Fit", "No Fit", "No Fit"]
    y_pred = ["Good Fit", "Good Fit", "No Fit", "No Fit", "No Fit", "No Fit"]
    rep = cx.report("m", y_true, y_pred)
    assert rep.accuracy == pytest.approx(4 / 6)
    assert rep.macro_f1 == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
    assert rep.per_class["Potential Fit"]["f1"] == 0.0


def test_confusion_keeps_a_fixed_shape_when_a_class_is_never_predicted():
    """A collapsed model must show as a column of zeros, not as a smaller matrix.

    Otherwise the golden diff reports a shape error instead of the collapse.
    """
    rep = cx.report("m", ["Good Fit", "Potential Fit", "No Fit"], ["No Fit"] * 3)
    assert len(rep.confusion) == 3 and all(len(row) == 3 for row in rep.confusion)
    assert rep.labels == LABELS


def test_a_model_that_only_predicts_the_majority_does_not_beat_the_floor():
    y_true = ["No Fit"] * 6 + ["Good Fit"] * 4
    rep = cx.report("m", y_true, ["No Fit"] * 10)
    assert not rep.beats_floor_on_accuracy and not rep.beats_floor_on_macro_f1


def test_a_label_outside_the_scheme_raises():
    """Otherwise accuracy counts a row that the confusion matrix and floor do not.

    Three of the four reported figures would then describe a different set of rows
    than the fourth, and nothing would say so.
    """
    with pytest.raises(ValueError, match="not in"):
        cx.report("m", ["Good Fit", "Excellent Fit"], ["No Fit", "No Fit"])
    with pytest.raises(ValueError, match="not in"):
        cx.report("m", ["Good Fit", "No Fit"], ["No Fit", "Maybe"])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="predictions"):
        cx.report("m", ["No Fit"] * 3, ["No Fit"] * 2)


def test_floor_ties_break_deterministically():
    """Equal counts must not make the floor depend on input order."""
    a = cx.majority_floor(["Good Fit", "No Fit"])
    b = cx.majority_floor(["No Fit", "Good Fit"])
    assert a.majority_class == b.majority_class == "Good Fit"
