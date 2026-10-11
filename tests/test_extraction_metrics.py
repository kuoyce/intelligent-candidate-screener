"""`extraction.metrics`: strict and partial span scores on a hand-built example."""
from __future__ import annotations

import numpy as np
import pytest

from candidate_screener.extraction.metrics import prf, score, unit_counts

GOLD = [(0, 2), (5, 7), (10, 12)]
PRED = [(0, 2), (5, 6), (20, 22)]


def test_unit_counts_by_hand():
    # strict: only (0,2) matches exactly. partial: (0,2),(5,6) touch gold; gold (0,2),(5,7)
    # are touched.
    assert unit_counts(PRED, GOLD).tolist() == [3, 3, 1, 2, 2]


def test_strict_and_partial_prf_by_hand():
    m = prf(unit_counts(PRED, GOLD))
    assert m["strict"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert m["partial"] == pytest.approx([2 / 3, 2 / 3, 2 / 3])


def test_partial_precision_and_recall_count_different_sides():
    """One long prediction covering two gold spans: precision 1/1, recall 2/2."""
    m = prf(unit_counts([(0, 10)], [(0, 3), (5, 9)]))
    assert m["partial"] == pytest.approx([1.0, 1.0, 1.0])
    assert m["strict"] == pytest.approx([0.0, 0.0, 0.0])


def test_touching_is_not_overlapping():
    assert unit_counts([(0, 5)], [(5, 9)]).tolist() == [1, 1, 0, 0, 0]


def test_zero_over_zero_is_zero_not_nan():
    m = prf(unit_counts([], []))
    assert not np.isnan(m["strict"]).any() and m["strict"].tolist() == [0, 0, 0]


def test_duplicate_spans_count_once():
    assert unit_counts([(0, 2), (0, 2)], [(0, 2)]).tolist() == [1, 1, 1, 1, 1]


def test_score_sums_units_and_carries_n():
    counts = np.array([unit_counts(PRED, GOLD), unit_counts([(1, 3)], [(1, 3)])])
    s = score(counts, n_boot=2000)
    assert (s["n_units"], s["n_pred"], s["n_gold"]) == (2, 4, 4)
    assert s["strict"]["f1"] == pytest.approx(0.5)
    lo, hi = s["strict"]["f1_ci"]
    assert lo <= s["strict"]["f1"] <= hi


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    counts = np.array([unit_counts(
        [(i, i + 2) for i in rng.integers(0, 20, 3)], [(i, i + 2) for i in rng.integers(0, 20, 3)])
        for _ in range(40)])
    assert score(counts, seed=0, n_boot=1000) == score(counts, seed=0, n_boot=1000)
    assert score(counts, seed=0, n_boot=1000) != score(counts, seed=1, n_boot=1000)


def test_paired_delta_is_zero_for_identical_systems_and_signed_otherwise():
    from candidate_screener.extraction.metrics import paired_delta
    a = np.array([unit_counts(PRED, GOLD)] * 10)
    b = np.array([unit_counts(GOLD, GOLD)] * 10)
    same = paired_delta(a, a, n_boot=500)
    assert same["strict"]["f1_delta"] == 0 and same["strict"]["f1_delta_ci"] == [0.0, 0.0]
    better = paired_delta(a, b, n_boot=500)
    assert better["strict"]["f1_delta"] == pytest.approx(2 / 3, abs=1e-4)
    with pytest.raises(ValueError):
        paired_delta(a, b[:3])
