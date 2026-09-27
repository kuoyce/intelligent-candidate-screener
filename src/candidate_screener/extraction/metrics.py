"""Span-level precision / recall / F1, strict and partial, with a bootstrap CI (5.2, 5.3).

- **strict**: a predicted span counts only if a gold span has exactly its boundaries.
- **partial**: a predicted span counts if it overlaps any gold span (precision), and a
  gold span counts if any prediction overlaps it (recall).

Counts are kept per *unit* (a sentence or a document) so the CI resamples units, the
thing that was sampled, never spans. Spans are half-open `(start, end)` pairs.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Column order of a unit's count vector.
FIELDS = ("n_pred", "n_gold", "strict_tp", "partial_pred_tp", "partial_gold_tp")
N_BOOT = 10_000
CI_LEVEL = 0.95


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def unit_counts(pred: Iterable[tuple[int, int]], gold: Iterable[tuple[int, int]]) -> np.ndarray:
    """The five counts of one unit. Duplicate spans are counted once."""
    p, g = set(map(tuple, pred)), set(map(tuple, gold))
    return np.array([
        len(p), len(g), len(p & g),
        sum(any(overlaps(x, y) for y in g) for x in p),
        sum(any(overlaps(x, y) for x in p) for y in g),
    ], dtype=np.int64)


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def prf(totals: np.ndarray) -> dict[str, np.ndarray]:
    """P/R/F1 from summed counts (last axis = `FIELDS`); 0/0 is scored 0."""
    t = np.asarray(totals)
    out = {}
    for name, tp_p, tp_g in (("strict", t[..., 2], t[..., 2]),
                             ("partial", t[..., 3], t[..., 4])):
        p, r = _div(tp_p, t[..., 0]), _div(tp_g, t[..., 1])
        out[name] = np.stack([p, r, _div(2 * p * r, p + r)], axis=-1)
    return out


def score(counts: np.ndarray, *, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Point estimates and percentile-bootstrap CIs over units.

    `counts` is `(n_units, 5)`. The record carries `n_units`, `n_pred` and `n_gold`
    beside every figure, so no F1 travels without the size it was measured at.
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1, len(FIELDS))
    n = len(counts)
    point = prf(counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    chunk = max(1, 2_000_000 // max(n, 1))
    done = 0
    while done < n_boot and n:
        b = min(chunk, n_boot - done)
        sums = counts[rng.integers(0, n, size=(b, n))].sum(axis=1)
        for k, v in prf(sums).items():
            draws[k].append(v)
        done += b
    lo, hi = 100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2
    out: dict = {"n_units": int(n), "n_pred": int(counts[:, 0].sum()),
                 "n_gold": int(counts[:, 1].sum())}
    for k, v in point.items():
        d = np.concatenate(draws[k]) if draws[k] else np.zeros((1, 3))
        block = {}
        for j, name in enumerate(("precision", "recall", "f1")):
            block[name] = round(float(v[j]), 4)
            block[f"{name}_ci"] = [round(float(np.percentile(d[:, j], lo)), 4),
                                   round(float(np.percentile(d[:, j], hi)), 4)]
        out[k] = block
    return out


def paired_delta(a: np.ndarray, b: np.ndarray, *, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """F1(b) − F1(a) with a paired bootstrap: both systems see the same resampled units."""
    a = np.asarray(a, dtype=np.int64).reshape(-1, len(FIELDS))
    b = np.asarray(b, dtype=np.int64).reshape(-1, len(FIELDS))
    if len(a) != len(b):
        raise ValueError("paired delta needs the same units for both systems")
    n = len(a)
    pa, pb = prf(a.sum(axis=0)), prf(b.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in pa}
    chunk = max(1, 2_000_000 // max(n, 1))
    done = 0
    while done < n_boot and n:
        k_ = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(k_, n))
        da, db = prf(a[idx].sum(axis=1)), prf(b[idx].sum(axis=1))
        for k in draws:
            draws[k].append(db[k][:, 2] - da[k][:, 2])
        done += k_
    out: dict = {"n_units": int(n)}
    for k in pa:
        d = np.concatenate(draws[k]) if draws[k] else np.zeros(1)
        out[k] = {"f1_delta": round(float(pb[k][2] - pa[k][2]), 4),
                  "f1_delta_ci": [round(float(np.percentile(d, 2.5)), 4),
                                  round(float(np.percentile(d, 97.5)), 4)]}
    return out
