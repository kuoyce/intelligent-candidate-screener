"""Bootstrap over queries (JDs), never over pairs.

A JD's ten candidates share a reader, a posting and a pool, so they are not independent
observations. Resampling pairs would give intervals several times too narrow. Every
interval in the in-domain evaluation comes from here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

N_BOOT = 10_000
ALPHA = 0.05


def group_indices(groups: pd.Series) -> list[np.ndarray]:
    """Row positions of each group, in sorted group order (so a seed means one thing)."""
    codes, uniques = pd.factorize(groups, sort=True)
    return [np.flatnonzero(codes == k) for k in range(len(uniques))]


def pair_stat_ci(frame: pd.DataFrame, stat: Callable[..., float], columns: Sequence[str],
                 group: str = "jd_id", n_boot: int = N_BOOT, seed: int = 0
                 ) -> tuple[float, float]:
    """Percentile CI of a pair-level statistic, resampling whole groups with replacement.

    `stat` receives one numpy array per name in `columns`, row-aligned, so the inner loop
    never touches pandas: 10,000 resamples take milliseconds, not minutes.
    """
    arrays = [frame[c].to_numpy() for c in columns]
    blocks = group_indices(frame[group])
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        values[b] = stat(*(arr[rows] for arr in arrays))
    low, high = np.nanpercentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def mean_ci(per_group: pd.Series, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the mean of one value per group (e.g. per-JD AUC)."""
    values = per_group.sort_index().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def binary_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def binary_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two 0/1 arrays; NaN when chance agreement is total."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")
