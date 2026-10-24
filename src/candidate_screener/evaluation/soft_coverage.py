"""Soft-matched coverage columns (plan 2026-10-13-soft-matching-in-the-score).

For a pair, each JD concept j has s(j) = its best cosine to any CV concept (0 if the CV has none)
and counts as matched when s(j) >= t. The three exact columns of the 11-feature set are replaced:

    coverage             weight-matched share (weights as W0: IDF, unknown phrases flat 1)
    coverage_unweighted  matched share
    log_matched          log(1 + #matched)

Text and concept sets only; no label is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ("coverage", "coverage_unweighted", "log_matched")


class SoftSims:
    """Best-cosine vectors and JD-concept weights for one pair frame, computed once."""

    def __init__(self, pairs: pd.DataFrame, extractor, freq, vec: dict):
        jd = {t: sorted(extractor.concepts(t, "jd")) for t in pairs.jd_text.unique()}
        cv = {t: sorted(extractor.concepts(t, "cv")) for t in pairs.cv_text.unique()}
        w = {t: np.array([freq.weight(j) for j in js]) for t, js in jd.items()}
        self.index = pairs.pair_id.to_numpy()
        self.weights, self.sims = [], []
        for jt, ct in zip(pairs.jd_text, pairs.cv_text):
            J, C = jd[jt], cv[ct]
            self.weights.append(w[jt])
            if not J or not C:
                self.sims.append(np.zeros(len(J)))
            else:
                self.sims.append((np.array([vec[j] for j in J]) @ np.array([vec[c] for c in C]).T).max(axis=1))

    def columns(self, t: float) -> pd.DataFrame:
        rows = []
        for w, s in zip(self.weights, self.sims):
            hit = s >= t
            rows.append((float(w[hit].sum() / w.sum()) if len(w) else 0.0,
                         float(hit.mean()) if len(w) else 0.0, float(np.log1p(hit.sum()))))
        return pd.DataFrame(rows, columns=COLUMNS, index=self.index)


def replace_columns(X: pd.DataFrame, soft: pd.DataFrame) -> pd.DataFrame:
    """A copy of the feature frame with the three coverage columns taken from `soft`."""
    out = X.copy()
    for c in COLUMNS:
        if c in out:
            out[c] = soft[c].reindex(out.index)
    return out
