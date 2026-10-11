"""The one interface every ladder rung implements, and the card it reports itself with.

A rung scores `(JD, CV)` pairs: higher means a better fit. It never sees a label from the
final set (D46). Whatever it trains on, it loads through `fit`.
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd

#: Columns a pair frame carries. A2 fills all of them; A1 has no metadata (NaN).
PAIR_COLUMNS = ("pair_id", "jd_id", "cv_id", "jd_text", "cv_text",
                "jd_keyword", "cv_keyword", "jd_exp_band", "cv_exp_band")


class System(Protocol):
    name: str
    rung: str

    def fit(self, train: pd.DataFrame | None = None) -> None: ...
    def score(self, pairs: pd.DataFrame) -> pd.Series: ...
    def card(self) -> dict: ...


class BaseSystem:
    """Timing and the card, so a rung only writes `_fit` and `_score`."""

    name = "base"
    rung = "?"
    stage = ""
    trained_on = "nothing"
    can_explain = False
    deterministic = True

    def __init__(self) -> None:
        self.fit_seconds = 0.0
        self.score_seconds = 0.0
        self.scored_pairs = 0

    def _fit(self, train: pd.DataFrame | None) -> None:
        """Override for rungs that learn something."""

    def _score(self, pairs: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame | None = None) -> None:
        start = time.perf_counter()
        self._fit(train)
        self.fit_seconds = time.perf_counter() - start

    def score(self, pairs: pd.DataFrame) -> pd.Series:
        start = time.perf_counter()
        scores = self._score(pairs)
        self.score_seconds += time.perf_counter() - start
        self.scored_pairs += len(pairs)
        # Values, not a Series: `pd.Series(series, index=...)` re-aligns on the old index
        # and fills NaN wherever the labels differ — a silent reorder, caught once here.
        values = np.asarray(scores.to_numpy() if hasattr(scores, "to_numpy") else scores,
                            dtype=float)
        if len(values) != len(pairs):
            raise ValueError(f"{self.name} returned {len(values)} scores for {len(pairs)} pairs")
        scores = pd.Series(values, index=pairs.pair_id.to_numpy(), name=self.name)
        if scores.isna().any():
            raise ValueError(f"{self.name} returned NaN scores — a missing score would "
                             "rank arbitrarily and silently")
        return scores

    def params(self) -> int:
        return 0

    def card(self) -> dict:
        per_pair = 1000 * self.score_seconds / self.scored_pairs if self.scored_pairs else None
        return {"name": self.name, "rung": self.rung, "stage": self.stage,
                "trained_on": self.trained_on, "params": self.params(),
                "can_explain": self.can_explain, "deterministic": self.deterministic,
                "fit_seconds": round(self.fit_seconds, 3),
                "ms_per_pair": round(per_pair, 4) if per_pair is not None else None,
                "ms_per_500_candidate_role": round(500 * per_pair, 1) if per_pair else None}
