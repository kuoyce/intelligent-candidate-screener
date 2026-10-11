"""The stage-1 interface (plan 2026-10-06, 02 §4): one JD, a pool of CV texts, a shortlist.

Fixed before either stage is built, so retrieval and reranking develop independently.
Any ladder rung becomes a retriever through `SystemRetriever`: it scores the JD against
every pool CV and returns the indices of the top `k`. Ties keep pool order, so a
shortlist never depends on sort stability.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from candidate_screener.matching.base import BaseSystem


class Retriever(Protocol):
    def shortlist(self, jd: str, pool: Sequence[str], k: int) -> list[int]: ...


class SystemRetriever:
    """A fitted `BaseSystem` as a `Retriever`. The pair frame it builds carries text only,
    so a rung that reads a metadata field fails here rather than reading NaN."""

    def __init__(self, system: BaseSystem) -> None:
        self.system = system
        self.name = system.name

    def scores(self, jd: str, pool: Sequence[str]) -> np.ndarray:
        n = len(pool)
        pairs = pd.DataFrame({"pair_id": [f"q__{i}" for i in range(n)], "jd_id": ["q"] * n,
                              "cv_id": [str(i) for i in range(n)], "jd_text": [jd] * n,
                              "cv_text": list(pool)})
        return self.system.score(pairs).to_numpy()

    def shortlist(self, jd: str, pool: Sequence[str], k: int) -> list[int]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if not len(pool):
            return []
        s = self.scores(jd, pool)
        order = np.lexsort((np.arange(len(s)), -s))            # score desc, then pool order
        return [int(i) for i in order[:k]]
