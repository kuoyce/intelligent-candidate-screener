"""The stage-2 interface (plan 2026-10-06, 02 §4): one JD, its CVs, one score per CV.

Fixed beside `matching.retriever.Retriever`, so the two stages develop independently.
Higher means a better fit, and scores are comparable within one JD only. Any fitted
ladder rung becomes a reranker through `SystemReranker`; the trained rerankers
(`rerank.models`) are themselves `BaseSystem`s, so the ladder tools take them unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from candidate_screener.matching.base import BaseSystem


class Reranker(Protocol):
    def score(self, jd: str, cvs: Sequence[str]) -> list[float]: ...


class SystemReranker:
    """A fitted `BaseSystem` as a `Reranker`. The pair frame carries text only, so a rung
    that reads a metadata field fails here rather than reading NaN."""

    def __init__(self, system: BaseSystem) -> None:
        self.system = system
        self.name = system.name

    def score(self, jd: str, cvs: Sequence[str]) -> list[float]:
        n = len(cvs)
        if not n:
            return []
        pairs = pd.DataFrame({"pair_id": [f"q__{i}" for i in range(n)], "jd_id": ["q"] * n,
                              "cv_id": [str(i) for i in range(n)], "jd_text": [jd] * n,
                              "cv_text": list(cvs)})
        return [float(x) for x in self.system.score(pairs).to_numpy()]
