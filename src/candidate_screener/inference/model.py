"""L1 frozen: the reported system's stage-2 model as numbers (plan 2026-10-23-inference-api).

L1 is `rerank.learners.BoundedRidge` on F12, fitted on every `chat-3` pair (D62, D63). The
evaluation code refits it from chat-3 every time it is used (`chat3.fit_stage2`); `freeze()` fits
it once, exactly as `chat3.l1_pool` does, and writes the coefficients to `RECORD`. Inference then
reads the record and never refits. The record is an artifact of an existing fit, not new training.

Features are computed on the **full 500-CV chat-3 pools** before the chat-3 pairs are selected,
because `tfidf_scaled` is min-max scaled within the pool.

    uv run python -m candidate_screener.inference --freeze
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROJECT_ROOT

RECORD = OUTPUT / "rerank" / "l1-chat3-model.json"
PLAN = "plan/2026-10-23-inference-api"


@dataclass(frozen=True)
class L1Model:
    """`intercept + ((x - mu) / sd) @ coef`, the maths of `BoundedRidge.predict`."""

    columns: tuple[str, ...]
    mu: tuple[float, ...]
    sd: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    alpha: float
    meta: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_ridge(cls, ridge, meta: dict | None = None) -> "L1Model":
        return cls(tuple(ridge.columns), tuple(map(float, ridge.mu)), tuple(map(float, ridge.sd)),
                   tuple(map(float, ridge.coef)), float(ridge.intercept), float(ridge.alpha),
                   dict(meta or {}))

    @classmethod
    def load(cls, path: Path = RECORD) -> "L1Model":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["columns"]), tuple(d["mu"]), tuple(d["sd"]), tuple(d["coef"]),
                   d["intercept"], d["alpha"], d.get("meta", {}))

    def save(self, path: Path = RECORD) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"columns": list(self.columns), "mu": list(self.mu), "sd": list(self.sd),
               "coef": list(self.coef), "intercept": self.intercept, "alpha": self.alpha,
               "meta": self.meta}
        # json writes floats with repr, so the round trip is exact.
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return (X[list(self.columns)].to_numpy(dtype=float) - np.asarray(self.mu)) / np.asarray(self.sd)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self._z(X) @ np.asarray(self.coef)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Each column's exact share of `predict - intercept`, row-aligned with `X`."""
        return pd.DataFrame(self._z(X) * np.asarray(self.coef), index=X.index, columns=list(self.columns))


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def freeze(path: Path = RECORD) -> L1Model:
    """Fit chat-3's winner once, as `chat3.l1_pool` does; write and verify the record."""
    from candidate_screener.evaluation import chat3
    from candidate_screener.matching import learned
    from candidate_screener.rerank import learners
    win = json.loads(chat3.SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    if win != {"learner": "L1", "features": "F12"}:
        raise ValueError(f"chat-3's winner is {win}, not L1 on F12")
    chat = chat3.targets()
    X = learned.Features()(chat3.pairs())
    ridge = chat3.fit_stage2(chat, X, win["learner"], win["features"])
    meta = {"learner": learners.L1, "features": win["features"], "trained_on": "chat-3 (D62)",
            "pairs": int(len(chat)), "jds": int(chat.jd_id.nunique()),
            "train_pair_ids_sha256": chat3._sha(chat.pair_id), "select_record": "output/rerank/chat3-select.json",
            "features_on": "the full 500-CV chat-3 pools (tfidf_scaled min-max within the pool)",
            "commit": _commit(), "plan": PLAN}
    model = L1Model.from_ridge(ridge, meta)
    model.save(path)
    back = L1Model.load(path)
    Xc = X.loc[chat.pair_id.to_numpy()]
    gap = float(np.max(np.abs(back.predict(Xc) - ridge.predict(Xc))))
    if gap > 1e-9:
        raise ValueError(f"the frozen L1 differs from the fit by {gap}")
    print(f"L1 frozen -> {path} ({len(chat)} pairs, max |frozen - fit| = {gap:.3g})")
    return back
