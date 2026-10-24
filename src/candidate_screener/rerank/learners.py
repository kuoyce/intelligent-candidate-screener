"""L1 and L2: stage-2 learners on F12, fitted on `chat-1` scores (plan 2026-10-18, step 3).

- **L1 linear:** ridge regression (alpha 1, on features standardised in the training fold), with
  the match signals in `CONSTRAINED` held to non-negative coefficients; the descriptors in
  `FREE` take any sign.
- **L2 boosting:** `HistGradientBoostingRegressor` at its defaults (random_state 0), with a
  monotonic increasing constraint on `CONSTRAINED`.

The split (owner, 18 Oct 2026): a higher match signal never lowers the score; concept counts and
the experience descriptors are controls whose sign the data sets. Nothing is tuned.

Nothing here reads a human label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONSTRAINED = ("role_match", "experience_compatible", "tfidf_scaled", "coverage",
               "coverage_unweighted", "log_matched", "skill_overlap", "sbert")
FREE = ("log_jd_concepts", "log_cv_concepts", "exp_gap", "exp_known")
L1_ALPHA = 1.0
L1, L2 = "L1 linear (ridge, non-negative match signals)", "L2 boosting (monotone match signals)"


class BoundedRidge:
    """Ridge on standardised columns; coefficients of `CONSTRAINED` columns bounded below by 0.

    Solved exactly as bounded least squares on the augmented system [X; sqrt(alpha) I], so with
    no bound active it is sklearn's `Ridge(alpha)` on the same standardised columns."""

    def __init__(self, columns, alpha: float = L1_ALPHA) -> None:
        self.columns, self.alpha = list(columns), alpha

    def fit(self, X: pd.DataFrame, y) -> "BoundedRidge":
        from scipy.optimize import lsq_linear
        A = X[self.columns].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu, sd = A.mean(0), A.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (A - self.mu) / self.sd
        self.intercept = float(y.mean())
        k = len(self.columns)
        lower = np.array([0.0 if c in CONSTRAINED else -np.inf for c in self.columns])
        res = lsq_linear(np.vstack([Z, np.sqrt(self.alpha) * np.eye(k)]),
                         np.concatenate([y - self.intercept, np.zeros(k)]),
                         bounds=(lower, np.full(k, np.inf)), method="bvls")
        if not res.success:
            raise RuntimeError(f"bounded ridge did not converge: {res.message}")
        self.coef = res.x
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.columns].to_numpy(dtype=float) - self.mu) / self.sd
        return self.intercept + Z @ self.coef

    def coefficients(self) -> dict:
        """Standardised coefficients by column."""
        return {c: float(v) for c, v in zip(self.columns, self.coef)}


class MonotoneBoosting:
    """`HistGradientBoostingRegressor` defaults, increasing in every `CONSTRAINED` column."""

    def __init__(self, columns) -> None:
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y) -> "MonotoneBoosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        cst = [1 if c in CONSTRAINED else 0 for c in self.columns]
        self.model = HistGradientBoostingRegressor(monotonic_cst=cst, random_state=0)
        self.model.fit(X[self.columns].to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns].to_numpy(dtype=float))


LEARNERS = {L1: BoundedRidge, L2: MonotoneBoosting}


def out_of_fold(kind: str, X: pd.DataFrame, chat: pd.DataFrame, columns,
                train_source: str = "x2_top100") -> tuple[pd.Series, list]:
    """Per fold, fit `kind` on the other folds' `train_source` rows and predict this fold's rows.

    `X` is indexed by pair id; `chat` carries `pair_id`, `fold`, `score`, `source`. Returns the
    out-of-fold scores (by pair id, `chat` order) and the fitted models, by fold."""
    out, models = pd.Series(np.nan, index=chat.pair_id.to_numpy()), []
    for k in sorted(chat.fold.unique()):
        tr = chat[(chat.fold != k) & (chat.source == train_source)]
        te = chat[chat.fold == k]
        m = LEARNERS[kind](columns).fit(X.loc[tr.pair_id.to_numpy()], tr.score.to_numpy())
        out.loc[te.pair_id.to_numpy()] = m.predict(X.loc[te.pair_id.to_numpy()])
        models.append(m)
    if out.isna().any():
        raise ValueError("a pair has no out-of-fold score")
    return out, models
