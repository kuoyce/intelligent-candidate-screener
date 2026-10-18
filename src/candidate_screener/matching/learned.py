"""Stage-1 classifier (plan 2026-10-18-classifier-then-rerank, step 1; D61).

A logistic regression on text-only features, fitted on the D50 batch's Fit / No Fit labels
(run 11, Fit = Good Fit or Potential Fit; D51 as amended by D61). Its stage-1 output is the
set of CVs it calls Fit: probability >= `THRESHOLD` (0.5, the untuned default). Nothing is
tuned; the fit is `CLASSIFIER_CONFIG` (lbfgs, balanced classes), as notebook 19 part 4 fitted it.

Two feature sets:

- **F12** = the 11 terms of `a1_weights.ALL_FEATURES` plus `sbert`, SBERT weak's raw cosine
  (X4's text model, chunk-mean embeddings; not min-max scaled);
- **base 4** = `a1_weights.FEATURES`, the notebook 19 reproduction.

Every feature reads JD text and CV text only (D53). `tfidf_scaled` is min-max scaled within
each JD of the frame it is computed on: over 10 CVs on D50 and on the final set, over 500 on
a pool, exactly as X2's own text term is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from candidate_screener.baselines.classifier import CLASSIFIER_CONFIG
from candidate_screener.evaluation import a1_weights as aw
from candidate_screener.matching.base import BaseSystem

TEXT_VARIANT = "v2"
THRESHOLD = 0.5
F12 = aw.ALL_FEATURES + ("sbert",)
BASE4 = aw.FEATURES
FEATURE_SETS = {"F12": F12, "base 4": BASE4}
TRAINED_ON = ("D50 batch, run 11 (claude-haiku-4-5, instrument 5d986c82): 1,000 pairs over "
              "100 A2-train JDs, Fit = Good Fit or Potential Fit (D61)")


class Features:
    """F12 for any pair frame (`pair_id`, `jd_id`, `jd_text`, `cv_text`), indexed by `pair_id`.

    One instance holds the fitted parts (X2's rule and TF-IDF, the concept IDF, SBERT weak), so
    the frames of a run share them. Built lazily: constructing it loads nothing."""

    def __init__(self) -> None:
        self.scorer = None
        self.sbert = None

    def _load(self) -> None:
        if self.scorer is None:
            from candidate_screener.matching import sbert
            self.scorer = aw.build_scorer()
            self.sbert = sbert.SbertWeak(TEXT_VARIANT)
            self.sbert.fit(None)

    def __call__(self, pairs: pd.DataFrame, columns=F12) -> pd.DataFrame:
        self._load()
        x = aw.features(self.scorer, pairs)
        if set(columns) - set(aw.FEATURES):
            x = x.join(aw.extra_features(self.scorer, pairs))
        if "sbert" in columns:
            s = self.sbert.score(pairs[["pair_id", "jd_id", "cv_id", "jd_text", "cv_text"]])
            x["sbert"] = s.loc[x.index].to_numpy()
        return x[list(columns)]


class LearnedStage1(BaseSystem):
    """S1: logistic regression on `columns`, fitted on D50 unless `fit_features` is called."""

    rung, stage, can_explain = "S1", "Stage 1 — learned classifier", True
    trained_on = TRAINED_ON

    def __init__(self, set_name: str = "F12", features: Features | None = None) -> None:
        super().__init__()
        self.set_name, self.columns = set_name, FEATURE_SETS[set_name]
        self.features = features or Features()
        self.name = f"S1 logistic {set_name} (D50)"
        self.model: LogisticRegression | None = None
        self.train_sd: pd.Series | None = None

    def fit_features(self, X: pd.DataFrame, y: pd.Series) -> "LearnedStage1":
        """Fit on a ready feature frame; `y` is 1 (Fit) / 0 (No Fit), aligned by index."""
        cols = list(self.columns)
        clf = LogisticRegression(**CLASSIFIER_CONFIG)
        clf.fit(X[cols].to_numpy(), y.loc[X.index].to_numpy())
        if int(np.max(clf.n_iter_)) >= CLASSIFIER_CONFIG["max_iter"]:
            raise RuntimeError("lbfgs hit max_iter: the fit is a function of the cap, not of the data")
        self.model, self.train_sd = clf, X[cols].std(ddof=0)
        return self

    def _fit(self, train):
        if train is not None:
            raise ValueError("S1 fits on the D50 batch only; use fit_features for other data")
        pairs, y = aw.d50_pairs()
        self.fit_features(self.features(pairs, self.columns), y)

    def proba(self, X: pd.DataFrame) -> pd.Series:
        """P(Fit) for a ready feature frame."""
        if self.model is None:
            raise ValueError("S1 is not fitted")
        p = self.model.predict_proba(X[list(self.columns)].to_numpy())[:, 1]
        return pd.Series(p, index=X.index)

    def passes(self, X: pd.DataFrame) -> pd.Series:
        """The stage-1 output: True where P(Fit) >= `THRESHOLD`."""
        return self.proba(X) >= THRESHOLD

    def _score(self, pairs):
        return self.proba(self.features(pairs, self.columns)).to_numpy()

    def coefficients(self) -> dict:
        """Raw coefficients, and standardised ones (raw x the feature's SD on the training set)."""
        raw = pd.Series(self.model.coef_[0], index=list(self.columns))
        return {"intercept": float(self.model.intercept_[0]),
                "raw": {k: float(v) for k, v in raw.items()},
                "standardised": {k: float(v) for k, v in (raw * self.train_sd).items()},
                "train_sd": {k: float(v) for k, v in self.train_sd.items()},
                "n_iter": int(np.max(self.model.n_iter_))}

    def params(self) -> int:
        return len(self.columns) + 1
