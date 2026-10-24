"""Stage-1 classifier (plan 2026-10-18-classifier-then-rerank, step 1; D61).

A logistic regression on text-only features, fitted on the D50 batch's Fit / No Fit labels
(run 11, Fit = Good Fit or Potential Fit; D51 as amended by D61). Its stage-1 output is the
set of CVs it calls Fit: probability >= `THRESHOLD` (0.5, the untuned default). Nothing is
tuned; the fit is `CLASSIFIER_CONFIG` (lbfgs, balanced classes), as notebook 19 part 4 fitted it.

Two feature sets:

- **F12** = the 11 terms of `a1_weights.ALL_FEATURES` plus `sbert`, SBERT weak's raw cosine
  (X4's text model, chunk-mean embeddings; not min-max scaled);
- **base 4** = `a1_weights.FEATURES`, the notebook 19 reproduction.
- **F9** (step 4b) = F12 without `coverage_unweighted`, `skill_overlap` and `log_matched`, which
  are near-copies of `coverage` (Spearman 0.95-1.00 on D50; VIFs 130-177).

Step 4b's extra columns (`EXTRA`): `title_role_match` (N1: Z1L's role term, the JD title's family
on a lookup hit, else the JD text's; the title is allowed, D54), `no_experience` (N3/F2: the CV
states no professional experience and the JD asks for years), `sbert_zs` (N4: zero-shot SBERT
cosine). `Features(strip_negated=True)` is N3/F1: negated clauses leave the CV text before any
feature is computed. `Features(concepts="R+N")` is N2: the concept terms come from the seed rules
united with the MiniLM tagger's cached spans (`extraction.concept_sources`), which must already
hold every text (`feature_arms --tag-n2`).

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
F9 = tuple(c for c in F12 if c not in ("coverage_unweighted", "skill_overlap", "log_matched"))
EXTRA = ("title_role_match", "no_experience", "sbert_zs")
FEATURE_SETS = {"F12": F12, "base 4": BASE4, "F9": F9}
TRAINED_ON = ("D50 batch, run 11 (claude-haiku-4-5, instrument 5d986c82): 1,000 pairs over "
              "100 A2-train JDs, Fit = Good Fit or Potential Fit (D61)")


class Features:
    """F12 for any pair frame (`pair_id`, `jd_id`, `jd_text`, `cv_text`), indexed by `pair_id`.

    One instance holds the fitted parts (X2's rule and TF-IDF, the concept IDF, SBERT weak), so
    the frames of a run share them. Built lazily: constructing it loads nothing."""

    def __init__(self, strip_negated: bool = False, concepts: str = "R") -> None:
        if concepts not in ("R", "R+N"):
            raise ValueError("concepts must be 'R' or 'R+N'")
        self.strip_negated, self.concepts = strip_negated, concepts
        self.scorer = self.sbert = self.sbert_zs = self.title_rule = None

    def _load(self) -> None:
        if self.scorer is None:
            from candidate_screener.matching import sbert
            extractor = None
            if self.concepts == "R+N":
                from candidate_screener.extraction import concept_sources as cs

                def untagged(text, side):
                    raise KeyError("a text has no cached MiniLM spans: run feature_arms --tag-n2 first")
                extractor = cs.make_extractors({"minilm": cs.CachedEntities("minilm", untagged)})["R+N"]
            self.scorer = aw.build_scorer(extractor=extractor)
            self.sbert = sbert.SbertWeak(TEXT_VARIANT)
            self.sbert.fit(None)

    def __call__(self, pairs: pd.DataFrame, columns=F12, reference=None) -> pd.DataFrame:
        """`reference`: "present" in CV date ranges (None = `experience.REFERENCE_DATE`)."""
        self._load()
        if self.strip_negated:
            from candidate_screener.evaluation.score_defects import strip_negated
            pairs = pairs.assign(cv_text=pairs.cv_text.map(strip_negated))
        rule, before = self.scorer.rule, self.scorer.rule.reference
        reference = before if reference is None else reference
        rule.reference = reference
        try:
            x = aw.features(self.scorer, pairs)
        finally:
            rule.reference = before
        if set(columns) - set(aw.FEATURES):
            x = x.join(aw.extra_features(self.scorer, pairs, reference=reference))
        text = pairs[["pair_id", "jd_id", "cv_id", "jd_text", "cv_text"]]
        if "sbert" in columns:
            x["sbert"] = self.sbert.score(text).loc[x.index].to_numpy()
        if "sbert_zs" in columns:
            if self.sbert_zs is None:
                from candidate_screener.matching import sbert
                self.sbert_zs = sbert.SbertZeroShot(TEXT_VARIANT)
                self.sbert_zs.fit(None)
            x["sbert_zs"] = self.sbert_zs.score(text).loc[x.index].to_numpy()
        if "title_role_match" in columns:
            if self.title_rule is None:
                from candidate_screener.matching.role_filter import SoftRoleWeight
                self.title_rule = SoftRoleWeight("lookup_or_text", 1.0, self.scorer.rule.role_model)
                self.title_rule.fit(None)
            x["title_role_match"] = self.title_rule.role_match(pairs.reset_index(drop=True))
        if "no_experience" in columns:
            from candidate_screener.evaluation.score_defects import f2_experience
            ones = pd.Series(1.0, index=range(len(pairs)))
            f2 = f2_experience(ones, pairs.jd_text.reset_index(drop=True), pairs.cv_text.reset_index(drop=True))
            x["no_experience"] = 1.0 - f2.to_numpy()
        return x[list(columns)]


class LearnedStage1(BaseSystem):
    """S1: logistic regression on `columns`, fitted on D50 unless `fit_features` is called."""

    rung, stage, can_explain = "S1", "Stage 1 — learned classifier", True
    trained_on = TRAINED_ON

    def __init__(self, set_name: str = "F12", features: Features | None = None, columns=None) -> None:
        super().__init__()
        self.set_name = set_name
        self.columns = tuple(columns) if columns is not None else FEATURE_SETS[set_name]
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
