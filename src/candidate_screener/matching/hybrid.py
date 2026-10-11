"""Rung 5: a learned combination of the text rungs, under the fixed metadata level.

Trained on the A1 pairs held out from the A1 cross-encoder's training (fold 0 of the
doubly-disjoint CV), so no component has seen its training rows. It is a standardised,
L2 logistic regression with C fixed at 1.0. Metadata cannot be learned there (A1 has
none), so the score is `metadata level + P(fit | text scores)`: the metadata rule decides
the level, and the learned text combination orders candidates within it.

Not trained on the weak pairs: their label is a function of keyword and experience, so a
combiner fitted there would relearn the metadata rule and ignore the text.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from candidate_screener.matching import corpus
from candidate_screener.matching.base import BaseSystem


class LearnedHybrid(BaseSystem):
    name, rung, stage = "metadata + learned text combination", "5", "hybrid (learned)"
    trained_on = "A1 train fold 0 (held out from 4a): logistic weights over text-rung scores"
    can_explain = False

    def __init__(self, components: list) -> None:
        super().__init__()
        self.factories = components

    def _fit(self, train):
        from candidate_screener.data import fit_split
        self.components = [f() for f in self.factories]
        for c in self.components:
            c.fit(None)
        a1 = corpus.a1_pairs("train")
        _, held = fit_split.fold_masks(a1.rename(columns={"cv_id": "resume_id"}), 5, 0, 0)
        held = a1[held]
        X = self._features(held)
        self.scaler = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=1.0, max_iter=5000).fit(self.scaler.transform(X), held.fit)
        self.weights = dict(zip([c.name for c in self.components], self.lr.coef_[0].round(4)))

    def _features(self, pairs):
        return np.column_stack([c.score(pairs).to_numpy() for c in self.components])

    def _score(self, pairs):
        p = self.lr.predict_proba(self.scaler.transform(self._features(pairs)))[:, 1]
        meta = 2 * corpus.keyword_match(pairs).to_numpy() + corpus.experience_compatible(pairs).to_numpy()
        return meta + 0.999 * p

    def params(self):
        return int(self.lr.coef_.size + 1)
