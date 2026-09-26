"""Stage 2 rungs: TF-IDF and BM25, reusing the frozen baseline's own functions, and the
hybrid that adds a text score to the metadata rule.

On A2 the document frequencies come from a seeded sample of the **train** region (D19),
never from the 600 evaluation pairs. Text is used as written — no skill-boost augment,
which belongs to the skill-overlap rung.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from candidate_screener.baselines import bm25, tfidf
from candidate_screener.matching import corpus
from candidate_screener.matching.base import BaseSystem


def a2_train_texts() -> tuple[pd.Series, pd.Series]:
    jd, cv = corpus.a2_train_documents()
    return cv.cv_text, jd.jd_text


@lru_cache(maxsize=1)
def a2_vectorizer():
    """The A2 TF-IDF space, fitted once per process and shared by every rung using it."""
    return tfidf.fit_shared_vectorizer(*a2_train_texts())


@lru_cache(maxsize=1)
def a2_bm25():
    return bm25.fit_bm25(a2_train_texts()[0])


class Tfidf(BaseSystem):
    name, rung, stage = "TF-IDF", "2", "Stage 2 — lexical"
    trained_on = "IDF from 20k+20k A2 train-region documents"
    can_explain = True

    def _fit(self, train):
        self.vectorizer = (tfidf.fit_shared_vectorizer(*train) if train is not None
                           else a2_vectorizer())

    def _score(self, pairs):
        return tfidf.pair_cosine_scores(self.vectorizer, pairs.cv_text, pairs.jd_text)

    def params(self):
        return len(self.vectorizer.vocabulary_)


class Bm25(BaseSystem):
    name, rung, stage = "BM25", "2", "Stage 2 — lexical"
    trained_on = "IDF/avgdl from 20k A2 train-region CVs"
    can_explain = True

    def _fit(self, train):
        self.model = bm25.fit_bm25(train[0]) if train is not None else a2_bm25()

    def _score(self, pairs):
        return bm25.pair_bm25_scores(self.model, pairs.cv_text, pairs.jd_text)

    def params(self):
        return len(self.model.idf)


class MetadataPlus(BaseSystem):
    """`metadata rules + text similarity in [0, 1]`: the metadata level decides, the text
    orders candidates within it. The text score is min-max scaled per JD so any rung's
    scale fits under one metadata step."""
    stage = "hybrid"
    can_explain = True

    def __init__(self, text_system: BaseSystem, rung: str) -> None:
        super().__init__()
        self.text = text_system
        self.name = f"metadata + {text_system.name}"
        self.rung = rung
        self.trained_on = text_system.trained_on

    def fit(self, train=None):
        self.text.fit(train)
        self.fit_seconds = self.text.fit_seconds

    def _score(self, pairs):
        text = self.text.score(pairs).to_numpy()
        frame = pd.DataFrame({"jd": pairs.jd_id.to_numpy(), "t": text})
        lo = frame.groupby("jd").t.transform("min")
        span = frame.groupby("jd").t.transform("max") - lo
        scaled = ((frame.t - lo) / span.where(span > 0, 1)).clip(0, 1) * 0.999
        meta = 2 * corpus.keyword_match(pairs).to_numpy() + corpus.experience_compatible(pairs).to_numpy()
        return meta + scaled.to_numpy()

    def params(self):
        return self.text.params()
