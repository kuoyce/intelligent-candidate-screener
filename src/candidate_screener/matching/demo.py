"""Small text-only illustration; distinct from every frozen benchmark ranker."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from candidate_screener.baselines.tfidf import pair_cosine_scores
from .base import BaseSystem


class IllustrationTfidf(BaseSystem):
    """Fit a single shared vocabulary on explicitly supplied demonstration text.

    This unsupervised submission-pool fit is a UI illustration, never a held-out
    benchmark or an adopted retrieval model. No dataset or weight download is needed.
    """
    name = "Illustration TF-IDF (submission pool)"
    rung = "illustration"
    stage = "Draft text-only illustration"
    trained_on = "Explicit demonstration/submission text only; no fit labels"

    def _fit(self, train):
        if train is None:
            raise ValueError("Illustration requires explicit JD/CV texts")
        if hasattr(train, "columns"):
            texts = list(train.jd_text) + list(train.cv_text)
        else:
            texts = list(train)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Nonempty JD/CV texts required")
        self.vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, max_features=5000,
                                         ngram_range=(1, 2), stop_words="english")
        try:
            self.vectorizer.fit(list(dict.fromkeys(texts)))
        except ValueError as exc:
            raise ValueError("No readable terms in submission text") from exc

    def _score(self, pairs):
        if not hasattr(self, "vectorizer"):
            raise ValueError("Fit illustration on explicit text before scoring")
        return pair_cosine_scores(self.vectorizer, pairs.cv_text, pairs.jd_text)

    def params(self):
        return len(self.vectorizer.vocabulary_) if hasattr(self, "vectorizer") else 0
