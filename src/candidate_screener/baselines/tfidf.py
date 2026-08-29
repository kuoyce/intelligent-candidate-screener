"""TF-IDF cosine over **one shared vector space** — the bug this module prevents.

The baseline originally fitted one `TfidfVectorizer` on resumes and a second on job
descriptions, then took the cosine between the two resulting vectors. Each fit builds
its own vocabulary→index map, so the same word lands on a different dimension in each
vector: the cosine then compares unrelated axes and returns near-random noise. It did
not raise, did not change a row count, and produced a plausible score for months.

The API here makes that unwritable. `fit_shared_vectorizer` returns exactly **one**
vectoriser and `pair_cosine_scores` takes exactly one, so there is no signature
through which a caller can hold a resume vectoriser and a JD vectoriser at once.
`test_tfidf.py` asserts the invariant anyway — a document scored against itself is
1.0 in a shared space and is not in two separate ones.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#: The configuration the baseline is defined by. Recorded verbatim into
#: `baseline-metrics.json`, so a change here is a visible check failure.
TFIDF_CONFIG: dict = {
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.8,
    "ngram_range": (1, 2),
    "lowercase": True,
    "stop_words": "english",
}


def fit_shared_vectorizer(resume_texts: Iterable[str],
                          jd_texts: Iterable[str]) -> TfidfVectorizer:
    """Fit one vectoriser on the unique **train** resumes and JDs together.

    Both sides must share a vocabulary→index map for the cosine to mean anything;
    fitting on the union is how that is guaranteed. Deduplication is
    order-preserving, so the fit depends on the corpus and not on how many pairs
    happen to mention a given document.

    Fit on train only. A vectoriser fitted over test documents would leak their
    document frequencies into the representation the model is scored on.
    """
    corpus = pd.concat([pd.Series(list(resume_texts), dtype="object"),
                        pd.Series(list(jd_texts), dtype="object")]).drop_duplicates()
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    vectorizer.fit(corpus)
    return vectorizer


def pair_cosine_scores(vectorizer: TfidfVectorizer, resume_texts: Iterable[str],
                       jd_texts: Iterable[str]) -> np.ndarray:
    """Cosine similarity per `(resume, jd)` pair, in the vectoriser's single space.

    The per-document vector cache lives inside this function rather than at module
    level: A1 pairs ~450 resumes against ~250 JDs, so a row-wise `transform` would
    re-vectorise the same document up to a hundred times. Caching is keyed by the
    full document text and is a pure speed-up — `test_tfidf.py` asserts cached and
    uncached scoring agree.
    """
    cache: dict[str, object] = {}

    def vector(text: str):
        if text not in cache:
            cache[text] = vectorizer.transform([text])
        return cache[text]

    return np.array([float(cosine_similarity(vector(r), vector(j))[0, 0])
                     for r, j in zip(resume_texts, jd_texts)], dtype=float)


def vocabulary_digest(vectorizer: TfidfVectorizer) -> str:
    """sha256 over the sorted feature names *(design spec §3, row 5)*.

    `max_features=5000` keeps the top terms by document frequency through an
    unstable `argsort`, so *which* of several equally-frequent terms survives at the
    5,000 boundary is an implementation detail of numpy's sort. Digesting the
    vocabulary turns a boundary shift into an explicit, diagnosable failure instead
    of a metric that moved by a hair for no stated reason.
    """
    names = sorted(vectorizer.get_feature_names_out().tolist())
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def sklearn_token_pattern() -> str:
    """sklearn's default pattern, so `test_lexical.py` can assert parity against
    `lexical.TOKEN_PATTERN` without reaching into sklearn itself.

    Deliberately *not* asserted at import time: a sklearn bump that changed the
    default should surface as one readable test failure, not as an ImportError that
    takes the whole package down before anyone can diagnose it.
    """
    return TfidfVectorizer().token_pattern
