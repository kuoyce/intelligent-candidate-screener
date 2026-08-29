"""BM25 scored from the corpus statistics, never from a corpus lookup *(D24)*.

`BM25Okapi.get_scores(query)` can only score documents that were part of the corpus
it was fitted on — it indexes into `self.doc_freqs`, built once at fit time. The
baseline originally exploited that by fitting on the train resumes and then finding
each row's score by looking its resume up *by exact text match* in that corpus.

Under the shipped `cnamuangtoun` partition, 99.8% of test resumes also appear in
train, so the lookup nearly always hit and the design looked correct. Under the
leak-free split it never hits — leak-free means zero resume overlap, by construction
— so **every BM25 test score was exactly `0.0`**. The third bug of the same shape as
the other two: no exception, no row-count change, a plausible number.

The fix is not a better lookup, it is removing the need for one. BM25's formula needs
only a document's own term frequencies and length, plus the corpus-fitted `idf` and
`avgdl`. So `fit_bm25` is read for those two statistics and nothing else, and
`pair_bm25_scores` scores **any** document, in or out of the fit corpus — the same
"fit statistics on train, apply to any document" contract `TfidfVectorizer.transform`
already has. There is no lookup step left for a future edit to get wrong.

Verified against `rank_bm25`'s own `get_scores` source before implementing.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from candidate_screener.baselines.lexical import tokenize


def fit_bm25(resume_texts: Iterable[str]) -> BM25Okapi:
    """Fit `idf` and `avgdl` on the unique **train** resumes.

    Only those two attributes are read downstream; the per-document frequencies
    `BM25Okapi` also stores are deliberately never consulted, because consulting
    them is what limits scoring to the fit corpus.
    """
    corpus = [tokenize(text) for text in resume_texts]
    return BM25Okapi(corpus)


def score_pair(model: BM25Okapi, query_tokens: list[str],
               doc_tokens: list[str]) -> float:
    """Okapi BM25 of one query against one document, from the model's statistics.

    This is `BM25Okapi.get_scores`' arithmetic with the corpus-membership assumption
    removed: term frequency and length come from `doc_tokens`, `idf` and `avgdl`
    from the fitted model. A term the fit corpus never saw contributes `0.0` — its
    `idf` is simply unknown, which is the honest answer and not an error.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        f = doc_freq.get(term, 0)
        if f == 0:
            continue
        idf = model.idf.get(term, 0.0)
        score += idf * (f * (model.k1 + 1)) / (
            f + model.k1 * (1 - model.b + model.b * doc_len / model.avgdl)
        )
    return float(score)


def pair_bm25_scores(model: BM25Okapi, resume_texts: Iterable[str],
                     jd_texts: Iterable[str]) -> np.ndarray:
    """Score every `(resume, jd)` pair — JD as query, resume as document.

    Both token caches live inside this call. Without them, scoring re-tokenises the
    same JD once per candidate rather than once per unique JD; at ~250 JDs over
    ~4,000 pairs that is a correctness-scale cost, not a micro-optimisation.
    Caching is keyed by full text and is pure — `test_bm25.py` asserts cached and
    uncached scoring agree.
    """
    query_cache: dict[str, list[str]] = {}
    doc_cache: dict[str, list[str]] = {}

    def cached(cache: dict[str, list[str]], text: str) -> list[str]:
        if text not in cache:
            cache[text] = tokenize(text)
        return cache[text]

    return np.array([score_pair(model, cached(query_cache, j), cached(doc_cache, r))
                     for r, j in zip(resume_texts, jd_texts)], dtype=float)
