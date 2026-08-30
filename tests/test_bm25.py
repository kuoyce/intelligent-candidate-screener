"""`baselines.bm25` — scoring documents the fit corpus never saw.

`test_bm25_scores_document_outside_fit_corpus` is one of the three named bug
regressions (design spec §5.2, D24).
"""
from __future__ import annotations

import numpy as np

from candidate_screener.baselines import bm25


def test_bm25_scores_document_outside_fit_corpus(resumes, jds, held_out_resume):
    """REGRESSION: the corpus-lookup bug, reduced to one assertion.

    Fit on a corpus that excludes this resume, then score it against a JD it
    genuinely overlaps with. Under the old design — `BM25Okapi.get_scores` plus a
    text→row lookup into the fit corpus — this is exactly 0.0 for **every** document
    held out of the fit corpus, which under the leak-free split is every test resume
    by construction.
    """
    model = bm25.fit_bm25(resumes)
    assert held_out_resume not in resumes
    score = bm25.pair_bm25_scores(model, [held_out_resume], [jds[0]])[0]
    assert score > 0.0


def test_term_overlap_drives_the_score(resumes, jds, held_out_resume):
    """The score is not merely non-zero, it tracks overlap in the right direction."""
    model = bm25.fit_bm25(resumes)
    relevant, irrelevant = bm25.pair_bm25_scores(
        model, [held_out_resume] * 2, [jds[0], jds[1]])
    assert relevant > irrelevant


def test_terms_unseen_at_fit_time_contribute_nothing(resumes, jds):
    """An unknown term has no idf. Contributing 0.0 is the honest answer, not an error."""
    model = bm25.fit_bm25(resumes)
    assert bm25.pair_bm25_scores(model, ["zzz_unseen_term"], ["zzz_unseen_term"])[0] == 0.0


def test_empty_side_scores_zero(resumes):
    model = bm25.fit_bm25(resumes)
    assert list(bm25.pair_bm25_scores(model, ["", resumes[0]], [resumes[0], ""])) == [0.0, 0.0]


def test_cached_and_uncached_scoring_agree(resumes, jds):
    """Design spec §3 row 10: the token caches must be arithmetically invisible."""
    model = bm25.fit_bm25(resumes)
    repeated_q = [jds[0]] * 4
    cached = bm25.pair_bm25_scores(model, resumes[:4], repeated_q)
    one_at_a_time = np.array([bm25.pair_bm25_scores(model, [r], [q])[0]
                              for r, q in zip(resumes[:4], repeated_q)])
    assert np.array_equal(cached, one_at_a_time)


def test_fit_is_row_order_invariant(resumes, jds):
    """Design spec §3 row 4, BM25 side: idf and avgdl are corpus statistics."""
    a, b = bm25.fit_bm25(resumes), bm25.fit_bm25(resumes[::-1])
    assert a.idf == b.idf and a.avgdl == b.avgdl
    assert np.array_equal(bm25.pair_bm25_scores(a, resumes, [jds[0]] * len(resumes)),
                          bm25.pair_bm25_scores(b, resumes, [jds[0]] * len(resumes)))


def test_score_pair_matches_the_library_on_a_corpus_document(resumes):
    """The reimplementation is BM25Okapi's own arithmetic, not an approximation of it.

    Scored against a document that *is* in the fit corpus, where `get_scores` is
    valid, the two must agree — that is what makes generalising it to held-out
    documents a removal of an assumption rather than a change of model.
    """
    from candidate_screener.baselines.lexical import tokenize
    model = bm25.fit_bm25(resumes)
    query = "python sql aws"
    library = model.get_scores(tokenize(query))
    ours = bm25.pair_bm25_scores(model, resumes, [query] * len(resumes))
    assert np.allclose(library, ours)
