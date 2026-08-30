"""`baselines.tfidf` — the shared vector space.

`test_shared_space_cosine_of_identical_text_is_one` is one of the three named bug
regressions (design spec §5.2).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from candidate_screener.baselines import tfidf


def test_shared_space_cosine_of_identical_text_is_one(resumes, jds):
    """REGRESSION: the two-vectoriser bug, reduced to one assertion.

    A document scored against itself is 1.0 in a shared space. Fit resumes and JDs
    separately and it is not, because the two vocabulary→index maps disagree and the
    cosine compares unrelated axes.
    """
    vec = tfidf.fit_shared_vectorizer(resumes, jds)
    scores = tfidf.pair_cosine_scores(vec, resumes, resumes)
    assert np.allclose(scores, 1.0)


def test_a_document_with_no_in_vocabulary_term_scores_zero(resumes, jds):
    """The one way the invariant above can pass for the wrong reason, pinned.

    `min_df=2` drops terms seen once, so a document made only of such terms
    vectorises to all-zeros and scores 0.0 against itself. That is the configuration
    behaving correctly — but a corpus built carelessly enough to trigger it would
    make the shared-space regression vacuous, so it is asserted rather than assumed.
    """
    vec = tfidf.fit_shared_vectorizer(resumes, jds)
    orphan = "zygomorphic thaumaturgy quinquagenarian"
    assert tfidf.pair_cosine_scores(vec, [orphan], [orphan])[0] == 0.0


def test_overlapping_pair_outscores_an_unrelated_one(resumes, jds):
    """A 4-document corpus where the expected ranking is obvious by inspection."""
    vec = tfidf.fit_shared_vectorizer(resumes, jds)
    matched = tfidf.pair_cosine_scores(vec, [resumes[0]], [jds[0]])[0]
    unrelated = tfidf.pair_cosine_scores(vec, [resumes[0]], [jds[1]])[0]
    assert matched > unrelated


def test_vocabulary_and_scores_survive_a_row_reordering(resumes, jds):
    """Design spec §3 row 4: no ordering may depend on `set` iteration order.

    Fitting from a reordered corpus must give an identical vocabulary and identical
    scores, or a re-run would drift with `PYTHONHASHSEED` — sometimes.
    """
    a = tfidf.fit_shared_vectorizer(resumes, jds)
    b = tfidf.fit_shared_vectorizer(resumes[::-1], jds[::-1])
    assert tfidf.vocabulary_digest(a) == tfidf.vocabulary_digest(b)
    assert np.array_equal(tfidf.pair_cosine_scores(a, resumes, [jds[0]] * len(resumes)),
                          tfidf.pair_cosine_scores(b, resumes, [jds[0]] * len(resumes)))


def test_repeated_documents_do_not_change_the_fit(resumes, jds):
    """Deduplication is what stops a popular JD dominating the document frequencies."""
    a = tfidf.fit_shared_vectorizer(resumes, jds)
    b = tfidf.fit_shared_vectorizer(resumes + resumes, jds)
    assert tfidf.vocabulary_digest(a) == tfidf.vocabulary_digest(b)


def test_cached_and_uncached_scoring_agree(resumes, jds):
    """The per-document cache is a speed-up, so it must be arithmetically invisible."""
    vec = tfidf.fit_shared_vectorizer(resumes, jds)
    repeated = [resumes[0]] * 4
    cached = tfidf.pair_cosine_scores(vec, repeated, [jds[0]] * 4)
    one_at_a_time = np.array([tfidf.pair_cosine_scores(vec, [r], [jds[0]])[0]
                              for r in repeated])
    assert np.array_equal(cached, one_at_a_time)


def test_config_is_honoured(resumes, jds):
    """`min_df=2` and `ngram_range=(1,2)` are part of what the baseline *is*."""
    vec = tfidf.fit_shared_vectorizer(resumes, jds)
    assert vec.min_df == 2 and vec.ngram_range == (1, 2)
    assert vec.max_features == tfidf.TFIDF_CONFIG["max_features"]
    # min_df=2 must actually bite: a term in one document only is excluded.
    assert "terraform" not in vec.vocabulary_


def test_max_features_truncates(resumes, jds):
    """The `max_features` boundary is real, which is why the vocabulary is digested."""
    small = dict(tfidf.TFIDF_CONFIG, max_features=3, min_df=1)
    vec = TfidfVectorizer(**small).fit(resumes + jds)
    assert len(vec.get_feature_names_out()) == 3


def test_vocabulary_digest_changes_with_the_vocabulary(resumes, jds):
    a = tfidf.fit_shared_vectorizer(resumes, jds)
    # Two *distinct* documents: the corpus is deduplicated before fitting, so a
    # repeated string still has df=1 and min_df=2 would exclude its terms.
    b = tfidf.fit_shared_vectorizer(
        resumes, jds + ["quantum annealing lattice solver",
                        "lattice quantum annealing hardware"])
    assert tfidf.vocabulary_digest(a) != tfidf.vocabulary_digest(b)
