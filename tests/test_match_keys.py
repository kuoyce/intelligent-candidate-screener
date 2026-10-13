"""Match keys of Part B (plan 2026-10-13-coverage-weighting-variants): synthetic phrases only."""
import numpy as np

from candidate_screener.evaluation import match_keys as mk


def test_k1_folds_punctuation_plural_and_js():
    assert mk.k1("ci/cd") == mk.k1("CICD")
    assert mk.k1("node.js") == mk.k1("nodejs") == mk.k1("Node")
    assert mk.k1("APIs") == mk.k1("api")
    assert mk.k1("c++") != mk.k1("c") and mk.k1("c#") != mk.k1("c")      # symbols stay distinguishing
    assert mk.k1("aws") == "aws" and mk.k1("sass") == "sass"             # short or -ss words are not clipped


def test_k0_misses_what_k1_finds():
    J, C = {"unknown:ci/cd", "unknown:kafka"}, {"unknown:cicd", "unknown:kafka"}
    assert mk.matched_exact(J, C) == {"unknown:kafka"}
    assert mk.matched_k1(J, C) == {"unknown:ci/cd", "unknown:kafka"}


def test_k2_containment_needs_four_characters_in_the_shorter_phrase():
    assert mk.matched_k2({"unknown:react"}, {"unknown:react developer"}) == {"unknown:react"}
    assert mk.matched_k2({"unknown:go"}, {"unknown:go developer"}) == set()     # 'go' is too short
    assert mk.matched_k2({"unknown:sql"}, {"unknown:sql server"}) == set()      # 'sql' is 3 characters


def test_embedding_match_uses_the_threshold():
    vec = {"a": np.array([1.0, 0.0]), "b": np.array([0.8, 0.6]), "c": np.array([0.0, 1.0])}
    assert mk.matched_embedding({"a"}, {"b"}, vec, 0.8) == {"a"}                # cosine 0.8
    assert mk.matched_embedding({"a"}, {"b"}, vec, 0.9) == set()
    assert mk.matched_embedding({"a"}, {"c"}, vec, 0.7) == set()
    assert mk.matched_embedding(set(), {"b"}, vec, 0.7) == set()
