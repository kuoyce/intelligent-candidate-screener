"""Weighting variants of the coverage term (plan 2026-10-13-coverage-weighting-variants)."""
import math

import pandas as pd
import pytest

from candidate_screener.evaluation import coverage_variants as cv
from candidate_screener.matching.entity_weighted import DocumentFrequency


class Fixed:
    def __init__(self, by_text): self.by_text = by_text
    def concepts(self, text, side="cv"): return frozenset(self.by_text[text])


def fitted(n_common=10):
    """10 CVs: 'kafka' (known) in 1, 'unknown:english' in all, 'unknown:rare' in 2."""
    cvs = {f"c{i}": {"unknown:english"} | ({"kafka"} if i == 0 else set()) | ({"unknown:rare"} if i < 2 else set())
           for i in range(n_common)}
    ex = Fixed(cvs)
    docs = pd.DataFrame({"id": list(cvs), "text": list(cvs), "region": "train", "doc_type": "cv"})
    return DocumentFrequency(ex).fit(docs)


def idf(df, n=10):
    return min(5.0, 1 + math.log((n + 1) / (df + 1)))


def test_default_is_w0_flat_unknown_weight():
    f = fitted()
    assert f.weight("unknown:english") == 1.0 and f.weight("unknown:rare") == 1.0
    assert f.weight("kafka") == pytest.approx(idf(1))


def test_w1_weights_unknown_phrases_by_df():
    f = fitted().variant(unknown="idf")
    assert f.weight("unknown:english") == pytest.approx(idf(10))
    assert f.weight("unknown:english") < f.weight("unknown:rare")


def test_w3_floor_keeps_rare_unknowns_at_one_but_not_known_concepts():
    f = fitted().variant(unknown="idf", min_df=5)
    assert f.weight("unknown:rare") == 1.0                 # df 2 < 5
    assert f.weight("kafka") == pytest.approx(idf(1))      # known concept untouched
    assert f.weight("unknown:english") == pytest.approx(idf(10))


def test_w2_top_k_uses_heaviest_concepts_and_lexical_ties():
    f = fitted().variant(unknown="idf", top_k=2)
    jd = {"kafka", "unknown:rare", "unknown:english"}
    # heaviest two: kafka (df 1) then unknown:rare (df 2); english is dropped from both sides
    full = (idf(1) + idf(2)) / (idf(1) + idf(2))
    assert f.coverage(jd, {"kafka", "unknown:rare"}) == pytest.approx(full)
    assert f.coverage(jd, {"unknown:english"}) == 0.0


def test_variant_shares_fitted_counts():
    base = fitted(); v = base.variant(unknown="idf")
    assert v.df is base.df and v.fitted and v.n_documents == base.n_documents


def test_split_is_deterministic_disjoint_and_sized():
    ids = [f"j{i}" for i in range(100)]
    s1, h1 = cv.split_jds(ids); s2, h2 = cv.split_jds(ids)
    assert (s1, h1) == (s2, h2) and not set(s1) & set(h1) and (len(s1), len(h1)) == (70, 30)


def test_pick_prefers_simpler_within_tie_and_takes_a_clear_winner():
    assert cv.pick({"W0": 0.70, "W1": 0.701, "W2": 0.7015, "W3": 0.7}) == "W0"
    assert cv.pick({"W0": 0.70, "W1": 0.71, "W2": 0.72, "W3": 0.715}) == "W2"
    assert cv.pick({"W0": 0.70, "W1": 0.72, "W2": 0.7205, "W3": 0.719}) == "W1"   # all within 0.002


def test_w2b_breaks_weight_ties_by_jd_mentions_not_alphabet():
    f = fitted().variant(unknown="idf", top_k=1, mention_ties=True)
    jd = {"unknown:aaa", "unknown:zzz"}                       # equal weight (df 0, at the cap)
    assert f.coverage(jd, {"unknown:zzz"}, jd_mentions={"unknown:zzz": 3, "unknown:aaa": 1}) == pytest.approx(1.0)
    assert f.coverage(jd, {"unknown:aaa"}, jd_mentions={"unknown:zzz": 3, "unknown:aaa": 1}) == 0.0
    plain = fitted().variant(unknown="idf", top_k=1)           # W2: alphabetical
    assert plain.coverage(jd, {"unknown:aaa"}, jd_mentions={"unknown:zzz": 3, "unknown:aaa": 1}) == pytest.approx(1.0)


def test_extractor_mentions_count_each_unqualified_mention():
    from candidate_screener.matching.entity_weighted import ConceptExtractor
    rows = [{"label": "SKILL", "surface": "Kafka", "context_status": "unqualified"},
            {"label": "SKILL", "surface": "kafka", "context_status": "unqualified"},
            {"label": "SKILL", "surface": "Redis", "context_status": "negated"}]
    ex = ConceptExtractor(entities=lambda t, s: rows, aliases={})
    assert ex.mentions("x", "jd") == {"unknown:kafka": 2}
    assert ex.concepts("x", "jd") == {"unknown:kafka"}
