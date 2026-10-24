"""Concept sources and the Stage 1 gate (plan 2026-10-13): synthetic text only, no models."""
import pandas as pd

from candidate_screener.evaluation import concept_overlap as co
from candidate_screener.extraction import concept_sources as cs
from candidate_screener.matching.entity_weighted import ConceptExtractor


class FakeTagger:
    def predict_spans(self, text):
        return [(text.index(w), text.index(w) + len(w)) for w in ("Kafka", "Spark") if w in text]


def rows_of(text):
    return cs.minilm_rows(FakeTagger())(text, "cv")


def test_cache_computes_once_and_keeps_only_skill_rows(tmp_path):
    calls = []

    def compute(text, side):
        calls.append(text)
        return [{"label": "SKILL", "surface": "x", "context_status": "unqualified"},
                {"label": "TITLE", "surface": "dev", "context_status": "unqualified"}]
    c = cs.CachedEntities("t", compute, tmp_path)
    assert [r["label"] for r in c("a doc")] == ["SKILL"]
    c("a doc"); assert len(calls) == 1
    c.save()
    again = cs.CachedEntities("t", None, tmp_path)         # reload: no compute needed
    assert again("a doc") == c("a doc")


def test_minilm_rows_carry_the_rules_context():
    assert rows_of("I know Kafka.")[0]["context_status"] == "unqualified"
    assert rows_of("no Kafka experience")[0]["context_status"] == "negated"


def test_negated_tagger_mention_is_not_a_concept(tmp_path):
    src = cs.CachedEntities("m", cs.minilm_rows(FakeTagger()), tmp_path)
    ex = ConceptExtractor(entities=src, aliases={})
    assert ex.concepts("Strong Kafka skills", "cv") == {"unknown:kafka"}
    assert ex.concepts("without Kafka", "cv") == frozenset()


class Const:
    def __init__(self, by_text): self.by_text = by_text
    def concepts(self, text, side="cv"): return frozenset(self.by_text.get(text, ()))


def test_union_is_the_set_union():
    u = cs.UnionConcepts(Const({"t": {"a"}}), Const({"t": {"b"}}))
    assert u.concepts("t") == {"a", "b"}


def pairs():
    return pd.DataFrame({"jd_id": ["j1", "j1", "j2"], "cv_id": ["c1", "c2", "c3"],
                         "jd_text": ["J1", "J1", "J2"], "cv_text": ["C1", "C2", "C3"]})


def test_overlap_stats_count_pairs_without_shared_concepts():
    ex = Const({"J1": {"a", "b"}, "J2": {"x"}, "C1": {"a"}, "C2": {"z"}, "C3": set()})
    s = co.overlap_stats(ex, pairs())
    assert s["pairs"] == 3 and s["no_overlap_share"] == 2 / 3
    assert s["jd_le3_share"] == 1.0 and s["cv_zero_share"] == 1 / 3


def test_gate_needs_five_points_and_ignores_the_baseline():
    stats = {"R": {"no_overlap_share": 0.53}, "G": {"no_overlap_share": 0.47},
             "N": {"no_overlap_share": 0.49}}
    g = co.gate(stats)
    assert g["passing"] == ["G"] and round(g["cut_pp"]["N"], 6) == 4.0


def test_gate_boundary_cases():
    stats = {"R": {"no_overlap_share": 0.50}, "N": {"no_overlap_share": 0.455},   # 4.5 pp: fails
             "G": {"no_overlap_share": 0.44}, "X": {"no_overlap_share": 0.40}}    # 6 pp, 10 pp: pass
    assert co.gate(stats)["passing"] == ["G", "X"]
