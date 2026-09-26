"""The ladder: which rungs are scored, in order, and what each is compared against.

`status` is `exploratory` for a rung whose configuration was looked at before the decision
rule was registered, and `held-out` for one built after it. Only held-out rungs test the
rule (`plan/2026-09-26-top-down-reset/06-registration.md`). The comparisons are the
registered C1-C11. C12 lives in the retrieval pilot.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from candidate_screener.matching import cross_encoder, hybrid, lexical, rules, sbert


@dataclass(frozen=True)
class Rung:
    factory: Callable
    status: str


def _plus(factory: Callable, rung: str) -> Callable:
    return lambda: lexical.MetadataPlus(factory(), rung)


TEXT_RUNGS = {
    "TF-IDF": lexical.Tfidf,
    "BM25": lexical.Bm25,
    "SBERT (zero-shot)": sbert.SbertZeroShot,
    "SBERT (weak fine-tuned)": sbert.SbertWeak,
    "cross-encoder (A1)": cross_encoder.CrossEncoderA1,
    "cross-encoder (weak)": cross_encoder.CrossEncoderWeak,
}


def _skill_overlap_available() -> bool:
    try:
        from candidate_screener.extraction.rules import default_matcher
        default_matcher()
        return True
    except Exception:                                        # extraction not built yet
        return False


def rungs() -> list[Rung]:
    out = [
        Rung(rules.Random, "exploratory"),
        Rung(rules.KeywordMatch, "exploratory"),
        Rung(rules.Metadata, "exploratory"),
    ]
    if _skill_overlap_available():
        out += [Rung(rules.SkillOverlap, "held-out"),
                Rung(_plus(rules.SkillOverlap, "1a+1b"), "held-out")]
    out += [
        Rung(lexical.Tfidf, "exploratory"),
        Rung(lexical.Bm25, "exploratory"),
        Rung(_plus(lexical.Tfidf, "1a+2"), "exploratory"),
        Rung(_plus(lexical.Bm25, "1a+2"), "exploratory"),
        Rung(sbert.SbertZeroShot, "exploratory"),
        Rung(_plus(sbert.SbertZeroShot, "1a+3a"), "exploratory"),
        Rung(sbert.SbertWeak, "held-out"),
        Rung(_plus(sbert.SbertWeak, "1a+3b"), "held-out"),
        Rung(cross_encoder.CrossEncoderA1, "held-out"),
        Rung(_plus(cross_encoder.CrossEncoderA1, "1a+4a"), "held-out"),
        Rung(cross_encoder.CrossEncoderWeak, "held-out"),
        Rung(_plus(cross_encoder.CrossEncoderWeak, "1a+4b"), "held-out"),
        Rung(lambda: hybrid.LearnedHybrid(list(TEXT_RUNGS.values())), "held-out"),
    ]
    return out


def classifiers() -> dict:
    return {"metadata rule (Fit iff keyword match)": (rules.metadata_classifier, "exploratory")}


def comparisons() -> list[tuple[str, str, str]]:
    """(id, B, A): does B earn its place over the cheaper A? (registration §3)."""
    return [
        ("C1", "skill overlap", "random"),
        ("C2", "metadata + skill overlap", "metadata rules"),
        ("C3", "SBERT (zero-shot)", "TF-IDF"),
        ("C4", "metadata + SBERT (zero-shot)", "metadata + TF-IDF"),
        ("C5", "SBERT (weak fine-tuned)", "TF-IDF"),
        ("C6", "metadata + SBERT (weak fine-tuned)", "metadata + TF-IDF"),
        ("C7", "cross-encoder (A1)", "TF-IDF"),
        ("C8", "metadata + cross-encoder (A1)", "metadata + TF-IDF"),
        ("C9", "cross-encoder (weak)", "TF-IDF"),
        ("C10", "metadata + cross-encoder (weak)", "metadata + TF-IDF"),
        ("C11", "metadata + learned text combination", "metadata + TF-IDF"),
        # reported, decide nothing
        ("E1", "metadata rules", "random"),
        ("E2", "TF-IDF", "metadata rules"),
        ("E3", "BM25", "TF-IDF"),
        ("E4", "metadata + TF-IDF", "metadata rules"),
        ("E5", "metadata + BM25", "metadata + TF-IDF"),
        ("E6", "SBERT (weak fine-tuned)", "SBERT (zero-shot)"),
        ("E7", "cross-encoder (weak)", "cross-encoder (A1)"),
    ]


def retrievers() -> list[Callable]:
    """Candidates for the 'retrieve' half, on the 500-candidate pools (no cross-encoders:
    scoring 30,000 pairs jointly is exactly what retrieve-then-rank avoids)."""
    return [rules.Random, rules.Metadata, lexical.Tfidf, lexical.Bm25, sbert.SbertZeroShot,
            sbert.SbertWeak, _plus(lexical.Tfidf, "1a+2"), _plus(sbert.SbertZeroShot, "1a+3a"),
            _plus(sbert.SbertWeak, "1a+3b")]


def rerankers() -> list[tuple[str, Callable]]:
    """(retriever name, reranker): C12 re-ranks the retriever's top 50."""
    out = []
    for retriever in ("metadata + TF-IDF", "TF-IDF"):
        out += [(retriever, cross_encoder.CrossEncoderA1),
                (retriever, cross_encoder.CrossEncoderWeak)]
    return out
