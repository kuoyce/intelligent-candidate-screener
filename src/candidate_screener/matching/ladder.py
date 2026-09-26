"""The ladder: which rungs are scored, in order, and what each is compared against.

`status` is `exploratory` for a rung whose configuration was looked at before the decision
rule was registered (plan 01 §5.4), and `held-out` for one built after it. Only held-out
rungs test the rule (`plan/2026-09-26-top-down-reset/06-registration.md`).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from candidate_screener.matching import lexical, rules


@dataclass(frozen=True)
class Rung:
    factory: Callable
    status: str


def rungs() -> list[Rung]:
    return [
        Rung(rules.Random, "exploratory"),
        Rung(rules.KeywordMatch, "exploratory"),
        Rung(rules.Metadata, "exploratory"),
        Rung(lexical.Tfidf, "exploratory"),
        Rung(lexical.Bm25, "exploratory"),
        Rung(lambda: lexical.MetadataPlus(lexical.Tfidf(), "1a+2"), "exploratory"),
        Rung(lambda: lexical.MetadataPlus(lexical.Bm25(), "1a+2"), "exploratory"),
    ]


def classifiers() -> dict:
    return {"metadata rule (Fit iff keyword match)": (rules.metadata_classifier, "exploratory")}


def comparisons() -> list[tuple[str, str]]:
    """(B, A): does B earn its place over the cheaper A?"""
    return [
        ("metadata rules", "random"),
        ("TF-IDF", "metadata rules"),
        ("BM25", "TF-IDF"),
        ("metadata + TF-IDF", "metadata rules"),
        ("metadata + BM25", "metadata + TF-IDF"),
    ]
