"""Registered bounded entity coverage extension of X2; no evaluation labels read."""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

from candidate_screener.matching.base import BaseSystem
from candidate_screener.matching.extracted import ExtractedRule
from candidate_screener.matching.lexical import Tfidf
from candidate_screener.matching.role_filter import per_jd_minmax


def normalized_phrase(value: str) -> str:
    """Case/space normalization preserves meaningful punctuation (C++, C#, .NET)."""
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


class ConceptExtractor:
    """Unique affirmative concepts from seed rules or supplied GLiNER predictions.

    ``entities(text, side)`` returns entity dictionaries, never raw model scores.
    Unknown phrases match only their exact normalized spelling.
    """

    def __init__(self, entities=None, aliases=None, matcher=None):
        self.entities = entities
        self.matcher = matcher
        self.aliases = None if aliases is None else {
            normalized_phrase(k): v for k, v in aliases.items()}
        self._cache = {}

    def _prepare(self):
        from candidate_screener.extraction.rules import SkillMatcher, load_vocab
        if self.aliases is None:
            vocabulary = load_vocab()
            self.aliases = {}
            for canonical, alias in zip(vocabulary.canonical, vocabulary.alias):
                self.aliases.setdefault(normalized_phrase(alias), canonical)
                self.aliases.setdefault(normalized_phrase(canonical), canonical)
            if self.matcher is None:
                self.matcher = SkillMatcher(vocabulary)
        elif self.entities is None and self.matcher is None:
            self.matcher = SkillMatcher(pd.DataFrame(
                [(v, k) for k, v in self.aliases.items()], columns=["canonical", "alias"]))

    def concepts(self, text: str, side: str = "cv") -> frozenset[str]:
        if side not in {"cv", "jd"}:
            raise ValueError("side must be jd or cv")
        key = (str(text), side)
        if key not in self._cache:
            self._prepare()
            if self.entities is None:
                from candidate_screener.extraction.rules import _context
                self._cache[key] = frozenset(
                    match.canonical for match in self.matcher.find(str(text))
                    if _context(str(text), match.start, match.end, "text") == "unqualified")
                return self._cache[key]
            else:
                rows = self.entities(str(text), side)
            out = set()
            for row in rows:
                if row.get("label") not in {"SKILL", "KNOWLEDGE"}:
                    continue
                if row.get("context_status") != "unqualified":
                    continue
                phrase = normalized_phrase(row.get("surface", row.get("text", "")))
                if phrase:
                    out.add(self.aliases.get(phrase, "unknown:" + phrase))
            self._cache[key] = frozenset(out)
        return self._cache[key]


class DocumentFrequency:
    """Train-only CV document frequency; repeated rows and mentions count once."""

    def __init__(self, extractor: ConceptExtractor, unknown: str = "flat",
                 min_df: int = 0, top_k: int | None = None):
        """Weighting variants (plan 2026-10-13-coverage-weighting-variants); defaults are W0.

        ``unknown="idf"`` weights unknown phrases by train df (W1); ``min_df`` keeps an unknown
        phrase seen in fewer train CVs at weight 1 (W3); ``top_k`` keeps only the JD's k
        heaviest concepts in numerator and denominator (W2)."""
        if unknown not in {"flat", "idf"}:
            raise ValueError("unknown must be 'flat' or 'idf'")
        if min_df < 0 or (top_k is not None and top_k < 1):
            raise ValueError("min_df must be >= 0 and top_k >= 1")
        self.unknown, self.min_df, self.top_k = unknown, min_df, top_k
        self.extractor = extractor
        self.df = Counter()
        self.n_documents = 0
        self.fitted = False

    def fit(self, documents: pd.DataFrame):
        if not {"id", "text", "region"}.issubset(documents.columns):
            raise ValueError("IDF requires id, text, region columns")
        if documents.empty or not documents.region.eq("train").all():
            raise ValueError("IDF requires nonempty train-region CV documents only")
        if "doc_type" in documents and not documents.doc_type.eq("cv").all():
            raise ValueError("IDF requires CV documents only")
        if documents.id.isna().any() or documents.text.isna().any():
            raise ValueError("Training document ids/text cannot be missing")
        if documents.groupby("id").text.nunique().gt(1).any():
            raise ValueError("One training document id has conflicting text")
        unique = documents.drop_duplicates("id")
        counts = Counter()
        for text in unique.text:
            counts.update(self.extractor.concepts(str(text), "cv"))
        self.df, self.n_documents, self.fitted = counts, len(unique), True
        return self

    def variant(self, unknown: str = "flat", min_df: int = 0, top_k: int | None = None):
        """A view with another weighting mode over the same fitted counts."""
        v = DocumentFrequency(self.extractor, unknown, min_df, top_k)
        v.df, v.n_documents, v.fitted = self.df, self.n_documents, self.fitted
        return v

    def weight(self, concept: str) -> float:
        if not self.fitted:
            raise ValueError("IDF has not been fitted on training CVs")
        if concept.startswith("unknown:") and (self.unknown == "flat" or self.df[concept] < self.min_df):
            return 1.0
        return min(5.0, 1 + math.log((self.n_documents + 1) / (self.df[concept] + 1)))

    def coverage(self, jd_concepts, cv_concepts, weighted=True) -> float:
        wanted, held = set(jd_concepts), set(cv_concepts)
        if not wanted:
            return 0.0
        weights = {s: self.weight(s) if weighted else 1.0 for s in wanted}
        if self.top_k is not None and len(weights) > self.top_k:
            keep = sorted(weights, key=lambda s: (-weights[s], s))[:self.top_k]
            weights = {s: weights[s] for s in keep}
            wanted = set(keep)
        return sum(weights[s] for s in wanted & held) / sum(weights.values())


class EntityWeightedX2(BaseSystem):
    """X2 plus bounded coverage; alpha zero retains the identical X2 formula."""

    rung, stage, can_explain = "EW", "entity-weighted lexical", True

    def __init__(self, alpha=0.0, title=False, weighted=True, extractor=None,
                 frequencies=None, rule=None, text_system=None):
        super().__init__()
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        self.alpha, self.title, self.weighted = float(alpha), title, weighted
        self.extractor = extractor or ConceptExtractor()
        self.frequencies = frequencies or DocumentFrequency(self.extractor)
        self.rule = rule or ExtractedRule()
        self.text = text_system or Tfidf(text_variant="v2")
        self.name = f"entity X2 (alpha={alpha:g}, title={title}, weighted={weighted})"

    def _fit(self, train):
        if train is not None:
            self.frequencies.fit(train)
        if self.alpha and self.weighted and not self.frequencies.fitted:
            raise ValueError("Weighted coverage requires explicit training CV frequencies")
        self.rule.fit(None)
        self.text.fit(None)
        self.trained_on = f"{self.rule.trained_on}; {self.text.trained_on}; train CV IDF"

    def inputs(self, pairs):
        x = pairs.copy()
        if self.title:
            if "jd_title" not in x:
                raise ValueError("Title variant requires given JD title")
            x["jd_text"] = x.jd_title.fillna("").astype(str) + "\n" + x.jd_text
        return x

    def components(self, pairs):
        x = self.inputs(pairs)
        c = self.rule.components(x).reset_index(drop=True)
        c["tfidf_scaled"] = per_jd_minmax(x.jd_id.to_numpy(), self.text.score(x).to_numpy())
        jd = [self.extractor.concepts(t, "jd") for t in x.jd_text]
        cv = [self.extractor.concepts(t, "cv") for t in x.cv_text]
        c["coverage"] = [self.frequencies.coverage(j, v, self.weighted)
                         for j, v in zip(jd, cv)]
        c["jd_concepts_available"] = [bool(j) for j in jd]
        c["cv_concepts_available"] = [bool(v) for v in cv]
        return c

    def _score(self, pairs):
        if self.alpha == 0:
            # Do not require extraction or IDF for the unchanged control.
            x = self.inputs(pairs)
            c = self.rule.components(x)
            t = per_jd_minmax(x.jd_id.to_numpy(), self.text.score(x).to_numpy())
            return 2 * c.role_match.to_numpy() + c.experience_compatible.to_numpy() + .999 * t
        c = self.components(pairs)
        return (2 * c.role_match.to_numpy() + c.experience_compatible.to_numpy()
                + .999 * ((1 - self.alpha) * c.tfidf_scaled.to_numpy()
                          + self.alpha * c.coverage.to_numpy()))

    def params(self):
        return self.text.params()
