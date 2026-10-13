"""Alternative concept sources for the coverage feature: GLiNER and our MiniLM skill tagger.

`ConceptExtractor(entities=...)` takes a callable `(text, side) -> entity rows`. Here each
source is wrapped as a disk-cached callable keyed by the text's SHA-256, so a model runs once
per distinct document. Only SKILL / KNOWLEDGE rows are kept, since those are all the coverage
feature reads. Caches are model outputs under `data/processed/` and are never committed (D23).
Text in, spans out: no label file is read (plan 2026-10-13-gliner-concepts-in-learned-weights).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from candidate_screener.config import PROCESSED

CACHE_DIR = PROCESSED / "extraction" / "concept-cache"
KEPT = {"SKILL", "KNOWLEDGE"}
SOURCES = ("gliner", "minilm")


def text_key(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class CachedEntities:
    """`source(text, side)` -> rows `{label, surface, context_status}`, computed once per text."""

    def __init__(self, name: str, compute, cache_dir: Path = CACHE_DIR):
        self.name, self.compute = name, compute
        self.path = Path(cache_dir) / f"{name}.json"
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.dirty = False

    def __call__(self, text: str, side: str = "cv") -> list[dict]:
        key = text_key(text)
        if key not in self.rows:
            self.rows[key] = [r for r in self.compute(str(text), side) if r["label"] in KEPT]
            self.dirty = True
        return self.rows[key]

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, separators=(",", ":")))
        tmp.replace(self.path)
        self.dirty = False


def minilm_rows(tagger):
    """Wrap `ner.NerTagger`: its (start, end) spans become SKILL rows with the rules context."""
    from candidate_screener.extraction.rules import _context

    def compute(text: str, side: str) -> list[dict]:
        return [{"label": "SKILL", "surface": text[a:b], "context_status": _context(text, a, b, "text")}
                for a, b in tagger.predict_spans(text)]
    return compute


def gliner_rows(model, threshold: float = 0.5):
    """Wrap a loaded GLiNER model; context is the rules-v1 heuristic that the adapter applies."""
    from candidate_screener.extraction.transformer import extract_transformer_document

    def compute(text: str, side: str) -> list[dict]:
        doc = {"doc_id": text_key(text)[:16], "source": "concept-cache", "doc_type": side,
               "fields": {"text": text}}
        records, _ = extract_transformer_document(doc, model, threshold=threshold)
        return [{"label": r["label"], "surface": r["surface"], "context_status": r["context_status"]}
                for r in records]
    return compute


class UnionConcepts:
    """Concepts of several extractors, merged. Exposes the one method the scorer calls."""

    def __init__(self, *extractors):
        self.extractors = extractors

    def concepts(self, text: str, side: str = "cv") -> frozenset[str]:
        return frozenset().union(*(e.concepts(text, side) for e in self.extractors))


def make_extractors(sources: dict[str, CachedEntities]) -> dict:
    """The five arms of the plan: R (seed rules), G, N, R+G, R+N."""
    from candidate_screener.matching.entity_weighted import ConceptExtractor
    r = ConceptExtractor()
    g = ConceptExtractor(entities=sources["gliner"]) if "gliner" in sources else None
    n = ConceptExtractor(entities=sources["minilm"]) if "minilm" in sources else None
    arms = {"R": r}
    if g:
        arms["G"], arms["R+G"] = g, UnionConcepts(r, g)
    if n:
        arms["N"], arms["R+N"] = n, UnionConcepts(r, n)
    return arms
