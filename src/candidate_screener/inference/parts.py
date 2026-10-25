"""Fitted parts saved for inference, so start-up never refits (plan 2026-10-23-inference-api).

`a1_weights.build_scorer` refits two parts at every start: the concept document frequencies
(over 20k train CVs minus the dev-pool CVs, which calls `dev_pools.build()`) and the TF-IDF space
(`lexical.a2_vectorizer`, 20k + 20k train documents). `cache_parts()` saves both once, without
pickle, under `data/processed/inference/parts/<fingerprint>/`; `load_parts()` rebuilds them.

- `concept_df.parquet` (concept, df) + `parts.json` (`n_documents`, config, provenance);
- `tfidf_vocabulary.json` (terms in column order) + `tfidf_idf.npy`.

The fingerprint digests the saved content, the role model's fingerprint and the vocabulary file,
so a retrained role model or a changed vocabulary gets a new directory.

    uv run python -m candidate_screener.inference --cache-parts
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED

ROOT = PROCESSED / "inference" / "parts"
FILES = ("concept_df.parquet", "tfidf_vocabulary.json", "tfidf_idf.npy")


@dataclass(frozen=True)
class Parts:
    frequencies: object        # entity_weighted.DocumentFrequency, fitted
    vectorizer: object         # sklearn TfidfVectorizer, transform-ready
    fingerprint: str
    info: dict


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_parts(out: Path, frequencies, vectorizer, provenance: dict) -> str:
    """Write the parts to `out/<fingerprint>/` and point `out/LATEST` at it; return the fingerprint."""
    from candidate_screener.baselines.tfidf import TFIDF_CONFIG
    tmp = Path(out) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sorted(frequencies.df.items()), columns=["concept", "df"])
    df.to_parquet(tmp / FILES[0], index=False)
    vocab = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    (tmp / FILES[1]).write_text(json.dumps(vocab), encoding="utf-8")
    np.save(tmp / FILES[2], np.asarray(vectorizer.idf_, dtype=float))
    digests = {f: _sha(tmp / f) for f in FILES}
    config = {k: vectorizer.get_params()[k] for k in TFIDF_CONFIG}
    info = {"n_documents": int(frequencies.n_documents), "tfidf_config": config,
            "files": digests, **provenance}
    fp = hashlib.sha256(json.dumps({"files": digests, **provenance}, sort_keys=True,
                                   default=str).encode()).hexdigest()[:12]
    (tmp / "parts.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
    final = Path(out) / fp
    if final.exists():
        for p in final.iterdir():
            p.unlink()
        final.rmdir()
    tmp.rename(final)
    (Path(out) / "LATEST").write_text(fp + "\n", encoding="utf-8")
    return fp


def load_parts(fingerprint: str | None = None, root: Path = ROOT, extractor=None) -> Parts:
    """The saved concept frequencies and TF-IDF space, rebuilt; `extractor` defaults to the seed rules."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from candidate_screener.matching.entity_weighted import ConceptExtractor, DocumentFrequency
    root = Path(root)
    if fingerprint is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(f"no inference parts under {root}: run "
                                    "`uv run python -m candidate_screener.inference --cache-parts`")
        fingerprint = latest.read_text(encoding="utf-8").strip()
    d = root / fingerprint
    info = json.loads((d / "parts.json").read_text(encoding="utf-8"))
    for f, digest in info["files"].items():
        if _sha(d / f) != digest:
            raise ValueError(f"{d / f} does not match its recorded digest")
    freq = DocumentFrequency(extractor or ConceptExtractor())
    table = pd.read_parquet(d / FILES[0])
    freq.df = Counter(dict(zip(table.concept, table.df.astype(int))))
    freq.n_documents, freq.fitted = int(info["n_documents"]), True
    config = dict(info["tfidf_config"])
    config["ngram_range"] = tuple(config["ngram_range"])
    vec = TfidfVectorizer(**config)
    vocab = json.loads((d / FILES[1]).read_text(encoding="utf-8"))
    vec.vocabulary_ = {t: i for i, t in enumerate(vocab)}
    vec.idf_ = np.load(d / FILES[2])
    return Parts(freq, vec, fingerprint, info)


def cache_parts(root: Path = ROOT) -> str:
    """Fit the parts once, exactly as `a1_weights.build_scorer` does, and save them."""
    from candidate_screener.evaluation import a1_weights
    from candidate_screener.extraction.rules import VOCAB_CSV
    from candidate_screener.baselines.tfidf import vocabulary_digest
    scorer = a1_weights.build_scorer()
    vec = scorer.text.vectorizer
    provenance = {"role_model": scorer.rule.role_model.fingerprint, "vocabulary_sha256": _sha(VOCAB_CSV),
                  "tfidf_vocabulary_digest": vocabulary_digest(vec),
                  "source": "a1_weights.build_scorer(): concept df over 20k A2 train CVs minus dev-pool "
                            "CVs; TF-IDF = lexical.a2_vectorizer('v2')"}
    fp = save_parts(root, scorer.frequencies, vec, provenance)
    print(f"inference parts -> {Path(root) / fp} ({scorer.frequencies.n_documents} CVs, "
          f"{len(scorer.frequencies.df)} concepts, {len(vec.vocabulary_)} TF-IDF terms)")
    return fp
