"""Alias normalisation: string matching against SBERT nearest label (task 5.4).

The gold set is free: an ESCO alternative label names its concept, so holding out a
seeded sample of alternative labels and mapping each back measures how well a method
recovers the canonical skill for a surface form it has never seen. A held-out label is
used only if it belongs to exactly one concept and is not itself (after normalisation)
any concept's preferred label, so every query has one unambiguous answer.

Methods, each predicting one concept per query:

- `exact`, `normalised` (casefold, punctuation to space), `token_overlap` (best Jaccard
  over normalised tokens; no shared token = no prediction), and `string`, the cascade of
  the three in that order;
- `sbert`: nearest index label by cosine, with the cached `all-MiniLM-L6-v2`.

`exact` and `normalised` score 0 by construction (no held-out label equals an index
label after normalisation); they stay in the record to show the cascade has no trivial
hits. A string-then-SBERT hybrid would therefore equal `sbert`, so it is not reported.

Index settings: `preferred` (one preferred label per concept, the plan's definition)
and `aliases` (plus every alternative label that was not held out).
"""
from __future__ import annotations

import collections
import os
import re

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

N_HELD_OUT = 1000
METHODS = ("exact", "normalised", "token_overlap", "string", "sbert")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def norm(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def gold_candidates(skills: pd.DataFrame) -> pd.DataFrame:
    """Unambiguous `(alt_label, concept, preferred)` rows from an ESCO skills frame."""
    sk = skills.drop_duplicates("conceptUri")
    preferred_norm = set(sk.preferredLabel.map(norm))
    rows = [(a.strip(), uri, pref) for uri, pref, alts in
            zip(sk.conceptUri, sk.preferredLabel, sk.altLabels)
            for a in str(alts).split("\n") if a.strip()]
    df = pd.DataFrame(rows, columns=["alt_label", "concept", "preferred"])
    df["key"] = df.alt_label.map(norm)
    owners = df.groupby("key").concept.nunique()
    df = df[df.key.map(owners).eq(1) & ~df.key.isin(preferred_norm) & df.key.ne("")]
    return df.drop_duplicates("key").sort_values(["key", "concept"]).reset_index(drop=True)


def held_out(skills: pd.DataFrame, n: int = N_HELD_OUT, seed: int = 0) -> pd.DataFrame:
    cand = gold_candidates(skills)
    idx = np.random.default_rng(seed).choice(len(cand), size=min(n, len(cand)), replace=False)
    return cand.iloc[np.sort(idx)].reset_index(drop=True)


def build_index(skills: pd.DataFrame, queries: pd.DataFrame, setting: str) -> pd.DataFrame:
    """`(label, concept)` strings a method may match against; held-out labels excluded."""
    sk = skills.drop_duplicates("conceptUri")
    rows = list(zip(sk.preferredLabel, sk.conceptUri))
    if setting == "aliases":
        banned = set(queries.key)
        rows += [(a.strip(), uri) for uri, alts in zip(sk.conceptUri, sk.altLabels)
                 for a in str(alts).split("\n") if a.strip() and norm(a) not in banned]
    elif setting != "preferred":
        raise ValueError(setting)
    idx = pd.DataFrame(rows, columns=["label", "concept"]).drop_duplicates()
    return idx.sort_values(["label", "concept"]).reset_index(drop=True)


def string_predictions(queries: list[str], index: pd.DataFrame) -> dict[str, list]:
    """Per query, the concept each string method predicts (None = no prediction)."""
    exact = dict(zip(index.label[::-1], index.concept[::-1]))  # first row wins
    normed = dict(zip(index.label.map(norm)[::-1], index.concept[::-1]))
    tok_sets = [frozenset(norm(l).split()) for l in index.label]
    inverted: dict[str, list[int]] = collections.defaultdict(list)
    for i, toks in enumerate(tok_sets):
        for t in toks:
            inverted[t].append(i)
    out = {m: [] for m in ("exact", "normalised", "token_overlap", "string")}
    for q in queries:
        e, n = exact.get(q), normed.get(norm(q))
        qt = frozenset(norm(q).split())
        best, best_i = 0.0, None
        for i in sorted({i for t in qt for i in inverted.get(t, ())}):
            j = len(qt & tok_sets[i]) / len(qt | tok_sets[i])
            if j > best:
                best, best_i = j, i
        t = None if best_i is None else index.concept.iloc[best_i]
        out["exact"].append(e)
        out["normalised"].append(e or n)
        out["token_overlap"].append(t)
        out["string"].append(e or n or t)
    return out


def sbert_predictions(queries: list[str], index: pd.DataFrame, encoder) -> list:
    q = encoder(queries)
    x = encoder(list(index.label))
    best = (q @ x.T).argmax(axis=1)
    return [index.concept.iloc[i] for i in best]


def minilm_encoder(threads: int = 4):
    """`texts -> L2-normalised float32 matrix`, from the cached MiniLM."""
    import torch
    from sentence_transformers import SentenceTransformer

    from candidate_screener.extraction.ner import base_model_path
    torch.set_num_threads(threads)
    model = SentenceTransformer(base_model_path(), device="cpu", local_files_only=True)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=256, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    return encode


def accuracy(correct: np.ndarray, seed: int = 0, n_boot: int = 10_000) -> dict:
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([correct[rng.integers(0, n, size=(1000, n))].mean(axis=1)
                            for _ in range(n_boot // 1000)])
    return {"n": int(n), "top1": round(float(correct.mean()), 4),
            "top1_ci": [round(float(np.percentile(boots, 2.5)), 4),
                        round(float(np.percentile(boots, 97.5)), 4)]}


def evaluate(skills: pd.DataFrame, setting: str, encoder=None, seed: int = 0,
             n: int = N_HELD_OUT) -> dict:
    """Top-1 accuracy of every method on one scope/index setting."""
    q = held_out(skills, n, seed)
    index = build_index(skills, q, setting)
    preds = string_predictions(list(q.alt_label), index)
    if encoder is not None:
        preds["sbert"] = sbert_predictions(list(q.alt_label), index, encoder)
    gold = list(q.concept)
    ok_by = {m: np.array([p == g for p, g in zip(preds[m], gold)]) for m in preds}
    out = {"n_queries": int(len(q)), "n_concepts": int(skills.conceptUri.nunique()),
           "n_index_labels": int(len(index)), "n_candidates": int(len(gold_candidates(skills))),
           "methods": {}}
    for m in METHODS:
        if m in preds:
            out["methods"][m] = {**accuracy(ok_by[m], seed),
                                 "coverage": round(float(np.mean([p is not None for p in preds[m]])), 4)}
    if "sbert" in ok_by:
        out["sbert_minus_string"] = paired_accuracy_delta(ok_by["string"], ok_by["sbert"], seed)
    return out


def paired_accuracy_delta(a: np.ndarray, b: np.ndarray, seed: int = 0,
                          n_boot: int = 10_000) -> dict:
    """acc(b) − acc(a), resampling the same queries for both methods."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, n, size=(1000, n))].mean(axis=1)
                            for _ in range(n_boot // 1000)])
    return {"n": int(n), "delta": round(float(d.mean()), 4),
            "delta_ci": [round(float(np.percentile(boots, 2.5)), 4),
                         round(float(np.percentile(boots, 97.5)), 4)]}
