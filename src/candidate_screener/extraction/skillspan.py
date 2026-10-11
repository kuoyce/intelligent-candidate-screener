"""SkillSpan (B2) as the extraction benchmark: sentences, gold spans, training tags.

SkillSpan ships tokens with two independent BIO layers, `tags_skill` and
`tags_knowledge`. The headline label is **SKILL = skill ∪ knowledge**: as gold, the
union keeps both layers' spans (they overlap in ~2% of spans, so a system emitting
non-overlapping spans has a small strict ceiling below 1). As a *training* target the
union must be one tagging, so a cross-layer overlap keeps the longer span (tie: skill).

Evaluation is on the token grid: a character-level prediction is snapped outward to
the SkillSpan tokens it touches, so the rules matcher and the tagger are scored on the
same units.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from candidate_screener.config import RAW

SPLITS = ("train", "validation", "test")
LAYERS = ("skill", "knowledge")
SCOPES = ("union", "skill", "knowledge")


@lru_cache(maxsize=None)
def load(split: str) -> pd.DataFrame:
    files = sorted((RAW / "skillspan").glob(f"*-{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no SkillSpan {split} parquet under {RAW / 'skillspan'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in ("tokens", "tags_skill", "tags_knowledge"):
        df[c] = df[c].map(lambda v: [str(x) for x in v])
    return df


def bio_spans(tags: list[str]) -> list[tuple[int, int]]:
    """Token spans `(start, end)` from BIO; an `I` with no open span starts one."""
    out, start = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "B" or t == "O" or (t == "I" and start is None):
            if start is not None:
                out.append((start, i))
                start = None
            if t in ("B", "I"):
                start = i
    return out


def gold_spans(row, scope: str) -> set[tuple[int, int]]:
    layers = LAYERS if scope == "union" else (scope,)
    return {s for layer in layers for s in bio_spans(row[f"tags_{layer}"])}


def join_tokens(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Tokens joined by single spaces, with each token's character offsets."""
    offsets, pos = [], 0
    for t in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t) + 1
    return " ".join(tokens), offsets


def snap(char_spans, offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Character spans → token spans they touch, merging any that then overlap."""
    ranges = []
    for cs, ce in char_spans:
        hit = [i for i, (s, e) in enumerate(offsets) if s < ce and cs < e]
        if hit:
            ranges.append((hit[0], hit[-1] + 1))
    merged: list[list[int]] = []
    for s, e in sorted(ranges):
        if merged and s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return {(s, e) for s, e in merged}


def training_tags(row) -> list[str]:
    """One BIO tagging of the union; a cross-layer overlap keeps the longer span."""
    spans = [(s, e, 0) for s, e in bio_spans(row["tags_skill"])]
    spans += [(s, e, 1) for s, e in bio_spans(row["tags_knowledge"])]
    spans.sort(key=lambda x: (-(x[1] - x[0]), x[2], x[0]))
    tags = ["O"] * len(row["tokens"])
    taken = [False] * len(tags)
    for s, e, _ in spans:
        if any(taken[s:e]):
            continue
        for i in range(s, e):
            tags[i] = "B" if i == s else "I"
            taken[i] = True
    return tags


@lru_cache(maxsize=None)
def prepared(split: str, scope: str) -> tuple:
    """`(text, token offsets, gold token spans)` per sentence, computed once per split."""
    out = []
    for toks, sk, kn in zip(*(load(split)[c] for c in ("tokens", "tags_skill", "tags_knowledge"))):
        text, offs = join_tokens(toks)
        out.append((text, offs, frozenset(gold_spans(
            {"tags_skill": sk, "tags_knowledge": kn}, scope))))
    return tuple(out)
