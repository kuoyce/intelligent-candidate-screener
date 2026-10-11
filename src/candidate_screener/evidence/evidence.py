"""Per-requirement evidence from a CV, quoted verbatim or not at all.

    matched    the CV names the same canonical skill (exact or alias): its span is quoted
    supported  no name match, but a CV sentence is semantically close (SBERT cosine >= tau):
               that sentence is quoted
    missing    neither

**Faithfulness is an invariant, not an estimate:** every quoted span is a verbatim slice
of the CV, and `assert_faithful` refuses a result where it is not. `tau` is fixed on
SkillSpan validation (`calibrate_tau`), never on A2 (D46).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from candidate_screener.config import OUTPUT
from candidate_screener.evidence.requirements import Requirement

TAU_RECORD = OUTPUT / "evidence" / "tau.json"
SENTENCE = re.compile(r"[^\n.;•●▪|]+")


@dataclass(frozen=True)
class Evidence:
    requirement: str
    status: str                  # matched | supported | missing
    quote: str | None
    start: int | None
    end: int | None
    similarity: float | None


def sentences(text: str, min_chars: int = 12) -> list[tuple[int, int, str]]:
    """Character-offset sentence spans (newline, full stop, semicolon, bullet)."""
    out = []
    for m in SENTENCE.finditer(text):
        raw = m.group(0)
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if len(body) >= min_chars:
            start = m.start() + lead
            out.append((start, start + len(body), body))
    return out


def tau() -> float:
    if not TAU_RECORD.exists():
        raise FileNotFoundError(f"{TAU_RECORD} missing — run `python -m "
                                "candidate_screener.evidence.evidence --calibrate`")
    return float(json.loads(TAU_RECORD.read_text())["tau"])


def gather(requirements: list[Requirement], cv_text: str, matcher, encoder=None,
           threshold: float | None = None) -> list[Evidence]:
    """Evidence for every requirement. Without an encoder, only exact/alias matches count."""
    by_canonical: dict[str, tuple[int, int, str]] = {}
    for m in matcher.find(cv_text):
        by_canonical.setdefault(m.canonical, (m.start, m.end, m.text))
    unmatched = [r for r in requirements if r.canonical not in by_canonical]
    support: dict[str, tuple[float, tuple[int, int, str]]] = {}
    spans = sentences(cv_text)
    if encoder is not None and unmatched and spans:
        req_vecs = encoder.encode([r.text for r in unmatched])
        sent_vecs = encoder.encode([s[2] for s in spans])
        sims = req_vecs @ sent_vecs.T
        cut = tau() if threshold is None else threshold
        for i, r in enumerate(unmatched):
            j = int(np.argmax(sims[i]))
            if sims[i, j] >= cut:
                support[r.canonical] = (float(sims[i, j]), spans[j])
    out = []
    for r in requirements:
        if r.canonical in by_canonical:
            s, e, t = by_canonical[r.canonical]
            out.append(Evidence(r.canonical, "matched", t, s, e, None))
        elif r.canonical in support:
            sim, (s, e, t) = support[r.canonical]
            out.append(Evidence(r.canonical, "supported", t, s, e, round(sim, 4)))
        else:
            out.append(Evidence(r.canonical, "missing", None, None, None, None))
    assert_faithful(out, cv_text)
    return out


def assert_faithful(evidence: list[Evidence], cv_text: str) -> None:
    for ev in evidence:
        if ev.status == "missing":
            continue
        if ev.start is None or cv_text[ev.start:ev.end] != ev.quote:
            raise ValueError(f"evidence for {ev.requirement!r} is not a verbatim slice of the "
                             "CV — refusing to show a quote the document does not contain")


def coverage(evidence: list[Evidence]) -> float | None:
    if not evidence:
        return None
    return sum(ev.status != "missing" for ev in evidence) / len(evidence)


def as_dicts(evidence: list[Evidence]) -> list[dict]:
    return [asdict(ev) for ev in evidence]


# --- tau, fixed on SkillSpan validation ----------------------------------------------

def calibrate_tau(seed: int = 0, n: int = 1500) -> dict:
    """Skill phrase vs its own sentence (positive) or another sentence (negative), from
    SkillSpan validation; tau maximises F1 of 'this sentence supports this skill'."""
    import pandas as pd
    from candidate_screener.config import RAW
    from candidate_screener.matching.sbert import Encoder
    from candidate_screener.matching.neural import MODEL_ID
    val = pd.read_parquet(RAW / "skillspan" / "default-validation-0000.parquet")
    pairs = []
    for tokens, a, b in zip(val.tokens, val.tags_skill, val.tags_knowledge):
        spans, cur = [], []
        for tok, t1, t2 in zip(tokens, a, b):
            tag = "B" if "B" in (t1[:1], t2[:1]) else ("I" if "I" in (t1[:1], t2[:1]) else "O")
            if tag == "B" or (tag == "I" and not cur):
                if cur:
                    spans.append(" ".join(cur))
                cur = [tok]
            elif tag == "I":
                cur.append(tok)
            else:
                if cur:
                    spans.append(" ".join(cur))
                cur = []
        if cur:
            spans.append(" ".join(cur))
        sentence = " ".join(tokens)
        pairs += [(s, sentence) for s in spans]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    phrases = [pairs[i][0] for i in idx]
    own = [pairs[i][1] for i in idx]
    other = [pairs[j][1] for j in rng.permutation(idx)]
    keep = [k for k in range(len(idx)) if own[k] != other[k]]
    enc = Encoder(MODEL_ID, "minilm-zero-shot-evidence")
    p = enc.encode([phrases[k] for k in keep])
    pos = np.einsum("ij,ij->i", p, enc.encode([own[k] for k in keep]))
    neg = np.einsum("ij,ij->i", p, enc.encode([other[k] for k in keep]))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    grid = np.round(np.arange(0.10, 0.80, 0.01), 2)
    f1 = []
    for t in grid:
        pred = s >= t
        tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
        f1.append(2 * tp / (2 * tp + fp + fn))
    best = int(np.argmax(f1))
    return {"tau": float(grid[best]), "f1_at_tau": round(float(f1[best]), 4),
            "n_positive": int(len(pos)), "n_negative": int(len(neg)),
            "source": "SkillSpan validation: skill phrase vs its own sentence / a random other",
            "model": MODEL_ID, "seed": seed}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        rec = calibrate_tau()
        TAU_RECORD.parent.mkdir(parents=True, exist_ok=True)
        TAU_RECORD.write_text(json.dumps(rec, indent=2) + "\n")
        print(rec)
