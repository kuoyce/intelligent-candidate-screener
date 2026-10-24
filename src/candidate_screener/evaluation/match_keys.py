"""Part B of plan 2026-10-13-coverage-weighting-variants: what does exact-string matching miss?

A diagnostic on the cached R+N concepts. For each match key the JD's concepts are matched
against the CV's concepts, and we report how many pairs gain overlap against the current exact
key (K0). Nothing here is adopted or fed to a model, and nothing is chosen on it.

    K0 exact concept id | K1 alphanumeric key with plural and `.js` folding
    K2 K1 or token containment (shorter phrase >= 4 chars) | K3 embedding cosine >= 0.7 / 0.8 / 0.9

    uv run python -m candidate_screener.evaluation.match_keys
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED
from candidate_screener.evaluation.coverage_variants import per_jd_auc

RECORD = OUTPUT / "extraction" / "match-keys.json"
SAMPLES_FILE = PROCESSED / "extraction" / "match-keys-samples.json"   # phrases from real documents: not committed
THRESHOLDS = (0.7, 0.8, 0.9)
KEYS = ("K0", "K1", "K2") + tuple(f"K3@{t}" for t in THRESHOLDS)
SAMPLES = 30


def phrase(concept: str) -> str:
    return concept[len("unknown:"):] if concept.startswith("unknown:") else concept


def tokens(p: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9+#]+", p.casefold()))


def k1(p: str) -> str:
    s = re.sub(r"[^a-z0-9+#]", "", p.casefold())
    if len(s) > 4 and s.endswith("js"):
        s = s[:-2]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def contains(a: frozenset, b: frozenset) -> bool:
    """One token set holds the other, and the shorter phrase has >= 4 characters."""
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return bool(small) and small <= big and len(" ".join(sorted(small))) >= 4


def matched_exact(J, C) -> set:
    return set(J) & set(C)


def matched_k1(J, C) -> set:
    keys = {k1(phrase(c)) for c in C}
    return {j for j in J if k1(phrase(j)) in keys}


def matched_k2(J, C) -> set:
    hit = matched_k1(J, C)
    toks = [tokens(phrase(c)) for c in C]
    for j in set(J) - hit:
        tj = tokens(phrase(j))
        if any(contains(tj, t) for t in toks):
            hit.add(j)
    return hit


def matched_embedding(J, C, vec: dict, threshold: float) -> set:
    if not J or not C:
        return set()
    J, C = list(J), list(C)
    sim = np.array([vec[j] for j in J]) @ np.array([vec[c] for c in C]).T
    return {j for j, m in zip(J, sim.max(axis=1)) if m >= threshold}


def embed(phrases) -> dict:
    from sentence_transformers import SentenceTransformer
    from candidate_screener.extraction import ner
    model = SentenceTransformer(ner.base_model_path(), device="cpu")
    ps = sorted(phrases)
    v = model.encode([phrase(p) for p in ps], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(ps, v))


def match_table(pairs: pd.DataFrame, extractor, vec: dict | None) -> dict[str, list[set]]:
    """key -> per-pair set of matched JD concepts."""
    jd = {t: extractor.concepts(t, "jd") for t in pairs.jd_text.unique()}
    cv = {t: extractor.concepts(t, "cv") for t in pairs.cv_text.unique()}
    out = {k: [] for k in KEYS}
    for jt, ct in zip(pairs.jd_text, pairs.cv_text):
        J, C = jd[jt], cv[ct]
        out["K0"].append(matched_exact(J, C)); out["K1"].append(matched_k1(J, C)); out["K2"].append(matched_k2(J, C))
        for t in THRESHOLDS:
            out[f"K3@{t}"].append(matched_embedding(J, C, vec, t) if vec else set())
    return out


def summarise(pairs: pd.DataFrame, y: pd.Series, extractor, matches: dict) -> dict:
    jd_n = {t: len(extractor.concepts(t, "jd")) for t in pairs.jd_text.unique()}
    size = np.array([max(jd_n[t], 1) for t in pairs.jd_text])
    ids = pairs.pair_id.to_numpy(); yy = y.loc[ids].to_numpy(); jd = pairs.set_index("pair_id").jd_id
    base = np.array([len(m) for m in matches["K0"]])
    res = {}
    for k, ms in matches.items():
        n = np.array([len(m) for m in ms])
        gained, still = (base == 0) & (n > 0), n == 0
        res[k] = {"no_overlap_share": float((n == 0).mean()), "mean_matched": float(n.mean()),
                  "pairs_gained_vs_K0": int(gained.sum()),
                  "fit_rate_gained": float(yy[gained].mean()) if gained.any() else None,
                  "fit_rate_still_empty": float(yy[still].mean()) if still.any() else None,
                  "coverage_only_auc": float(per_jd_auc(pd.Series(n / size, index=ids), y, jd).mean())}
    return res


def samples(pairs: pd.DataFrame, extractor, matches: dict, seed: int = 0) -> dict:
    """Up to SAMPLES (JD phrase, matched CV phrase) examples per key that K0 does not match."""
    from candidate_screener.annotation.redact import redact_text
    jd = {t: extractor.concepts(t, "jd") for t in pairs.jd_text.unique()}
    cv = {t: extractor.concepts(t, "cv") for t in pairs.cv_text.unique()}
    rng = np.random.default_rng(seed); out = {}
    for k in KEYS[1:]:
        cand = []
        for i, (jt, ct) in enumerate(zip(pairs.jd_text, pairs.cv_text)):
            for j in matches[k][i] - matches["K0"][i]:
                cand.append((phrase(j), sorted(phrase(c) for c in cv[ct] if k1(phrase(c)) == k1(phrase(j)) or contains(tokens(phrase(c)), tokens(phrase(j))))[:1]))
        pick = [cand[i] for i in rng.permutation(len(cand))[:SAMPLES]]
        out[k] = [[redact_text(a), redact_text(b[0]) if b else "(embedding match)"] for a, b in pick]
    return out


def run() -> dict:
    from candidate_screener.evaluation import a1_weights as aw, final_set, indomain
    from candidate_screener.extraction import concept_sources as cs
    from candidate_screener.matching import corpus
    ex = cs.make_extractors({"minilm": cs.CachedEntities("minilm", None)})["R+N"]
    lab = indomain.labels(final_set.load())
    a2 = corpus.a2_pairs(text_variant="v2").set_index("pair_id").loc[lab.index].reset_index()
    d50, y50 = aw.d50_pairs()
    llm = aw.llm_labels()
    a1 = corpus.a1_pairs("test").drop_duplicates("pair_id").set_index("pair_id", drop=False).loc[llm.index]
    sets = {"A2 final": (a2, lab.y), "D50": (d50, y50), "A1-LLM": (a1, llm)}
    phrases = set()
    for p, _ in sets.values():
        for t in p.jd_text.unique(): phrases |= set(ex.concepts(t, "jd"))
        for t in p.cv_text.unique(): phrases |= set(ex.concepts(t, "cv"))
    vec = embed(phrases)
    out = {"keys": KEYS, "distinct_phrases": len(phrases), "sets": {}, "samples": {}}
    for name, (pairs, y) in sets.items():
        m = match_table(pairs, ex, vec)
        out["sets"][name] = summarise(pairs, y, ex, m)
        out["samples"][name] = samples(pairs, ex, m)
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_FILE.write_text(json.dumps(result.pop("samples"), indent=2, sort_keys=True) + "\n")
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(RECORD, SAMPLES_FILE)


if __name__ == "__main__":
    main()
