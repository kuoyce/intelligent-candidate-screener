"""Part C of plan 2026-10-13-coverage-weighting-variants (amendment 1): which embedding threshold?

Coverage-only per-JD AUC of concept-level embedding matching at thresholds 0.50 ... 0.90, R+N,
paired against exact matching (K0). The 70 D50 selection JDs choose; the 30 D50 held-out JDs and
the A1-LLM pairs check. **A2 final is not computed** (it was seen at 0.7 / 0.8 / 0.9 in Part B).

    uv run python -m candidate_screener.evaluation.match_threshold
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import coverage_variants as cv, match_keys as mk

RECORD = OUTPUT / "extraction" / "match-threshold.json"
GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(9))
TIE = 0.002


def best_sims(pairs: pd.DataFrame, extractor, vec: dict) -> list[np.ndarray]:
    """Per pair: for each JD concept, its best cosine to any CV concept (0 if the CV has none)."""
    jd = {t: sorted(extractor.concepts(t, "jd")) for t in pairs.jd_text.unique()}
    cvc = {t: sorted(extractor.concepts(t, "cv")) for t in pairs.cv_text.unique()}
    out = []
    for jt, ct in zip(pairs.jd_text, pairs.cv_text):
        J, C = jd[jt], cvc[ct]
        if not J:
            out.append(np.zeros(0)); continue
        if not C:
            out.append(np.zeros(len(J))); continue
        out.append((np.array([vec[j] for j in J]) @ np.array([vec[c] for c in C]).T).max(axis=1))
    return out


def coverage_at(sims: list[np.ndarray], t: float, pair_ids) -> pd.Series:
    return pd.Series([float((s >= t).mean()) if len(s) else 0.0 for s in sims], index=pair_ids)


def pick_threshold(selection_auc: dict[float, float]) -> dict:
    """Highest selection AUC; ties within TIE go to the higher threshold; flag a grid-edge maximum."""
    best = max(selection_auc.values())
    chosen = max(t for t, a in selection_auc.items() if best - a <= TIE)
    return {"threshold": chosen, "at_lower_grid_edge": bool(chosen == min(selection_auc))}


def run() -> dict:
    from candidate_screener.extraction import concept_sources as cs
    ex = cs.make_extractors({"minilm": cs.CachedEntities("minilm", None)})["R+N"]
    sets = cv.evaluation_sets()
    phrases = set()
    for pairs, _ in sets.values():
        for t in pairs.jd_text.unique(): phrases |= set(ex.concepts(t, "jd"))
        for t in pairs.cv_text.unique(): phrases |= set(ex.concepts(t, "cv"))
    vec = mk.embed(phrases)
    out = {"grid": GRID, "tie": TIE, "sets": {}, "keys": {}}
    auc = {}
    for name, (pairs, y) in sets.items():
        ids = pairs.pair_id.to_numpy(); jd = pairs.set_index("pair_id").jd_id
        m = mk.match_table(pairs, ex, None)
        size = np.array([max(len(ex.concepts(t, "jd")), 1) for t in pairs.jd_text])
        for k in ("K0", "K1", "K2"):
            n = np.array([len(x) for x in m[k]])
            auc[(k, name)] = cv.per_jd_auc(pd.Series(n / size, index=ids), y, jd)
        sims = best_sims(pairs, ex, vec)
        for t in GRID:
            auc[(t, name)] = cv.per_jd_auc(coverage_at(sims, t, ids), y, jd)
    names = list(sets)
    for key in ("K0", "K1", "K2") + GRID:
        out["keys"][str(key)] = {n: {"auc": float(auc[(key, n)].mean()),
                                    **({} if key == "K0" else cv.paired(auc[(key, n)], auc[("K0", n)]))} for n in names}
    out["guidance"] = pick_threshold({t: out["keys"][str(t)][names[0]]["auc"] for t in GRID})
    t = str(out["guidance"]["threshold"])
    out["guidance"]["positive_vs_K0_on_both_heldout"] = bool(all(out["keys"][t][n]["delta"] > 0 for n in names[1:]))
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["guidance"]), RECORD)


if __name__ == "__main__":
    main()
