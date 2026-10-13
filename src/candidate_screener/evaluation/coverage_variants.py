"""Part A of plan 2026-10-13-coverage-weighting-variants: weighting variants of the coverage term.

Coverage-only per-JD AUC (no learned weights) of W0-W3 for the seed-rule arm R and the R+N arm,
on D50 selection JDs (70), D50 held-out JDs (30) and the A1-LLM pairs. Labels are read by this
scorer only; **A2 final never appears here** (selection discipline of the plan). The owner picks;
`guidance` only encodes the plan's tie rule and the D50-specific-win flag.

    uv run python -m candidate_screener.evaluation.coverage_variants
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import resample

RECORD = OUTPUT / "extraction" / "coverage-variants.json"
VARIANTS = {"W0": {}, "W1": {"unknown": "idf"}, "W2": {"unknown": "idf", "top_k": 10},
            "W3": {"unknown": "idf", "min_df": 5}}
SIMPLICITY = ("W0", "W1", "W3", "W2")
MIN_GAIN, TIE = 0.005, 0.002
ARMS = ("R", "R+N")


def split_jds(jd_ids, n_select: int = 70, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic selection / held-out split of sorted JD ids."""
    ids = sorted(set(jd_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    pick = [ids[i] for i in order]
    return sorted(pick[:n_select]), sorted(pick[n_select:])


def coverage_scores(freq, pairs: pd.DataFrame, extractor) -> pd.Series:
    jd = {t: extractor.concepts(t, "jd") for t in pairs.jd_text.unique()}
    cv = {t: extractor.concepts(t, "cv") for t in pairs.cv_text.unique()}
    return pd.Series([freq.coverage(jd[j], cv[c], True) for j, c in zip(pairs.jd_text, pairs.cv_text)],
                     index=pairs.pair_id.to_numpy())


def per_jd_auc(scores: pd.Series, y: pd.Series, jd: pd.Series) -> pd.Series:
    d = pd.DataFrame({"s": scores, "y": y.loc[scores.index], "j": jd.loc[scores.index]})
    return pd.Series({j: roc_auc_score(g.y, g.s) for j, g in d.groupby("j") if g.y.nunique() == 2})


def paired(a: pd.Series, b: pd.Series, seed: int = 0) -> dict:
    """Mean per-JD difference a - b over shared JDs, with a JD-resampled CI."""
    diff = (a - b).dropna()
    low, high = resample.mean_ci(diff, seed=seed)
    return {"delta": float(diff.mean()), "ci": [float(low), float(high)], "n_jd": int(len(diff))}


def pick(selection_auc: dict[str, float]) -> str:
    """Highest selection AUC; ties within TIE go to the simpler variant."""
    best = max(selection_auc.values())
    close = [v for v in SIMPLICITY if v in selection_auc and best - selection_auc[v] <= TIE]
    return close[0]


def evaluation_sets():
    """name -> (pairs, y) with text columns; D50 split into selection and held-out JDs."""
    from candidate_screener.evaluation import a1_weights as aw
    from candidate_screener.matching import corpus
    d50, y50 = aw.d50_pairs()
    sel, held = split_jds(d50.jd_id)
    llm = aw.llm_labels()
    a1 = corpus.a1_pairs("test").drop_duplicates("pair_id").set_index("pair_id", drop=False).loc[llm.index]
    return {"D50 selection (70 JDs)": (d50[d50.jd_id.isin(sel)], y50),
            "D50 held-out (30 JDs)": (d50[d50.jd_id.isin(held)], y50),
            "A1-LLM (639 pairs)": (a1, llm)}


def idf_frame() -> pd.DataFrame:
    from candidate_screener.evaluation.concept_overlap import idf_documents
    d = idf_documents()
    return pd.DataFrame({"id": d.cv_id, "text": d.cv_text, "region": "train", "doc_type": "cv"})


def run() -> dict:
    from candidate_screener.extraction import concept_sources as cs
    from candidate_screener.matching.entity_weighted import DocumentFrequency
    sets = evaluation_sets()
    arms = cs.make_extractors({"minilm": cs.CachedEntities("minilm", None)})
    idf = idf_frame()
    out = {"variants": VARIANTS, "min_gain": MIN_GAIN, "tie": TIE, "arms": {}}
    for arm in ARMS:
        ext = arms[arm]
        base = DocumentFrequency(ext).fit(idf)
        auc = {}
        for v, kw in VARIANTS.items():
            freq = base.variant(**kw)
            for name, (pairs, y) in sets.items():
                auc[(v, name)] = per_jd_auc(coverage_scores(freq, pairs, ext), y, pairs.set_index("pair_id").jd_id)
        table = {}
        for v in VARIANTS:
            table[v] = {name: {"auc": float(auc[(v, name)].mean()),
                               **({} if v == "W0" else paired(auc[(v, name)], auc[("W0", name)]))}
                        for name in sets}
        sel_name = next(iter(sets))
        chosen = pick({v: table[v][sel_name]["auc"] for v in VARIANTS})
        guidance = {"selected_by_rule": chosen}
        if chosen != "W0":
            guidance["gain_on_selection"] = table[chosen][sel_name]["delta"]
            guidance["meets_min_gain"] = bool(table[chosen][sel_name]["delta"] >= MIN_GAIN)
            guidance["positive_on_both_heldout"] = bool(all(table[chosen][n]["delta"] > 0 for n in list(sets)[1:]))
        out["arms"][arm] = {"table": table, "guidance": guidance}
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({a: r["guidance"] for a, r in result["arms"].items()}, indent=1), RECORD)


if __name__ == "__main__":
    main()
