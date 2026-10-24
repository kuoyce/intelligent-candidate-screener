"""Every model scored against the LLM's 0-100 scores (owner request, 17 Oct 2026), exploratory.

D43 makes the human grades the only reference, and `ordering.reference` refuses the LLM. The owner
asked to see the LLM as a reference anyway, before deciding whether to change that rule. This
module reports only: nothing is fitted, tuned or selected on it, and it writes its own record.

Two pair sets:

- **final set** (600 A2 pairs), reference = the committed LLM series (runs 9/10), on `full` and
  `llm_free` (the human-split pairs dropped);
- **D50 validation** (200 run-11 pairs, 20 JDs), reference = the run-11 score. No model trained on
  these pairs, but R1-R3 were *selected* on them, and every model is from the same A2 train
  region (the IDF, the role model and SBERT weak's weak pairs may include these documents).

Metric: per-JD pairwise agreement with the reference (pairs whose reference scores differ; a
system tie counts 0.5) and nDCG with the 0-100 score as gain; mean over JDs, JD-resampled CI.
Each model is tagged by what it learned from (`LEARNED_FROM`). Score digests are checked against
the committed stage-1 and stage-2 records where those exist.

    uv run python -m candidate_screener.evaluation.llm_reference
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import ordering, resample

RECORD = OUTPUT / "matching" / "llm-reference.json"
LLM_TRAINED = "trained on LLM scores (D50, run 11)"
NO_LLM = "no LLM labels"
ORACLE = "no LLM labels; metadata oracle (not allowed at inference, D53)"
HUMAN = "human annotator"
#: What each system learned from, by name. R1-R3 are the only ones that saw an LLM score.
LEARNED_FROM = {"R1": LLM_TRAINED, "R2": LLM_TRAINED, "R3": LLM_TRAINED,
                "metadata rules": ORACLE, "metadata + TF-IDF": ORACLE, "yc": HUMAN, "van": HUMAN}


def per_jd(ref: pd.Series, scores: pd.Series, jd: pd.Series, metric: str) -> pd.Series:
    """Per-JD agreement of `scores` with `ref` over the pairs both cover; JDs with one ref value skipped."""
    d = pd.DataFrame({"r": ref, "s": scores, "j": jd}).dropna()
    fn = ordering.pairwise if metric == "pairwise" else ordering.ndcg
    return pd.Series({j: fn(g.r.to_numpy(), g.s.to_numpy()) for j, g in d.groupby("j", sort=True)
                      if g.r.nunique() > 1}, dtype=float).dropna()


def figure(ref, scores, jd, metric, seed=0) -> dict:
    v = per_jd(ref, scores, jd, metric)
    lo, hi = resample.mean_ci(v, seed=seed)
    return {"value": float(v.mean()), "ci": [lo, hi], "n_jd": int(len(v)),
            "n_pairs": int(pd.concat([ref, scores], axis=1).dropna().shape[0])}


def paired_vs(ref, b, a, jd, metric, seed=0) -> dict:
    d = (per_jd(ref, b, jd, metric) - per_jd(ref, a, jd, metric)).dropna()
    lo, hi = resample.mean_ci(d, seed=seed)
    return {"delta": float(d.mean()), "ci": [lo, hi], "n_jd": int(len(d))}


def _rerankers(pairs: pd.DataFrame, check: dict | None) -> dict:
    """R1-R3 from their cached selected weights (never trains); digests checked when `check`."""
    from candidate_screener.evaluation import stage2
    from candidate_screener.matching import neural
    from candidate_screener.rerank import batch, models
    s2 = json.loads(stage2.RECORD.read_text())
    digest = neural.file_digest(batch.SCORES_CSV)
    out = {}
    for r in (models.R1, models.R2, models.R3):
        config = models.grid(r)[s2["batch"]["rungs"][r]["selected"]]
        path = models.weights_dir(config, digest)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"{r}: no cached weights at {path}; this module never trains")
        system = models.system_for(config, path)
        system.fit(None)
        out[r] = system.score(pairs).set_axis(pairs.pair_id.to_numpy())
        del system
        models._release()
        if check is not None and ordering._digest(out[r].set_axis(check["index"])) != s2["systems"][r]["scores_sha256"]:
            raise ValueError(f"{r}: scores differ from the committed stage-2 record")
    return out


def _text_systems(pairs: pd.DataFrame, with_metadata: bool, check: bool) -> dict:
    from candidate_screener.evaluation import stage1
    committed = json.loads(stage1.RECORD.read_text())["systems"]
    out = {}
    for spec in stage1.specs():
        system = spec.factory()
        if system.name in ("metadata rules", "metadata + TF-IDF") and not with_metadata:
            continue
        system.fit(None)
        s = system.score(pairs)
        if check and stage1._sha(s) != committed[system.name]["scores_sha256"]:
            raise ValueError(f"{system.name}: scores differ from the committed stage-1 record")
        out[system.name] = s.set_axis(pairs.pair_id.to_numpy())
    return out


def block(ref: pd.Series, scores: dict, jd: pd.Series, scopes: dict, seed: int) -> dict:
    x2 = scores["extracted rule + TF-IDF"]
    out = {}
    for name, s in scores.items():
        out[name] = {"learned_from": LEARNED_FROM.get(name, NO_LLM), "cells": []}
        for (scope, subset), ids in scopes.items():
            for metric in ("pairwise", "ndcg"):
                k = ref.index.intersection(ids)
                cell = {"scope": scope, "subset": subset, "metric": metric,
                        **figure(ref.loc[k], s.reindex(k), jd.loc[k], metric, seed)}
                if name != "extracted rule + TF-IDF":
                    cell["vs_x2"] = paired_vs(ref.loc[k], s.reindex(k), x2.reindex(k), jd.loc[k], metric, seed)
                out[name]["cells"].append(cell)
    return out


def run(seed: int = 0) -> dict:
    from candidate_screener.evaluation import final_set, indomain, stage2
    from candidate_screener.matching import corpus
    from candidate_screener.rerank import batch
    # --- final set: reference = LLM runs 9/10 ---
    frame = final_set.load()
    lab = indomain.labels(frame)
    pairs = corpus.a2_pairs(text_variant="v2").assign(keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()
    scores = _text_systems(pairs, with_metadata=True, check=True)
    p2 = corpus.a2_pairs(text_variant=stage2.TEXT_VARIANT).set_index("pair_id").loc[frame.pair_id].reset_index()
    scores |= _rerankers(p2, {"index": p2.pair_id.to_numpy()})
    grades = ordering.grade_frame(frame)
    scores["yc"], scores["van"] = grades.yc, grades.van
    ref = ordering.llm_scores()
    f = frame.set_index("pair_id")
    clean = f.index[f.tier.isin(final_set.LLM_FREE_TIERS)]
    scopes = {}
    for subset, ids in (("full", f.index), ("llm_free", clean)):
        for scope in ("generic", "targeted", "total"):
            scopes[(scope, subset)] = ids if scope == "total" else ids.intersection(f.index[f.stratum == scope])
    final = block(ref, scores, f.jd_id, scopes, seed)
    # --- D50 validation: reference = run-11 score ---
    d = batch.load()
    val = d[d.split == "validation"].reset_index(drop=True)
    vs = _text_systems(val.assign(keyword_match=np.nan), with_metadata=False, check=False)
    vs |= _rerankers(val, None)
    vref = val.set_index("pair_id").score.astype(float)
    d50 = block(vref, vs, val.set_index("pair_id").jd_id, {("total", "all"): vref.index}, seed)
    return {"status": "exploratory, owner request 17 Oct 2026; the LLM as reference breaks D43 for reporting only",
            "final_set": {"reference": "LLM runs 9/10 (instrument 5d986c82, haiku), 0-100", "systems": final},
            "d50_validation": {"reference": "LLM run 11, 0-100", "n_pairs": int(len(val)),
                               "n_jds": int(val.jd_id.nunique()), "systems": d50}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    result = run(ap.parse_args(argv).seed)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
