"""Pool scaling: how much does ranking a small upload pool move L1's order? (plan
2026-10-23-inference-api, step 2; exploratory.)

The chatbot ranks uploads only, so its pools are small. `tfidf_scaled` is the TF-IDF cosine
min-max scaled within the pool, and L1 was fitted on rows of 500-CV pools. Two options:

- **A** scale over the uploaded CVs only;
- **B** scale over a 500-CV pool (the tested condition; the uploads padded with background CVs).

Every other F12 feature is per pair, so A and B differ in `tfidf_scaled` alone. Within a pool
both are affine maps of the same cosine, so A changes the order only by changing how much
TF-IDF weighs against the other features.

Design (fixed before anything was computed):

- **Data:** A2 train-region dev pools only (`dev_pools.build()`), never the final set (D46).
  50 JDs drawn with seed 0 from the 400 dev-pool JDs; each has a 500-CV pool.
- **B:** `learned.Features()` on each full pool, L1 = `inference.model.L1Model.load()`. B's
  score of a CV is its L1 score inside its 500 pool.
- **Sub-pools:** n in {5, 10, 20, 50, 100}, R = 20 per JD per n, seeded, in two scenarios:
  `random` (uniform from the 500) and `relevant` (uniform from the pool's L1 top 50, ties by
  pair id: applicants are likelier to be relevant). `relevant` has no n = 100 (the top 50
  cannot supply it), and at n = 50 every draw is the whole top 50.
- **A:** within the sub-pool, `tfidf_scaled` = min-max of the raw TF-IDF cosine, then L1.
  The per-pair claim is checked in code: full features are recomputed on two sub-pools and
  every other F12 column must equal the 500-pool value.
- **Per sub-pool, A vs B on the same CVs:** Kendall tau (tau-b); top-1 agreement (ties by
  pair id); overlap of the top k sets, k = min(10, n // 2); mean |score_A - score_B|.
  Averaged per JD, then the mean over JDs with a 95% bootstrap CI over JDs
  (`resample.mean_ci`, 1,000 resamples, seed 0). No strata: dev pools have none.
- **Context:** X2 (2 role_match + experience_compatible + 0.999 tfidf_scaled) under the same
  A/B switch.

Reads no label. Record: `output/inference/pool-scaling.json`.

    uv run python -m candidate_screener.evaluation.pool_scaling
    uv run python -m candidate_screener.evaluation.pool_scaling --n-jds 1 --no-write   # timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT

RECORD = OUTPUT / "inference" / "pool-scaling.json"
SEED = 0
N_JDS = 50
SIZES = (5, 10, 20, 50, 100)
REPS = 20
TOP_RELEVANT = 50
N_BOOT = 1_000
SCENARIOS = ("random", "relevant")
METRICS = ("kendall_tau", "top1_agree", "topk_overlap", "mean_abs_delta")
CONFIG = {"seed": SEED, "n_jds": N_JDS, "sizes": list(SIZES), "reps": REPS,
          "relevant_from_top": TOP_RELEVANT, "n_boot": N_BOOT, "text_variant": "v2",
          "data": "A2 train-region dev pools (dev_pools.build), 500 CVs per JD; no final-set JD"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def minmax(v: np.ndarray) -> np.ndarray:
    """`v` scaled into [0, 1]; a constant vector gets 0 (as `role_filter.per_jd_minmax`)."""
    v = np.asarray(v, dtype=float)
    span = v.max() - v.min()
    return np.clip((v - v.min()) / (span if span > 0 else 1.0), 0, 1)


def order(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Positions by score descending, ties by id."""
    return np.lexsort((np.asarray(ids), -np.asarray(scores, dtype=float)))


def rescale(X: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    """Option A: the same rows with `tfidf_scaled` min-maxed over these rows' raw cosine."""
    return X.assign(tfidf_scaled=minmax(raw))


def compare(ids: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """A vs B on the same CVs: Kendall tau-b, top-1 agreement, top-k overlap, mean |a - b|."""
    from scipy.stats import kendalltau
    ids, a, b = np.asarray(ids), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    k = max(1, min(10, len(ids) // 2))
    oa, ob = order(ids, a), order(ids, b)
    tau = kendalltau(a, b).statistic
    return {"kendall_tau": float(tau) if np.isfinite(tau) else 1.0,
            "top1_agree": float(oa[0] == ob[0]),
            "topk_overlap": len(set(oa[:k]) & set(ob[:k])) / k,
            "mean_abs_delta": float(np.mean(np.abs(a - b)))}


def x2_score(X: pd.DataFrame) -> np.ndarray:
    return (2 * X.role_match.to_numpy() + X.experience_compatible.to_numpy()
            + 0.999 * X.tfidf_scaled.to_numpy())


def draws(ids: np.ndarray, b_scores: np.ndarray, n: int, scenario: str,
          rng: np.random.Generator) -> np.ndarray | None:
    """Positions of one sub-pool of size `n`, or None when the scenario cannot supply it."""
    source = np.arange(len(ids)) if scenario == "random" else order(ids, b_scores)[:TOP_RELEVANT]
    if n > len(source):
        return None
    return np.sort(rng.choice(source, n, replace=False))


def pick_jds(pools: pd.DataFrame, n: int = N_JDS, seed: int = SEED) -> list[str]:
    jds = np.array(sorted(pools.jd_id.unique()))
    return sorted(np.random.default_rng(seed).choice(jds, n, replace=False).tolist())


def subpool_rows(frame: pd.DataFrame, X: pd.DataFrame, raw: pd.Series, l1,
                 seed: int = SEED) -> pd.DataFrame:
    """One row per (JD, scenario, n, rep) for L1 and X2: the four metrics of A vs B."""
    rows = []
    for j, (jd, g) in enumerate(frame.groupby("jd_id", sort=True)):
        ids = g.pair_id.to_numpy()
        xb = X.loc[ids]
        r = raw.loc[ids].to_numpy()
        b = {"L1": l1.predict(xb), "X2": x2_score(xb)}
        for s, scenario in enumerate(SCENARIOS):
            for n in SIZES:
                rng = np.random.default_rng([seed, j, s, n])
                for rep in range(REPS):
                    pos = draws(ids, b["L1"], n, scenario, rng)
                    if pos is None:
                        break
                    xa = rescale(xb.iloc[pos], r[pos])
                    a = {"L1": l1.predict(xa), "X2": x2_score(xa)}
                    for system in ("L1", "X2"):
                        rows.append({"jd_id": jd, "scenario": scenario, "n": n, "rep": rep,
                                     "system": system, **compare(ids[pos], a[system], b[system][pos])})
    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> list[dict]:
    from candidate_screener.evaluation.resample import mean_ci
    out = []
    for (system, scenario, n), g in rows.groupby(["system", "scenario", "n"], sort=True):
        per_jd = g.groupby("jd_id")[list(METRICS)].mean()
        row = {"system": system, "scenario": scenario, "n": int(n), "jds": int(len(per_jd)),
               "reps_per_jd": int(g.groupby("jd_id").size().max())}
        for m in METRICS:
            lo, hi = mean_ci(per_jd[m], n_boot=n_boot, seed=seed)
            row[m] = {"value": round(float(per_jd[m].mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        out.append(row)
    return out


def verify_per_pair(feats, frame: pd.DataFrame, X: pd.DataFrame, raw: pd.Series, l1) -> dict:
    """Recompute full features on two sub-pools of the first JD: only `tfidf_scaled` may differ
    from the 500-pool value, and it must equal `rescale`'s."""
    from candidate_screener.matching import learned
    jd, g = next(iter(frame.groupby("jd_id", sort=True)))
    ids = g.pair_id.to_numpy()
    rng = np.random.default_rng([SEED, 999])
    checks = {}
    for name, pos in (("random_10", draws(ids, None, 10, "random", rng)),
                      ("relevant_5", draws(ids, l1.predict(X.loc[ids]), 5, "relevant", rng))):
        sub = g.iloc[pos]
        fresh = feats(sub, learned.F12).loc[sub.pair_id.to_numpy()]
        full = X.loc[sub.pair_id.to_numpy(), list(learned.F12)]
        diff = (fresh - full).abs().max()
        others = float(diff.drop("tfidf_scaled").max())
        ours = rescale(full, raw.loc[sub.pair_id.to_numpy()].to_numpy()).tfidf_scaled.to_numpy()
        checks[name] = {"max_diff_other_features": others,
                        "max_diff_tfidf_scaled_vs_500_pool": float(diff["tfidf_scaled"]),
                        "max_diff_tfidf_scaled_vs_rescale": float(np.abs(fresh.tfidf_scaled.to_numpy() - ours).max())}
        if others > 1e-12 or checks[name]["max_diff_tfidf_scaled_vs_rescale"] > 1e-12:
            raise ValueError(f"sub-pool features are not per pair as assumed: {checks[name]}")
    return checks


def run(n_jds: int = N_JDS) -> dict:
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.inference.model import L1Model, RECORD as L1_RECORD
    from candidate_screener.matching import learned
    from candidate_screener.matching.role_filter import per_jd_minmax
    t0 = time.time()
    dev = dev_pools.build()
    jds = pick_jds(dev["pools"], n_jds)
    frame = dev_pools.pairs(dev)
    frame = frame[frame.jd_id.isin(jds)].reset_index(drop=True)
    _log(f"{len(jds)} JDs, {len(frame)} pairs")
    l1 = L1Model.load()
    feats = learned.Features()
    t1 = time.time()
    X = feats(frame, learned.F12)
    feature_seconds = time.time() - t1
    _log(f"features: {feature_seconds:.0f} s")
    raw = pd.Series(feats.scorer.text.score(frame[["pair_id", "jd_id", "cv_id", "jd_text", "cv_text"]]),
                    index=frame.pair_id.to_numpy())
    again = per_jd_minmax(frame.jd_id.to_numpy(), raw.loc[frame.pair_id.to_numpy()].to_numpy())
    if np.abs(again - X.loc[frame.pair_id.to_numpy()].tfidf_scaled.to_numpy()).max() > 1e-12:
        raise ValueError("raw TF-IDF cosine does not reproduce the 500-pool tfidf_scaled")
    per_pair = verify_per_pair(feats, frame, X, raw, l1)
    rows = subpool_rows(frame, X, raw, l1)
    _log(f"{len(rows)} comparisons")
    return {"plan": "plan/2026-10-23-inference-api (step 2, pool scaling)", "status": "exploratory",
            "question": "A (tfidf_scaled min-maxed over the uploads) vs B (over a 500-CV pool): "
                        "how different is the order of the same CVs?",
            "config": {**CONFIG, "n_jds": len(jds)},
            "strata": "none: dev pools have no generic/targeted split",
            "jd_ids_sha256": _sha(jds), "pairs": int(len(frame)),
            "l1_record_sha256": hashlib.sha256(L1_RECORD.read_bytes()).hexdigest(),
            "per_pair_check": per_pair,
            "notes": ["relevant: no n = 100 (top 50 only); at n = 50 every draw is the whole top 50",
                      "Kendall tau-b; a sub-pool with a constant score counts as tau 1"],
            "results": summarise(rows),
            "seconds": {"features": round(feature_seconds, 1), "total": round(time.time() - t0, 1)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jds", type=int, default=N_JDS, help="JDs to draw (timing runs use 1)")
    ap.add_argument("--no-write", action="store_true", help="print, do not write the record")
    args = ap.parse_args(argv)
    doc = run(args.n_jds)
    for r in doc["results"]:
        print(r["system"], r["scenario"], r["n"], {m: r[m]["value"] for m in METRICS})
    print("seconds:", doc["seconds"])
    if not args.no_write:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
