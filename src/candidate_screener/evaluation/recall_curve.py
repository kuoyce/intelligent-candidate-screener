"""Stage-1 recall@K curve for X2 (plan 2026-10-17-llm-rerank, step 1), exploratory.

Stage 1 shortlists, stage 2 (the LLM) reranks the shortlist. The shortlist size K sets both how
many fits survive and how many LLM calls stage 2 costs (K per JD). This measures X2's recall at
a range of K on two sets of 500-CV pools:

- **dev-select** (150 A2 train-region JDs, `dev_pools`): where K is **chosen** (D46). Relevance
  is the role-family proxy (same oracle family), not fit (Q-R3).
- **final pools** (60 A2 final-set JDs, `retrieval_pilot`): reported, never used to choose.
  Relevance = a final-set Fit among the 10 judged CVs; the 490 distractors are unjudged and
  counted as not relevant, so this is an upper bound on recall.

X2's final-pool scores are checked against the stage-1 record's digest.

    uv run python -m candidate_screener.evaluation.recall_curve
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import resample

RECORD = OUTPUT / "matching" / "stage1-recall-curve.json"
KS = (10, 20, 50, 100, 150, 200, 250)
X2 = "extracted rule + TF-IDF"


def curve(frame: pd.DataFrame, scores: pd.Series, rel: pd.Series, ks=KS, seed: int = 0) -> list[dict]:
    """Mean per-JD recall@k with a JD-resampled CI, for every k in `ks`."""
    from candidate_screener.evaluation import retrieval_pilot
    out = []
    for k in ks:
        v = retrieval_pilot.recall_at(frame, scores, rel, k)
        lo, hi = resample.mean_ci(v, seed=seed)
        out.append({"k": k, "recall": float(v.mean()), "ci": [lo, hi], "n_jd": int(len(v)),
                    "share_of_pool": k / retrieval_pilot.POOL_SIZE})
    return out


def _x2():
    from candidate_screener.evaluation import stage1
    spec = next(s for s in stage1.specs() if s.factory().name == X2)
    system = spec.factory()
    system.fit(None)
    return system


def run(seed: int = 0) -> dict:
    from candidate_screener.evaluation import dev_pools, retrieval_pilot, stage1
    system = _x2()
    # final pools (report only)
    pools = retrieval_pilot.build_pools(seed, stage1.TEXT_VARIANT)
    s = system.score(pools)
    committed = json.loads(stage1.RECORD.read_text())["systems"][X2]["pool_scores_sha256"]
    if stage1._sha(s) != committed:
        raise ValueError("X2 pool scores differ from the committed stage-1 record")
    rel = retrieval_pilot.relevant(pools)
    final = {}
    for scope in ("generic", "targeted", "total"):
        m = (pools.stratum.to_numpy() == scope) if scope != "total" else slice(None)
        final[scope] = curve(pools[m], pd.Series(s.to_numpy()[m]), pd.Series(rel.to_numpy()[m]), seed=seed)
    fits = rel.groupby(pools.jd_id.to_numpy()).sum()
    # dev-select (where K is chosen)
    dev = dev_pools.build()
    frame = dev_pools.pairs(dev, dev_pools.SPLITS[1]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no {dev_pools.SPLITS[1]} pairs")
    ds = system.score(frame)
    drel = dev_pools.labels(dev, frame)
    return {"status": "exploratory; K is chosen on dev-select only (D46)", "retriever": X2, "pool_size": retrieval_pilot.POOL_SIZE,
            "final_pools": {"relevance": "final-set Fit among 10 judged CVs; 490 unjudged counted not relevant (upper bound)",
                            "fits_per_jd": {"mean": float(fits.mean()), "min": int(fits.min()), "max": int(fits.max())},
                            "curve": final},
            "dev_select": {"relevance": "same oracle role family (proxy, not fit)", "n_jds": int(frame.jd_id.nunique()),
                           "relevant_per_jd": float(drel.groupby(frame.jd_id.to_numpy()).sum().mean()),
                           "curve": curve(frame, pd.Series(ds.to_numpy(), index=frame.index), pd.Series(drel.to_numpy(), index=frame.index), seed=seed)}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    result = run(ap.parse_args(argv).seed)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
