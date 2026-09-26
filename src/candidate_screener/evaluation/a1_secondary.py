"""A1 as the secondary benchmark (D40): the text rungs on A1 test, Fit / No Fit only (D47).

A1's labels do not reproduce (A13): two blind readers agree with them at chance. Every
figure here therefore measures agreement with A1's unknown labelling process, not
screening quality, and travels with that line. No "% of attainable" is quoted (D26
retired). The A1 cross-encoder (4a) is trained on A1 train and is in-distribution here.

Reported: per-JD AUC on the 659 test pairs; Recall@10 and Precision@5 over the N100
pools under the `graded` definition (= Fit); and, for rungs that emit Fit / No Fit,
accuracy and binary macro-F1 against the majority floor.

Usage:
    uv run python -m candidate_screener.evaluation.a1_secondary --seed 0 [--check]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import indomain, metrics, resample

RECORD = OUTPUT / "matching" / "a1-metrics.json"
A13 = ("A1's labels do not reproduce (A13): blind readers agree with them at chance. "
       "These figures measure agreement with A1's labelling process, not screening quality.")


def pool_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pools = metrics_pools()
    n100 = pools[pools.pool_variant == "N100"]
    frame = pd.DataFrame({
        "pair_id": n100.query_jd_id + "__" + n100.candidate_resume_id,
        "jd_id": n100.query_jd_id, "cv_id": n100.candidate_resume_id,
        "jd_text": n100.job_description_text, "cv_text": n100.resume_text,
        "jd_keyword": np.nan, "cv_keyword": np.nan, "jd_exp_band": np.nan, "cv_exp_band": np.nan})
    return pools, frame.reset_index(drop=True)


def metrics_pools() -> pd.DataFrame:
    from candidate_screener.evaluation.retrieval import load_pools
    return load_pools()


def evaluate(rungs: list, seed: int = 0) -> dict:
    from candidate_screener.matching import corpus
    test = corpus.a1_pairs("test")
    pools, pool_frame = pool_pairs()
    floor_acc = float(max(test.fit.mean(), 1 - test.fit.mean()))
    record = {"seed": seed, "note_a13": A13, "test_pairs": int(len(test)),
              "test_jds": int(test.jd_id.nunique()), "fit_rate": round(float(test.fit.mean()), 6),
              "majority_floor_accuracy": round(floor_acc, 6), "systems": {}}
    for factory in rungs:
        system = factory()
        system.fit(None)
        s = system.score(test)
        auc = indomain.per_jd_auc(test.fit.set_axis(test.pair_id), s, test.jd_id.set_axis(test.pair_id))
        low, high = resample.mean_ci(auc, seed=seed)
        block = {"rung": system.rung, "per_jd_auc": {
            "value": round(float(auc.mean()), 6), "ci": [round(low, 6), round(high, 6)],
            "n_jd": int(len(auc))}}
        ps = system.score(pool_frame)
        scores = pd.DataFrame({"query_jd_id": pool_frame.jd_id, "candidate_resume_id": pool_frame.cv_id,
                               "score": ps.to_numpy()})
        for metric, k in (("Recall", 10), ("Precision", 5)):
            f = metrics.score(pools, scores, "N100", metric, k, "graded", seed)
            block[f"{metric.lower()}@{k}_graded_N100"] = {
                "value": round(f.value, 6), "ci": [round(f.ci_low, 6), round(f.ci_high, 6)],
                "n_queries": f.n_queries}
        if hasattr(system, "classify"):
            calls = (system.classify(test) == "Fit").astype(int).to_numpy()
            stats = indomain._cls_stats(test.fit.to_numpy(), calls)
            block["classifier"] = {"threshold": round(system.threshold, 6),
                                   "accuracy": round(stats["accuracy"], 6),
                                   "macro_f1": round(stats["macro_f1"], 6),
                                   "kappa": round(stats["kappa"], 6)}
        if "A1" in system.name:
            block["note"] = "trained on A1 train: in-distribution here"
        record["systems"][system.name] = block
    return record


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    from candidate_screener.matching import ladder, rules
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    record = evaluate([rules.Random, *ladder.TEXT_RUNGS.values()], args.seed)
    if args.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    for name, b in record["systems"].items():
        print(f"  {name:28s} AUC {b['per_jd_auc']['value']:.3f}  R@10 {b['recall@10_graded_N100']['value']:.3f}"
              + (f"  macroF1 {b['classifier']['macro_f1']:.3f}" if "classifier" in b else ""))
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
