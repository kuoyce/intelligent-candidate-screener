"""Stage 2: the rerankers trained on the LLM batch, scored as orderings (plan 2026-10-06
P8; 06-registration §2, §4-§6 and amendment A3, §8).

Three parts, one record (`output/matching/stage2-metrics.json`):

1. **Batch (train / validation).** R1, R2 and R3 (`rerank.models`) are trained on run 11's
   train split and selected on its validation split: the whole grid with each point's
   validation pairwise agreement, the selected point, R1 vs R2 (the selected
   cross-encoder), and the stop rule (A3.7). Nothing in this part reads the final set.
2. **Final set (8.4).** Every system as an ordering (`evaluation.ordering`): `yc`, `van`,
   `mean`, `yc_single`; `full`, `llm_free`, `fit_only`; nDCG and pairwise; generic /
   targeted / total. Systems: random (expected), TF-IDF, SBERT zero-shot and weak (all
   `text_v2`), R1, R2, R3, and the LLM score as a ranker (read on `llm_free`, D43/D51).
   O1-O5 under the registration §3 rule in the A3.8 cell (pairwise, `yc`, `full`; the
   LLM-dependent clause on `llm_free`), with the `mean` verdict beside; §5 selection.
3. **End to end (8.5, decides nothing).** On the pilot's 500-CV pools (`text_v2`): X2 ranks
   (S2 missed: "below S2"), the adopted stage-2 system re-orders X2's top 50
   (`retrieval_pilot.rerank_top`). Recall@10 of final-set Fits for both, the paired Δ,
   the ordering on judged CVs in each top 10, and the unjudged CVs there, counted.

Usage:
    uv run python -m candidate_screener.evaluation.stage2 [--seed 0]
    uv run python -m candidate_screener.evaluation.stage2 --check           # cold: retrains
    uv run python -m candidate_screener.evaluation.stage2 --check --warm    # cached weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, ordering, resample, retrieval_pilot
from candidate_screener.matching.selection import verdict_of

TEXT_VARIANT = "v2"
RECORD = OUTPUT / "matching" / "stage2-metrics.json"
COSTS = OUTPUT / "matching" / "stage2-costs.json"
DECIMALS = 6

RANDOM, TFIDF = ordering.RANDOM, "TF-IDF"
SBERT_ZS, SBERT_WEAK = "SBERT (zero-shot)", "SBERT (weak fine-tuned)"
LLM = ordering.LLM
X2 = "extracted rule + TF-IDF"
CE = "selected cross-encoder"     # alias resolved to R1's or R2's name in the record

#: A3.8: the verdict cell. Keys follow `selection.verdict_of` ("all" = the full subset).
VERDICT_REF, VERDICT_METRIC = "yc", "pairwise"
CELLS = {("total", "all"): ("total", "full"), ("generic", "all"): ("generic", "full"),
         ("targeted", "all"): ("targeted", "full"), ("total", "llm_free"): ("total", "llm_free")}
#: Registration §4 stage 2: (id, B, A, decides selection?). CE = the selected cross-encoder.
COMPARISONS = (("O1", CE, SBERT_WEAK, True), ("O2", CE, TFIDF, False),
               ("O3", "R3", SBERT_WEAK, True), ("O4", CE, "R3", True))
#: §5 cost order (cheapest first) and who may displace whom (A3.9).
COST_ORDER = (SBERT_WEAK, "R3", CE)
BEATS = ((CE, SBERT_WEAK, "O1"), ("R3", SBERT_WEAK, "O3"), (CE, "R3", "O4"))
SELECTION_NOTE = (
    "Registration §5 as A3.9 reads it: walk SBERT weak < R3 < selected cross-encoder; a "
    "candidate failing O5 is skipped; otherwise it is adopted unless a dearer O5-passing "
    "candidate beats it with verdict `justified` (O1, O3, O4; yc, pairwise, full). If no "
    "reranker beats SBERT weak, stage 2 is SBERT weak ordering.")
TOP = 10


def _r(x: float) -> float:
    return round(float(x), DECIMALS) + 0.0


def _sha(s: pd.Series) -> str:
    return hashlib.sha256(np.round(s.sort_index().to_numpy(float), 9).tobytes()).hexdigest()


# --- verdicts and selection ----------------------------------------------------------------

def verdict(b: pd.Series | None, a: pd.Series | None, grades: pd.DataFrame, ref: str,
            seed: int = 0) -> dict:
    """Registration §3 on the A3.8 cell for one reference, with the cells it read."""
    cells = {key: ordering.paired_delta(b, a, grades, ref, VERDICT_METRIC, scope, subset,
                                        seed)
             for key, (scope, subset) in CELLS.items()}
    return {"verdict": verdict_of(cells),
            "cells": [{k: v for k, v in c.items() if k != "metric"} for c in cells.values()]}


def beats_random(b: pd.Series, grades: pd.DataFrame, ref: str, seed: int = 0) -> dict:
    """O5 (A2): paired per-JD Δ vs the random expectation, total, full; CI lower > 0."""
    d = ordering.paired_delta(b, None, grades, ref, VERDICT_METRIC, "total", "full", seed)
    return {"delta": d["delta"], "ci": d["ci"], "n_jd": d["n_jd"],
            "passes": bool(d["ci"] is not None and d["ci"][0] > 0)}


def select(o5: dict[str, bool], verdicts: dict[tuple[str, str], str]) -> dict:
    """§5 as A3.9 reads it (the same procedure as `stage1.select`)."""
    trail = []
    for name in COST_ORDER:
        if not o5[name]:
            trail.append({"system": name, "passes_o5": False, "beaten_by": []})
            continue
        checks = [{"b": b, "via": via, "verdict": verdicts[(b, a)]}
                  for b, a, via in BEATS if a == name and o5[b]]
        beaten = [c["b"] for c in checks if c["verdict"] == "justified"]
        trail.append({"system": name, "passes_o5": True, "checks": checks, "beaten_by": beaten})
        if not beaten:
            return {"selected": name, "trail": trail, "note": SELECTION_NOTE}
    return {"selected": None, "trail": trail, "note": SELECTION_NOTE}


# --- 8.5: end to end -----------------------------------------------------------------------

def top_k(pools: pd.DataFrame, scores: pd.Series, k: int = TOP) -> pd.DataFrame:
    """Each JD's top k by `scores` (ties: pair_id), with the scores attached."""
    df = pools[["pair_id", "jd_id", "stratum"]].assign(s=scores.loc[pools.pair_id].to_numpy())
    df = df.sort_values(["jd_id", "s", "pair_id"], ascending=[True, False, True])
    return df.groupby("jd_id", sort=True).head(k)


def shortlist_ordering(top: pd.DataFrame, grades: pd.DataFrame, seed: int = 0) -> dict:
    """The ordering figure on the judged CVs inside each top k (`yc`, full), plus the
    unjudged CVs there, counted and never scored (02 §4)."""
    judged = top[top.pair_id.isin(grades.index)]
    sub = grades.loc[judged.pair_id]
    scores = pd.Series(judged.s.to_numpy(), index=judged.pair_id.to_numpy())
    out = {"unjudged_in_top": int((~top.pair_id.isin(grades.index)).sum()),
           "judged_in_top": int(len(judged)), "jds": int(top.jd_id.nunique()), "figures": []}
    for metric in ordering.METRICS:
        for scope in final_set.SCOPES:
            values, excluded = ordering.per_jd(final_set.in_scope(sub, scope), "yc", scores,
                                               metric)
            out["figures"].append({"metric": metric, "scope": scope, "reference": "yc",
                                   **ordering.summarise(values, seed),
                                   "n_jd_excluded": excluded})
    return out


def recall_delta(b: pd.Series, a: pd.Series, pools: pd.DataFrame, rel: pd.Series, k: int,
                 seed: int = 0) -> list[dict]:
    out = []
    for scope in final_set.SCOPES:
        sub = pools if scope == "total" else pools[pools.stratum == scope]
        idx = sub.pair_id.to_numpy()
        diff = (retrieval_pilot.recall_at(sub, b.loc[idx], rel.loc[idx], k)
                - retrieval_pilot.recall_at(sub, a.loc[idx], rel.loc[idx], k)).dropna()
        low, high = resample.mean_ci(diff, seed=seed)
        out.append({"metric": f"recall@{k}", "scope": scope, "delta": _r(diff.mean()),
                    "ci": [_r(low), _r(high)], "n_jd": int(len(diff))})
    return out


def end_to_end(rerankers: dict[str, object], grades: pd.DataFrame, seed: int = 0) -> dict:
    from candidate_screener.matching import extracted, lexical
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    rel = retrieval_pilot.relevant(pools)
    x2 = extracted.ExtractedPlus(lexical.Tfidf(TEXT_VARIANT), "X2")
    x2.fit(None)
    base = x2.score(pools)
    rows = {X2: base}
    for label, system in rerankers.items():
        rows[f"{X2} -> {label}"] = retrieval_pilot.rerank_top(pools, base, system,
                                                              retrieval_pilot.SHORTLIST)
    out = {"retriever": X2, "retriever_status": "below S2 (R@50 0.779 < 0.80; Q-P7-2, A3.4)",
           "shortlist": retrieval_pilot.SHORTLIST, "top": TOP,
           "pool_size": retrieval_pilot.POOL_SIZE, "n_pools": int(pools.jd_id.nunique()),
           "note": ("Relevant = final-set Fit among the 10 judged CVs per pool; the 490 "
                    "distractors are unjudged and assumed not fits (upper-bound recall). "
                    "Decides nothing (06 §4)."),
           "systems": {}}
    for name, s in rows.items():
        out["systems"][name] = {
            "scores_sha256": _sha(s),
            "recall": retrieval_pilot.figures(name, pools, s, rel, ks=(TOP,), seed=seed),
            "top_ordering": shortlist_ordering(top_k(pools, s), grades, seed)}
        if name != X2:
            out["systems"][name]["recall_delta_vs_x2"] = recall_delta(s, base, pools, rel, TOP,
                                                                      seed)
    return out


# --- the record ----------------------------------------------------------------------------

def train_part(root: Path, seed: int = 0) -> tuple[dict, dict, dict]:
    """(fitted rungs, the batch block of the record, costs). Reads the batch only."""
    from candidate_screener.matching import neural
    from candidate_screener.rerank import batch, models
    frame = batch.load()
    digest = neural.file_digest(batch.SCORES_CSV)
    fitted = {r: models.fit_rung(r, frame, digest, root, seed) for r in (models.R1, models.R2,
                                                                         models.R3)}
    best = {r: f.points[f.selected]["validation_pairwise"] for r, f in fitted.items()}
    ce = models.R1 if best[models.R1] >= best[models.R2] else models.R2
    stop = {r: models.vs_random(f.validation, seed) for r, f in fitted.items()}
    block = {
        "scores_file": batch.SCORES_CSV.name,
        "scores_sha256": hashlib.sha256(batch.SCORES_CSV.read_bytes()).hexdigest(),
        "run": batch.RUN, "text_variant": batch.TEXT_VARIANT,
        "summary": batch.score_summary(frame),
        "preference_pairs_train": batch.gap_counts(frame[frame.split == "train"],
                                                   models.DELTAS),
        "selection_metric": ("per-JD pairwise agreement with the run-11 score on the "
                             "validation JDs (reference ties excluded, score tie 0.5); "
                             "earlier grid point wins a tie (A3.6)"),
        "rungs": {r: {"name": models.NAMES[r], "grid": f.points, "selected": f.selected,
                      "selected_form": models.best_form(f)} for r, f in fitted.items()},
        "selected_cross_encoder": ce,
        "stop_rule": {"per_rung": stop,
                      "insufficient": not any(v["beats_random"] for v in stop.values()),
                      "note": "06 §6 / A3.7: insufficient only if every rung fails; the "
                              "batch is not extended either way"},
    }
    costs = {r: f.costs for r, f in fitted.items()}
    return fitted, block, costs


def evaluate(seed: int = 0, root: Path | None = None) -> tuple[dict, dict]:
    from candidate_screener.matching import corpus
    from candidate_screener.rerank import models
    fitted, batch_block, costs = train_part(root or models.WEIGHTS, seed)
    ce_rung = batch_block["selected_cross_encoder"]

    # --- the final set: read only from here on (A3.11) ---
    frame = final_set.load()
    grades = ordering.grade_frame(frame)
    scores, skipped = ordering.score_systems(TEXT_VARIANT)
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).set_index("pair_id").loc[
        frame.pair_id].reset_index()
    names = {}
    for r, f in fitted.items():
        scores[r] = f.system.score(pairs)
        names[r] = f.system.name
    alias = {CE: ce_rung}

    def get(name):
        return scores[alias.get(name, name)]

    systems = {}
    for key, s in scores.items():
        systems[key] = {"name": names.get(key, key), "scores_sha256": ordering._digest(s),
                        "figures": ordering.figures(names.get(key, key), s, grades, seed)}
    deltas = []
    for b in (*fitted, LLM):
        for a in (TFIDF, SBERT_WEAK):
            if b in scores and a in scores:
                deltas += [{"b": b, "a": a, **ordering.paired_delta(
                    scores[b], scores[a], grades, ref, metric, scope, subset, seed)}
                    for ref in ordering.REFERENCES for subset in ordering.SUBSETS
                    for metric in ordering.METRICS for scope in final_set.SCOPES]
    o5 = {name: {ref: beats_random(get(name), grades, ref, seed) for ref in ("yc", "mean")}
          for name in (*fitted, SBERT_WEAK, TFIDF, SBERT_ZS)}
    comparisons, verdicts = [], {}
    for cid, b, a, decides in COMPARISONS:
        v = {ref: verdict(get(b), get(a), grades, ref, seed) for ref in ("yc", "mean")}
        comparisons.append({"id": cid, "b": alias.get(b, b), "a": alias.get(a, a),
                            "decides_selection": decides, "verdict": v["yc"]["verdict"],
                            "verdict_mean": v["mean"]["verdict"],
                            "references_disagree": v["yc"]["verdict"] != v["mean"]["verdict"],
                            "cells": {ref: v[ref]["cells"] for ref in v}})
    for b, a, _ in BEATS:
        verdicts[(b, a)] = verdict(get(b), get(a), grades, VERDICT_REF, seed)["verdict"]
    o5_pass = {name: o5[alias.get(name, name)]["yc"]["passes"] for name in COST_ORDER}
    selection = select(o5_pass, verdicts)
    adopted = alias.get(selection["selected"], selection["selected"])

    # --- 8.5 ---
    lookup = {r: f.system for r, f in fitted.items()}
    if adopted == SBERT_WEAK or adopted is None:
        from candidate_screener.matching import sbert
        weak = sbert.SbertWeak(TEXT_VARIANT)
        weak.fit(None)
        lookup[SBERT_WEAK] = weak
    rerankers = {}
    if adopted is not None:
        rerankers[f"{adopted} (adopted)"] = lookup[adopted]
    if adopted != ce_rung:
        rerankers[f"{ce_rung} (selected cross-encoder, not adopted)"] = lookup[ce_rung]
    e2e = end_to_end(rerankers, grades, seed)

    record = {
        "seed": seed,
        "text_variant": TEXT_VARIANT,
        "registration": "plan/2026-10-06-extraction-and-rerank/06-registration.md (A2, A3)",
        "status": {"R1": "held-out", "R2": "held-out", "R3": "held-out",
                   TFIDF: "exploratory", SBERT_ZS: "exploratory", SBERT_WEAK: "exploratory",
                   LLM: "exploratory (read on llm_free, D43)", RANDOM: "exact expectation"},
        "batch": batch_block,
        "reference": {
            "file": final_set.FINAL_CSV.name, "rule_version": int(frame.rule_version.iloc[0]),
            "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest(),
            "grades": "human raw labels: Good Fit=2, Potential Fit=1, No Fit=0 (D52); "
                      "the LLM is never the reference",
            "pairs": {ref: int(grades[ref].notna().sum()) for ref in ordering.REFERENCES}},
        "note": ("Per-JD nDCG and pairwise agreement as `evaluation.ordering`; 95% CI "
                 f"resampling JDs ({resample.N_BOOT}). Verdicts: registration §3 on pairwise, "
                 "yc, full (A3.8), LLM-dependent clause on llm_free; the mean-reference "
                 "verdict beside. The run-11 judge saw Position; the rerankers read text_v2 "
                 "(A3.1)."),
        "skipped": skipped,
        "systems": systems,
        "deltas": deltas,
        "o5": o5,
        "comparisons": comparisons,
        "selection": {**selection, "adopted_system": adopted},
        "end_to_end": e2e,
    }
    return record, costs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def _headline(record: dict) -> None:
    for key, block in record["systems"].items():
        f = {(x["scope"]): x for x in block["figures"] if x["reference"] == "yc"
             and x["subset"] == "full" and x["metric"] == "pairwise"}
        cells = " ".join(f"{s} {f[s]['value']:.3f}" for s in final_set.SCOPES
                         if f[s]["value"] is not None)
        print(f"  {block['name']:42s} yc pairwise {cells}")
    for c in record["comparisons"]:
        print(f"  {c['id']} {c['b']} vs {c['a']}: {c['verdict']} (mean: {c['verdict_mean']})")
    print(f"  stop rule insufficient: {record['batch']['stop_rule']['insufficient']}")
    print(f"  adopted: {record['selection']['adopted_system']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    ap.add_argument("--warm", action="store_true",
                    help="with --check: reuse the cached weights (D55 warm tier); without "
                         "it the check retrains every grid point in a fresh directory")
    args = ap.parse_args(argv)
    if args.warm and not args.check:
        ap.error("--warm is only valid with --check")
    if args.check and not args.warm:
        from candidate_screener.rerank import models
        models.WEIGHTS.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="cold-", dir=models.WEIGHTS))
        try:
            record, _ = evaluate(args.seed, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    else:
        record, costs = evaluate(args.seed)
    if args.check:
        committed = RECORD.read_text(encoding="utf-8") if RECORD.exists() else ""
        same = committed == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        if not same and committed:
            old = json.loads(committed)
            for key in record:
                if old.get(key) != record[key]:
                    print(f"  differs: {key}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    COSTS.write_text(render(costs), encoding="utf-8")
    _headline(record)
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
