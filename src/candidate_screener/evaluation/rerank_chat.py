"""S3 of plan 2026-10-18-classifier-then-rerank: stage-2 rerankers trained and scored on `chat-1` (D61).

**Task.** For each of the 60 final-set JDs, reorder X2's top 100 (the 6,000 `chat-1` pairs) so
the chat's top 10 is in the top 10. `chat-1` is the target and the reference for this task only
(D61); folds are by JD (`chat_targets.folds`), and every learned score is out of fold.

**Candidates.** Not trained on `chat-1`: X2's order, the S1 classifier's probability, and R1-R3
as stored (trained on D50 run 11; R1/R2 cut pairs at 384 tokens). Trained per fold on F12
(features of the 500-CV pool frame, as stage 1 sees them): L1, L2 (`rerank.learners`), three
ablations of the better of the two, and L3 (`rerank.chunked`).

**Metrics** per JD, mean over JDs, JD-resampled CI, generic / targeted / total, paired delta
against X2's order: tie-aware Recall@10 against the chat's top 10 (the bar), nDCG@10 with gain =
chat score, chat No Fits in the top 10. **Human check** (exploratory): the out-of-fold scores on
the judged CVs inside the shortlists, per-JD AUC (final label) and pairwise on yc's grades.

**End to end.** S1's passed set -> the best candidate -> top 10, beside X2 alone, S1 alone and
X2's top 100 -> the same candidate. The reference is the chat's top 10 over every chat-scored
CV of the JD (X2's 100 plus the 20-pair extension); a JD whose passed set is under 10 returns
fewer. `RULES` was fixed before this module first ran.

    uv run python -m candidate_screener.evaluation.rerank_chat           # write the record
    uv run python -m candidate_screener.evaluation.rerank_chat --check   # cold: retrains L3 (~9 h)
    uv run python -m candidate_screener.evaluation.rerank_chat --check --warm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED
from candidate_screener.evaluation import final_set, indomain, ordering

RECORD = OUTPUT / "rerank" / "chat-rerank-eval.json"
COSTS = OUTPUT / "rerank" / "chat-rerank-costs.json"
CACHE = PROCESSED / "rerank" / "chat-eval"
TEXT_VARIANT = "v2"
DECIMALS = 6
BAR = 0.05
METRICS = ("recall", "ndcg", "no_fits")
ABLATE = ("tfidf_scaled", "role_match", "sbert")
HUMAN_REF = "yc"
RULES = {
    "bar": "a candidate passes when its paired Recall@10 delta against X2's order (total) is "
           ">= +0.05 and the CI's lower end is above 0",
    "best": "the candidate with the highest out-of-fold Recall@10 (total) among S1, L1, L2, R3, "
            "R1, R2, L3; a tie (6 dp) goes to the earlier, cheaper one. Chosen on the same "
            "figures it is then reported with: mildly optimistic (owner, 18 Oct 2026)",
    "ablations": "of the better of L1 and L2 on Recall@10 (total), ties to L1: without "
                 "tfidf_scaled, without role_match, without sbert; diagnostic, never 'best'",
    "features": "F12 computed on each JD's 500-CV pool frame (tfidf_scaled min-max over the "
                "pool, as S1 sees it), then the chat pairs' rows",
    "constraints": "L1 non-negative and L2 monotone increasing on the 8 match signals; the 4 "
                   "descriptors (log_jd_concepts, log_cv_concepts, exp_gap, exp_known) free",
    "learning_curve": "skipped (owner, 18 Oct 2026)",
    "end_to_end": "stage 1 = S1's passed set (P >= 0.5, owner 18 Oct 2026); reference = chat's "
                  "top 10 over all chat-scored CVs of the JD, tie-aware; fewer than 10 "
                  "returned counts the misses",
    "human_check": "exploratory: judged CVs inside the shortlists, per-JD AUC on the final label "
                   "and pairwise on yc's grades, full and llm_free",
    "fixed": "18 Oct 2026, in this module, before its first run",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _r(v) -> float:
    return round(float(v), DECIMALS) + 0.0


def _sha(s: pd.Series) -> str:
    return hashlib.sha256(np.round(s.to_numpy(dtype=float), 9).tobytes()).hexdigest()


#: R1-R3 are re-scored by stored models, and re-scoring moves the last bits of R1's and R2's floats
#: (D55 cold run, 21 Oct 2026: every figure equal, the score digests not). Their fingerprint is
#: each JD's top 10, which is what Recall@10 reads; every other candidate keeps `_sha`.
REINFERRED = ("R1", "R2", "R3")


def _top10_sha(frame: pd.DataFrame, s: pd.Series) -> str:
    """Digest of each JD's top 10 by `s` (ties by pair id): `frame`'s pairs, JD by JD."""
    from candidate_screener.rerank.chat_targets import top_k_mask
    ids = []
    for _, g in frame.sort_values("pair_id").groupby("jd_id", sort=True):
        p = g.pair_id.to_numpy()
        ids.extend(sorted(p[top_k_mask(p, s.loc[p].to_numpy(dtype=float))]))
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# --- summaries (pure) -----------------------------------------------------------------------

def summary(per: pd.DataFrame, stratum: pd.Series, seed: int = 0) -> dict:
    """Mean and CI of each metric per scope; `per` is indexed by JD."""
    from candidate_screener.evaluation.stage1_classifier import by_scope, mean_block
    return {m: by_scope(per, stratum, lambda r, m=m: mean_block(r[m].astype(float), seed))
            for m in per.columns if m in METRICS}


def deltas(per_b: pd.DataFrame, per_a: pd.DataFrame, stratum: pd.Series, seed: int = 0) -> dict:
    """Paired per-JD B - A of each metric, per scope."""
    d = per_b[list(METRICS)].astype(float) - per_a.loc[per_b.index, list(METRICS)].astype(float)
    return summary(d, stratum, seed)


def passes(delta: dict) -> bool:
    tot = delta["recall"]["total"]
    return bool(tot["value"] >= BAR and tot["ci"][0] > 0)


def pick_best(recall_total: dict[str, float], order) -> str:
    """Highest Recall@10 (6 dp); a tie goes to the earlier name in `order`."""
    return max(order, key=lambda n: (round(recall_total[n], DECIMALS), -order.index(n)))


# --- inputs ---------------------------------------------------------------------------------

def _stored(chat: pd.DataFrame, warm: bool) -> dict[str, pd.Series]:
    """R1-R3 as the stage-2 record selected them, scored on the chat pairs."""
    from candidate_screener.evaluation import stage2
    from candidate_screener.rerank import models
    rungs = json.loads(stage2.RECORD.read_text(encoding="utf-8"))["batch"]["rungs"]
    pair_digest = hashlib.sha256("\n".join(chat.pair_id).encode()).hexdigest()[:12]
    out = {}
    for r in (models.R1, models.R2, models.R3):
        cfg = models.grid(r)[rungs[r]["selected"]]
        path = models.WEIGHTS / rungs[r]["grid"][rungs[r]["selected"]]["weights"]
        cache = CACHE / f"{path.name}-{pair_digest}.parquet"
        if warm and cache.exists():
            s = pd.read_parquet(cache).set_index("pair_id").score
        else:
            _log(f"{r}: scoring {len(chat)} chat pairs with {path.name}")
            system = models.system_for(cfg, path)
            system.fit(None)
            s = system.score(chat[["pair_id", "jd_id", "cv_id", "jd_text", "cv_text"]])
            CACHE.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"pair_id": s.index, "score": s.to_numpy()}).to_parquet(cache, index=False)
            del system
        out[f"{r} {models.NAMES[r]}"] = s.loc[chat.pair_id.to_numpy()]
    return out


def evaluate(seed: int = 0, warm: bool = True, l3_root: Path | None = None) -> tuple[dict, dict]:
    from candidate_screener.evaluation import retrieval_pilot, stage1, stage1_classifier as sc
    from candidate_screener.matching import corpus, learned
    from candidate_screener.rerank import chat_targets as ct, chunked, learners

    chat = chunked.chat_frame()
    train = chat[chat.source == ct.TRAIN_SOURCE]
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum

    _log("pools: X2 and F12")
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    x2 = sc._x2()
    ps_x2 = x2.score(pools)
    committed = json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"][stage1.X2]
    if _sha(ps_x2) != committed["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the stage-1 record")
    top = sc.top_k(pools, ps_x2, sc.TOP)
    if set(pools.pair_id[top]) != set(train.pair_id):
        raise ValueError("X2's top 100 is not the set of pairs chat-1 scored")
    feats = learned.Features()
    s1 = sc.fit_arms(feats)[3][sc.PRIMARY]
    Xp = feats(pools)
    sc._same_as_x2(Xp, pools, ps_x2, "the final pools")
    p_pool = s1.proba(Xp)
    passed_ids = set(p_pool.index[p_pool >= learned.THRESHOLD])
    if not passed_ids <= set(chat.pair_id):
        raise ValueError(f"{len(passed_ids - set(chat.pair_id))} S1 passed CVs have no chat score")
    X = Xp.loc[chat.pair_id.to_numpy()]
    ids = chat.pair_id.to_numpy()

    scores = {stage1.X2: ps_x2.loc[ids], s1.name: p_pool.loc[ids]}
    names = {"S1": s1.name}
    _log("R1-R3 (stored)")
    stored = _stored(chat, warm)
    scores.update(stored)
    for key, full in zip(("R1", "R2", "R3"), stored):
        names[key] = full

    _log("L1, L2 out of fold")
    cols = list(learned.F12)
    fitted = {}
    for key, kind in (("L1", learners.L1), ("L2", learners.L2)):
        scores[kind], fitted[kind] = learners.out_of_fold(kind, X, chat, cols)
        names[key] = kind

    _log("L3 out of fold")
    l3, l3_info = chunked.oof(chat, l3_root or chunked.ROOT)
    scores[chunked.NAME] = l3
    names["L3"] = chunked.NAME

    def per(name: str, frame: pd.DataFrame = train, returned=None) -> pd.DataFrame:
        return ct.per_jd(frame, scores[name], returned)

    base = per(stage1.X2)
    reinferred = {names[k] for k in REINFERRED}
    rows = {}
    for name in scores:
        p = per(name)
        rows[name] = {"summary": summary(p, stratum, seed),
                      "delta_vs_x2": None if name == stage1.X2 else deltas(p, base, stratum, seed),
                      **({"top10_sha256": _top10_sha(train, scores[name])} if name in reinferred
                         else {"scores_sha256": _sha(scores[name])})}
        if name != stage1.X2:
            rows[name]["passes_bar"] = passes(rows[name]["delta_vs_x2"])
    better = pick_best({n: rows[names[n]]["summary"]["recall"]["total"]["value"] for n in ("L1", "L2")},
                       ["L1", "L2"])
    for f in ABLATE:
        name = f"{names[better]} without {f}"
        kind = learners.L1 if better == "L1" else learners.L2
        scores[name], _ = learners.out_of_fold(kind, X, chat, [c for c in cols if c != f])
        p = per(name)
        rows[name] = {"summary": summary(p, stratum, seed), "delta_vs_x2": deltas(p, base, stratum, seed),
                      "delta_vs_full": deltas(p, per(names[better]), stratum, seed),
                      "scores_sha256": _sha(scores[name]), "ablation_of": names[better]}
    order = ["S1", "L1", "L2", "R3", "R1", "R2", "L3"]
    best_key = pick_best({k: rows[names[k]]["summary"]["recall"]["total"]["value"] for k in order}, order)
    best = names[best_key]

    _log("human check (exploratory)")
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = indomain.labels(frame)
    judged = lab.index.intersection(pd.Index(train.pair_id))
    lab_j = lab.loc[judged]
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).assign(
        keyword_match=lambda d: corpus.keyword_match(d))
    grades = ordering.grade_frame(frame).loc[judged]
    human, unscoreable = {}, []
    with indomain.auc_memo():
        for name in scores:
            s = scores[name].loc[judged]
            block = {"auc": [f.as_dict() for f in indomain.rank_figures(
                name, s, lab_j, pairs, rule_version, seed, unscoreable)]}
            block["pairwise_" + HUMAN_REF] = {
                sub: {scope: ordering.summarise(ordering.per_jd(
                    ordering.restrict(grades, scope, sub), HUMAN_REF, s, "pairwise")[0], seed)
                    for scope in final_set.SCOPES} for sub in ("full", "llm_free")}
            if name != stage1.X2:
                a = scores[stage1.X2].loc[judged]
                block["auc_delta_vs_x2"] = [indomain.paired_delta(s, a, lab_j, pairs, "auc", scope,
                                                                  sub, seed)
                                            for sub in indomain.SUBSETS for scope in final_set.SCOPES]
                block["pairwise_delta_vs_x2"] = [ordering.paired_delta(s, a, grades, HUMAN_REF,
                                                                       "pairwise", scope, sub, seed)
                                                 for sub in ("full", "llm_free")
                                                 for scope in final_set.SCOPES]
            human[name] = block

    _log("end to end")
    rel = chat.pair_id.isin(set(lab.index[lab.y == 1])).to_numpy()
    passed = pd.Series(chat.pair_id.isin(passed_ids).to_numpy(), index=ids)
    in_top = pd.Series((chat.source == ct.TRAIN_SOURCE).to_numpy(), index=ids)
    pipelines = {"X2 alone": (stage1.X2, None),
                 "S1 alone (passed set, by probability)": (s1.name, passed),
                 f"X2 top 100 -> {best_key}": (best, in_top),
                 f"S1 passed set -> {best_key}": (best, passed)}
    e2e, per_e2e = {}, {}
    for label, (name, returned) in pipelines.items():
        p = per(name, chat, returned)
        fits = {}
        for jd, g in chat.groupby("jd_id", sort=True):
            r = None if returned is None else returned.loc[g.pair_id.to_numpy()].to_numpy()
            s = scores[name].loc[g.pair_id.to_numpy()].to_numpy()
            keep = np.ones(len(g), bool) if r is None else r
            m = ct.top_k_mask(g.pair_id.to_numpy(), np.where(keep, s, -np.inf)) & keep
            fits[jd] = int(rel[g.index.to_numpy()][m].sum())
        p["judged_fits_in_top10"] = pd.Series(fits)
        per_e2e[label] = p
        e2e[label] = {"system": name,
                      "summary": summary(p, stratum, seed),
                      "returned_per_jd": {"median": _r(p.returned.median()), "min": int(p.returned.min()),
                                          "jds_under_10": int((p.returned < 10).sum())},
                      "judged_fits_in_top10_total": int(p.judged_fits_in_top10.sum())}
    ref = "X2 alone"
    for label in pipelines:
        if label != ref:
            e2e[label]["delta_vs_x2_alone"] = deltas(per_e2e[label], per_e2e[ref], stratum, seed)
    e2e[f"S1 passed set -> {best_key}"]["delta_vs_x2_top100_pipeline"] = deltas(
        per_e2e[f"S1 passed set -> {best_key}"], per_e2e[f"X2 top 100 -> {best_key}"], stratum, seed)

    l1_coef = pd.DataFrame([m.coefficients() for m in fitted[learners.L1]])
    record = {
        "plan": "plan/2026-10-18-classifier-then-rerank (step 3, D61)",
        "status": ("chat-1 is the target and the reference for this task only (D61); learned "
                   "scores are out of fold by JD; the human check is exploratory"),
        "rules": RULES, "bar": BAR,
        "chat": {"pairs_train": int(len(train)), "pairs_extension": int((chat.source != ct.TRAIN_SOURCE).sum()),
                 "jds": int(chat.jd_id.nunique()),
                 "folds": {int(k): {"jds": int(g.jd_id.nunique()),
                                    "generic": int(g[g.stratum == "generic"].jd_id.nunique()),
                                    "targeted": int(g[g.stratum == "targeted"].jd_id.nunique())}
                           for k, g in chat.groupby("fold")},
                 "score_mean": _r(train.score.mean()),
                 "no_fit_share": _r((train.label == "No Fit").mean())},
        "features": cols, "constrained": list(learners.CONSTRAINED), "free": list(learners.FREE),
        "l1_standardised_coefficients_mean_over_folds": {c: _r(v) for c, v in l1_coef.mean().items()},
        "l3_config": chunked.CONFIG,
        "candidates": rows,
        "best": {"key": best_key, "name": best, "better_of_l1_l2": better},
        "human_check": {"reference": HUMAN_REF, "judged_pairs": int(len(judged)),
                        "systems": human, "unscoreable": unscoreable},
        "end_to_end": e2e,
        "note": ("Recall@10 is tie-aware against the chat's top 10 (r/t on a tie at the 10th "
                 "score). chat-1 was run once, with batch drift and multiples-of-5 scores, so no "
                 "model's Recall@10 can reach 1 (no noise ceiling; accepted limitation)."),
    }
    costs = {"l3_folds": l3_info}
    return record, costs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    ap.add_argument("--warm", action="store_true",
                    help="with --check: reuse the stored R1-R3 scores and L3's out-of-fold "
                         "scores (D55 warm tier); without it the check re-infers and retrains")
    a = ap.parse_args(argv)
    if a.warm and not a.check:
        ap.error("--warm is only valid with --check")
    cold = a.check and not a.warm
    root = Path(tempfile.mkdtemp(prefix="chat-l3-cold-", dir=PROCESSED / "rerank")) if cold else None
    try:
        record, costs = evaluate(a.seed, warm=not cold, l3_root=root)
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
    if a.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    COSTS.write_text(render(costs), encoding="utf-8")
    for name, b in record["candidates"].items():
        r = b["summary"]["recall"]
        d = b["delta_vs_x2"]["recall"]["total"] if b["delta_vs_x2"] else None
        print(f"  {name[:58]:58s} R@10 {r['generic']['value']:.3f}/{r['targeted']['value']:.3f}/"
              f"{r['total']['value']:.3f}" + (f"  delta {d['value']:+.3f} {d['ci']}"
                                               f"  {'PASS' if b.get('passes_bar') else ''}" if d else ""))
    print(f"  best: {record['best']}")
    for label, b in record["end_to_end"].items():
        print(f"  {label:45s} R@10 {b['summary']['recall']['total']['value']:.3f}  "
              f"returned median {b['returned_per_jd']['median']}")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
