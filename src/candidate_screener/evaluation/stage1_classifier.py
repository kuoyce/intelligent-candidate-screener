"""S1 and S2 of plan 2026-10-18-classifier-then-rerank: the learned stage-1 classifier (D61).

**S1.** `matching.learned.LearnedStage1`, a logistic regression fitted on the 1,000 D50 pairs,
on F12 and (second arm) on the base 4. Its stage-1 output is the set it calls Fit (P >= 0.5).
Beside X2, with generic / targeted / total and JD-resampled CIs:

1. per-JD AUC on the final set, `all` and `llm_free`, paired delta S1 - X2;
2. recall of the passed set on the 500-CV final pools (judged Fits kept / judged Fits; the 490
   unjudged distractors count as not relevant), against **X2 cut at the same size per JD**;
3. the passed-set size per JD, on the final pools and on dev-select (with the role-family
   proxy recall there, size-matched the same way).

**S2.** Per JD, the share of a stage-1 shortlist inside X2's top 100 (the CVs `chat-1` scored),
for two shortlists: the passed set, and S1's top 100 by probability (open question 6). Also the
passed CVs with no chat score, and the final set's judged Fits each shortlist keeps.

`RULES` holds every bar and reading, fixed before this module first ran. X2's final-set and
pool scores must reproduce the stage-1 record's digests, and X2's top 100 must be exactly the
pairs `chat-1` scored. Record: `output/matching/stage1-classifier.json`.

    uv run python -m candidate_screener.evaluation.stage1_classifier
    uv run python -m candidate_screener.evaluation.stage1_classifier --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, indomain, resample

RECORD = OUTPUT / "matching" / "stage1-classifier.json"
TEXT_VARIANT = "v2"
DECIMALS = 6
ARMS = ("F12", "base 4")
PRIMARY = "F12"
S1_BAR = -0.06
S2_BAR = 0.80
TOP = 100
PASSED, TOP100 = "passed set", "top 100 by probability"
RULES = {
    "s1_model": "the F12 arm is S1; the base-4 arm is the notebook 19 reproduction, reported",
    "s1_bar": "S1 passes when the paired delta S1 - X2 is >= -0.06 on both per-JD AUC (total, "
              "all) and size-matched recall of the passed set (total); point estimates decide, "
              "CIs are stated beside (the S2 convention of stage1.py)",
    "size_matched": "per JD, X2 keeps its top n CVs where n is the size of S1's passed set; "
                    "ties by pair id; a JD with an empty passed set scores 0 for both",
    "s2_bar": "mean per-JD overlap >= 0.80 (point estimate; CI stated); overlap = share of the "
              "shortlist inside X2's top 100; JDs with an empty shortlist are counted, not averaged",
    "s2_shortlist": "open question 6: S2 is judged on S1's top 100 by probability when the 0.5 "
                    "threshold passes far more than 100 CVs, read as a median passed-set size "
                    "per JD (final pools) above 125, where even a passed set nested around X2's "
                    "top 100 cannot reach the 0.80 bar; otherwise on the passed set. Both shortlists "
                    "are always reported",
    "fixed": "18 Oct 2026, in this module, before its first run",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _r(v: float) -> float:
    return round(float(v), DECIMALS)


def _sha(s: pd.Series) -> str:
    return hashlib.sha256(np.round(s.to_numpy(), 9).tobytes()).hexdigest()


# --- per-JD pieces (pure) -------------------------------------------------------------------

def ranks(frame: pd.DataFrame, scores) -> np.ndarray:
    """0-based rank of each row within its JD: score descending, ties by pair id."""
    df = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "pid": frame.pair_id.to_numpy(),
                       "s": np.asarray(scores, dtype=float), "row": np.arange(len(frame))})
    df = df.sort_values(["jd", "s", "pid"], ascending=[True, False, True])
    out = np.empty(len(frame), dtype=int)
    out[df.row.to_numpy()] = df.groupby("jd").cumcount().to_numpy()
    return out


def top_k(frame: pd.DataFrame, scores, k) -> np.ndarray:
    """True for each JD's top `k` rows (`k` a number, or a per-JD mapping)."""
    kk = frame.jd_id.map(k).to_numpy() if isinstance(k, (dict, pd.Series)) else k
    return ranks(frame, scores) < kk


def size_matched(frame: pd.DataFrame, passed, x2, rel) -> pd.DataFrame:
    """Per JD: passed-set size, relevant CVs, and those kept by S1's passed set and by X2's top n.

    Recall columns are NaN for a JD with no relevant CV (it is left out of the means)."""
    passed = np.asarray(passed, dtype=bool)
    n = pd.Series(passed).groupby(frame.jd_id.to_numpy()).sum()
    x2_keep = top_k(frame, x2, n)
    rel = np.asarray(rel, dtype=int)
    d = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "rel": rel, "s1": passed * rel,
                      "x2": x2_keep * rel}).groupby("jd").sum()
    d.insert(0, "n_pass", n.loc[d.index].astype(int))
    has = d.rel > 0
    d["recall_s1"] = (d.s1 / d.rel).where(has)
    d["recall_x2"] = (d.x2 / d.rel).where(has)
    return d


def overlap(frame: pd.DataFrame, shortlist, reference) -> pd.DataFrame:
    """Per JD: shortlist size, how many lie in `reference`, the share (NaN when empty)."""
    s, r = np.asarray(shortlist, dtype=bool), np.asarray(reference, dtype=bool)
    d = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "size": s, "inside": s & r}).groupby("jd").sum()
    d["outside"] = d["size"] - d["inside"]
    d["overlap"] = (d["inside"] / d["size"]).where(d["size"] > 0)
    return d.astype({"size": int, "inside": int, "outside": int})


def mean_block(values: pd.Series, seed: int = 0) -> dict:
    v = values.dropna()
    lo, hi = resample.mean_ci(v, seed=seed)
    return {"value": _r(v.mean()), "ci": [_r(lo), _r(hi)], "n_jd": int(len(v))}


def sizes(n: pd.Series) -> dict:
    q = np.quantile(n, [0, 0.25, 0.5, 0.75, 1])
    return {"median": _r(q[2]), "min": int(q[0]), "q25": _r(q[1]), "q75": _r(q[3]),
            "max": int(q[4]), "mean": _r(n.mean()), "n_jd": int(len(n)),
            "jds_over_100": int((n > TOP).sum()), "jds_empty": int((n == 0).sum())}


def by_scope(per_jd: pd.DataFrame, stratum: pd.Series, fn) -> dict:
    """`fn(rows)` for generic, targeted and total; `stratum` maps JD -> stratum."""
    out = {}
    for scope in final_set.SCOPES:
        rows = per_jd if scope == "total" else per_jd[stratum.loc[per_jd.index] == scope]
        out[scope] = fn(rows)
    return out


def s2_shortlist(median_passed: float) -> str:
    return TOP100 if median_passed > 125 else PASSED


# --- the record -----------------------------------------------------------------------------

def _x2():
    from candidate_screener.matching import extracted, lexical
    x2 = extracted.ExtractedPlus(lexical.Tfidf(TEXT_VARIANT), "X2")
    x2.fit(None)
    return x2


def _same_as_x2(X: pd.DataFrame, frame: pd.DataFrame, x2: pd.Series, where: str) -> None:
    """X2's hand weights over F12's first three terms must give X2's own scores, row by row."""
    from candidate_screener.evaluation import a1_weights as aw
    hand = aw.linear_score(X, aw.HAND_WEIGHTS).loc[frame.pair_id.to_numpy()].to_numpy()
    if not np.allclose(hand, np.asarray(x2, dtype=float), rtol=0, atol=1e-9):
        raise ValueError(f"F12's X2 terms do not reproduce X2 on {where}: features misaligned")


def _chat_pairs() -> dict[str, set[str]]:
    """The pairs `chat-1` scored as X2's top 100, by JD (the S1 extension excluded)."""
    from candidate_screener.rerank import chat_targets
    d = chat_targets.load()
    d = d[d.source == chat_targets.TRAIN_SOURCE]
    return {j: set(g.pair_id) for j, g in d.groupby("jd_id")}


def fit_arms(feats):
    """Both arms fitted on the D50 batch: (D50 pairs, labels, features, {arm: system})."""
    from candidate_screener.evaluation import a1_weights as aw
    from candidate_screener.matching import learned
    _log("D50: features and fits")
    d50, y50 = aw.d50_pairs()
    X50 = feats(d50)
    return d50, y50, X50, {a: learned.LearnedStage1(a, feats).fit_features(X50, y50) for a in ARMS}


def passed_set(seed: int = 0, arm: str = PRIMARY, threshold: float | None = None) -> pd.DataFrame:
    """S1's stage-1 output on the 60 final pools: `jd_id`, `cv_id`, `pair_id`, `stratum`, `p_fit`.

    `threshold` defaults to `learned.THRESHOLD` (0.5); step 4a uses 0.3."""
    from candidate_screener.evaluation import retrieval_pilot
    from candidate_screener.matching import learned
    feats = learned.Features()
    m = fit_arms(feats)[3][arm]
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    p = m.proba(feats(pools)).loc[pools.pair_id.to_numpy()].to_numpy()
    out = pools[["jd_id", "cv_id", "pair_id", "stratum"]].assign(p_fit=p)
    t = learned.THRESHOLD if threshold is None else threshold
    return out[p >= t].reset_index(drop=True)


def evaluate(seed: int = 0) -> dict:
    from candidate_screener.evaluation import dev_pools, retrieval_pilot, stage1
    from candidate_screener.matching import corpus, learned
    committed = json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"][stage1.X2]
    feats = learned.Features()
    d50, y50, X50, arms = fit_arms(feats)
    fits = {a: {**m.coefficients(), "train_pass_rate": _r(m.passes(X50).mean())}
            for a, m in arms.items()}

    _log("final set: AUC")
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = indomain.labels(frame)
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).assign(
        keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()
    x2 = _x2()
    s_x2 = x2.score(pairs)
    if _sha(s_x2) != committed["scores_sha256"]:
        raise ValueError("X2 final-set scores differ from the stage-1 record")
    Xf = feats(pairs)
    _same_as_x2(Xf, pairs, s_x2, "the final set")
    scores = {stage1.X2: s_x2}
    scores.update({m.name: m.proba(Xf).loc[pairs.pair_id.to_numpy()] for m in arms.values()})
    unscoreable: list = []
    auc = {}
    with indomain.auc_memo():
        for name, s in scores.items():
            figs = [f.as_dict() for f in indomain.rank_figures(name, s, lab, pairs, rule_version,
                                                                seed, unscoreable)]
            indomain.assert_total_has_strata(figs)
            auc[name] = figs
        auc_deltas = {m.name: [indomain.paired_delta(scores[m.name], s_x2, lab, pairs, metric,
                                                     scope, subset, seed)
                               for subset in indomain.SUBSETS for scope in final_set.SCOPES
                               for metric in indomain.RANK_METRICS] for m in arms.values()}

    _log("final pools: features")
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    rel = retrieval_pilot.relevant(pools).to_numpy()
    ps_x2 = x2.score(pools)
    if _sha(ps_x2) != committed["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the stage-1 record")
    Xp = feats(pools)
    _same_as_x2(Xp, pools, ps_x2, "the final pools")
    stratum = pools.drop_duplicates("jd_id").set_index("jd_id").stratum
    x2_top = top_k(pools, ps_x2, TOP)
    chat = _chat_pairs()
    top_sets = {j: set(g.pair_id) for j, g in pools[x2_top].groupby("jd_id")}
    if top_sets != chat:
        raise ValueError("X2's top 100 is not the set of pairs chat-1 scored")
    x2_kept = pd.Series(x2_top * rel).groupby(pools.jd_id.to_numpy()).sum()

    _log("dev-select: features")
    dev = dev_pools.build()
    dframe = dev_pools.pairs(dev, dev_pools.SPLITS[1]).reset_index(drop=True)
    drel = dev_pools.labels(dev, dframe).to_numpy()
    ds_x2 = x2.score(dframe).to_numpy()
    Xd = feats(dframe)
    _same_as_x2(Xd, dframe, ds_x2, "dev-select")

    s1, s2, per_jd_out = {}, {}, None
    for a, m in arms.items():
        p = m.proba(Xp).loc[pools.pair_id.to_numpy()].to_numpy()
        passed = p >= learned.THRESHOLD
        pj = size_matched(pools, passed, ps_x2.to_numpy(), rel)
        recall = by_scope(pj, stratum, lambda r: {
            "s1": mean_block(r.recall_s1, seed), "x2_size_matched": mean_block(r.recall_x2, seed),
            "delta": mean_block(r.recall_s1 - r.recall_x2, seed)})
        dp = m.proba(Xd).loc[dframe.pair_id.to_numpy()].to_numpy()
        dj = size_matched(dframe, dp >= learned.THRESHOLD, ds_x2, drel)
        auc_total = next(d for d in auc_deltas[m.name] if d["metric"] == "auc"
                         and d["scope"] == "total" and d["subset"] == "all")
        rec_total = recall["total"]["delta"]
        s1[a] = {
            "name": m.name, "fit": fits[a],
            "final_pools": {"passed_size": by_scope(pj, stratum, lambda r: sizes(r.n_pass)),
                            "recall": recall},
            "dev_select": {"passed_size": sizes(dj.n_pass),
                           "proxy_recall": {"s1": mean_block(dj.recall_s1, seed),
                                            "x2_size_matched": mean_block(dj.recall_x2, seed),
                                            "delta": mean_block(dj.recall_s1 - dj.recall_x2, seed)}},
            "bar": {"auc_delta_total": auc_total["delta"], "auc_ci": auc_total["ci"],
                    "recall_delta_total": rec_total["value"], "recall_ci": rec_total["ci"],
                    "threshold": S1_BAR,
                    "verdict": "pass" if (auc_total["delta"] >= S1_BAR
                                          and rec_total["value"] >= S1_BAR) else "fail"}}

        top = top_k(pools, p, TOP)
        lists = {PASSED: passed, TOP100: top}
        block = {}
        for label, mask in lists.items():
            o = overlap(pools, mask, x2_top)
            kept = pd.Series(mask * rel).groupby(pools.jd_id.to_numpy()).sum()
            block[label] = {
                "overlap": by_scope(o, stratum, lambda r: mean_block(r.overlap, seed)),
                "jds_empty": int((o["size"] == 0).sum()),
                "no_chat_score_per_jd": {"mean": _r(o.outside.mean()),
                                         "median": _r(o.outside.median()),
                                         "max": int(o.outside.max()),
                                         "total": int(o.outside.sum())},
                "judged_fits_kept": {"total": int(kept.sum()),
                                     "x2_top100_total": int(x2_kept.sum()),
                                     "judged_fits": int(rel.sum())}}
            tot = block[label]["overlap"]["total"]
            block[label]["verdict"] = "pass" if tot["value"] >= S2_BAR else "fail"
        median = s1[a]["final_pools"]["passed_size"]["total"]["median"]
        governs = s2_shortlist(median)
        s2[a] = {"governs": governs, "verdict": block[governs]["verdict"], "shortlists": block}
        if a == PRIMARY:
            o = overlap(pools, passed, x2_top)
            per_jd_out = [{"jd_id": j, "stratum": stratum[j], "n_pass": int(r.n_pass),
                           "fits": int(r.rel), "kept_s1": int(r.s1), "kept_x2_size_matched": int(r.x2),
                           "inside_x2_top100": int(o.inside[j])} for j, r in pj.iterrows()]

    return {
        "plan": "plan/2026-10-18-classifier-then-rerank (steps 1-2, D61)",
        "status": ("S1 and S2 bars fixed in the plan and in `rules` before this ran; the final set "
                   "is scored, never fitted on (D46): S1 fits on D50 only"),
        "rules": RULES,
        "text_variant": TEXT_VARIANT,
        "trained_on": learned.TRAINED_ON,
        "features": {"F12": list(learned.F12), "base 4": list(learned.BASE4),
                     "note": ("sbert = SBERT weak raw cosine; tfidf_scaled is min-max scaled per "
                              "JD within the frame scored (10 CVs on D50 and the final set, 500 "
                              "on a pool), as in X2")},
        "d50": {"pairs": int(len(d50)), "jds": int(d50.jd_id.nunique()),
                "fit_rate": _r(y50.mean())},
        "reference": {"file": final_set.FINAL_CSV.name, "rule_version": rule_version,
                      "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest()},
        "seed": seed,
        "x2_reproduces_stage1": True,
        "x2_top100_is_chat1": True,
        "note": ("AUC per JD over JDs with both classes, CIs resample JDs "
                 f"({resample.N_BOOT:,}). Pool recall over JDs with a judged Fit; the 490 "
                 "unjudged distractors count as not relevant (upper bound). Dev-select relevance "
                 "is the role-family proxy (same oracle family), not fit."),
        "auc": auc, "auc_deltas_vs_x2": auc_deltas, "unscoreable": unscoreable,
        "s1": s1,
        "s2": s2,
        "per_jd_f12": per_jd_out,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    a = ap.parse_args(argv)
    record = evaluate(a.seed)
    if a.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    for a_, b in record["s1"].items():
        sz = b["final_pools"]["passed_size"]["total"]
        print(f"  S1 {a_:7s} AUC delta {b['bar']['auc_delta_total']:+.4f} {b['bar']['auc_ci']}  "
              f"recall delta {b['bar']['recall_delta_total']:+.4f} {b['bar']['recall_ci']}  "
              f"passed median {sz['median']} [{sz['min']}, {sz['max']}]  -> {b['bar']['verdict']}")
    for a_, b in record["s2"].items():
        cells = "  ".join(f"{k}: {v['overlap']['total']['value']:.3f}"
                          for k, v in b["shortlists"].items())
        print(f"  S2 {a_:7s} {cells}  governs {b['governs']} -> {b['verdict']}")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
