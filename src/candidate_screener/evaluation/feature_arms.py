"""Step 4b of plan 2026-10-18-classifier-then-rerank: feature arms for both stages.

Arms (fixed in the plan before this ran): A0 F12 (step 3's features), A1 F9 (F12 without the
three near-copies of `coverage`), and on top of F9 one change each: A2 + `title_role_match`
(N1), A3 + `no_experience` (N3 F2), A4 F9 on CV text with negated clauses removed (N3 F1),
A5 + `sbert_zs` (N4), A6 F9 with the concept terms from R+N (seed rules + MiniLM tagger; N2).
N2 was first skipped (tagging 26,155 documents timed at ~80 min, over the agreed ~1 h), then
reinstated by the owner on 18 Oct 2026 before any arm's result existed; `--tag-n2` fills the
tagger cache first.

- **Stage 1** (reported, decides nothing): logistic regression as S1, 5-fold JD cross-validation
  on the D50 batch (100 JDs); per-JD AUC on held-out JDs, paired against A0 and A1. Never the
  final set (D46).
- **Stage 2** (decides): L2 out of fold on `chat-1` exactly as step 3 (X2's top 100, JD folds),
  Recall@10 against the chat's top 10. Adoption (the plan's rule): F9 if its Recall@10 delta vs
  F12 is >= -0.005; an added change if its paired delta vs F9 is > 0 with the CI above 0.

The **final arm** is F9 (or F12) plus every adopted change, scored the same way. When it is not
F12, its S1 is refitted on D50, step 4a's rule is re-applied on dev-select (the largest list
with a median <= 100 CVs per JD over t in 0.5 / 0.4 / 0.3 / 0.2), and the CVs its passed set
holds without a chat score are counted (a second chat round, plan question 20).

    uv run python -m candidate_screener.evaluation.feature_arms --tag-n2   # once, ~80 min
    uv run python -m candidate_screener.evaluation.feature_arms
    uv run python -m candidate_screener.evaluation.feature_arms --check
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import resample

RECORD = OUTPUT / "rerank" / "feature-arms.json"
TEXT_VARIANT = "v2"
DECIMALS = 6
N_FOLDS = 5
SEED = 0
F9_TOLERANCE = -0.005
GRID = (0.5, 0.4, 0.3, 0.2)
BUDGET = 100
RULES = {
    "arms": "A0 F12; A1 F9; A2 F9 + title_role_match (N1); A3 F9 + no_experience (N3 F2); "
            "A4 F9 on negation-stripped CV text (N3 F1); A5 F9 + sbert_zs (N4); A6 F9 with R+N "
            "concepts (N2, reinstated by the owner before any result)",
    "stage_1": "logistic as S1, 5-fold JD CV on D50, per-JD AUC on held-out JDs; reported only",
    "stage_2": "L2 out of fold on chat-1 (step 3's folds and reference); Recall@10 decides",
    "adoption": "F9 if Recall@10 delta vs F12 >= -0.005; a change if its paired delta vs F9 > 0 "
                "with the CI above 0; the final arm is F9 (or F12) plus every adopted change",
    "re_apply_4a": "if the final arm is not F12: refit S1 on D50, choose t on dev-select as the "
                   "largest list with a median <= 100 CVs per JD over t in (0.5, 0.4, 0.3, 0.2), "
                   "count the passed CVs without a chat score",
    "fixed": "18 Oct 2026, in the plan and this module, before its first run",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _r(v) -> float:
    return round(float(v), DECIMALS) + 0.0


def jd_folds(jds, n: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """JD -> fold, a seeded deal over the sorted ids."""
    jds = np.array(sorted(set(jds)))
    perm = np.random.default_rng(seed).permutation(len(jds))
    return pd.Series(np.arange(len(jds)) % n, index=jds[perm]).sort_index()


def stage1_cv(X: pd.DataFrame, y: pd.Series, jd: pd.Series, columns) -> pd.Series:
    """Out-of-fold P(Fit) on D50: each JD scored by a logistic fit without its fold."""
    from candidate_screener.matching import learned
    fold = jd.map(jd_folds(jd))
    out = pd.Series(np.nan, index=X.index)
    for k in range(N_FOLDS):
        tr, te = X.index[(fold != k).to_numpy()], X.index[(fold == k).to_numpy()]
        m = learned.LearnedStage1("arm", features=object(), columns=columns).fit_features(X.loc[tr], y)
        out.loc[te] = m.proba(X.loc[te]).to_numpy()
    return out


def mean_ci(v: pd.Series, seed: int = SEED) -> dict:
    v = v.dropna()
    lo, hi = resample.mean_ci(v, seed=seed)
    return {"value": _r(v.mean()), "ci": [_r(lo), _r(hi)], "n_jd": int(len(v))}


def adopt(deltas_vs_f9: dict[str, dict], f9_vs_f12: dict) -> dict:
    """The plan's rule, mechanically. Deltas are Recall@10 total blocks (`value`, `ci`)."""
    use_f9 = f9_vs_f12["value"] >= F9_TOLERANCE
    changes = [a for a, d in deltas_vs_f9.items() if d["value"] > 0 and d["ci"][0] > 0]
    return {"base": "F9" if use_f9 else "F12", "changes": changes}


def _titles(pairs: pd.DataFrame, title_of: dict) -> pd.DataFrame:
    return pairs.assign(jd_title=pairs.jd_id.map(title_of))


def evaluate(seed: int = SEED) -> dict:
    from candidate_screener.data.profile import load_split
    from candidate_screener.evaluation import a1_weights as aw, dev_pools, indomain, retrieval_pilot
    from candidate_screener.evaluation import rerank_chat as rc, stage1_classifier as sc
    from candidate_screener.matching import corpus, learned
    from candidate_screener.rerank import chat_targets as ct, chunked, learners

    d50, y50 = aw.d50_pairs()
    d50_titles = load_split("djinni-jd", "train").set_index("id").Position
    d50 = _titles(d50, d50_titles.reindex(d50.jd_id.unique()).to_dict())
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    a2 = corpus.a2_pairs(text_variant=TEXT_VARIANT)
    pools = _titles(pools, dict(zip(a2.jd_id, a2.jd_position)))
    chat = chunked.chat_frame()
    train = chat[chat.source == ct.TRAIN_SOURCE]
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    cols_all = list(learned.F12) + list(learned.EXTRA)
    frames: dict[tuple[bool, str], tuple] = {}

    def frame(strip: bool, concepts: str):
        """(Features, D50 frame, pool frame) for one text and concept setting, built once."""
        if (strip, concepts) not in frames:
            _log(f"features: strip_negated={strip}, concepts={concepts}")
            f = learned.Features(strip_negated=strip, concepts=concepts)
            frames[(strip, concepts)] = (f, f(d50, cols_all), f(pools, cols_all))
        return frames[(strip, concepts)]

    base50, basep = frame(False, "R")[1], frame(False, "R")[2]
    x2 = sc._x2().score(pools)
    sc._same_as_x2(basep, pools, x2, "the final pools")

    base, stripped, rn = (False, "R"), (True, "R"), (False, "R+N")
    arms = {"A0 F12": (base, list(learned.F12)), "A1 F9": (base, list(learned.F9)),
            "A2 F9 + title_role_match": (base, list(learned.F9) + ["title_role_match"]),
            "A3 F9 + no_experience": (base, list(learned.F9) + ["no_experience"]),
            "A4 F9, negation-stripped CVs": (stripped, list(learned.F9)),
            "A5 F9 + sbert_zs": (base, list(learned.F9) + ["sbert_zs"]),
            "A6 F9, R+N concepts": (rn, list(learned.F9))}

    def score_arm(setting, cols) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        _, X50, Xp = frame(*setting)
        p50 = stage1_cv(X50, y50, d50.jd_id, cols)
        auc = indomain.per_jd_auc(y50.loc[p50.index], p50, d50.jd_id.loc[p50.index])
        l2, _ = learners.out_of_fold(learners.L2, Xp.loc[chat.pair_id.to_numpy()], chat, cols)
        return auc, ct.per_jd(train, l2), l2

    results, per = {}, {}
    for arm, (setting, cols) in arms.items():
        _log(f"arm {arm}")
        auc, p2, l2 = score_arm(setting, cols)
        per[arm] = (auc, p2)
        results[arm] = {"features": {"strip_negated": setting[0], "concepts": setting[1]}, "columns": cols,
                        "stage1_d50_cv_auc": mean_ci(auc, seed),
                        "stage2_l2": rc.summary(p2, stratum, seed),
                        "l2_scores_sha256": rc._sha(l2)}
    s3 = json.loads(rc.RECORD.read_text(encoding="utf-8"))["candidates"][learners.L2]["scores_sha256"]
    if results["A0 F12"]["l2_scores_sha256"] != s3:
        raise ValueError("A0 does not reproduce step 3's L2")
    for arm in arms:
        for ref in ("A0 F12", "A1 F9"):
            if arm == ref:
                continue
            results[arm][f"stage1_auc_delta_vs_{ref[:2]}"] = mean_ci(per[arm][0] - per[ref][0], seed)
            results[arm][f"stage2_delta_vs_{ref[:2]}"] = rc.deltas(per[arm][1], per[ref][1], stratum, seed)

    decision = adopt({a: results[a]["stage2_delta_vs_A1"]["recall"]["total"] for a in arms
                      if a not in ("A0 F12", "A1 F9")},
                     results["A1 F9"]["stage2_delta_vs_A0"]["recall"]["total"])
    setting = (any(a.startswith("A4") for a in decision["changes"]),
               "R+N" if any(a.startswith("A6") for a in decision["changes"]) else "R")
    base_cols = list(learned.F9) if decision["base"] == "F9" else list(learned.F12)
    extra = [arms[a][1][-1] for a in decision["changes"] if a[:2] in ("A2", "A3", "A5")]
    final_cols = base_cols + extra
    final = {"base": decision["base"], "changes": decision["changes"],
             "features": {"strip_negated": setting[0], "concepts": setting[1]},
             "columns": final_cols, "is_f12": final_cols == list(learned.F12) and setting == base}
    if not final["is_f12"]:
        _log("final arm")
        auc, p2, _ = score_arm(setting, final_cols)
        final.update({"stage1_d50_cv_auc": mean_ci(auc, seed), "stage2_l2": rc.summary(p2, stratum, seed),
                      "stage2_delta_vs_A0": rc.deltas(p2, per["A0 F12"][1], stratum, seed)})
        _log("re-apply 4a on dev-select")
        f, X50, Xp = frame(*setting)
        s1 = learned.LearnedStage1("final", f, final_cols).fit_features(X50, y50)
        dev = dev_pools.build()
        dframe = dev_pools.pairs(dev, dev_pools.SPLITS[1]).reset_index(drop=True)
        pdv = s1.proba(f(dframe, final_cols)).loc[dframe.pair_id.to_numpy()].to_numpy()
        med = {t: float(pd.Series(pdv >= t).groupby(dframe.jd_id.to_numpy()).sum().median()) for t in GRID}
        t = min((t for t in GRID if med[t] <= BUDGET), default=max(GRID))
        pp = s1.proba(Xp).loc[pools.pair_id.to_numpy()]
        passed = set(pp.index[pp.to_numpy() >= t])
        from candidate_screener.rerank import chat_rerank as cr
        plan = pd.read_csv(cr.PLAN)
        planned = set(plan.pair_id)
        n = pd.Series(pp.to_numpy() >= t).groupby(pools.jd_id.to_numpy()).sum()
        final["re_applied_4a"] = {"dev_select_median_by_t": {str(k): v for k, v in med.items()}, "t": t,
                                  "final_pools_passed": sc.sizes(n),
                                  "passed_not_planned_for_chat": int(len(passed - planned))}
    return {"plan": "plan/2026-10-18-classifier-then-rerank (step 4b)", "rules": RULES, "seed": seed,
            "d50": {"pairs": int(len(d50)), "jds": int(d50.jd_id.nunique()),
                    "titles_missing": int(d50.jd_title.isna().sum())},
            "arms": results, "adoption": decision, "final": final}


def tag_n2(seed: int = SEED) -> None:
    """Tag every JD and CV text arm A6 reads (D50, the final pools, dev-select) with the MiniLM
    tagger, into `concept_sources`' cache (saved every 250 documents)."""
    from candidate_screener.evaluation import a1_weights as aw, concept_overlap as co, dev_pools, retrieval_pilot
    d50, _ = aw.d50_pairs()
    dev = dev_pools.build()
    sets = {"d50": d50, "pools": retrieval_pilot.build_pools(seed, TEXT_VARIANT),
            "dev-select": dev_pools.pairs(dev, dev_pools.SPLITS[1])}
    co.extract(sets, co.open_sources(("minilm",)))


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    ap.add_argument("--tag-n2", action="store_true", help="fill the MiniLM tagger cache for N2, then stop")
    a = ap.parse_args(argv)
    if a.tag_n2:
        tag_n2(a.seed)
        return 0
    record = evaluate(a.seed)
    if a.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    for arm, b in record["arms"].items():
        r = b["stage2_l2"]["recall"]["total"]
        d = b.get("stage2_delta_vs_A1", {}).get("recall", {}).get("total")
        print(f"  {arm:32s} S1 D50-CV AUC {b['stage1_d50_cv_auc']['value']:.3f}  L2 R@10 {r['value']:.3f}"
              + (f"  vs F9 {d['value']:+.3f} {d['ci']}" if d else ""))
    print(f"  adoption: {record['adoption']}")
    print(f"  final: { {k: v for k, v in record['final'].items() if k in ('columns', 'is_f12', 're_applied_4a')} }")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
