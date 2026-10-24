"""Step 4a of plan 2026-10-18-classifier-then-rerank: a lower stage-1 threshold, end to end.

S1 (F12, fitted on D50 as in step 1) passes the CVs with P >= t. The operating point **t = 0.3**
was fixed before this ran, from a dev-select size budget (median <= 100 CVs per JD); t = 0.5
and 0.4 are reported beside it as a curve, t = 0.2 as sizes only. Stage 2 is step 3's L2 (the
same five fold models, not retrained): each JD is scored by the model that never saw it.

**Reference.** The chat's latest score per pair over every chat-scored CV of the JD (`chat-1`
plus the step-4a fresh chats, `chat-2`; owner 18 Oct 2026). Per JD and arm:

- **ceiling**: the share of the chat's top 10 (tie-aware) inside the list stage 1 hands over;
- **Recall@10**, nDCG@10 and chat No Fits of L2's top 10 of that list;
- list size, and judged Fits kept (stage-1 recall on the human final set, exploratory here).

Paired against X2 alone, X2's top 100 -> L2 and t = 0.5 -> L2. **Bar:** t = 0.3 -> L2 against X2
alone, Recall@10 delta >= +0.05 with the CI above 0. The anchors' drift (their `chat-2` score
against `chat-1`) is reported; how to use it is decided after it is seen (owner).

    uv run python -m candidate_screener.evaluation.stage1_threshold
    uv run python -m candidate_screener.evaluation.stage1_threshold --check
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, resample

RECORD = OUTPUT / "rerank" / "threshold-ceiling.json"
TEXT_VARIANT = "v2"
DECIMALS = 6
OPERATING_POINT = 0.3
CURVE = (0.5, 0.4, 0.3)
SIZE_ONLY = (0.2,)
BAR = 0.05
X2_ALONE, X2_L2 = "X2 alone", "X2 top 100 -> L2"
RULES = {
    "operating_point": "t = 0.3: the largest list within a dev-select budget of a median <= 100 "
                       "CVs per JD (dev-select medians 21 / 31 / 55 / 122.5 at t = 0.5 / 0.4 / "
                       "0.3 / 0.2); sizes only, fixed before any ceiling or recall was computed",
    "bar": "t = 0.3 -> L2 against X2 alone: Recall@10 delta (total) >= +0.05 with the CI's "
           "lower end above 0 (the S3 bar)",
    "stage_2": "step 3's L2 fold models unchanged; a JD is scored by the model of its own fold",
    "reference": "latest chat score per pair across chat-1 and chat-2 (owner 18 Oct 2026); "
                 "every CV an arm hands over must be chat-scored",
    "anchors": "drift reported only; any correction is decided after it is seen (owner) and "
               "labelled post-hoc",
    "f9_follow_up": "after step 4b adopted F9 (by its pre-registered rule): stage 1 stays S1 on F12, "
                    "L2 is refitted on F9 in the same five folds (owner, option B, 18 Oct 2026); "
                    "rows marked (F9) are reported beside, with the same bar, and the registered "
                    "row is unchanged",
    "fixed": "18 Oct 2026, before chat-2 was scored",
}


def _r(v) -> float:
    return round(float(v), DECIMALS) + 0.0


def ceiling(ids, chat, returned, k: int = 10) -> float:
    """Share of the chat's top k (tie-aware) inside `returned`: Recall@k of a perfect stage 2."""
    from candidate_screener.rerank import chat_targets as ct
    w = ct.chat_weights(np.asarray(chat, dtype=float), k)
    return float(w[np.asarray(returned, bool)].sum() / min(k, len(w)))


def arm_per_jd(ref: pd.DataFrame, system: pd.Series, returned: pd.Series | None) -> pd.DataFrame:
    """`ct.per_jd` plus the ceiling, per JD of `ref` (one row per chat-scored pair)."""
    from candidate_screener.rerank import chat_targets as ct
    per = ct.per_jd(ref, system, returned)
    if returned is not None:
        per["ceiling"] = pd.Series({jd: ceiling(g.pair_id, g.score, returned.loc[g.pair_id.to_numpy()])
                                    for jd, g in ref.groupby("jd_id", sort=True)})
    return per


def drift(anchors: pd.DataFrame, stratum: pd.Series, seed: int = 0) -> dict:
    """The anchors' chat-2 score minus their chat-1 score: per-JD mean offset, agreement."""
    from scipy.stats import spearmanr
    from candidate_screener.evaluation.stage1_classifier import by_scope, mean_block
    a = anchors.assign(d=anchors.again - anchors.first)
    per = pd.DataFrame({
        "offset": a.groupby("jd_id").d.mean(),
        "abs_diff": a.assign(x=a.d.abs()).groupby("jd_id").x.mean(),
        "spearman": a.groupby("jd_id").apply(lambda g: spearmanr(g["first"], g["again"])[0]
                                             if g["first"].nunique() > 1 else np.nan),
        "label_agreement": a.assign(x=a.first_label == a.again_label).groupby("jd_id").x.mean()})
    return {"anchors": int(len(a)), "jds": int(a.jd_id.nunique()),
            **{m: by_scope(per, stratum, lambda r, m=m: mean_block(r[m].astype(float), seed))
               for m in per.columns}}


def retest(pairs: pd.DataFrame, top100: set, stratum: pd.Series, seed: int = 0) -> dict:
    """The accidental `chat-1r` retest (JDs 2-6 re-scored, owner kept it apart): how well the chat
    agrees with itself. Exploratory; five JDs. Recall@10 of the retest's top 10 against chat-1's,
    over each JD's X2 top 100: a rough noise ceiling for every Recall@10 here."""
    from candidate_screener.rerank import chat_targets as ct
    rec = {}
    for jd, g in pairs[pairs.pair_id.isin(top100)].groupby("jd_id", sort=True):
        g = g.sort_values("pair_id")
        rec[jd] = ct.recall_at_k(g.pair_id.to_numpy(), g["first"].to_numpy(), g["again"].to_numpy())
    r = pd.Series(rec)
    return {**drift(pairs, stratum, seed), "recall_at_10_vs_chat1": {
        "value": _r(r.mean()), "per_jd": [_r(v) for v in r.to_numpy()], "n_jd": int(len(r))},
        "note": "exploratory, 5 JDs: an accident, not a designed retest"}


def evaluate(seed: int = 0) -> dict:
    from candidate_screener.evaluation import rerank_chat as rc, retrieval_pilot, stage1, \
        stage1_classifier as sc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_rerank as cr, chat_targets as ct, chunked, learners

    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    x2 = sc._x2()
    ps_x2 = x2.score(pools)
    committed = json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"][stage1.X2]
    if rc._sha(ps_x2) != committed["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the stage-1 record")
    feats = learned.Features()
    s1 = sc.fit_arms(feats)[3][sc.PRIMARY]
    Xp = feats(pools)
    sc._same_as_x2(Xp, pools, ps_x2, "the final pools")
    p = s1.proba(Xp)

    # stage 2: step 3's L2 fold models, applied to every pool pair of each fold's JDs
    chat1 = chunked.chat_frame()
    fold_of = chat1.drop_duplicates("jd_id").set_index("jd_id").fold
    l2_chat, models = learners.out_of_fold(learners.L2, Xp.loc[chat1.pair_id.to_numpy()], chat1,
                                           list(learned.F12))
    s3 = json.loads(rc.RECORD.read_text(encoding="utf-8"))["candidates"][learners.L2]["scores_sha256"]
    if rc._sha(l2_chat) != s3:
        raise ValueError("L2's fold models do not reproduce step 3's scores")
    l2_f9_chat, models_f9 = learners.out_of_fold(learners.L2, Xp.loc[chat1.pair_id.to_numpy()], chat1,
                                                 list(learned.F9))
    from candidate_screener.evaluation import feature_arms
    a1 = json.loads(feature_arms.RECORD.read_text(encoding="utf-8"))["arms"]["A1 F9"]["l2_scores_sha256"]
    if rc._sha(l2_f9_chat) != a1:
        raise ValueError("L2 on F9 does not reproduce step 4b's A1 arm")
    folds = sorted(chat1.fold.unique())
    pool_fold = pools.jd_id.map(fold_of).to_numpy()

    def on_pools(fold_models) -> pd.Series:
        out = pd.Series(np.nan, index=pools.pair_id.to_numpy())
        for k, m in zip(folds, fold_models):
            rows = pools.pair_id.to_numpy()[pool_fold == k]
            out.loc[rows] = m.predict(Xp.loc[rows])
        return out
    l2, l2_f9 = on_pools(models), on_pools(models_f9)

    # reference: latest chat score per pair; every handed-over CV must be in it
    ref = ct.load_latest()
    stratum = ref.drop_duplicates("jd_id").set_index("jd_id").stratum
    ids = ref.pair_id.to_numpy()
    x2_top = set(pools.pair_id[sc.top_k(pools, ps_x2, sc.TOP)])
    lists = {X2_L2: x2_top, **{f"t = {t}": set(p.index[p >= t]) for t in CURVE}}
    f9_rows = {f"{X2_L2} (F9)": (X2_L2, l2_f9), f"t = {OPERATING_POINT} (F9)": (f"t = {OPERATING_POINT}", l2_f9)}
    for name, members in lists.items():
        missing = members - set(ids)
        if missing:
            raise ValueError(f"{name}: {len(missing)} handed-over CVs have no chat score")
    rel = retrieval_pilot.relevant(pools)
    fits_total = rel.groupby(pools.jd_id.to_numpy()).sum()

    per, size_of = {X2_ALONE: arm_per_jd(ref, ps_x2.loc[ids], None)}, {}
    for name, members in lists.items():
        returned = pd.Series(ref.pair_id.isin(members).to_numpy(), index=ids)
        per[name] = arm_per_jd(ref, l2.loc[ids], returned)
        jd = pools.jd_id[pools.pair_id.isin(members).to_numpy()]
        size_of[name] = jd.value_counts().reindex(per[name].index).fillna(0).astype(int)
        kept = rel.loc[pools.pair_id[pools.pair_id.isin(members).to_numpy()].to_numpy()]
        per[name]["fits_kept"] = kept.groupby(jd.to_numpy()).sum().reindex(per[name].index).fillna(0)
        per[name]["stage1_recall"] = (per[name].fits_kept / fits_total.loc[per[name].index]).where(
            fits_total.loc[per[name].index] > 0)
    for name, (base, scores) in f9_rows.items():
        returned = pd.Series(ref.pair_id.isin(lists[base]).to_numpy(), index=ids)
        per[name] = arm_per_jd(ref, scores.loc[ids], returned)
        per[name][["fits_kept", "stage1_recall"]] = per[base][["fits_kept", "stage1_recall"]]
        size_of[name] = size_of[base]

    from candidate_screener.evaluation.rerank_chat import deltas, summary
    from candidate_screener.evaluation.stage1_classifier import by_scope, mean_block, sizes
    arms = {}
    for name, d in per.items():
        block = {"summary": summary(d, stratum, seed),
                 "returned": sizes(d.returned) if name == X2_ALONE else sizes(size_of[name])}
        if "ceiling" in d:
            block["ceiling"] = by_scope(d, stratum, lambda r: mean_block(r.ceiling, seed))
            block["stage1_recall"] = by_scope(d, stratum, lambda r: mean_block(r.stage1_recall, seed))
            block["judged_fits_kept_total"] = int(d.fits_kept.sum())
        if name != X2_ALONE:
            block["delta_vs_x2_alone"] = deltas(d, per[X2_ALONE], stratum, seed)
        if name in f9_rows:
            block["delta_vs_f12_l2"] = deltas(d, per[f9_rows[name][0]], stratum, seed)
            if name.startswith("t = "):
                block["delta_vs_x2_top100_l2"] = deltas(d, per[f"{X2_L2} (F9)"], stratum, seed)
        elif name.startswith("t = "):
            block["delta_vs_x2_top100_l2"] = deltas(d, per[X2_L2], stratum, seed)
            c = (d.ceiling - per[X2_L2].ceiling)
            block["ceiling_delta_vs_x2_top100"] = by_scope(pd.DataFrame({"c": c}), stratum,
                                                           lambda r: mean_block(r.c, seed))
            if name != "t = 0.5" and name not in f9_rows:
                block["delta_vs_t050"] = deltas(d, per["t = 0.5"], stratum, seed)
        arms[name] = block
    op = arms[f"t = {OPERATING_POINT}"]["delta_vs_x2_alone"]["recall"]["total"]
    size_only = {}
    for t in SIZE_ONLY:
        m = p >= t
        n = pd.Series(m.to_numpy()).groupby(pools.jd_id.to_numpy()).sum()
        size_only[f"t = {t}"] = {**sizes(n), "unscored": int(len(set(p.index[m]) - set(ids)))}
    return {
        "plan": "plan/2026-10-18-classifier-then-rerank (step 4a)",
        "rules": RULES, "seed": seed,
        "reference": {"pairs": int(len(ref)), "by_run": ref.groupby("run").size().astype(int).to_dict(),
                      "by_source": ref.groupby("source").size().astype(int).to_dict(),
                      "replies_sha256": hashlib.sha256(cr.REPLIES.read_bytes()).hexdigest()},
        "arms": arms,
        "size_only": size_only,
        "bar": {"arm": f"t = {OPERATING_POINT} -> L2 vs X2 alone", "recall_delta": op["value"],
                "ci": op["ci"], "threshold": BAR,
                "verdict": "pass" if op["value"] >= BAR and op["ci"][0] > 0 else "fail"},
        "f9_follow_up": {"arm": f"t = {OPERATING_POINT} (F9) vs X2 alone",
                         "recall_delta": arms[f"t = {OPERATING_POINT} (F9)"]["delta_vs_x2_alone"]["recall"]["total"],
                         "meets_bar": bool(rc.passes(arms[f"t = {OPERATING_POINT} (F9)"]["delta_vs_x2_alone"]))},
        "drift": drift(ct.anchor_pairs(), stratum, seed),
        "retest": retest(ct.anchor_pairs("chat-1r"), set(chat1.pair_id[chat1.source == ct.TRAIN_SOURCE]),
                         stratum, seed),
        "note": ("Recall@10 and the ceiling are tie-aware against the chat's top 10 over every "
                 "chat-scored CV of the JD. Stage-1 recall counts judged final-set Fits kept "
                 f"(unjudged pool CVs count as not relevant; {resample.N_BOOT:,} JD resamples)."),
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
    for name, b in record["arms"].items():
        r = b["summary"]["recall"]["total"]
        c = b.get("ceiling", {}).get("total")
        print(f"  {name:20s} R@10 {r['value']:.3f}" + (f"  ceiling {c['value']:.3f}" if c else "")
              + f"  returned median {b['returned']['median']}")
    print(f"  bar: {record['bar']}")
    print(f"  drift offset: {record['drift']['offset']['total']}")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
