"""chat-3: the stage-2 training set of plan 2026-10-21-chat3-holdout (D62).

48 A2 **train-region** JDs, scored by the owner in claude.ai on X2's top 100 of a 500-CV pool,
with `chat-1`'s listwise instrument. Stage 2 is chosen and fitted on them alone, then tested once
on the 60 final-set JDs (`chat-1` as the reference).

- **JDs** (seed 0) come from the dev documents (`dev_pools.dev_documents_from`: P3's validation
  part minus its calibration draw, minus weak pairs, train region only), eligible as in
  `dev_pools.draw_pools` (a title; at least 8 CVs of the keyword), outside the 400 dev-pool JDs
  and D50's JDs. 32 `generic`, uniform; then 16 `targeted` from the final set's four targeted
  families, in its proportions (`TARGETED`).
- **Pools:** `dev_pools`' recipe over every dev CV: 8 of the JD's keyword, 2 of another, 490
  random (the final set's: 10 labelled picks, 80% keyword match, + 490 random).
- **Shortlist:** X2 (digest-guarded against the stage-1 record), top 100 per JD, as a
  `chat_rerank` plan with its own run label and files (`chat_rerank.CHAT3_*`).

Ids and digests go to the record; texts stay in `data/` (git-ignored). `--build` and `--select`
read no human label; `--test` reads the final set only to score (rule 3), after the fit.

    uv run python -m candidate_screener.evaluation.chat3 --build
    uv run python -m candidate_screener.evaluation.chat3 --select   # §4, once chat-3 is complete
    uv run python -m candidate_screener.evaluation.chat3 --test     # §5, once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED

RECORD = OUTPUT / "rerank" / "chat3-pools.json"
CACHE = PROCESSED / "rerank" / "chat3"
SEED = 0
N_GENERIC = 32
#: The final set's targeted stratum is Business Analyst 7, Data Science 6, Data Engineer 5,
#: Data Analyst 2 (of 20); scaled to 16 by largest remainder.
TARGETED = {"Business Analyst": 6, "Data Science": 5, "Data Engineer": 4, "Data Analyst": 1}
POOL_SIZE, N_SAME, N_OTHER = 500, 8, 2
K = 100
CONFIG = {"seed": SEED, "n_generic": N_GENERIC, "targeted": TARGETED, "pool_size": POOL_SIZE,
          "n_same_keyword": N_SAME, "n_other_keyword": N_OTHER, "k": K, "text_variant": "v2"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# --- pure ----------------------------------------------------------------------------------

def eligible_jds(jd: pd.DataFrame, cv: pd.DataFrame, n_same: int = N_SAME) -> pd.DataFrame:
    """JDs with a title and at least `n_same` CVs of their keyword (`dev_pools.draw_pools`' rule)."""
    from candidate_screener.evaluation.dev_pools import _missing_title
    n_kw = cv["Primary Keyword"].value_counts()
    ok = ~jd.title.map(_missing_title) & jd["Primary Keyword"].map(lambda k: n_kw.get(k, 0) >= n_same)
    return jd[ok].sort_values("id").reset_index(drop=True)


def draw_jds(eligible: pd.DataFrame, excluded: set, n_generic: int = N_GENERIC,
             targeted: Mapping[str, int] = TARGETED, seed: int = SEED) -> pd.DataFrame:
    """`(jd_id, stratum)`: `n_generic` uniform from `eligible` minus `excluded`, then per family of
    `targeted` its quota from what is left. Raises when a family is short."""
    pool = eligible[~eligible["id"].isin(excluded)].sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    generic = pool["id"].to_numpy()[np.sort(rng.permutation(len(pool))[:n_generic])]
    rows = [(j, "generic") for j in generic]
    left = pool[~pool["id"].isin(set(generic))]
    for family, n in targeted.items():
        ids = left[left["Primary Keyword"] == family]["id"].to_numpy()
        if len(ids) < n:
            raise ValueError(f"{family}: {len(ids)} eligible JDs for a quota of {n}")
        rows.extend((j, "targeted") for j in ids[np.sort(rng.permutation(len(ids))[:n])])
    return pd.DataFrame(rows, columns=["jd_id", "stratum"])


def draw_pools(jds: pd.DataFrame, jd: pd.DataFrame, cv: pd.DataFrame, pool_size: int = POOL_SIZE,
               n_same: int = N_SAME, n_other: int = N_OTHER, seed: int = SEED) -> pd.DataFrame:
    """`(jd_id, cv_id, stratum, slot)`: per JD of `jds`, `n_same` CVs of its keyword, `n_other` of
    another, then random CVs up to `pool_size`, no CV twice (`dev_pools.draw_pools`' recipe)."""
    from candidate_screener.evaluation.dev_pools import SLOTS
    cv = cv.sort_values("id").reset_index(drop=True)
    all_ids, cv_kw = cv["id"].to_numpy(), cv["Primary Keyword"].to_numpy()
    by_kw = {k: g["id"].to_numpy() for k, g in cv.groupby("Primary Keyword", sort=True)}
    kw_of = dict(zip(jd["id"], jd["Primary Keyword"]))
    rows = []
    for k, (jd_id, stratum) in enumerate(zip(jds.jd_id, jds.stratum)):
        r = np.random.default_rng([seed, k])
        kw = kw_of[jd_id]
        same = r.choice(by_kw[kw], n_same, replace=False)
        other = r.choice(all_ids[cv_kw != kw], n_other, replace=False)
        rest = all_ids[~np.isin(all_ids, np.concatenate([same, other]))]
        rand = r.choice(rest, pool_size - n_same - n_other, replace=False)
        for slot, ids in zip(SLOTS, (same, other, rand)):
            rows.extend((jd_id, c, stratum, slot) for c in ids)
    return pd.DataFrame(rows, columns=["jd_id", "cv_id", "stratum", "slot"])


def overlaps(frame: pd.DataFrame, forbidden: Mapping[str, Mapping[str, set]]) -> dict[str, int]:
    """Per named id set: how many of `frame`'s JDs (`jd`) and CVs (`cv`) fall in it. All 0 = clean."""
    jds, cvs = set(frame.jd_id), set(frame.cv_id)
    return {name: len(jds & sides.get("jd", set())) + len(cvs & sides.get("cv", set()))
            for name, sides in forbidden.items()}


# --- build ---------------------------------------------------------------------------------

def _forbidden() -> dict[str, dict[str, set]]:
    from candidate_screener.evaluation import dev_pools, final_set
    from candidate_screener.rerank import scoring_batch as sb
    fs = pd.read_csv(final_set.PAIRS_CSV, usecols=["jd_id", "cv_id"])       # ids only, no labels
    d50 = pd.read_csv(sb.BATCH_MANIFEST, usecols=["jd_id"])
    return {"eval_region": dev_pools._eval_ids(), "weak_pairs": dev_pools._weak_ids(),
            "final_set": {"jd": set(fs.jd_id), "cv": set(fs.cv_id)},
            "d50_jds": {"jd": set(d50.jd_id)},
            "dev_pool_jds": {"jd": set(dev_pools.build()["pools"].jd_id)}}


def _x2(frame: pd.DataFrame) -> pd.Series:
    """X2's scores on `frame`, after checking X2 still reproduces the stage-1 record."""
    from candidate_screener.evaluation import retrieval_pilot, stage1
    from candidate_screener.rerank import chat_rerank as cr
    spec = next(s for s in stage1.specs() if s.factory().name == cr.X2)
    system = spec.factory()
    system.fit(None)
    final = system.score(retrieval_pilot.build_pools(SEED, stage1.TEXT_VARIANT))
    if stage1._sha(final) != json.loads(stage1.RECORD.read_text())["systems"][cr.X2]["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the committed stage-1 record")
    return system.score(frame)


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(pair frame with texts, chat-3 plan, record). Refuses any id it must not hold."""
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.rerank import chat_rerank as cr
    docs, prep, _ = dev_pools._prepared()
    dev = dev_pools.dev_documents_from(docs, prep)
    jd, cv = dev["jd"], dev["cv"]
    forbidden = _forbidden()
    excluded = forbidden["dev_pool_jds"]["jd"] | forbidden["d50_jds"]["jd"]
    elig = eligible_jds(jd, cv)
    jds = draw_jds(elig, excluded)
    _log(f"{len(jds)} JDs from {len(elig)} eligible ({len(elig[elig['id'].isin(excluded)])} excluded)")
    pools = draw_pools(jds, jd, cv)
    hits = overlaps(pools, forbidden)
    if any(hits.values()):
        raise ValueError(f"chat-3 pools hold forbidden ids: {hits}")
    jdi, cvi = jd.set_index("id"), cv.set_index("id")
    frame = pools.assign(pair_id=pools.jd_id + "__" + pools.cv_id,
                         jd_title=pools.jd_id.map(jdi.title), jd_text=pools.jd_id.map(jdi.text),
                         cv_text=pools.cv_id.map(cvi.text), jd_keyword=pools.jd_id.map(jdi["Primary Keyword"]))
    _log("X2 scores")
    x2 = _x2(frame)
    plan = cr.plan_from(frame, x2.loc[frame.pair_id.to_numpy()].to_numpy(), k=K)
    plan = plan.assign(run=cr.CHAT3_RUN, source="x2_top100")
    fam = frame.drop_duplicates("jd_id").groupby(["stratum", "jd_keyword"]).size()
    record = {
        "plan": "plan/2026-10-21-chat3-holdout (D62)", "config": CONFIG,
        "jds": {"total": int(frame.jd_id.nunique()),
                "by_stratum": {k: int(v) for k, v in jds.stratum.value_counts().sort_index().items()},
                "targeted_by_family": {k: int(v) for (s, k), v in fam.items() if s == "targeted"},
                "eligible": int(len(elig)), "excluded_eligible": int(elig["id"].isin(excluded).sum()),
                "ids": sorted(map(str, jds.jd_id))},
        "pairs": {"pool": int(len(frame)), "shortlist": int(len(plan)),
                  "batches": int(plan.groupby("jd_id").batch.nunique().sum()),
                  "pool_ids_sha256": _sha(frame.pair_id), "shortlist_ids_sha256": _sha(plan.pair_id)},
        "forbidden_overlap": hits, "x2_guard": "stage-1 record reproduced",
        "instrument_sha256": cr.instrument_sha256(), "run": cr.CHAT3_RUN}
    return frame, plan, record


def pairs() -> pd.DataFrame:
    """The cached chat-3 pair frame (`pair_id`, `jd_id`, `cv_id`, `stratum`, `jd_title`, `jd_text`,
    `cv_text`, ...). `--build` writes it."""
    path = CACHE / f"pools-{json.loads(RECORD.read_text())['pairs']['pool_ids_sha256'][:12]}.parquet"
    return pd.read_parquet(path)


# --- choose and fit stage 2 on chat-3 (plan §4) ----------------------------------------------

SELECT_RECORD = OUTPUT / "rerank" / "chat3-select.json"
TEST_RECORD = OUTPUT / "rerank" / "chat3-test.json"
#: §4's menu, in tie order: a tie (3 dp) goes to the earlier arm (F9 before F12, L1 before L2).
MENU = (("L1", "F9"), ("L2", "F9"), ("L1", "F12"), ("L2", "F12"))
BAR = 0.05          # rule 1: S3's bar vs X2's order (owner, 18 Oct 2026)
STAGE2 = "X2 top 100 -> stage 2 (chat-3)"
S1_ORDER, OLD_L2 = "X2 top 100 -> S1", "X2 top 100 -> L2 (chat-1, out of fold)"


def join_scores(plan: pd.DataFrame, replies: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """`plan`'s pairs with their chat-3 `score` and `label`. Refuses another run's or another
    instrument's rows, and an incomplete collection. Pure."""
    from candidate_screener.rerank import chat_rerank as cr
    d = replies[replies.run == cr.CHAT3_RUN]
    if set(d.instrument_sha256) != {instrument}:
        raise ValueError("chat-3 replies are not all from the build's instrument")
    out = plan[["pair_id", "jd_id", "cv_id", "stratum", "source"]].merge(
        d[["pair_id", "score", "label"]], on="pair_id", how="inner", validate="one_to_one")
    missing = set(plan.pair_id) - set(out.pair_id)
    if missing:
        raise ValueError(f"{len(missing)} of {len(plan)} chat-3 pairs have no score yet")
    return out.sort_values("pair_id").reset_index(drop=True)


def choose(totals: Mapping[tuple, float], menu=MENU) -> tuple:
    """The arm with the highest out-of-fold Recall@10 (total, 3 dp); a tie goes to the earlier arm."""
    return max(menu, key=lambda a: (round(totals[a], 3), -menu.index(a)))


def targets() -> pd.DataFrame:
    """chat-3, complete: one row per planned pair with its score, label and JD fold."""
    from candidate_screener.rerank import chat_rerank as cr, chat_targets as ct
    chat = join_scores(pd.read_csv(cr.CHAT3_PLAN), cr.latest(cr.CHAT3_REPLIES),
                       json.loads(RECORD.read_text())["instrument_sha256"])
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    return chat.assign(fold=chat.jd_id.map(ct.folds(stratum)))


def _kind(learner: str) -> str:
    from candidate_screener.rerank import learners
    return {"L1": learners.L1, "L2": learners.L2}[learner]


def select(seed: int = SEED) -> dict:
    """§4: each menu arm out of fold on chat-3 (5 JD folds), Recall@10 against chat-3's top 10."""
    from candidate_screener.evaluation import rerank_chat as rc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_rerank as cr, chat_targets as ct, learners
    chat = targets()
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    _log(f"chat-3: {len(chat)} pairs, {chat.jd_id.nunique()} JDs; features on the pools")
    X = learned.Features()(pairs()).loc[chat.pair_id.to_numpy()]
    x2 = pd.read_csv(cr.CHAT3_PLAN).set_index("pair_id").x2
    arms = {"X2's order": ct.per_jd(chat, x2)}
    for learner, fs in MENU:
        s, _ = learners.out_of_fold(_kind(learner), X, chat, list(learned.FEATURE_SETS[fs]))
        arms[f"{learner} {fs}"] = ct.per_jd(chat, s)
    win = choose({a: rc.summary(arms[f"{a[0]} {a[1]}"], stratum, seed)["recall"]["total"]["value"]
                  for a in MENU})
    return {"plan": "plan/2026-10-21-chat3-holdout (§4)", "seed": seed,
            "chat_3": {"pairs": int(len(chat)), "jds": int(chat.jd_id.nunique()),
                       "by_stratum": {k: int(v) for k, v in stratum.value_counts().sort_index().items()},
                       "mean_score": round(float(chat.score.mean()), 3),
                       "labels": {k: int(v) for k, v in chat.label.value_counts().sort_index().items()}},
            "menu": [f"{a} {b}" for a, b in MENU],
            "rule": "highest out-of-fold Recall@10 (total, 3 dp); a tie goes to the earlier menu arm",
            "arms": {n: {"summary": rc.summary(p, stratum, seed)} for n, p in arms.items()},
            "winner": {"learner": win[0], "features": win[1]}}


def fit_stage2(chat: pd.DataFrame, X: pd.DataFrame, learner: str, features: str):
    """The winner, fitted once on every chat-3 pair. Reads nothing but its arguments."""
    from candidate_screener.matching import learned
    from candidate_screener.rerank import learners
    return learners.LEARNERS[_kind(learner)](list(learned.FEATURE_SETS[features])).fit(
        X.loc[chat.pair_id.to_numpy()], chat.score.to_numpy())


def rules(results: dict, human: dict) -> dict:
    """The three rules of plan §5, fixed before the test was scored. Pure."""
    d1 = results[STAGE2]["delta_vs_X2"]["recall"]["total"]
    d2 = results[STAGE2][f"delta_vs_{S1_ORDER}"]["recall"]["total"]
    d3 = next(f for f in human["systems"][STAGE2]["auc_delta_vs_X2"]
              if f["scope"] == "total" and f["subset"] == "all")
    return {"1 headline: Recall@10 vs X2 >= +0.05, CI above 0":
                {"delta": d1, "passes": bool(d1["value"] >= BAR and d1["ci"][0] > 0)},
            "2 chat vs chat-free: Recall@10 vs S1's order > 0, CI above 0":
                {"delta": d2, "passes": bool(d2["value"] > 0 and d2["ci"][0] > 0)},
            "3 humans: per-JD AUC (final label, all) vs X2 > 0, CI above 0":
                {"delta": {"value": d3["delta"], "ci": d3["ci"], "n_jd": d3["n_jd"]},
                 "passes": bool(d3["delta"] > 0 and d3["ci"][0] > 0)}}


def test(seed: int = SEED, write=None) -> dict:
    """§5: fit §4's winner on chat-3, then score the 60 final-set JDs once against `chat-1`."""
    from candidate_screener.evaluation import rerank_chat as rc, rerank_embed, retrieval_pilot, \
        stage1_classifier as sc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, chunked
    win = json.loads(SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    feats = learned.Features()
    chat3 = targets()
    model = fit_stage2(chat3, feats(pairs()), win["learner"], win["features"])
    _log(f"stage 2 = {win['learner']} {win['features']}, fitted on {len(chat3)} chat-3 pairs; the test")
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    ids = chat.pair_id.to_numpy()
    x2, old_l2 = rerank_embed._x2_and_l2(chat_all, seed)       # both digest-guarded
    pools = retrieval_pilot.build_pools(seed, rerank_embed.TEXT_VARIANT)
    Xp = feats(pools)
    stage2 = pd.Series(model.predict(Xp.loc[ids]), index=ids)
    p = sc.fit_arms(feats)[3][sc.PRIMARY].proba(Xp).loc[pools.pair_id.to_numpy()]
    systems = {"X2": x2, STAGE2: stage2, S1_ORDER: p, OLD_L2: old_l2}
    per = {n: ct.per_jd(chat, s) for n, s in systems.items()}
    results = {}
    for n in systems:
        results[n] = {"summary": rc.summary(per[n], stratum, seed)}
        for ref in ("X2", S1_ORDER, STAGE2):
            if n != ref:
                results[n][f"delta_vs_{ref}"] = rc.deltas(per[n], per[ref].loc[per[n].index], stratum, seed)
    _log("human check")
    human = rerank_embed._human(systems, chat, ("X2", S1_ORDER), seed)
    doc = {"plan": "plan/2026-10-21-chat3-holdout (§5)", "seed": seed,
           "stage_2": {**win, "fitted_on": f"chat-3, {len(chat3)} pairs, {chat3.jd_id.nunique()} JDs"},
           "reference": "chat-1 (X2's top 100 of the 60 final-set JDs)", "jds": int(chat.jd_id.nunique()),
           "rules": rules(results, human), "results": results, "human": human}
    if write is not None:
        write(doc)
    return doc


COMPARE_RECORD = OUTPUT / "rerank" / "chat3-vs-c2.json"
C2_ALONE, C2_S1, C2_STAGE2 = "X2 top 100 -> C2", "X2 top 100 -> C2 + S1", "X2 top 100 -> C2 + stage 2 (chat-3)"
C2_L2, C2_CASCADE = "X2 top 100 -> C2 + L2 (chat-1)", "L2 (chat-1) top 30 -> C2 + L2"


def compare_c2(seed: int = SEED) -> dict:
    """Exploratory (owner, 22 Oct 2026: "compare this against the C2 results"): §5's arms beside the
    zero-shot C2 arms of plan 2026-10-18 steps 5b and 5d, on the same 60 JDs and `chat-1`
    reference, paired against stage 2 (chat-3). C2 + stage 2 (chat-3) is new and decides nothing.
    Uses only cached C2 scores; refits §4's winner exactly as `test` did and checks it reproduces."""
    from candidate_screener.evaluation import rerank_chat as rc, rerank_embed as re_, retrieval_pilot, \
        stage1_classifier as sc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, chunked, cross
    win = json.loads(SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    tested = json.loads(TEST_RECORD.read_text(encoding="utf-8"))
    feats = learned.Features()
    chat3 = targets()
    model = fit_stage2(chat3, feats(pairs()), win["learner"], win["features"])
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    fold_of = chat.drop_duplicates("jd_id").set_index("jd_id").fold
    ids = chat.pair_id.to_numpy()
    x2, l2 = re_._x2_and_l2(chat_all, seed)
    pools = retrieval_pilot.build_pools(seed, re_.TEXT_VARIANT)
    Xp = feats(pools)
    stage2 = pd.Series(model.predict(Xp.loc[ids]), index=ids)
    p = sc.fit_arms(feats)[3][sc.PRIMARY].proba(Xp).loc[pools.pair_id.to_numpy()]
    c2 = cross.score_pairs(chat, re_.CASCADE_SYSTEM, cached_only=True)
    systems = {"X2": x2, C2_ALONE: c2, S1_ORDER: p, C2_S1: re_.cascade_scores(chat, p, c2, k=K)[0],
               STAGE2: stage2, C2_STAGE2: re_.cascade_scores(chat, stage2, c2, k=K)[0],
               OLD_L2: l2, C2_L2: re_.cascade_scores(chat, l2, c2, k=K)[0],
               C2_CASCADE: re_.cascade_scores(chat, l2, c2, k=re_.CASCADE_K)[0]}
    per = {n: ct.per_jd(chat, s) for n, s in systems.items()}
    if round(per[STAGE2].recall.mean(), 6) != tested["results"][STAGE2]["summary"]["recall"]["total"]["value"]:
        raise ValueError("the refitted stage 2 does not reproduce §5's Recall@10")

    def scored(jds) -> dict:
        res = {}
        for n in systems:
            q = per[n].loc[per[n].index.intersection(jds)]
            res[n] = {"summary": rc.summary(q, stratum, seed)}
            for ref in (STAGE2, "X2"):
                if n != ref:
                    res[n][f"delta_vs_{ref}"] = rc.deltas(q, per[ref].loc[q.index], stratum, seed)
        return res
    held = fold_of.index[fold_of.isin(re_.HELD_OUT_FOLDS)]
    _log("human check (exploratory)")
    return {"plan": "plan/2026-10-21-chat3-holdout (exploratory comparison with steps 5b-5d)",
            "status": "exploratory", "seed": seed, "stage_2": win, "reference": "chat-1",
            "note": "C2's fold-0 scores were seen in the 5a pilot; C2 + stage 2 (chat-3) is new",
            "all_60": scored(fold_of.index), "folds_1_4": scored(held),
            "human": re_._human(systems, chat, ("X2", STAGE2), seed)}


COVERAGE_RECORD = OUTPUT / "rerank" / "chat3-stage1-coverage.json"
COVERAGE_K = (10, 20, 30, 50, 100)
PIPELINE = "X2 top 100 -> stage 2 (chat-3)"


def pool_coverage(pool: pd.DataFrame, ref: pd.DataFrame, system: pd.Series, k: int) -> pd.DataFrame:
    """Per JD: `coverage`, the share of the chat's tie-aware top 10 (over `ref`'s scored CVs) inside
    `system`'s top `k` of the **whole pool** (ties by pair id); `scored`, the share of that top `k`
    with a chat score. Unscored CVs count as misses, so a ranking that reaches past the scored set
    gets a lower bound. Pure."""
    from candidate_screener.rerank.chat_targets import K as TOP, chat_weights, top_k_mask
    refs = {jd: g for jd, g in ref.groupby("jd_id", sort=True)}
    out = {}
    for jd, g in pool.sort_values("pair_id").groupby("jd_id", sort=True):
        ids = g.pair_id.to_numpy()
        top = set(ids[top_k_mask(ids, system.loc[ids].to_numpy(dtype=float), k)])
        r = refs[jd].sort_values("pair_id")
        w = chat_weights(r.score.to_numpy())
        out[jd] = {"coverage": float(w[r.pair_id.isin(top).to_numpy()].sum() / min(TOP, len(r))),
                   "scored": float(len(top & set(r.pair_id)) / len(top))}
    return pd.DataFrame.from_dict(out, orient="index")


def _shortlist_then(pool: pd.DataFrame, first: pd.Series, second: pd.Series, k: int = K) -> pd.Series:
    """A pool-wide order: `first`'s top `k` per JD ordered by `second`, above the rest in `first`'s order."""
    from candidate_screener.evaluation.rerank_embed import cascade_scores
    return cascade_scores(pool, first, second, k=k)[1]


def stage1_coverage(seed: int = SEED) -> dict:
    """Exploratory (owner, 22 Oct 2026): candidate stage-1 rankings over the 500-CV pool, against the
    current pipeline. X2; S1 (D50); L1 F12 trained on chat-3 applied to all 500 (it saw only X2's top
    100 in training); X2 top 100 -> stage 2 (chat-3), and -> L2 (`chat-1`, out of fold) for context.
    Reference: `chat-1` + `chat-2`, the latest score per pair (step 4a's)."""
    from candidate_screener.evaluation import rerank_embed as re_, retrieval_pilot, stage1_classifier as sc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, chunked
    win = json.loads(SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    feats = learned.Features()
    model = fit_stage2(targets(), feats(pairs()), win["learner"], win["features"])
    ref = ct.load_latest()
    stratum = ref.drop_duplicates("jd_id").set_index("jd_id").stratum
    pools = retrieval_pilot.build_pools(seed, re_.TEXT_VARIANT)
    x2_chat, l2 = re_._x2_and_l2(chunked.chat_frame(), seed)
    x2 = sc._x2().score(pools)
    Xp = feats(pools)
    ids = pools.pair_id.to_numpy()
    l1_pool = pd.Series(model.predict(Xp.loc[ids]), index=ids)
    p = sc.fit_arms(feats)[3][sc.PRIMARY].proba(Xp).loc[ids]
    systems = {"X2 (500)": x2, "S1 (500)": p, f"{win['learner']} {win['features']} chat-3 (500)": l1_pool,
               PIPELINE: _shortlist_then(pools, x2, l1_pool),
               "X2 top 100 -> L2 (chat-1, out of fold)": _shortlist_then(pools, x2, l2.reindex(ids).fillna(-1e9))}
    if not (x2.loc[x2_chat.index] == x2_chat).all():
        raise ValueError("X2's pool scores and the guarded chat scores disagree")
    per = {n: {k: pool_coverage(pools, ref, s, k) for k in COVERAGE_K} for n, s in systems.items()}
    out = {}
    for n in systems:
        out[n] = {}
        for k in COVERAGE_K:
            c = per[n][k]
            out[n][str(k)] = {"coverage": re_.block(c.coverage, stratum, seed),
                              "scored_share": round(float(c.scored.mean()), 4)}
            if n != PIPELINE:
                out[n][str(k)]["delta_vs_pipeline"] = re_.block(c.coverage - per[PIPELINE][k].coverage, stratum, seed)
    return {"plan": "plan/2026-10-21-chat3-holdout (exploratory: stage-1 variants)", "status": "exploratory",
            "seed": seed, "reference": "chat-1 + chat-2, latest score per pair (step 4a)",
            "pairs": {"pool": int(len(pools)), "scored": int(len(ref)), "jds": int(ref.jd_id.nunique())},
            "metric": "share of the chat's tie-aware top 10 inside the system's top k of the 500-CV pool; "
                      "unscored CVs count as misses (a lower bound for rankings that reach past the scored set)",
            "k": list(COVERAGE_K), "systems": out}


def main(argv: list[str] | None = None) -> int:
    from candidate_screener.rerank import chat_rerank as cr
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="draw the 48 JDs, pools, X2 top 100, the chat-3 plan")
    ap.add_argument("--select", action="store_true", help="§4: choose stage 2 on chat-3 (needs all of it)")
    ap.add_argument("--test", action="store_true", help="§5: fit the winner, score the 60 JDs once")
    ap.add_argument("--compare-c2", action="store_true", help="exploratory: §5 beside the C2 arms of steps 5b-5d")
    ap.add_argument("--stage1-coverage", action="store_true", help="exploratory: stage-1 variants over the pool")
    args = ap.parse_args(argv)
    if args.stage1_coverage:
        doc = stage1_coverage()
        COVERAGE_RECORD.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for n, by_k in doc["systems"].items():
            print(f"{n:42s} " + "  ".join(f"k={k}: {v['coverage']['total']['value']:.3f} ({v['scored_share']:.2f})"
                                          for k, v in by_k.items()))
        return 0
    if args.compare_c2:
        doc = compare_c2()
        COMPARE_RECORD.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for n, r in doc["all_60"].items():
            d = r.get(f"delta_vs_{STAGE2}", {}).get("recall", {}).get("total")
            print(f"{n:40s} R@10 {r['summary']['recall']['total']['value']:.3f}"
                  + (f"  vs stage 2 {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d else ""))
        return 0
    if args.select:
        if TEST_RECORD.exists():
            raise SystemExit(f"{TEST_RECORD} exists: the choice was tested, never re-choose")
        doc = select()
        SELECT_RECORD.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for n, r in doc["arms"].items():
            print(f"{n:12s} R@10 {r['summary']['recall']['total']['value']:.3f}")
        print("winner:", doc["winner"])
        return 0
    if args.test:
        if TEST_RECORD.exists():
            raise SystemExit(f"{TEST_RECORD} exists: the test is scored once")
        doc = test(write=lambda d: TEST_RECORD.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8"))
        for n, r in doc["results"].items():
            print(f"{n:42s} R@10 {r['summary']['recall']['total']['value']:.3f}")
        for rule, r in doc["rules"].items():
            print(f"{'PASS' if r['passes'] else 'FAIL'}  {rule}: {r['delta']}")
        return 0
    if not args.build:
        ap.error("--build, --select or --test")
    if cr.CHAT3_REPLIES.exists():
        raise SystemExit(f"{cr.CHAT3_REPLIES} exists: chat-3 is being collected, never rebuild its plan")
    frame, plan, record = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE / f"pools-{record['pairs']['pool_ids_sha256'][:12]}.parquet", index=False)
    cr.CHAT3_PLAN.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(cr.CHAT3_PLAN, index=False)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _log(f"wrote {cr.CHAT3_PLAN} ({record['pairs']['batches']} batches) and {RECORD}")
    print(json.dumps({k: record[k] for k in ("jds", "pairs", "forbidden_overlap")}, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
