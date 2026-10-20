"""Step 5a pilot of plan 2026-10-18-classifier-then-rerank: E1 (Qwen3-Embedding-0.6B, zero-shot)
as a stage-2 signal on a sample of `chat-1` (exploratory).

Design fixed in the plan before scoring:
- **Sample:** fold 0's 12 JDs, X2's top 100 only. Per JD every CV in the chat's tie-aware top 10
  (`chat_weights > 0`) plus `N_OTHER` others drawn at random (seed 0) from the rest of its 100.
- **Systems on the same pairs:** X2; L2 on F9 out of fold (4b's A1, digest-guarded); E1 under the
  instruction variants I0-I3 (`embedder.INSTRUCTIONS`); U2 = the per-JD min-max sum of E1 and L2
  over the sampled CVs.
- **Metric:** per-JD AUC of the score for the chat's top 10 against the sampled others (Recall@10
  needs a full list); secondary, within-JD Spearman with the chat score. Mean over JDs, JD-resampled
  95% CI, generic / targeted / total, paired deltas vs X2 and vs L2.
- **Rule for a full run:** its instruction is the variant with the highest pilot AUC (total).

The cross-encoder pilot (`--pilot-cross`) scores `rerank.cross.SYSTEMS` on the same pairs,
beside X2, L2 and E1 I0, with the same metric. `--coverage`, `--cascade` (5b), `--s1-zero-shot`
(5c) and `--full-c2` (5d) are the later steps; their designs are in the plan.

    uv run python -m candidate_screener.evaluation.rerank_embed --pilot
    uv run python -m candidate_screener.evaluation.rerank_embed --pilot-cross
    uv run python -m candidate_screener.evaluation.rerank_embed --full-c2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT

PILOT_RECORD = OUTPUT / "rerank" / "embed-pilot.json"
CROSS_RECORD = OUTPUT / "rerank" / "cross-pilot.json"
COVERAGE_RECORD = OUTPUT / "rerank" / "topk-coverage.json"
COVERAGE_K = (10, 20, 30, 40, 50, 100)
CASCADE_RECORD = OUTPUT / "rerank" / "cascade-c2.json"
CASCADE_K = 30
CASCADE_SYSTEM = "C2 Qwen3-Reranker, fit"
HELD_OUT_FOLDS = (1, 2, 3, 4)
FUSED, C_ONLY = "L2 top 30 -> C2 + L2", "L2 top 30 -> C2"
S1_RECORD = OUTPUT / "rerank" / "s1-zero-shot.json"
S1_T, S1_CAP = 0.3, 30
S1_ALONE, S1_C2, S1_FUSED = "S1 alone", "S1 -> C2", "S1 -> C2 + S1"
X2_L2, X2_ALONE = "X2 top 100 -> L2 (F9)", "X2 alone"
FULL_RECORD = OUTPUT / "rerank" / "full-c2.json"
FULL_K = 100
FULL_C2, FULL_C2_L2 = "X2 top 100 -> C2", "X2 top 100 -> C2 + L2"
FULL_S1, FULL_C2_S1 = "X2 top 100 -> S1", "X2 top 100 -> C2 + S1"
TEXT_VARIANT = "v2"
PILOT_FOLD = 0
N_OTHER = 8
SEED = 0
DECIMALS = 6


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _r(v) -> float:
    return round(float(v), DECIMALS)


# --- pure helpers ---------------------------------------------------------------------------

def pilot_sample(chat: pd.DataFrame, fold: int = PILOT_FOLD, n_other: int = N_OTHER,
                 seed: int = SEED) -> pd.DataFrame:
    """Fold `fold`'s pairs: per JD the chat's tie-aware top 10 (`top`) and `n_other` others."""
    from candidate_screener.rerank.chat_targets import chat_weights
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in chat[chat.fold == fold].sort_values("pair_id").groupby("jd_id", sort=True):
        top = chat_weights(g.score.to_numpy()) > 0
        rest = g.pair_id.to_numpy()[~top]
        keep = set(g.pair_id.to_numpy()[top]) | set(rng.choice(rest, min(n_other, len(rest)), replace=False))
        parts.append(g[g.pair_id.isin(keep)].assign(top=lambda d: d.pair_id.isin(g.pair_id.to_numpy()[top])))
    return pd.concat(parts, ignore_index=True)


def per_jd_auc(sample: pd.DataFrame, system: pd.Series) -> pd.Series:
    """AUC of `system` for `top` within each JD (a tie counts half)."""
    from sklearn.metrics import roc_auc_score
    s = system.loc[sample.pair_id.to_numpy()].to_numpy(dtype=float)
    frame = pd.DataFrame({"jd": sample.jd_id.to_numpy(), "y": sample.top.to_numpy(), "s": s})
    return pd.Series({jd: roc_auc_score(g.y, g.s) for jd, g in frame.groupby("jd", sort=True)
                      if g.y.nunique() == 2}, dtype=float)


def per_jd_spearman(sample: pd.DataFrame, system: pd.Series) -> pd.Series:
    s = system.loc[sample.pair_id.to_numpy()].to_numpy(dtype=float)
    frame = pd.DataFrame({"jd": sample.jd_id.to_numpy(), "chat": sample.score.to_numpy(dtype=float), "s": s})
    return pd.Series({jd: g.chat.corr(g.s, method="spearman") for jd, g in frame.groupby("jd", sort=True)},
                     dtype=float)


def minmax_fuse(sample: pd.DataFrame, a: pd.Series, b: pd.Series) -> pd.Series:
    """Equal-weight sum of `a` and `b`, each min-max scaled within JD over `sample`'s pairs."""
    ids = sample.pair_id.to_numpy()
    frame = pd.DataFrame({"jd": sample.jd_id.to_numpy(), "a": a.loc[ids].to_numpy(dtype=float),
                          "b": b.loc[ids].to_numpy(dtype=float)}, index=ids)

    def scaled(col):
        g = frame.groupby("jd")[col]
        span = (g.transform("max") - g.transform("min")).replace(0, 1)
        return (frame[col] - g.transform("min")) / span
    return scaled("a") + scaled("b")


def block(per: pd.Series, stratum: pd.Series, seed: int = SEED) -> dict:
    from candidate_screener.evaluation.stage1_classifier import by_scope, mean_block
    return by_scope(per, stratum, lambda r: mean_block(r.astype(float), seed))


def coverage(chat: pd.DataFrame, system: pd.Series, k: int) -> pd.Series:
    """Per JD: the share of the chat's tie-aware top 10 inside `system`'s top `k` (ties by pair
    id), i.e. Recall@10 with a perfect reranker on that list. `k` = 10 is Recall@10 itself."""
    from candidate_screener.rerank.chat_targets import K, chat_weights, top_k_mask
    out = {}
    for jd, g in chat.sort_values("pair_id").groupby("jd_id", sort=True):
        ids = g.pair_id.to_numpy()
        w = chat_weights(g.score.to_numpy())
        out[jd] = float(w[top_k_mask(ids, system.loc[ids].to_numpy(dtype=float), k)].sum() / min(K, len(ids)))
    return pd.Series(out, dtype=float)


def _minmax(v: np.ndarray) -> np.ndarray:
    span = v.max() - v.min()
    return (v - v.min()) / (span if span > 0 else 1.0)


def cascade_scores(chat: pd.DataFrame, first: pd.Series, second: pd.Series, k: int = CASCADE_K
                   ) -> tuple[pd.Series, pd.Series]:
    """(fused, second only) over every pair of `chat`, by pair id. Per JD, `first`'s top `k` (ties
    by pair id) is reordered, by min-max `first` + min-max `second` within the k, or by `second`
    alone; the rest stay in `first`'s order below them. `second` needs only the top-k pairs."""
    from candidate_screener.rerank.chat_targets import top_k_mask
    fused = pd.Series(np.nan, index=chat.pair_id.to_numpy())
    alone = fused.copy()
    for _, g in chat.sort_values("pair_id").groupby("jd_id", sort=True):
        ids = g.pair_id.to_numpy()
        f = first.loc[ids].to_numpy(dtype=float)
        top = top_k_mask(ids, f, k)
        below = _minmax(f)[~top]                 # in [0, 1]; the top k sit above 10
        sec = _minmax(second.loc[ids[top]].to_numpy(dtype=float))
        fused.loc[ids[top]] = 10 + _minmax(f[top]) + sec
        alone.loc[ids[top]] = 10 + sec
        fused.loc[ids[~top]] = below
        alone.loc[ids[~top]] = below
    return fused, alone


def capped_list(frame: pd.DataFrame, p: pd.Series, t: float = S1_T, cap: int = S1_CAP) -> set:
    """Pair ids of each JD's pairs with `p` >= `t`, at most its top `cap` (ties by pair id)."""
    d = frame.assign(p=p.loc[frame.pair_id.to_numpy()].to_numpy())
    d = d[d.p >= t].sort_values(["jd_id", "p", "pair_id"], ascending=[True, False, True])
    return set(d.groupby("jd_id").head(cap).pair_id)


def best_variant(auc_total: dict[str, float]) -> str:
    """Highest pilot AUC (total, 6 dp); a tie goes to the earlier variant."""
    order = list(auc_total)
    return max(order, key=lambda v: (round(auc_total[v], DECIMALS), -order.index(v)))


# --- run ------------------------------------------------------------------------------------

def _x2_and_l2(chat: pd.DataFrame, seed: int = SEED) -> tuple[pd.Series, pd.Series]:
    """X2 and L2-on-F9 (out of fold) scores for every chat pair, each digest-guarded."""
    from candidate_screener.evaluation import feature_arms, rerank_chat as rc, retrieval_pilot, stage1, \
        stage1_classifier as sc
    from candidate_screener.matching import learned
    from candidate_screener.rerank import learners
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    x2 = sc._x2().score(pools)
    committed = json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"][stage1.X2]
    if rc._sha(x2) != committed["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the stage-1 record")
    Xp = learned.Features()(pools)
    l2, _ = learners.out_of_fold(learners.L2, Xp.loc[chat.pair_id.to_numpy()], chat, list(learned.F9))
    a1 = json.loads(feature_arms.RECORD.read_text(encoding="utf-8"))["arms"]["A1 F9"]["l2_scores_sha256"]
    if rc._sha(l2) != a1:
        raise ValueError("L2 on F9 does not reproduce step 4b's A1 arm")
    return x2.loc[chat.pair_id.to_numpy()], l2


def _inputs(seed: int = SEED):
    """The pilot's chat pairs, strata, sample, and X2 / L2 scores (shared by both pilots)."""
    from candidate_screener.rerank import chat_targets as ct, chunked
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    sample = pilot_sample(chat, seed=seed)
    _log(f"sample: {sample.jd_id.nunique()} JDs, {len(sample)} pairs, {sample.cv_id.nunique()} CVs, "
         f"{int(sample.top.sum())} in the chat's top 10")
    _log("X2 and L2 (F9) scores")
    x2, l2 = _x2_and_l2(chat_all, seed)   # every chat pair, as 4b scored them (the digest guard)
    return sample, stratum, x2, l2


def _sample_doc(sample: pd.DataFrame, stratum: pd.Series) -> dict:
    return {"fold": PILOT_FOLD, "n_other": N_OTHER, "jds": int(sample.jd_id.nunique()),
            "pairs": int(len(sample)), "cvs": int(sample.cv_id.nunique()),
            "chat_top10_pairs": int(sample.top.sum()),
            "by_stratum_jds": {k: int(v) for k, v in stratum.loc[sample.jd_id.unique()].value_counts().items()}}


def results_for(systems: dict[str, pd.Series], sample: pd.DataFrame, stratum: pd.Series, refs,
                seed: int = SEED) -> dict:
    """AUC and Spearman per system, and paired AUC deltas against each of `refs`."""
    auc = {n: per_jd_auc(sample, s) for n, s in systems.items()}
    rho = {n: per_jd_spearman(sample, s) for n, s in systems.items()}
    results = {}
    for n in systems:
        results[n] = {"auc_top10": block(auc[n], stratum, seed), "spearman": block(rho[n], stratum, seed)}
        for ref in refs:
            if n != ref:
                results[n][f"auc_delta_vs_{ref}"] = block(auc[n] - auc[ref], stratum, seed)
    return results


def _jds(sample: pd.DataFrame) -> pd.DataFrame:
    from candidate_screener.matching import corpus
    a2 = corpus.a2_pairs(text_variant=TEXT_VARIANT)
    title_of = dict(zip(a2.jd_id, a2.jd_position))
    jds = sample.drop_duplicates("jd_id")[["jd_id", "jd_text"]].assign(jd_title=lambda d: d.jd_id.map(title_of))
    if jds.jd_title.isna().any():
        raise ValueError("a sampled JD has no title")
    return jds


def pilot(seed: int = SEED) -> dict:
    from candidate_screener.rerank import embedder
    t_start = time.time()
    sample, stratum, x2, l2 = _inputs(seed)
    jds = _jds(sample)
    model = embedder.load()
    tok = model.tokenizer
    cv_tokens = int(sum(len(tok(t)["input_ids"]) for t in sample.drop_duplicates("cv_id").cv_text))
    t0 = time.time()
    cv_vecs = embedder.cv_vectors(sample[["cv_id", "cv_text"]], model)
    encode_s = time.time() - t0
    systems = {"X2": x2, "L2 (F9)": l2}
    for v in embedder.INSTRUCTIONS:
        _log(f"JD side, {v}")
        e = embedder.score(sample, embedder.jd_vectors(jds, v, model), cv_vecs)
        systems[f"E1 {v}"] = e
        systems[f"U2 {v} (E1 + L2)"] = minmax_fuse(sample, e, l2)

    results = results_for(systems, sample, stratum, ("X2", "L2 (F9)"), seed)
    totals = {v: results[f"E1 {v}"]["auc_top10"]["total"]["value"] for v in embedder.INSTRUCTIONS}
    chosen = best_variant(totals)
    go = {"variant": chosen,
          "e1_auc_delta_vs_x2": results[f"E1 {chosen}"]["auc_delta_vs_X2"]["total"]["value"],
          "u2_auc_delta_vs_l2": results[f"U2 {chosen} (E1 + L2)"]["auc_delta_vs_L2 (F9)"]["total"]["value"]}
    go["proposal_met"] = bool(go["e1_auc_delta_vs_x2"] > 0 or go["u2_auc_delta_vs_l2"] > 0)
    return {"plan": "plan/2026-10-18-classifier-then-rerank (step 5a pilot)", "status": "exploratory",
            "seed": seed, "config": embedder.CONFIG,
            "instructions": {v: {"task": t, "jd_title_first": w} for v, (t, w) in embedder.INSTRUCTIONS.items()},
            "sample": _sample_doc(sample, stratum),
            "cost": {"cv_tokens": cv_tokens, "encode_seconds": round(encode_s, 1),
                     "tokens_per_second": round(cv_tokens / encode_s, 1) if encode_s > 1 else None,
                     "total_seconds": round(time.time() - t_start, 1)},
            "metric": "per-JD AUC of the score for the chat's tie-aware top 10 vs the sampled others",
            "systems": results, "full_run_rule": go}


def cross_pilot(seed: int = SEED, write=None) -> dict:
    """C1-C3 (`cross.SYSTEMS`) on the embedder pilot's pairs, beside X2, L2 and E1 I0.

    `write(doc)` is called after each cross-encoder, so a long run leaves results as it goes."""
    from candidate_screener.rerank import cross, embedder
    sample, stratum, x2, l2 = _inputs(seed)
    model = embedder.load()
    cv_vecs = embedder.cv_vectors(sample[["cv_id", "cv_text"]], model)   # cached by the embedder pilot
    e1 = embedder.score(sample, embedder.jd_vectors(_jds(sample), "I0", model), cv_vecs)
    del model
    systems = {"X2": x2, "L2 (F9)": l2, "E1 I0": e1}
    refs = ("X2", "L2 (F9)", "E1 I0")
    seconds, doc = {}, {}
    for name in cross.SYSTEMS:
        t0 = time.time()
        c = cross.score_pairs(sample, name)
        seconds[name] = round(time.time() - t0, 1)
        systems[name] = c
        systems[f"U2 {name.split()[0]} (+ L2)"] = minmax_fuse(sample, c, l2)
        doc = {"plan": "plan/2026-10-18-classifier-then-rerank (step 5a cross-encoder pilot)",
               "status": "exploratory", "seed": seed,
               "configs": {n: cross.config(n) for n in cross.SYSTEMS}, "sample": _sample_doc(sample, stratum),
               "seconds": seconds, "done": list(seconds),
               "metric": "per-JD AUC of the score for the chat's tie-aware top 10 vs the sampled others",
               "systems": results_for(systems, sample, stratum, refs, seed)}
        if write is not None:
            write(doc)
    return doc


def coverage_doc(seed: int = SEED) -> dict:
    """How much of the chat's top 10 L2's (and X2's) top k holds, all 60 JDs: the ceiling of a
    cascade that reranks only that top k. Uses no cross-encoder score."""
    from candidate_screener.rerank import chat_targets as ct, chunked
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    x2, l2 = _x2_and_l2(chat_all, seed)
    out = {}
    for name, s in (("L2 (F9)", l2), ("X2", x2)):
        out[name] = {str(k): block(coverage(chat, s, k), stratum, seed) for k in COVERAGE_K}
    return {"plan": "plan/2026-10-18-classifier-then-rerank (step 5a, cascade cost)", "status": "exploratory",
            "pairs": int(len(chat)), "jds": int(chat.jd_id.nunique()), "k": list(COVERAGE_K),
            "metric": "share of the chat's tie-aware top 10 inside the system's top k (ties by pair id)",
            "coverage": out}


def _human(scores: dict[str, pd.Series], chat: pd.DataFrame, refs, seed: int = SEED) -> dict:
    """Exploratory: judged CVs in X2's top 100; per-JD AUC on the final label and pairwise on
    yc's grades, `full` and `llm_free`, paired against each of `refs` (as `rerank_chat`)."""
    from candidate_screener.evaluation import final_set, indomain, ordering
    from candidate_screener.evaluation.rerank_chat import HUMAN_REF
    from candidate_screener.matching import corpus
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = indomain.labels(frame)
    judged = lab.index.intersection(pd.Index(chat.pair_id))
    lab_j = lab.loc[judged]
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).assign(keyword_match=lambda d: corpus.keyword_match(d))
    grades = ordering.grade_frame(frame).loc[judged]
    out, unscoreable = {}, []
    with indomain.auc_memo():
        for name, s in scores.items():
            s = s.loc[judged]
            block_ = {"auc": [f.as_dict() for f in indomain.rank_figures(name, s, lab_j, pairs, rule_version,
                                                                         seed, unscoreable)],
                      "pairwise_" + HUMAN_REF: {sub: {scope: ordering.summarise(ordering.per_jd(
                          ordering.restrict(grades, scope, sub), HUMAN_REF, s, "pairwise")[0], seed)
                          for scope in final_set.SCOPES} for sub in ("full", "llm_free")}}
            for ref in refs:
                if ref == name:
                    continue
                a = scores[ref].loc[judged]
                block_[f"auc_delta_vs_{ref}"] = [indomain.paired_delta(s, a, lab_j, pairs, "auc", scope, sub, seed)
                                                 for sub in indomain.SUBSETS for scope in final_set.SCOPES]
                block_[f"pairwise_delta_vs_{ref}"] = [ordering.paired_delta(s, a, grades, HUMAN_REF, "pairwise",
                                                                            scope, sub, seed)
                                                      for sub in ("full", "llm_free") for scope in final_set.SCOPES]
            out[name] = block_
    return {"reference": HUMAN_REF, "judged_pairs": int(len(judged)), "systems": out,
            "unscoreable": unscoreable, "status": "exploratory"}


def cascade(seed: int = SEED, write=None) -> dict:
    """Step 5b: L2's top 30 per JD rescored by C2, fused with L2 (the decision arm) or alone."""
    from candidate_screener.evaluation import rerank_chat as rc
    from candidate_screener.rerank import chat_targets as ct, chunked, cross
    t_start = time.time()
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    fold_of = chat.drop_duplicates("jd_id").set_index("jd_id").fold
    _log("X2 and L2 (F9) scores")
    x2, l2 = _x2_and_l2(chat_all, seed)
    top = pd.concat([g[ct.top_k_mask(g.pair_id.to_numpy(), l2.loc[g.pair_id.to_numpy()].to_numpy(dtype=float),
                                     CASCADE_K)] for _, g in chat.sort_values("pair_id").groupby("jd_id", sort=True)])
    _log(f"stage 2b: {len(top)} pairs ({CASCADE_SYSTEM})")
    t0 = time.time()
    c = cross.score_pairs(top, CASCADE_SYSTEM)
    score_s = time.time() - t0
    fused, alone = cascade_scores(chat, l2, c)
    systems = {"X2": x2, "L2 (F9)": l2, FUSED: fused, C_ONLY: alone}
    per = {n: ct.per_jd(chat, s) for n, s in systems.items()}
    held = fold_of.index[fold_of.isin(HELD_OUT_FOLDS)]

    def scored(jds) -> dict:
        res = {}
        for n in systems:
            p = per[n].loc[per[n].index.intersection(jds)]
            res[n] = {"summary": rc.summary(p, stratum, seed)}
            for ref in ("L2 (F9)", "X2"):
                if n != ref:
                    res[n][f"delta_vs_{ref}"] = rc.deltas(p, per[ref].loc[p.index], stratum, seed)
        return res

    held_res, all_res = scored(held), scored(fold_of.index)
    d = held_res[FUSED]["delta_vs_L2 (F9)"]["recall"]["total"]
    doc = {"plan": "plan/2026-10-18-classifier-then-rerank (step 5b cascade)", "status": "exploratory",
           "seed": seed, "k": CASCADE_K, "stage_2b": cross.config(CASCADE_SYSTEM),
           "pairs": {"chat": int(len(chat)), "rescored": int(len(top)), "jds": int(chat.jd_id.nunique()),
                     "held_out_jds": int(len(held))},
           "decision": {"rule": "fused Recall@10 delta vs L2 F9 (total, folds 1-4) > 0 with the CI above 0",
                        "delta": d, "adopted": bool(d["value"] > 0 and d["ci"][0] > 0)},
           "held_out_folds_1_4": held_res, "all_60": all_res,
           "cost": {"stage_2b_seconds": round(score_s, 1), "total_seconds": None}}
    if write is not None:
        write(doc)
    _log("human check (exploratory)")
    doc["human_check"] = _human(systems, chat, ("L2 (F9)", "X2"), seed)
    doc["cost"]["total_seconds"] = round(time.time() - t_start, 1)
    if write is not None:
        write(doc)
    return doc


def s1_zero_shot(seed: int = SEED, write=None) -> dict:
    """Step 5c: S1's t = 0.3 list capped at 30, reordered by zero-shot C2 (alone or + S1); nothing
    in the pipeline is trained on `chat-1`, which only scores it (`ct.load_latest()`)."""
    from candidate_screener.evaluation import rerank_chat as rc, retrieval_pilot, stage1, \
        stage1_classifier as sc, stage1_threshold as st
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, cross
    t_start = time.time()
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    x2 = sc._x2().score(pools)
    if rc._sha(x2) != json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"][stage1.X2]["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the stage-1 record")
    feats = learned.Features()
    s1 = sc.fit_arms(feats)[3][sc.PRIMARY]
    Xp = feats(pools)
    p = s1.proba(Xp).loc[pools.pair_id.to_numpy()]
    n_full = pd.Series((p >= S1_T).to_numpy()).groupby(pools.jd_id.to_numpy()).sum()
    if sc.sizes(n_full)["median"] != json.loads(st.RECORD.read_text(encoding="utf-8"))["arms"][f"t = {S1_T}"]["returned"]["median"]:
        raise ValueError("S1's t = 0.3 list sizes differ from step 4a's record")
    members = capped_list(pools, p)

    ref = ct.load_latest()
    ids = ref.pair_id.to_numpy()
    stratum = ref.drop_duplicates("jd_id").set_index("jd_id").stratum
    fold_of = ct.folds(stratum)
    missing = members - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} listed CVs have no chat score")
    from candidate_screener.rerank import chunked
    _, l2 = _x2_and_l2(chunked.chat_frame(), seed)
    x2_top = set(pools.pair_id[sc.top_k(pools, x2, sc.TOP)])

    listed = pools[pools.pair_id.isin(members)]
    _log(f"stage 2: {len(listed)} pairs (C2)")
    t0 = time.time()
    c2 = cross.score_pairs(listed[["pair_id", "jd_text", "cv_text"]], CASCADE_SYSTEM)
    score_s = time.time() - t0
    fused, _ = cascade_scores(listed, p, c2, k=S1_CAP)          # the whole list is the "top k"
    low = -1e9

    def on_ref(s: pd.Series) -> pd.Series:
        return pd.Series(s.reindex(ids).fillna(low).to_numpy(), index=ids)
    in_list = pd.Series(ref.pair_id.isin(members).to_numpy(), index=ids)
    in_x2 = pd.Series(ref.pair_id.isin(x2_top).to_numpy(), index=ids)
    arms = {S1_ALONE: (on_ref(p.loc[list(members)]), in_list), S1_C2: (on_ref(c2), in_list),
            S1_FUSED: (on_ref(fused), in_list), X2_L2: (on_ref(l2), in_x2), X2_ALONE: (on_ref(x2), None)}
    per = {n: st.arm_per_jd(ref, s, r) for n, (s, r) in arms.items()}
    held = fold_of.index[fold_of.isin(HELD_OUT_FOLDS)]

    def scored(jds) -> dict:
        res = {}
        for n in arms:
            q = per[n].loc[per[n].index.intersection(jds)]
            res[n] = {"summary": rc.summary(q, stratum, seed)}
            if "ceiling" in q:
                res[n]["ceiling"] = block(q.ceiling, stratum, seed)
            for ref_name in (S1_ALONE, X2_L2, X2_ALONE):
                if n != ref_name:
                    res[n][f"delta_vs_{ref_name}"] = rc.deltas(q, per[ref_name].loc[q.index], stratum, seed)
        return res

    sizes = pd.Series(in_list.to_numpy()).groupby(ref.jd_id.to_numpy()).sum()
    d = scored(held)[S1_FUSED][f"delta_vs_{S1_ALONE}"]["recall"]["total"]
    doc = {"plan": "plan/2026-10-18-classifier-then-rerank (step 5c)", "status": "exploratory", "seed": seed,
           "stage_1": {"model": "S1, logistic on F12, fitted on D50", "t": S1_T, "cap": S1_CAP,
                       "list_sizes": sc.sizes(sizes)},
           "stage_2": cross.config(CASCADE_SYSTEM),
           "reference": {"runs": list(ct.REFERENCE_RUNS), "pairs": int(len(ref)),
                         "listed_from_chat_2": int(ref[ref.pair_id.isin(members)].run.eq("chat-2").sum()),
                         "chat_2_offset": json.loads(st.RECORD.read_text(encoding="utf-8"))["drift"]["offset"]},
           "decision": {"rule": f"{S1_FUSED} Recall@10 delta vs {S1_ALONE} (total, folds 1-4) > 0 with the CI above 0",
                        "delta": d, "adopted": bool(d["value"] > 0 and d["ci"][0] > 0)},
           "held_out_folds_1_4": scored(held), "all_60": scored(fold_of.index),
           "cost": {"stage_2_pairs": int(len(listed)), "stage_2_seconds": round(score_s, 1)}}
    if write is not None:
        write(doc)
    _log("human check (exploratory)")
    frame = ref[ref.pair_id.isin(members)]
    doc["human_check"] = _human({n: arms[n][0] for n in (S1_ALONE, S1_C2, S1_FUSED)}, frame, (S1_ALONE,), seed)
    doc["cost"]["total_seconds"] = round(time.time() - t_start, 1)
    if write is not None:
        write(doc)
    return doc


def full_c2(seed: int = SEED, write=None) -> dict:
    """Step 5d: C2 on every pair of X2's top 100 (no cut), alone (replace L2) or + L2 (enhance);
    exploratory, S1's order and C2 + S1 (nothing trained on `chat-1`). Guards run before scoring."""
    from candidate_screener.evaluation import rerank_chat as rc, retrieval_pilot, stage1_classifier as sc, \
        stage1_threshold as st
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, chunked, cross
    t_start = time.time()
    chat_all = chunked.chat_frame()
    chat = chat_all[chat_all.source == ct.TRAIN_SOURCE].reset_index(drop=True)
    if chat.groupby("jd_id").size().max() > FULL_K:
        raise ValueError(f"a JD has more than {FULL_K} chat pairs")
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    fold_of = chat.drop_duplicates("jd_id").set_index("jd_id").fold
    _log("X2 and L2 (F9) scores")
    x2, l2 = _x2_and_l2(chat_all, seed)
    _log("S1 probabilities")
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    feats = learned.Features()
    p = sc.fit_arms(feats)[3][sc.PRIMARY].proba(feats(pools)).loc[pools.pair_id.to_numpy()]
    n_full = pd.Series((p >= S1_T).to_numpy()).groupby(pools.jd_id.to_numpy()).sum()
    if sc.sizes(n_full)["median"] != json.loads(st.RECORD.read_text(encoding="utf-8"))["arms"][f"t = {S1_T}"]["returned"]["median"]:
        raise ValueError("S1's t = 0.3 list sizes differ from step 4a's record")

    _log(f"stage 2: {len(chat)} pairs ({CASCADE_SYSTEM})")
    t0 = time.time()
    c2 = cross.score_pairs(chat, CASCADE_SYSTEM)
    score_s = time.time() - t0
    with_l2, _ = cascade_scores(chat, l2, c2, k=FULL_K)     # the whole list is the "top k"
    with_s1, _ = cascade_scores(chat, p, c2, k=FULL_K)
    systems = {"X2": x2, "L2 (F9)": l2, FULL_C2: c2, FULL_C2_L2: with_l2, FULL_S1: p, FULL_C2_S1: with_s1}
    refs = ("L2 (F9)", "X2", FULL_S1)
    per = {n: ct.per_jd(chat, s) for n, s in systems.items()}
    held = fold_of.index[fold_of.isin(HELD_OUT_FOLDS)]

    def scored(jds) -> dict:
        res = {}
        for n in systems:
            q = per[n].loc[per[n].index.intersection(jds)]
            res[n] = {"summary": rc.summary(q, stratum, seed)}
            for ref in refs:
                if n != ref:
                    res[n][f"delta_vs_{ref}"] = rc.deltas(q, per[ref].loc[q.index], stratum, seed)
        return res

    held_res = scored(held)
    decisions = {}
    for arm in (FULL_C2, FULL_C2_L2):
        d = held_res[arm]["delta_vs_L2 (F9)"]["recall"]["total"]
        decisions[arm] = {"rule": f"{arm} Recall@10 delta vs L2 F9 (total, folds 1-4) > 0 with the CI above 0",
                          "delta": d, "adopted": bool(d["value"] > 0 and d["ci"][0] > 0)}
    doc = {"plan": "plan/2026-10-18-classifier-then-rerank (step 5d)", "status": "exploratory", "seed": seed,
           "stage_2": cross.config(CASCADE_SYSTEM),
           "pairs": {"chat": int(len(chat)), "scored": int(len(c2)), "jds": int(chat.jd_id.nunique()),
                     "held_out_jds": int(len(held))},
           "decisions": decisions, "no_rule": [FULL_S1, FULL_C2_S1],
           "held_out_folds_1_4": held_res, "all_60": scored(fold_of.index),
           "cost": {"stage_2_seconds": round(score_s, 1), "total_seconds": None}}
    if write is not None:
        write(doc)
    _log("human check (exploratory)")
    doc["human_check"] = _human(systems, chat, refs, seed)
    doc["cost"]["total_seconds"] = round(time.time() - t_start, 1)
    if write is not None:
        write(doc)
    return doc


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def _print(doc: dict) -> None:
    for n, r in doc["systems"].items():
        t = r["auc_top10"]["total"]
        print(f"{n:30s} AUC {t['value']:.3f} [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}]  "
              f"spearman {r['spearman']['total']['value']:.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the fold-0 sample pilot (E1)")
    ap.add_argument("--pilot-cross", action="store_true", help="C1-C3 on the same sample")
    ap.add_argument("--coverage", action="store_true", help="the chat's top 10 inside L2's / X2's top k")
    ap.add_argument("--cascade", action="store_true", help="step 5b: L2 top 30 -> C2 (+ L2)")
    ap.add_argument("--s1-zero-shot", action="store_true", help="step 5c: S1 list -> C2 (+ S1)")
    ap.add_argument("--full-c2", action="store_true", help="step 5d: C2 on all of X2's top 100")
    args = ap.parse_args(argv)
    if args.full_c2:
        def write(doc):
            FULL_RECORD.write_text(render(doc), encoding="utf-8")
            _log(f"wrote {FULL_RECORD}")
        doc = full_c2(write=write)
        for scope_name in ("held_out_folds_1_4", "all_60"):
            for n, r in doc[scope_name].items():
                t = r["summary"]["recall"]["total"]
                dl = r.get("delta_vs_L2 (F9)", {}).get("recall", {}).get("total")
                print(f"{scope_name:20s} {n:24s} R@10 {t['value']:.3f}"
                      + (f"  vs L2 {dl['value']:+.3f} [{dl['ci'][0]:+.3f}, {dl['ci'][1]:+.3f}]" if dl else ""))
        print("decisions:", doc["decisions"])
        return 0
    if args.s1_zero_shot:
        def write(doc):
            S1_RECORD.write_text(render(doc), encoding="utf-8")
            _log(f"wrote {S1_RECORD}")
        doc = s1_zero_shot(write=write)
        for scope_name in ("held_out_folds_1_4", "all_60"):
            for n, r in doc[scope_name].items():
                t = r["summary"]["recall"]["total"]
                dl = r.get(f"delta_vs_{S1_ALONE}", {}).get("recall", {}).get("total")
                print(f"{scope_name:20s} {n:24s} R@10 {t['value']:.3f}"
                      + (f"  vs S1 alone {dl['value']:+.3f} [{dl['ci'][0]:+.3f}, {dl['ci'][1]:+.3f}]" if dl else ""))
        print("decision:", doc["decision"])
        return 0
    if args.cascade:
        def write(doc):
            CASCADE_RECORD.write_text(render(doc), encoding="utf-8")
            _log(f"wrote {CASCADE_RECORD}")
        doc = cascade(write=write)
        for scope_name in ("held_out_folds_1_4", "all_60"):
            for n, r in doc[scope_name].items():
                t = r["summary"]["recall"]["total"]
                dl = r.get("delta_vs_L2 (F9)", {}).get("recall", {}).get("total")
                print(f"{scope_name:20s} {n:24s} R@10 {t['value']:.3f}"
                      + (f"  vs L2 {dl['value']:+.3f} [{dl['ci'][0]:+.3f}, {dl['ci'][1]:+.3f}]" if dl else ""))
        print("decision:", doc["decision"])
        return 0
    if args.coverage:
        doc = coverage_doc()
        COVERAGE_RECORD.write_text(render(doc), encoding="utf-8")
        _log(f"wrote {COVERAGE_RECORD}")
        for name, by_k in doc["coverage"].items():
            print(name, "  ".join(f"k={k}: {v['total']['value']:.3f} (gen {v['generic']['value']:.3f} / "
                                  f"tar {v['targeted']['value']:.3f})" for k, v in by_k.items()))
        return 0
    if args.pilot_cross:
        def write(doc):
            CROSS_RECORD.parent.mkdir(parents=True, exist_ok=True)
            CROSS_RECORD.write_text(render(doc), encoding="utf-8")
            _log(f"wrote {CROSS_RECORD} ({', '.join(doc['done'])})")
        _print(cross_pilot(write=write))
        return 0
    if not args.pilot:
        ap.error("--pilot or --pilot-cross; no full run is planned yet")
    doc = pilot()
    PILOT_RECORD.parent.mkdir(parents=True, exist_ok=True)
    PILOT_RECORD.write_text(render(doc), encoding="utf-8")
    _log(f"wrote {PILOT_RECORD}")
    _print(doc)
    print("full-run rule:", doc["full_run_rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
