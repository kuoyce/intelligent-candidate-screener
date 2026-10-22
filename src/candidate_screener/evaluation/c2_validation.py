"""C2 validation (plan 2026-10-22-c2-validation): stage 2's top k -> C2, chosen on chat-3.

**Menu** (fixed in the plan before scoring): within L1's top k of a JD, `mm(L1) + w * mm(C2)`,
each min-max scaled within those k; below the top k, L1's order. k in `KS`, w in `WEIGHTS`, beside
L1 alone. L1 is D63's model (F12, the chat-3 plan's §4 winner).

**Selection on chat-3 only:** L1 out of fold (5 JD folds), C2 on each JD's out-of-fold top 50,
Recall@10 against chat-3's tie-aware top 10. The winner has the highest total (3 dp); ties go to
the smaller k, then the smaller w. **Gate:** its Δ vs L1 alone must be > 0, or C2 is retired.

Nothing here reads a human label or `chat-1`.

    uv run python -m candidate_screener.evaluation.c2_validation --score    # step B (~3-5 h)
    uv run python -m candidate_screener.evaluation.c2_validation --select   # minutes
    uv run python -m candidate_screener.evaluation.c2_validation --check --warm  # minutes, cached C2
    uv run python -m candidate_screener.evaluation.c2_validation --check    # cold: re-scores every C2 pair
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED

SELECT_RECORD = OUTPUT / "rerank" / "c2-validation-select.json"
KS = (20, 30, 50)
WEIGHTS = (0.25, 0.5, 1.0)
SEED = 0
L1_ALONE = "L1 alone"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def option(k: int, w: float) -> str:
    return f"L1 top {k} -> L1 + {w} C2"


# --- pure ----------------------------------------------------------------------------------

def _minmax(v: np.ndarray) -> np.ndarray:
    span = v.max() - v.min()
    return (v - v.min()) / (span if span > 0 else 1.0)


def fused_order(frame: pd.DataFrame, first: pd.Series, second: pd.Series, k: int, w: float) -> pd.Series:
    """Per JD of `frame`: `first`'s top `k` (ties by pair id) ordered by `mm(first) + w * mm(second)`
    within the k, above the rest in `first`'s order. `second` needs only the top-k pairs."""
    from candidate_screener.rerank.chat_targets import top_k_mask
    out = pd.Series(np.nan, index=frame.pair_id.to_numpy())
    for _, g in frame.sort_values("pair_id").groupby("jd_id", sort=True):
        ids = g.pair_id.to_numpy()
        f = first.loc[ids].to_numpy(dtype=float)
        top = top_k_mask(ids, f, k)
        out.loc[ids[top]] = 10 + _minmax(f[top]) + w * _minmax(second.loc[ids[top]].to_numpy(dtype=float))
        out.loc[ids[~top]] = _minmax(f)[~top]
    return out


def choose(totals: dict[tuple[int, float], float], ks=KS, weights=WEIGHTS) -> tuple[int, float]:
    """The option with the highest total (3 dp); a tie goes to the smaller k, then the smaller w."""
    order = [(k, w) for k in ks for w in weights]
    return max(order, key=lambda o: (round(totals[o], 3), -order.index(o)))


def top_lists(frame: pd.DataFrame, scores: pd.Series, k: int) -> pd.DataFrame:
    """`frame`'s rows in each JD's top `k` by `scores` (ties by pair id)."""
    from candidate_screener.rerank.chat_targets import top_k_mask
    return pd.concat([g[top_k_mask(g.pair_id.to_numpy(), scores.loc[g.pair_id.to_numpy()].to_numpy(dtype=float), k)]
                      for _, g in frame.sort_values("pair_id").groupby("jd_id", sort=True)])


# --- chat-3 ---------------------------------------------------------------------------------

def oof_l1() -> tuple[pd.DataFrame, pd.Series]:
    """chat-3 (complete, with folds) and L1's out-of-fold scores on it, F12, as the chat-3 plan's §4."""
    from candidate_screener.evaluation import chat3
    from candidate_screener.matching import learned
    from candidate_screener.rerank import learners
    win = json.loads(chat3.SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    if (win["learner"], win["features"]) != ("L1", "F12"):
        raise ValueError(f"D63's model is L1 F12, the chat-3 record says {win}")
    chat = chat3.targets()
    X = learned.Features()(chat3.pairs()).loc[chat.pair_id.to_numpy()]
    s, _ = learners.out_of_fold(learners.L1, X, chat, list(learned.F12))
    return chat, s


def c2_pairs(chat: pd.DataFrame, l1: pd.Series) -> pd.DataFrame:
    """Each JD's out-of-fold top `max(KS)` with its texts: the pairs C2 scores."""
    from candidate_screener.evaluation import chat3
    top = top_lists(chat, l1, max(KS))
    text = chat3.pairs().set_index("pair_id")
    return top.assign(jd_text=top.pair_id.map(text.jd_text), cv_text=top.pair_id.map(text.cv_text))


def score(chat=None, l1=None, root: Path | None = None) -> pd.Series:
    from candidate_screener.evaluation import rerank_embed as re_
    from candidate_screener.rerank import cross
    if chat is None:
        chat, l1 = oof_l1()
    pairs = c2_pairs(chat, l1)
    _log(f"C2 on {len(pairs)} chat-3 pairs ({pairs.jd_id.nunique()} JDs x top {max(KS)})")
    return cross.score_pairs(pairs[["pair_id", "jd_text", "cv_text"]], re_.CASCADE_SYSTEM, root or cross.ROOT)


def select(seed: int = SEED, root: Path | None = None) -> dict:
    from candidate_screener.evaluation import rerank_chat as rc, rerank_embed as re_
    from candidate_screener.rerank import chat_targets as ct, cross
    chat, l1 = oof_l1()
    pairs = c2_pairs(chat, l1)
    c2 = cross.score_pairs(pairs[["pair_id", "jd_text", "cv_text"]], re_.CASCADE_SYSTEM, root or cross.ROOT,
                           cached_only=True)
    stratum = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    per = {L1_ALONE: ct.per_jd(chat, l1)}
    for k in KS:
        for w in WEIGHTS:
            per[option(k, w)] = ct.per_jd(chat, fused_order(chat, l1, c2, k, w))
    res = {n: {"summary": rc.summary(p, stratum, seed),
               **({} if n == L1_ALONE else {"delta_vs_L1": rc.deltas(p, per[L1_ALONE], stratum, seed)})}
           for n, p in per.items()}
    totals = {(k, w): res[option(k, w)]["summary"]["recall"]["total"]["value"] for k in KS for w in WEIGHTS}
    k, w = choose(totals)
    d = res[option(k, w)]["delta_vs_L1"]["recall"]["total"]
    return {"plan": "plan/2026-10-22-c2-validation (selection on chat-3)", "seed": seed,
            "menu": {"k": list(KS), "w": list(WEIGHTS)}, "c2_pairs": int(len(pairs)),
            "rule": "highest out-of-fold Recall@10 on chat-3 (total, 3 dp); ties to the smaller k, then w",
            "options": res, "winner": {"k": k, "w": w, "name": option(k, w)},
            "gate": {"rule": "the winner's Recall@10 delta vs L1 alone on chat-3 > 0", "delta": d,
                     "passes": bool(d["value"] > 0)}}


POOL_RECORD = OUTPUT / "rerank" / "c2-validation-pool.json"
#: The options compared on the test, whose top 10s are pooled for chat-4 (owner, decision 2).
POOLED = ("winner", L1_ALONE, "X2")


def compared_scores(seed: int = SEED, root: Path | None = None, cached_only: bool = False) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """The 60 final-set JDs' pools and the compared options' scores over all 500 CVs: L1 fitted once
    on all of chat-3 (checked against §5's 0.484413 of the chat-3 plan), the winner on its top k
    (C2 scored where not cached), X2. Reads `chat-1` only for that check."""
    from candidate_screener.evaluation import chat3, rerank_embed as re_, retrieval_pilot
    from candidate_screener.matching import learned
    from candidate_screener.rerank import chat_targets as ct, chunked, cross
    win = json.loads(SELECT_RECORD.read_text(encoding="utf-8"))["winner"]
    feats = learned.Features()
    model = chat3.fit_stage2(chat3.targets(), feats(chat3.pairs()), "L1", "F12")
    pools = retrieval_pilot.build_pools(seed, re_.TEXT_VARIANT)
    ids = pools.pair_id.to_numpy()
    x2 = chat3._x2(pools)
    l1 = pd.Series(model.predict(feats(pools).loc[ids]), index=ids)
    chat = chunked.chat_frame()
    tested = json.loads(chat3.TEST_RECORD.read_text(encoding="utf-8"))["results"][chat3.STAGE2]
    got = round(float(chat3.pool_coverage(pools, chat[chat.source == ct.TRAIN_SOURCE], l1, 10).coverage.mean()), 6)
    if got != tested["summary"]["recall"]["total"]["value"]:
        raise ValueError(f"L1 over the pools gives {got}, not the chat-3 test's figure")
    top = top_lists(pools, l1, win["k"])
    _log(f"C2 on the test's top {win['k']} lists ({len(top)} pairs; scored where not cached)")
    c2 = cross.score_pairs(top[["pair_id", "jd_text", "cv_text"]], re_.CASCADE_SYSTEM, root or cross.ROOT,
                           cached_only=cached_only)
    scores = {"winner": fused_order(pools, l1, c2, win["k"], win["w"]), L1_ALONE: l1, "X2": x2}
    return pools, scores, win


def pool(seed: int = SEED, root: Path | None = None, cached_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Decision 2: pool the compared options' top 10s; the CVs without a `chat-1` / `chat-2` score
    become `chat-4`, a fresh chat per JD with anchors. Computes no metric of the options."""
    import hashlib
    from candidate_screener.rerank import chat_rerank as cr, chat_targets as ct
    pools, scores, win = compared_scores(seed, root, cached_only)
    scored = ct.load_latest()
    have = set(scored.pair_id)
    tops = {n: top_lists(pools, scores[n], 10) for n in POOLED}
    pooled = pd.concat(tops.values()).drop_duplicates("pair_id")
    new = pooled[~pooled.pair_id.isin(have)]
    order = pd.read_csv(cr.PLAN).drop_duplicates("jd_id").set_index("jd_id").jd_order
    stratum = pools.drop_duplicates("jd_id").set_index("jd_id").stratum
    plan = cr.fresh_plan(scored[["pair_id", "jd_id", "cv_id", "score"]], new, cr.CHAT4_RUN, cr.CHAT4_PREFIX,
                         order, stratum, seed) if len(new) else pd.DataFrame()
    per_jd = (plan.groupby("jd_id").agg(new=("source", lambda s: int((s == "pooled").sum())),
                                        anchors=("source", lambda s: int((s == "anchor").sum())),
                                        batches=("batch", "nunique")) if len(plan) else pd.DataFrame())
    sha = lambda ids: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    record = {"plan": "plan/2026-10-22-c2-validation (test lists and chat-4; no metric computed)",
              "seed": seed, "winner": win, "pooled_options": list(POOLED),
              "unscored_in_top10": {n: int((~t.pair_id.isin(have)).sum()) for n, t in tops.items()},
              "chat_4": {"new_pairs": int(len(new)), "jds": int(new.jd_id.nunique()),
                         "anchors": int((plan.source == "anchor").sum()) if len(plan) else 0,
                         "batches": int(per_jd.batches.sum()) if len(plan) else 0,
                         "per_jd": {str(j): {k: int(v) for k, v in r.items()} for j, r in per_jd.iterrows()},
                         "new_pair_ids_sha256": sha(new.pair_id), "plan_ids_sha256": sha(plan.pair_id) if len(plan) else None},
              "instrument_sha256": cr.instrument_sha256(), "run": cr.CHAT4_RUN}
    return plan, record


TEST_RECORD = OUTPUT / "rerank" / "c2-validation-test.json"
BAR = 0.02


def test(seed: int = SEED, root: Path | None = None, cached_only: bool = False) -> dict:
    """The exploratory test (owner's decision 1): the winner vs L1 alone (and X2's order) on the 60
    final-set JDs. Reference A: `chat-1` + `chat-2` (+ `chat-4`, empty: the pooled top 10s were all
    scored), the latest score per pair; reference B: `chat-1` alone, unscored CVs as misses."""
    from candidate_screener.evaluation import chat3, rerank_chat as rc, rerank_embed as re_
    from candidate_screener.rerank import chat_targets as ct, chunked
    if json.loads(POOL_RECORD.read_text(encoding="utf-8"))["chat_4"]["new_pairs"]:
        raise NotImplementedError("chat-4 has pairs: merge its replies into reference A first")
    pools, scores, win = compared_scores(seed, root, cached_only)
    ref = ct.load_latest()
    stratum = ref.drop_duplicates("jd_id").set_index("jd_id").stratum
    chat = chunked.chat_frame()
    chat1 = chat[chat.source == ct.TRAIN_SOURCE]
    per, by_pool = {}, {}
    for n in POOLED:
        per[n] = ct.per_jd(ref, scores[n])
        by_pool[n] = chat3.pool_coverage(pools, ref, scores[n], 10)
        if not np.allclose(per[n].recall.loc[by_pool[n].index], by_pool[n].coverage):
            raise ValueError(f"{n}: a top-10 CV lies outside reference A")
    res = {}
    for n in POOLED:
        res[n] = {"reference_A": {"summary": rc.summary(per[n], stratum, seed)},
                  "reference_B_recall": re_.block(chat3.pool_coverage(pools, chat1, scores[n], 10).coverage,
                                                  stratum, seed)}
        if n != L1_ALONE:
            res[n]["reference_A"]["delta_vs_L1"] = rc.deltas(per[n], per[L1_ALONE], stratum, seed)
    b = {n: chat3.pool_coverage(pools, chat1, scores[n], 10).coverage for n in POOLED}
    res["winner"]["reference_B_delta_vs_L1"] = re_.block(b["winner"] - b[L1_ALONE], stratum, seed)
    d = res["winner"]["reference_A"]["delta_vs_L1"]["recall"]["total"]
    _log("human check (exploratory)")
    return {"plan": "plan/2026-10-22-c2-validation (test; exploratory: the 60 JDs had the chat-3 plan's §5)",
            "status": "exploratory", "seed": seed, "winner": win,
            "reference_A": "chat-1 + chat-2, latest score per pair (chat-4 empty)",
            "reference_B": "chat-1 alone, unscored CVs as misses",
            "bar": {"rule": f"winner Recall@10 delta vs L1 alone (total, reference A) >= +{BAR}, CI above 0",
                    "delta": d, "met": bool(d["value"] >= BAR and d["ci"][0] > 0)},
            "results": res,
            "human": re_._human({n: scores[n] for n in POOLED}, chat1, (L1_ALONE,), seed)}


def _render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def check(cold: bool = False, seed: int = SEED) -> dict[str, bool]:
    """D55: rebuild the selection, pooling and test records and compare byte for byte with disk;
    writes nothing. Warm uses the C2 cache and refuses to score; cold re-scores every C2 pair into
    a temporary cache (hours), then removes it."""
    from candidate_screener.rerank import chat_rerank as cr
    root = Path(tempfile.mkdtemp(prefix="c2-validation-cold-", dir=PROCESSED / "rerank")) if cold else None
    try:
        if cold:
            score(root=root)
        sel = select(seed, root)
        plan, pooled = pool(seed, root, cached_only=not cold)
        tested = test(seed, root, cached_only=not cold)
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
    return {SELECT_RECORD.name: SELECT_RECORD.read_text(encoding="utf-8") == _render(sel),
            POOL_RECORD.name: POOL_RECORD.read_text(encoding="utf-8") == _render(pooled),
            "chat-4 plan (empty: none on disk)": len(plan) == 0 and not cr.CHAT4_PLAN.exists(),
            TEST_RECORD.name: TEST_RECORD.read_text(encoding="utf-8") == _render(tested)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", action="store_true", help="step B: C2 on chat-3's out-of-fold top-50 lists")
    ap.add_argument("--select", action="store_true", help="the menu out of fold on chat-3, the winner, the gate")
    ap.add_argument("--pool", action="store_true", help="test lists and the chat-4 plan (no metric)")
    ap.add_argument("--test", action="store_true", help="the exploratory test, once")
    ap.add_argument("--check", action="store_true", help="rebuild select / pool / test and diff, write nothing")
    ap.add_argument("--warm", action="store_true", help="with --check: use the cached C2 scores (D55 warm tier)")
    args = ap.parse_args(argv)
    if args.warm and not args.check:
        ap.error("--warm is only valid with --check")
    if args.check:
        same = check(cold=not args.warm)
        for name, ok in same.items():
            print(f"[{'PASS' if ok else 'FAIL'}] rebuild reproduces {name}")
        return 0 if all(same.values()) else 1
    if args.test:
        if TEST_RECORD.exists():
            raise SystemExit(f"{TEST_RECORD} exists: the test is scored once")
        doc = test()
        TEST_RECORD.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for n, r in doc["results"].items():
            print(f"{n:10s} R@10 A {r['reference_A']['summary']['recall']['total']['value']:.3f}  "
                  f"B {r['reference_B_recall']['total']['value']:.3f}")
        print("bar:", doc["bar"])
        return 0
    if args.pool:
        from candidate_screener.rerank import chat_rerank as cr
        if not json.loads(SELECT_RECORD.read_text(encoding="utf-8"))["gate"]["passes"]:
            raise SystemExit("the chat-3 gate failed: C2 is retired, the test JDs are not touched")
        if cr.CHAT4_REPLIES.exists():
            raise SystemExit(f"{cr.CHAT4_REPLIES} exists: chat-4 is being collected, never rebuild its plan")
        plan, record = pool()
        if len(plan):
            plan.to_csv(cr.CHAT4_PLAN, index=False)
        POOL_RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("unscored_in_top10", "chat_4")}, indent=1)[:3000])
        return 0
    if args.score:
        s = score()
        _log(f"C2 scores: {len(s)} pairs, none missing: {bool(s.notna().all())}")
        return 0
    if args.select:
        doc = select()
        SELECT_RECORD.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for n, r in doc["options"].items():
            t = r["summary"]["recall"]["total"]
            d = r.get("delta_vs_L1", {}).get("recall", {}).get("total")
            print(f"{n:28s} R@10 {t['value']:.3f}" + (f"  vs L1 {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d else ""))
        print("winner:", doc["winner"], "gate:", doc["gate"]["passes"])
        return 0
    ap.error("--score, --select, --pool, --test or --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
