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
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT

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


def score(chat=None, l1=None) -> pd.Series:
    from candidate_screener.evaluation import rerank_embed as re_
    from candidate_screener.rerank import cross
    if chat is None:
        chat, l1 = oof_l1()
    pairs = c2_pairs(chat, l1)
    _log(f"C2 on {len(pairs)} chat-3 pairs ({pairs.jd_id.nunique()} JDs x top {max(KS)})")
    return cross.score_pairs(pairs[["pair_id", "jd_text", "cv_text"]], re_.CASCADE_SYSTEM)


def select(seed: int = SEED) -> dict:
    from candidate_screener.evaluation import rerank_chat as rc, rerank_embed as re_
    from candidate_screener.rerank import chat_targets as ct, cross
    chat, l1 = oof_l1()
    pairs = c2_pairs(chat, l1)
    c2 = cross.score_pairs(pairs[["pair_id", "jd_text", "cv_text"]], re_.CASCADE_SYSTEM, cached_only=True)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", action="store_true", help="step B: C2 on chat-3's out-of-fold top-50 lists")
    ap.add_argument("--select", action="store_true", help="the menu out of fold on chat-3, the winner, the gate")
    args = ap.parse_args(argv)
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
    ap.error("--score or --select")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
