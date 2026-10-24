"""`chat-1` as the stage-2 target and reference (plan 2026-10-18-classifier-then-rerank, step 3; D61).

- `load()`: one row per chat-scored pair (the latest reply wins), with the plan's `source`:
  `x2_top100` (the 6,000 training and scoring pairs) or `s1_passed` (the 20-pair extension,
  scored only, for the end-to-end run).
- `folds()`: 5 folds by JD, stratified (8 generic + 4 targeted JDs per fold), seeded.
- Per-JD top-10 metrics against the chat score: tie-aware Recall@10, nDCG@10 with gain = chat
  score, and chat No Fits in a top 10.

A system's top 10 is its 10 highest scores, ties by pair id; it may hold fewer than 10 CVs (a
short stage-1 list). The chat's top 10 is tie-aware: when *t* CVs tie at the 10th chat score
for *r* remaining slots, each counts *r / t*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 10
N_FOLDS = 5
SEED = 0
TRAIN_SOURCE = "x2_top100"


def _plan() -> pd.DataFrame:
    from candidate_screener.rerank import chat_rerank as cr
    plan = pd.read_csv(cr.PLAN)
    return plan.assign(source=plan["source"] if "source" in plan else TRAIN_SOURCE,
                       run=plan["run"].fillna(cr.RUN) if "run" in plan else cr.RUN)


def load(run: str = "chat-1") -> pd.DataFrame:
    """One run's chat-scored pairs: `pair_id`, `jd_id`, `cv_id`, `stratum`, `source`, `score`, `label`.

    The default, `chat-1`, is what step 3 trained and scored on; later runs never change it."""
    from candidate_screener.rerank import chat_rerank as cr
    plan = _plan()
    plan = plan[plan.run == run]
    d = cr.latest()
    d = d[d.run == run]
    if d.instrument_sha256.nunique() != 1:
        raise ValueError("chat replies pooled across instruments")
    out = plan[["pair_id", "jd_id", "cv_id", "stratum", "source"]].merge(
        d[["pair_id", "score", "label"]], on="pair_id", how="inner", validate="one_to_one")
    missing = set(plan.pair_id) - set(out.pair_id)
    if missing:
        raise ValueError(f"{len(missing)} planned pairs have no chat score")
    return out.sort_values("pair_id").reset_index(drop=True)


REFERENCE_RUNS = ("chat-1", "chat-2")


def load_latest(runs=REFERENCE_RUNS) -> pd.DataFrame:
    """Every chat-scored pair once, its **latest** score across `runs` (step 4's reference, owner
    18 Oct 2026; the retest `chat-1r` is not one of them). `source` is the pair's first plan row
    (an anchor keeps `x2_top100`); `run` is the run its score comes from."""
    from candidate_screener.rerank import chat_rerank as cr
    plan = _plan().drop_duplicates("pair_id", keep="first")
    d = cr.latest()
    d = d[d.run.isin(runs)]
    if d.instrument_sha256.nunique() != 1:
        raise ValueError("chat replies pooled across instruments")
    d = d.sort_values("saved_at", kind="stable").drop_duplicates("pair_id", keep="last")
    out = plan[["pair_id", "jd_id", "cv_id", "stratum", "source"]].merge(
        d[["pair_id", "score", "label", "run"]], on="pair_id", how="inner", validate="one_to_one")
    return out.sort_values("pair_id").reset_index(drop=True)


def anchor_pairs(again_run: str = "chat-2") -> pd.DataFrame:
    """Pairs scored in `chat-1` and again in `again_run`: `pair_id`, `jd_id`, `first`, `again` and
    their labels. `chat-2` gives the anchors' drift; `chat-1r` the accidental retest. Decides nothing."""
    from candidate_screener.rerank import chat_rerank as cr
    d = cr.latest()
    first = d[d.run == cr.RUN].set_index("pair_id")
    again = d[d.run == again_run].sort_values("saved_at", kind="stable").drop_duplicates("pair_id", keep="last")
    again = again[again.pair_id.isin(first.index)].set_index("pair_id")
    return pd.DataFrame({"jd_id": again.jd_id, "first": first.score.loc[again.index],
                         "again": again.score, "first_label": first.label.loc[again.index],
                         "again_label": again.label}).reset_index()


def folds(strata: pd.Series, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """JD -> fold (0..n-1). Each stratum is shuffled (seeded, from id order) and dealt round
    robin, so every fold holds the same number of JDs of each stratum when they divide."""
    rng = np.random.default_rng(seed)
    out = {}
    for stratum in sorted(strata.unique()):
        jds = np.array(sorted(strata.index[strata == stratum]))
        for i, j in enumerate(jds[rng.permutation(len(jds))]):
            out[j] = i % n_folds
    return pd.Series(out).loc[sorted(out)]


# --- per-JD top-10 metrics (pure) ----------------------------------------------------------

def _order(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Positions sorted by score descending, ties by id."""
    return np.lexsort((ids, -np.asarray(scores, dtype=float)))


def chat_weights(chat: np.ndarray, k: int = K) -> np.ndarray:
    """Each CV's share of the chat's top k: 1 above the k-th score, r/t on a tie at it."""
    chat = np.asarray(chat, dtype=float)
    if len(chat) <= k:
        return np.ones(len(chat))
    kth = np.sort(chat)[::-1][k - 1]
    above = chat > kth
    tied = chat == kth
    w = above.astype(float)
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def top_k_mask(ids, scores, k: int = K) -> np.ndarray:
    mask = np.zeros(len(ids), dtype=bool)
    mask[_order(np.asarray(ids), scores)[:k]] = True
    return mask


def recall_at_k(ids, chat, system, returned=None, k: int = K) -> float:
    """Share of the chat's top k (tie-aware) inside the system's top k.

    `returned` (bool) limits the system to the CVs it hands over (a stage-1 list); the
    reference is always the chat's top k over every chat-scored CV given."""
    ids, chat = np.asarray(ids), np.asarray(chat, dtype=float)
    w = chat_weights(chat, k)
    keep = np.ones(len(ids), bool) if returned is None else np.asarray(returned, bool)
    s = np.where(keep, np.asarray(system, dtype=float), -np.inf)
    top = top_k_mask(ids, s, k) & keep
    return float(w[top].sum() / min(k, len(ids)))


def ndcg_at_k(ids, chat, system, returned=None, k: int = K) -> float:
    """nDCG@k with gain = chat score; the ideal is the k highest chat scores given."""
    ids, chat = np.asarray(ids), np.asarray(chat, dtype=float)
    keep = np.ones(len(ids), bool) if returned is None else np.asarray(returned, bool)
    s = np.where(keep, np.asarray(system, dtype=float), -np.inf)
    order = [i for i in _order(ids, s)[:k] if keep[i]]
    disc = 1.0 / np.log2(np.arange(k) + 2.0)
    ideal = float((np.sort(chat)[::-1][:k] * disc[:min(k, len(chat))]).sum())
    return float((chat[order] * disc[:len(order)]).sum() / ideal) if ideal > 0 else float("nan")


def no_fits_at_k(ids, labels, system, returned=None, k: int = K) -> int:
    """Chat No Fits among the system's top k."""
    keep = np.ones(len(ids), bool) if returned is None else np.asarray(returned, bool)
    s = np.where(keep, np.asarray(system, dtype=float), -np.inf)
    top = top_k_mask(np.asarray(ids), s, k) & keep
    return int((np.asarray(labels)[top] == "No Fit").sum())


def per_jd(frame: pd.DataFrame, system: pd.Series, returned: pd.Series | None = None,
           k: int = K) -> pd.DataFrame:
    """`recall`, `ndcg`, `no_fits`, `returned` per JD of `frame` (`pair_id`, `jd_id`, `score`,
    `label`); `system` and `returned` are indexed by pair id."""
    rows = {}
    for jd, g in frame.groupby("jd_id", sort=True):
        ids = g.pair_id.to_numpy()
        s = system.loc[ids].to_numpy(dtype=float)
        r = None if returned is None else returned.loc[ids].to_numpy(dtype=bool)
        rows[jd] = {"recall": recall_at_k(ids, g.score, s, r, k),
                    "ndcg": ndcg_at_k(ids, g.score, s, r, k),
                    "no_fits": no_fits_at_k(ids, g.label, s, r, k),
                    "returned": int(min(k, len(ids) if r is None else r.sum()))}
    return pd.DataFrame.from_dict(rows, orient="index")
