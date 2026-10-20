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

Ids and digests go to the record; texts stay in `data/` (git-ignored). Nothing here reads a label.

    uv run python -m candidate_screener.evaluation.chat3 --build
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


def main(argv: list[str] | None = None) -> int:
    from candidate_screener.rerank import chat_rerank as cr
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="draw the 48 JDs, pools, X2 top 100, the chat-3 plan")
    args = ap.parse_args(argv)
    if not args.build:
        ap.error("--build")
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
