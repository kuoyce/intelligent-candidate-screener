"""The shortlist-recall development set (plan 2026-10-07-shortlist-recall, 01 §1, Q1).

Every option of the Y rungs is chosen here, never on the final set (D46). The documents
are A2 **train-region** documents (D19) that no model behind a Y rung was fitted on:

- the P3 role classifier's **validation part** (never in its fit part),
- minus its **calibration draw** per side (its temperature was fitted there),
- minus every document in `weak-pairs.csv` (SBERT weak trained on those),
- and, as a second guard, minus any eval-region id.

**Pools.** 400 JDs with a non-empty `Position` (the JD title, D54), each with a 500-CV
pool built like the pilot's `generic` design: 8 CVs of the JD's `Primary Keyword`, 2 of
another, then 490 random dev CVs. Split by JD into **dev-train** (250; fitting) and
**dev-select** (150; choosing among fixed options).

**Relevance (a proxy, Q-R3).** A CV is relevant to a JD when their oracle role families
are equal (`role_family(Primary Keyword, Position)`, P3's label). It measures agreement with
the metadata, not fit. Labels live in `labels()`, apart from the frame a rung scores.

This module never reads `a2-final-labels.csv` or `judgements.csv` (test).

    uv run python -m candidate_screener.evaluation.dev_pools --build   # cache + report
    uv run python -m candidate_screener.evaluation.dev_pools --check   # rebuild, diff the report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED

REPORT = OUTPUT / "matching" / "shortlist-dev-set.json"
CACHE = PROCESSED / "matching" / "shortlist-dev"
SEED = 0
N_JDS = 400
N_DEV_TRAIN = 250
POOL_SIZE = 500
N_SAME_KEYWORD = 8
N_OTHER_KEYWORD = 2
SPLITS = ("dev-train", "dev-select")
SLOTS = ("same_keyword", "other_keyword", "random")
CONFIG = {"seed": SEED, "n_jds": N_JDS, "n_dev_train": N_DEV_TRAIN, "pool_size": POOL_SIZE,
          "same_keyword": N_SAME_KEYWORD, "other_keyword": N_OTHER_KEYWORD,
          "documents": "P3 validation part - calibration draw - weak-pair documents - eval region",
          "jd_requires": "non-empty Position (jd_title, D54)",
          "relevance": "oracle role family equal (Q-R3)"}


def _missing_title(v: object) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


# --- documents ------------------------------------------------------------------------------

def select_dev_documents(docs: Mapping[str, pd.DataFrame], parts: Mapping[str, Mapping],
                         excluded_ids: Mapping[str, Iterable[str]]) -> dict[str, pd.DataFrame]:
    """Rows of `docs[side]` in the validation part, outside the calibration draw, and
    whose id is not in `excluded_ids[side]` (weak pairs, eval region). Pure.

    `parts[side]` carries positional index arrays `val` and `calib` into `docs[side]`, as
    `role_family.prepare` returns them."""
    out = {}
    for side, frame in docs.items():
        rows = np.setdiff1d(np.asarray(parts[side]["val"]), np.asarray(parts[side]["calib"]))
        keep = frame.iloc[np.sort(rows)]
        keep = keep[~keep["id"].isin(set(excluded_ids.get(side, ())))]
        out[side] = keep.reset_index(drop=True)
    return out


def _eval_ids() -> dict[str, set[str]]:
    from candidate_screener.data.a2_finetune import PARTITION_MANIFEST
    part = pd.read_csv(PARTITION_MANIFEST)
    ev = part[part.region == "eval"]
    return {"jd": set(ev[ev.doc_type == "jd"].doc_id), "cv": set(ev[ev.doc_type == "cv"].doc_id)}


def _weak_ids() -> dict[str, set[str]]:
    from candidate_screener.data import weak_pairs
    wp = weak_pairs.load()
    return {"jd": set(wp.jd_id), "cv": set(wp.cv_id)}


def _prepared():
    """Train-region documents, P3's `prepare` (its split and labels), the role model."""
    from candidate_screener.extraction import role_family as rf
    docs = rf.load_train_documents()
    prep = rf.prepare(docs)
    model = rf.load_role_model()
    fp = rf.config_fingerprint(prep)
    if fp != model.fingerprint:
        raise RuntimeError(f"P3 split fingerprint {fp} != role model {model.fingerprint}: "
                           "the dev set would not be the model's validation part")
    return docs, prep, model


def dev_documents_from(docs, prep) -> dict[str, pd.DataFrame]:
    """The dev documents with their oracle family (label) and, for JDs, the title."""
    labelled = {s: docs[s].assign(family=prep["y"][s].to_numpy()) for s in docs}
    ev, wk = _eval_ids(), _weak_ids()
    dev = select_dev_documents(labelled, prep["parts"],
                               {s: ev[s] | wk[s] for s in labelled})
    for s in dev:
        dev[s] = dev[s][dev[s].family.notna()].reset_index(drop=True)
    dev["jd"] = dev["jd"].assign(title=dev["jd"].Position)
    return dev


# --- pools ----------------------------------------------------------------------------------

def draw_pools(jd: pd.DataFrame, cv: pd.DataFrame, n_jds: int = N_JDS,
               n_dev_train: int = N_DEV_TRAIN, pool_size: int = POOL_SIZE,
               n_same: int = N_SAME_KEYWORD, n_other: int = N_OTHER_KEYWORD,
               seed: int = SEED) -> pd.DataFrame:
    """`(jd_id, cv_id, split, slot)` rows: `n_jds` pools of `pool_size`. Pure.

    A JD is eligible when its title is non-empty and its keyword has at least `n_same` CVs;
    JDs are drawn seeded from the id-sorted eligible set. Each pool: `n_same` CVs of the
    JD's keyword, `n_other` of other keywords, then random CVs (any keyword) up to
    `pool_size`, no CV twice. Reads `Primary Keyword` to build pools only, as the pilot did."""
    cv = cv.sort_values("id").reset_index(drop=True)
    by_kw = {k: g["id"].to_numpy() for k, g in cv.groupby("Primary Keyword", sort=True)}
    ok = jd[~jd.title.map(_missing_title)
            & jd["Primary Keyword"].map(lambda k: len(by_kw.get(k, ())) >= n_same)]
    ok = ok.sort_values("id")
    if len(ok) < n_jds:
        raise ValueError(f"only {len(ok)} eligible dev JDs for {n_jds} pools")
    rng = np.random.default_rng(seed)
    chosen = ok["id"].to_numpy()[np.sort(rng.permutation(len(ok))[:n_jds])]
    split_of = dict(zip(chosen[rng.permutation(n_jds)],
                        ["dev-train"] * n_dev_train + ["dev-select"] * (n_jds - n_dev_train)))
    kw_of = dict(zip(jd["id"], jd["Primary Keyword"]))
    all_ids = cv["id"].to_numpy()
    cv_kw = cv["Primary Keyword"].to_numpy()
    rows = []
    for k, jd_id in enumerate(chosen):
        r = np.random.default_rng([seed, k])
        kw = kw_of[jd_id]
        same = r.choice(by_kw[kw], n_same, replace=False)
        other = r.choice(all_ids[cv_kw != kw], n_other, replace=False)
        taken = set(same) | set(other)
        rest = np.array([i for i in all_ids if i not in taken])
        rand = r.choice(rest, pool_size - n_same - n_other, replace=False)
        for slot, ids in zip(SLOTS, (same, other, rand)):
            rows.extend((jd_id, c, split_of[jd_id], slot) for c in ids)
    return pd.DataFrame(rows, columns=["jd_id", "cv_id", "split", "slot"])


# --- cache and access -----------------------------------------------------------------------

def _key(model_fp: str) -> str:
    from candidate_screener.data import weak_pairs
    blob = json.dumps({"config": CONFIG, "role_model": model_fp,
                       "weak_pairs": hashlib.sha256(weak_pairs.MANIFEST.read_bytes()).hexdigest()},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build(force: bool = False) -> dict[str, pd.DataFrame]:
    """`{"jd", "cv", "pools"}`, cached under `data/processed/matching/shortlist-dev/`."""
    from candidate_screener.extraction import role_family as rf
    latest = rf.CACHE / "LATEST"
    key = _key(latest.read_text(encoding="utf-8").strip()) if latest.exists() else None
    paths = {n: CACHE / f"{n}-{key}.parquet" for n in ("jd", "cv", "pools")}
    if key and not force and all(p.exists() for p in paths.values()):
        return {n: pd.read_parquet(p) for n, p in paths.items()}
    docs, prep, model = _prepared()
    key = _key(model.fingerprint)
    paths = {n: CACHE / f"{n}-{key}.parquet" for n in ("jd", "cv", "pools")}
    dev = dev_documents_from(docs, prep)
    pools = draw_pools(dev["jd"], dev["cv"])
    used = {"jd": set(pools.jd_id), "cv": set(pools.cv_id)}
    out = {"jd": dev["jd"][dev["jd"]["id"].isin(used["jd"])].reset_index(drop=True),
           "cv": dev["cv"][dev["cv"]["id"].isin(used["cv"])].reset_index(drop=True),
           "pools": pools}
    CACHE.mkdir(parents=True, exist_ok=True)
    for n, frame in out.items():
        frame.to_parquet(paths[n], index=False)
    return out


def pairs(dev: Mapping[str, pd.DataFrame], split: str | None = None) -> pd.DataFrame:
    """The frame a rung scores: ids, `jd_title`, `jd_text`, `cv_text`. No label column."""
    p = dev["pools"] if split is None else dev["pools"][dev["pools"].split == split]
    jd, cv = dev["jd"].set_index("id"), dev["cv"].set_index("id")
    return pd.DataFrame({
        "pair_id": (p.jd_id + "__" + p.cv_id).to_numpy(), "jd_id": p.jd_id.to_numpy(),
        "cv_id": p.cv_id.to_numpy(), "split": p.split.to_numpy(),
        "jd_title": p.jd_id.map(jd.title).to_numpy(), "jd_text": p.jd_id.map(jd.text).to_numpy(),
        "cv_text": p.cv_id.map(cv.text).to_numpy()})


def labels(dev: Mapping[str, pd.DataFrame], frame: pd.DataFrame) -> pd.Series:
    """Proxy relevance per pair of `frame`: 1 when the oracle families are equal."""
    jf = frame.jd_id.map(dev["jd"].set_index("id").family)
    cf = frame.cv_id.map(dev["cv"].set_index("id").family)
    return pd.Series((jf.to_numpy() == cf.to_numpy()).astype(int), index=frame.pair_id.to_numpy())


# --- report ---------------------------------------------------------------------------------

def report(dev: Mapping[str, pd.DataFrame]) -> dict:
    """Sizes, relevant per pool, and the disjointness checks (each a named bool)."""
    from candidate_screener.extraction import role_family as rf
    docs, prep, model = _prepared()
    fit_ids = {s: set(docs[s]["id"].to_numpy()[prep["parts"][s]["fit"]]) for s in rf.SIDES}
    calib_ids = {s: set(docs[s]["id"].to_numpy()[prep["parts"][s]["calib"]]) for s in rf.SIDES}
    ev, wk = _eval_ids(), _weak_ids()
    used = {"jd": set(dev["pools"].jd_id), "cv": set(dev["pools"].cv_id)}
    available = dev_documents_from(docs, prep)
    checks = {}
    for s in rf.SIDES:
        checks[f"{s}_disjoint_from_role_classifier_fit_part"] = not (used[s] & fit_ids[s])
        checks[f"{s}_disjoint_from_calibration_draw"] = not (used[s] & calib_ids[s])
        checks[f"{s}_disjoint_from_weak_pairs"] = not (used[s] & wk[s])
        checks[f"{s}_disjoint_from_eval_region"] = not (used[s] & ev[s])
    p = dev["pools"]
    checks["every_pool_has_pool_size_distinct_cvs"] = bool(
        p.groupby("jd_id").cv_id.nunique().eq(POOL_SIZE).all())
    checks["dev_train_and_dev_select_share_no_jd"] = not (
        set(p[p.split == SPLITS[0]].jd_id) & set(p[p.split == SPLITS[1]].jd_id))
    frame = pairs(dev)
    rel = labels(dev, frame)
    per = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "split": frame.split.to_numpy(),
                        "slot": p.slot.to_numpy(), "r": rel.to_numpy()})
    by_pool = per.groupby(["split", "jd"]).r.sum()
    q = lambda s: [int(v) for v in np.quantile(s, [0, 0.25, 0.5, 0.75, 1])]
    return {
        "plan": "plan/2026-10-07-shortlist-recall (Q1, 01 §1)",
        "config": CONFIG, "role_model": model.fingerprint,
        "documents_available": {s: int(len(d)) for s, d in available.items()},
        "documents_in_pools": {"jd": int(len(used["jd"])), "cv": int(len(used["cv"]))},
        "pools": {sp: int(p[p.split == sp].jd_id.nunique()) for sp in SPLITS},
        "pairs": {sp: int((p.split == sp).sum()) for sp in SPLITS},
        "jd_oracle_families": int(dev["jd"].family.nunique()),
        "relevant_per_pool_quantiles_0_25_50_75_100": {sp: q(by_pool[sp]) for sp in SPLITS},
        "pools_with_no_relevant": {sp: int((by_pool[sp] == 0).sum()) for sp in SPLITS},
        "relevant_share_by_slot": {sl: round(float(per[per.slot == sl].r.mean()), 4)
                                   for sl in SLOTS},
        "pools_sha256": hashlib.sha256(p.to_csv(index=False).encode()).hexdigest(),
        "checks": checks,
        "note": ("Relevance is a proxy (same oracle role family), not fit; the final set is "
                 "the only test of fit. Nothing in this set comes from the eval region."),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="build the cache, write the report")
    g.add_argument("--check", action="store_true", help="rebuild from scratch, diff the report")
    a = ap.parse_args(argv)
    dev = build(force=a.check)
    rep = report(dev)
    bad = [k for k, v in rep["checks"].items() if not v]
    for k, v in rep["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if a.check:
        same = REPORT.exists() and REPORT.read_text(encoding="utf-8") == render(rep)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {REPORT.name}")
        return 0 if same and not bad else 1
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(rep), encoding="utf-8")
    print(f"-> {REPORT}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
