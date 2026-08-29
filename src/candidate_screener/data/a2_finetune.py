"""A2 train/eval partition and the data-science labelling shortlist *(decisions D14, D19)*.

**D18** restricted A2 to unlabelled domain-adaptive pretraining: "It never contributes fit
labels." **Q16** asked what happens once a stage needs more *labelled* A2 data than the
200-pair in-domain evaluation set (task 3.4), and deferred the answer until that happened.
It has: labelled A2 pairs are now wanted for supervised fine-tuning as well as evaluation.

Q16's own text names the fix, and this module is it: partition A2 at document level — every
JD `id` and every CV `id` independently — into a training-eligible region and an
evaluation-eligible region, **before** either side is sampled. Task 3.4's 200-pair set (D14:
not stratified by role family, drawn from all 41 families) must draw only from `eval`;
anything labelled for fine-tuning must draw only from `train`. That is exactly the doubly-
disjoint discipline `fit_split.py` already applies to A1 (D8) — reused here via
`assign_documents` rather than re-derived — so the same leakage the project fixed once for A1
cannot reopen on A2.

The shortlist itself answers a narrower, personal question: within the training region,
which JD/CV pairs are worth handing to a data-science-focused annotator first. It filters to
`Primary Keyword ∈ {Data Science, Data Engineer, Data Analyst}` — verified present on both
sides, and the only bucket `AI Engineer`/`ML Engineer` titles fold into — then bands by
experience and draws candidate pairs per band, mirroring `notebooks/03-djinni-in-domain.ipynb`
and the pair shape already specified for task 3.4 in `01-design-spec.md` §3.4.

Usage:
    uv run python -m candidate_screener.data.a2_finetune --partition --seed 0
    uv run python -m candidate_screener.data.a2_finetune --shortlist --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.data import fit_split as fs
from candidate_screener.data.fit_split import MANIFESTS
from candidate_screener.data.profile import load_split

PARTITION_MANIFEST = MANIFESTS / "a2-partition.csv"
PARTITION_REPORT = MANIFESTS / "a2-partition-report.json"
SHORTLIST_MANIFEST = MANIFESTS / "a2-datascience-shortlist.csv"
SHORTLIST_REPORT = MANIFESTS / "a2-shortlist-report.json"

#: The bucket every AI/ML-titled row folds into on the JD side (verified: no separate
#: "AI Engineer"/"ML Engineer" Primary Keyword exists).
DATASCIENCE_KEYWORDS = ("Data Science", "Data Engineer", "Data Analyst")

#: JD `Exp Years` ships as five categories, not the CV side's 0-11 numeric scale, so the two
#: cannot share a band function. This is the explicit, human-chosen mapping between them —
#: there is no `7+` JD category, so that CV band simply draws fewer/no JD matches, which is a
#: true property of how this field was collected, not a bug to work around.
JD_EXP_TO_BAND = {"no_exp": "0-1", "1y": "0-1", "2y": "2-3", "3y": "2-3", "5y": "4-6"}


def cv_experience_band(years: object) -> str | None:
    """Same banding as notebook 03's `band` lambda, promoted here so builder and
    notebook cannot silently drift apart."""
    if pd.isna(years):
        return None
    return "0-1" if years <= 1 else "2-3" if years <= 3 else "4-6" if years <= 6 else "7+"


# --- 1. partition -----------------------------------------------------------

def partition_ids(jd_ids: list[str], cv_ids: list[str], eval_fraction: float,
                  seed: int) -> tuple[pd.Series, pd.Series]:
    """Independently assign JD ids and CV ids to `eval` / `train`.

    Delegates the actual draw to `fit_split.assign_documents` — sort, permute, slice —
    rather than re-implementing it; only the eval/train naming differs from A1's
    test/val/train.
    """
    jd_seed, cv_seed = np.random.SeedSequence(seed).spawn(2)
    jd_region = fs.assign_documents(jd_ids, {"test": eval_fraction}, jd_seed) \
        .map({"test": "eval", "train": "train"}).rename("region")
    cv_region = fs.assign_documents(cv_ids, {"test": eval_fraction}, cv_seed) \
        .map({"test": "eval", "train": "train"}).rename("region")
    return jd_region, cv_region


def build_partition(eval_fraction: float, seed: int) -> dict:
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    jd_region, cv_region = partition_ids(
        sorted(jd["id"].unique()), sorted(cv["id"].unique()), eval_fraction, seed)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat([
        pd.DataFrame({"doc_id": jd_region.index, "doc_type": "jd", "region": jd_region.to_numpy()}),
        pd.DataFrame({"doc_id": cv_region.index, "doc_type": "cv", "region": cv_region.to_numpy()}),
    ]).sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True)
    manifest.to_csv(PARTITION_MANIFEST, index=False, lineterminator="\n")

    counts = manifest.groupby(["doc_type", "region"]).size().unstack(fill_value=0)
    report = {"seed": seed, "eval_fraction": eval_fraction,
              "counts": {t: {r: int(c) for r, c in row.items()} for t, row in counts.iterrows()}}
    PARTITION_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_partition(p: dict) -> None:
    print(f"\n=== A2 partition — eval_fraction={p['eval_fraction']:.0%} seed={p['seed']}")
    for t, row in p["counts"].items():
        print(f"  {t:<4} " + "  ".join(f"{r}={n}" for r, n in row.items()))
    print(f"  manifest -> {PARTITION_MANIFEST.name}   <-- task 3.4 must sample only from "
          f"'eval'; any fine-tuning labels must come only from 'train'")


# --- 2. shortlist -------------------------------------------------------------

def load_train_region() -> tuple[set[str], set[str]]:
    if not PARTITION_MANIFEST.exists():
        raise FileNotFoundError(
            f"{PARTITION_MANIFEST} missing — run --partition first, so the shortlist "
            "cannot cross into the region task 3.4's evaluation set is reserved from")
    manifest = pd.read_csv(PARTITION_MANIFEST)
    train = manifest[manifest.region == "train"]
    return (set(train[train.doc_type == "jd"].doc_id),
            set(train[train.doc_type == "cv"].doc_id))


def sample_shortlist(jd_f: pd.DataFrame, cv_f: pd.DataFrame, jds_per_cell: int,
                     per_jd: int, seed: int) -> pd.DataFrame:
    """Sample JD/CV pairs banded by (Primary Keyword, experience) — the pure core.

    `jd_f`/`cv_f` are already filtered to the training region and the keyword scope, and
    already carry an `exp_band` column. Banding, not a flat sample, so the shortlist
    contains plausible near-misses across seniority rather than only the largest keyword's
    most common band — the same rationale notebook 03 gives for banding the task 3.4 pools.
    """
    cells = sorted({(k, b) for k, b in zip(jd_f["Primary Keyword"], jd_f.exp_band) if b})
    seeds = np.random.SeedSequence(seed).spawn(len(cells))
    rows = []
    for (keyword, band), cell_seed in zip(cells, seeds):
        jd_pool = jd_f[(jd_f["Primary Keyword"] == keyword) & (jd_f.exp_band == band)]["id"].to_numpy()
        cv_pool = cv_f[(cv_f["Primary Keyword"] == keyword) & (cv_f.exp_band == band)]["id"].to_numpy()
        if len(jd_pool) == 0 or len(cv_pool) == 0:
            continue
        # One seed to pick the JDs, one per chosen JD to independently shuffle its CV
        # pool — spawned up front so the draw is deterministic regardless of how many
        # of the jds_per_cell slots this cell actually fills.
        jd_seed, *cv_seeds = cell_seed.spawn(1 + jds_per_cell)
        chosen_jds = jd_pool[np.random.default_rng(jd_seed).permutation(len(jd_pool))][:jds_per_cell]
        for query_jd, cv_seed in zip(chosen_jds, cv_seeds):
            order = cv_pool[np.random.default_rng(cv_seed).permutation(len(cv_pool))]
            for candidate_cv in order[:per_jd]:
                rows.append((f"{query_jd}__{candidate_cv}", query_jd, candidate_cv, keyword, band))

    out = pd.DataFrame(rows, columns=["pair_id", "jd_id", "cv_id", "primary_keyword", "exp_band"])
    return out.drop_duplicates("pair_id").sort_values(
        ["primary_keyword", "exp_band", "jd_id", "cv_id"], kind="stable").reset_index(drop=True)


def build_shortlist(keywords: tuple[str, ...], jds_per_cell: int, per_jd: int,
                    seed: int) -> pd.DataFrame:
    """Load A2, restrict to the training region and the keyword scope, band, and sample."""
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    train_jd_ids, train_cv_ids = load_train_region()

    jd_f = jd[jd["id"].isin(train_jd_ids) & jd["Primary Keyword"].isin(keywords)].copy()
    jd_f["exp_band"] = jd_f["Exp Years"].map(JD_EXP_TO_BAND)
    cv_f = cv[cv["id"].isin(train_cv_ids) & cv["Primary Keyword"].isin(keywords)].copy()
    cv_f["exp_band"] = cv_f["Experience Years"].map(cv_experience_band)

    return sample_shortlist(jd_f, cv_f, jds_per_cell, per_jd, seed)


def summarise_shortlist(shortlist: pd.DataFrame) -> dict:
    by_cell = shortlist.groupby(["primary_keyword", "exp_band"]).agg(
        pairs=("pair_id", "size"), jds=("jd_id", "nunique"), cvs=("cv_id", "nunique"))
    return {
        "pairs": int(len(shortlist)),
        "jds": int(shortlist.jd_id.nunique()),
        "cvs": int(shortlist.cv_id.nunique()),
        "by_cell": {f"{k}/{b}": {"pairs": int(r.pairs), "jds": int(r.jds), "cvs": int(r.cvs)}
                   for (k, b), r in by_cell.iterrows()},
    }


def build(keywords: tuple[str, ...], jds_per_cell: int, per_jd: int, seed: int) -> dict:
    shortlist = build_shortlist(keywords, jds_per_cell, per_jd, seed)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(SHORTLIST_MANIFEST, index=False, lineterminator="\n")
    report = {"seed": seed, "keywords": list(keywords), "jds_per_cell": jds_per_cell,
              "per_jd": per_jd, "summary": summarise_shortlist(shortlist)}
    SHORTLIST_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_shortlist(r: dict) -> None:
    s = r["summary"]
    print(f"\n=== Data-science shortlist — {s['pairs']} pairs, {s['jds']} JDs, {s['cvs']} CVs "
          f"(seed={r['seed']}, train region only)")
    for cell, c in s["by_cell"].items():
        print(f"  {cell:<22} pairs={c['pairs']:>4}  jds={c['jds']:>3}  cvs={c['cvs']:>3}")
    print(f"  manifest -> {SHORTLIST_MANIFEST.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", action="store_true", help="write a2-partition.csv")
    ap.add_argument("--shortlist", action="store_true", help="write a2-datascience-shortlist.csv")
    ap.add_argument("--eval-fraction", type=float, default=0.25)
    ap.add_argument("--keywords", nargs="+", default=list(DATASCIENCE_KEYWORDS))
    ap.add_argument("--jds-per-cell", type=int, default=15)
    ap.add_argument("--per-jd", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.partition or args.shortlist):
        ap.error("pass --partition and/or --shortlist")

    if args.partition:
        print_partition(build_partition(args.eval_fraction, args.seed))
    if args.shortlist:
        print_shortlist(build(tuple(args.keywords), args.jds_per_cell, args.per_jd, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
