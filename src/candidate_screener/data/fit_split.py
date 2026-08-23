"""Task 3.2 — the leak-free A1 split *(decisions D8, D16)*.

The shipped `cnamuangtoun` split leaks 99.8% of test resumes into train, so it is
replaced rather than repaired. **D16**: pool the shipped train and test first —
642 + 477 unique resumes dedupe to 643, because 476 of the test resumes *are*
train resumes, so there is no partition worth preserving — then split the pooled
corpus from scratch.

The split is **doubly disjoint**. Each resume and each JD is independently assigned
to `train`, `val` or `test`; a pair joins a split only when *both* of its documents
were assigned to it, and every cross pair is discarded. That discard is the price of
a genuinely leak-free evaluation: nothing in test shares a resume *or* a JD with
anything in train.

Yield falls roughly as the square of the hold-out fraction, so the fraction is not a
free parameter — `--survey` measures what each one buys, across seeds, before the
choice is made.

Usage:
    uv run python -m candidate_screener.data.fit_split --survey
    uv run python -m candidate_screener.data.fit_split --build --test-frac 0.30 --seed 0
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import DOCS_DATA, PROCESSED
from candidate_screener.data.ids import mint_ids
from candidate_screener.data.profile import load_split

MANIFESTS = DOCS_DATA / "manifests"
SPLIT_MANIFEST = MANIFESTS / "fit-split.csv"
YIELD_MANIFEST = MANIFESTS / "fit-split-yield.json"
CONFLICT_MANIFEST = MANIFESTS / "fit-pair-conflicts.csv"
FIT_OUT = PROCESSED / "fit"

#: Documents in `data/raw/fit/` — verified 19 Aug 2026, re-asserted on every build.
EXPECTED_RESUMES, EXPECTED_JDS, EXPECTED_ROWS = 643, 351, 8000

#: Assigned in this order, so raising `val_frac` never moves a document out of
#: `test`: the two-way and three-way yields at one seed are directly comparable.
EVAL_SPLITS = ("test", "val")

NOT_COMPARABLE = (
    "Results on this split are NOT comparable to any published number on the shipped "
    "cnamuangtoun train/test partition. That partition leaks 99.8% of its test resumes "
    "into train; this one is doubly disjoint. The incomparability is the accepted price "
    "of removing the leakage and must be stated wherever Stage 1-4 results appear."
)


# --- loading and pair hygiene ---------------------------------------------

def load_pooled() -> tuple[pd.DataFrame, dict]:
    """Pool the shipped splits, mint document IDs, and resolve duplicate pairs.

    Pooling surfaces 7 `(resume, jd)` pairs that appear twice — all of them inside
    the shipped *train* set, so this is a defect of the source, not of the pooling.
    6 of the 7 carry two different labels. A pair the publisher labelled two ways is
    not evidence in either direction, so all its rows are dropped; the one
    self-consistent duplicate keeps a single row. Every case is logged by ID.
    """
    frames = []
    for shipped in ("train", "test"):
        df = load_split("fit", shipped)
        frames.append(df.assign(shipped_split=shipped))
    pooled = pd.concat(frames, ignore_index=True)
    if len(pooled) != EXPECTED_ROWS:
        raise AssertionError(f"pooled {len(pooled)} rows, expected {EXPECTED_ROWS}")

    pooled["resume_id"] = mint_ids(pooled.resume_text, "resume")
    pooled["jd_id"] = mint_ids(pooled.job_description_text, "jd")

    n_res, n_jd = pooled.resume_id.nunique(), pooled.jd_id.nunique()
    if (n_res, n_jd) != (EXPECTED_RESUMES, EXPECTED_JDS):
        raise AssertionError(
            f"{n_res} unique resumes / {n_jd} unique JDs, expected "
            f"{EXPECTED_RESUMES} / {EXPECTED_JDS} — publisher re-upload?"
        )

    grouped = pooled.groupby(["resume_id", "jd_id"], sort=True)
    sizes, labels = grouped.size(), grouped.label.nunique()
    conflicting = labels[labels > 1].index
    duplicated = sizes[sizes > 1].index

    log = (
        pooled[pooled.set_index(["resume_id", "jd_id"]).index.isin(duplicated)]
        .groupby(["resume_id", "jd_id"])
        .agg(rows=("label", "size"),
             labels=("label", lambda s: " | ".join(sorted(s.unique()))),
             shipped_splits=("shipped_split", lambda s: " | ".join(sorted(s.unique()))))
        .reset_index()
    )
    log["resolution"] = np.where(log.labels.str.contains(r"\|"), "dropped_conflicting",
                                 "deduplicated_kept_one")

    keep = ~pooled.set_index(["resume_id", "jd_id"]).index.isin(conflicting)
    clean = (pooled[keep]
             .drop_duplicates(["resume_id", "jd_id"], keep="first")
             .sort_values(["jd_id", "resume_id"], kind="stable")
             .reset_index(drop=True))

    stats = {
        "pooled_rows": int(len(pooled)),
        "unique_resumes": int(n_res),
        "unique_jds": int(n_jd),
        "duplicate_pairs": int(len(duplicated)),
        "conflicting_pairs": int(len(conflicting)),
        "rows_dropped": int(len(pooled) - len(clean)),
        "usable_pairs": int(len(clean)),
        "label_counts": {k: int(v) for k, v in clean.label.value_counts().items()},
        "shipped_split_was_jd_disjoint": bool(
            not set(pooled[pooled.shipped_split == "train"].jd_id)
            & set(pooled[pooled.shipped_split == "test"].jd_id)),
        "shipped_test_resumes_also_in_train": int(len(
            set(pooled[pooled.shipped_split == "test"].resume_id)
            & set(pooled[pooled.shipped_split == "train"].resume_id))),
    }
    return clean, {"pooling": stats, "conflict_log": log}


# --- the split itself ------------------------------------------------------

def assign_documents(doc_ids: Sequence[str], fracs: Mapping[str, float],
                     seed: np.random.SeedSequence) -> pd.Series:
    """Assign each document to `test`, `val` or `train` by an unweighted draw.

    Documents are sorted by ID first, so the assignment depends on the corpus and
    the seed but never on row order in the source parquet.
    """
    ids = np.array(sorted(doc_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    out = np.full(len(ids), "train", dtype=object)
    cut = 0
    for name in EVAL_SPLITS:
        n = int(round(fracs.get(name, 0.0) * len(ids)))
        out[perm[cut:cut + n]] = name
        cut += n
    if cut > len(ids):
        raise ValueError(f"fractions sum above 1.0: {dict(fracs)}")
    return pd.Series(out, index=ids, name="split")


def split_pairs(pairs: pd.DataFrame, test_frac: float, val_frac: float,
                seed: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assign documents, then assign each pair to the split *both* its sides share."""
    fracs = {"test": test_frac, "val": val_frac}
    resume_seed, jd_seed = np.random.SeedSequence(seed).spawn(2)
    r_split = assign_documents(pairs.resume_id.unique(), fracs, resume_seed)
    j_split = assign_documents(pairs.jd_id.unique(), fracs, jd_seed)

    rs = pairs.resume_id.map(r_split).to_numpy()
    js = pairs.jd_id.map(j_split).to_numpy()
    out = pairs.assign(split=np.where(rs == js, rs, "discarded"))
    return out, r_split, j_split


def summarise(pairs: pd.DataFrame) -> dict:
    """Per-split pair, document and query counts. `jds_with_good_fit` is the binding
    number: it is *n* for every retrieval figure task 3.3 can report."""
    out = {}
    for name in ("train", "val", "test", "discarded"):
        sub = pairs[pairs.split == name]
        good = sub[sub.label == "Good Fit"]
        out[name] = {
            "pairs": int(len(sub)),
            "resumes": int(sub.resume_id.nunique()),
            "jds": int(sub.jd_id.nunique()),
            "jds_with_good_fit": int(good.jd_id.nunique()),
            "good_fit_pairs": int(len(good)),
            # Relevance density per query — task 3.3 needs it to decide which
            # Recall@k is attainable at all, and it is not recoverable later.
            "good_fit_per_jd_median": (
                int(good.groupby("jd_id").size().median()) if len(good) else 0),
        }
    return out


def cv_folds(train_pairs: pd.DataFrame, k: int, seed: int) -> list[dict]:
    """k doubly-disjoint folds *within* the train side.

    The fallback D16 anticipated: if a held-out val fold costs more train pairs than
    it is worth, tune by cross-validation here instead. Same disjointness rule, so a
    fold is as leak-free as the test set — it just spends nothing permanently.
    """
    resume_seed, jd_seed = np.random.SeedSequence(seed).spawn(2)
    r_ids, j_ids = np.array(sorted(train_pairs.resume_id.unique())), np.array(sorted(train_pairs.jd_id.unique()))
    r_bucket = pd.Series(np.random.default_rng(resume_seed).permutation(len(r_ids)) % k, index=r_ids)
    j_bucket = pd.Series(np.random.default_rng(jd_seed).permutation(len(j_ids)) % k, index=j_ids)
    rb = train_pairs.resume_id.map(r_bucket).to_numpy()
    jb = train_pairs.jd_id.map(j_bucket).to_numpy()

    folds = []
    for i in range(k):
        held = (rb == i) & (jb == i)
        fit = (rb != i) & (jb != i)
        ev = train_pairs[held]
        folds.append({
            "fold": i,
            "eval_pairs": int(held.sum()),
            "eval_jds": int(ev.jd_id.nunique()),
            "eval_jds_with_good_fit": int(ev[ev.label == "Good Fit"].jd_id.nunique()),
            "fit_pairs": int(fit.sum()),
        })
    return folds


# --- survey ----------------------------------------------------------------

def survey(pairs: pd.DataFrame, fractions: Sequence[float], seeds: Sequence[int],
           val_fracs: Sequence[float], cv_k: int) -> dict:
    """Measure what each hold-out fraction buys, and how much of that is seed luck.

    A single draw is not evidence at this corpus size: at a 25% hold-out the number
    of test JDs carrying a Good Fit — *n* for every retrieval metric — moves by more
    than a factor of two across seeds. The distribution is the finding, not the draw.
    """
    two_way = []
    for f in fractions:
        runs = [summarise(split_pairs(pairs, f, 0.0, s)[0]) for s in seeds]
        row: dict = {"test_frac": f, "seed0": runs[0], "seeds": len(seeds), "across_seeds": {}}
        for metric in ("pairs", "jds", "jds_with_good_fit"):
            vals = np.array([r["test"][metric] for r in runs])
            row["across_seeds"][f"test_{metric}"] = {
                "mean": round(float(vals.mean()), 1), "min": int(vals.min()), "max": int(vals.max())}
        vals = np.array([r["train"]["pairs"] for r in runs])
        row["across_seeds"]["train_pairs"] = {
            "mean": round(float(vals.mean()), 1), "min": int(vals.min()), "max": int(vals.max())}
        two_way.append(row)

    three_way = []
    for f in fractions:
        for v in val_fracs:
            if v <= 0 or f + v > 0.6:
                continue
            three_way.append({"test_frac": f, "val_frac": v,
                              "seed0": summarise(split_pairs(pairs, f, v, 0)[0])})

    cv = {}
    for f in fractions:
        assigned, _, _ = split_pairs(pairs, f, 0.0, 0)
        cv[f"{f:.2f}"] = cv_folds(assigned[assigned.split == "train"], cv_k, 0)

    return {"two_way": two_way, "three_way": three_way,
            "train_cross_validation": {"k": cv_k, "by_test_frac": cv}}


def print_survey(s: dict) -> None:
    print("\n=== Two-way doubly-disjoint hold-out — seed 0, and the spread over "
          f"{s['two_way'][0]['seeds']} seeds")
    print(f"  {'frac':>5} {'test pairs':>18} {'test JDs':>14} {'JDs w/ Good Fit':>22} {'train pairs':>18}")
    for r in s["two_way"]:
        a, z = r["across_seeds"], r["seed0"]
        def cell(key, seed0):
            return f"{seed0:>5} [{a[key]['min']}-{a[key]['max']}]"
        print(f"  {r['test_frac']:>5.0%} "
              f"{cell('test_pairs', z['test']['pairs']):>18} "
              f"{cell('test_jds', z['test']['jds']):>14} "
              f"{cell('test_jds_with_good_fit', z['test']['jds_with_good_fit']):>22} "
              f"{cell('train_pairs', z['train']['pairs']):>18}")
    print("  (cells are: seed-0 value [min-max across seeds])")

    print("\n=== Three-way — what a held-out val fold costs, seed 0")
    print(f"  {'test':>5} {'val':>5} | {'test pairs':>10} {'test good-JD':>12} | "
          f"{'val pairs':>9} {'val good-JD':>11} | {'train pairs':>11}")
    for r in s["three_way"]:
        z = r["seed0"]
        print(f"  {r['test_frac']:>5.0%} {r['val_frac']:>5.0%} | {z['test']['pairs']:>10} "
              f"{z['test']['jds_with_good_fit']:>12} | {z['val']['pairs']:>9} "
              f"{z['val']['jds_with_good_fit']:>11} | {z['train']['pairs']:>11}")

    k = s["train_cross_validation"]["k"]
    print(f"\n=== The alternative to a val fold: {k}-fold doubly-disjoint CV inside train")
    for frac, folds in s["train_cross_validation"]["by_test_frac"].items():
        ev = [f["eval_pairs"] for f in folds]
        gj = [f["eval_jds_with_good_fit"] for f in folds]
        print(f"  test_frac={float(frac):.0%}  eval pairs/fold {min(ev)}-{max(ev)} "
              f"(total {sum(ev)})  good-fit JDs/fold {min(gj)}-{max(gj)}  "
              f"fit pairs/fold {min(f['fit_pairs'] for f in folds)}-{max(f['fit_pairs'] for f in folds)}")


# --- build -----------------------------------------------------------------

def build(test_frac: float, val_frac: float, seed: int, write_parquet: bool = True) -> dict:
    pairs, meta = load_pooled()
    assigned, r_split, j_split = split_pairs(pairs, test_frac, val_frac, seed)
    counts = summarise(assigned)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat([
        pd.DataFrame({"doc_id": r_split.index, "doc_type": "resume", "split": r_split.to_numpy()}),
        pd.DataFrame({"doc_id": j_split.index, "doc_type": "jd", "split": j_split.to_numpy()}),
    ]).sort_values(["doc_type", "doc_id"], kind="stable")
    manifest.to_csv(SPLIT_MANIFEST, index=False, lineterminator="\n")
    meta["conflict_log"].to_csv(CONFLICT_MANIFEST, index=False, lineterminator="\n")

    yields = {
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "note_not_comparable": NOT_COMPARABLE,
        "note_discarded": (
            "Discarded pairs share exactly one document with another split. They are "
            "unusable by construction, not a bug: keeping them would leak a resume or a "
            "JD across the train/test boundary."
        ),
        "note_determinism": (
            "No timestamps are written into any manifest — the git history is the "
            "timestamp — so `build --seed N` is byte-reproducible and a diff is a real "
            "change in the data, not a re-run."
        ),
        "pooling": meta["pooling"],
        "counts": counts,
        "documents": {
            "resumes": {k: int(v) for k, v in r_split.value_counts().items()},
            "jds": {k: int(v) for k, v in j_split.value_counts().items()},
        },
    }
    YIELD_MANIFEST.write_text(json.dumps(yields, indent=2) + "\n", encoding="utf-8")

    if write_parquet:
        FIT_OUT.mkdir(parents=True, exist_ok=True)
        cols = ["resume_id", "jd_id", "resume_text", "job_description_text", "label"]
        for name in ("train", "val", "test"):
            sub = assigned[assigned.split == name][cols].reset_index(drop=True)
            path = FIT_OUT / f"{name}.parquet"
            if sub.empty:
                path.unlink(missing_ok=True)
                continue
            sub.to_parquet(path, index=False)

    return yields


def print_build(y: dict) -> None:
    p = y["pooling"]
    print(f"\n=== Pooled A1 — {p['pooled_rows']} rows, {p['unique_resumes']} resumes, "
          f"{p['unique_jds']} JDs")
    print(f"  shipped split leaked {p['shipped_test_resumes_also_in_train']} test resumes into "
          f"train; JD-disjoint: {p['shipped_split_was_jd_disjoint']}")
    print(f"  dropped {p['rows_dropped']} rows over {p['duplicate_pairs']} duplicate pairs "
          f"({p['conflicting_pairs']} of them labelled two ways) -> {p['usable_pairs']} usable")
    print(f"\n=== Split at test={y['test_frac']:.0%} val={y['val_frac']:.0%} seed={y['seed']}")
    for name, c in y["counts"].items():
        print(f"  {name:<10} pairs={c['pairs']:>5}  resumes={c['resumes']:>4}  jds={c['jds']:>4}  "
              f"JDs w/ Good Fit={c['jds_with_good_fit']:>3}")
    print(f"  manifests -> {SPLIT_MANIFEST.relative_to(SPLIT_MANIFEST.parents[3])}, "
          f"{YIELD_MANIFEST.name}, {CONFLICT_MANIFEST.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", action="store_true", help="yield table; writes nothing")
    ap.add_argument("--build", action="store_true", help="write the manifests and splits")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--survey-seeds", type=int, default=20)
    ap.add_argument("--cv-k", type=int, default=5)
    ap.add_argument("--survey-out", type=Path, default=MANIFESTS / "fit-split-survey.json")
    ap.add_argument("--no-parquet", action="store_true", help="manifests only")
    args = ap.parse_args()
    if not (args.survey or args.build):
        ap.error("pass --survey and/or --build")

    if args.survey:
        pairs, meta = load_pooled()
        s = survey(pairs, [0.15, 0.20, 0.25, 0.30, 0.35], list(range(args.survey_seeds)),
                   [0.10, 0.15], args.cv_k)
        s["pooling"] = meta["pooling"]
        print_survey(s)
        args.survey_out.parent.mkdir(parents=True, exist_ok=True)
        args.survey_out.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        print(f"\n  survey -> {args.survey_out}")

    if args.build:
        print_build(build(args.test_frac, args.val_frac, args.seed,
                          write_parquet=not args.no_parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
