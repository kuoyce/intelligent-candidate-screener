"""Weak supervision from the A2 **train** region (plan 02 §4.1, D19): pairs labelled by metadata.

    positive       same `Primary Keyword`, CV experience band meets the JD's      label 1
    hard negative  same keyword, CV experience band below the JD's               label 0
    easy negative  different keyword                                             label 0

JDs and CVs are split into `train` / `validation` **before** pairing, so the two splits
share no document. Only ids are committed (`docs/data/manifests/weak-pairs.csv`); text is
read from the seeded train-region sample in `matching.corpus`.

Known limitation, stated where it is built: the label is mostly "predict the keyword",
which the metadata rung gets for free. A model trained on it earns its place only by
beating metadata + text (registration C5-C10).

Usage:
    uv run python -m candidate_screener.data.weak_pairs --build
    uv run python -m candidate_screener.data.weak_pairs --check
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import DOCS_DATA
from candidate_screener.data.a2_finetune import PARTITION_MANIFEST
from candidate_screener.matching import corpus

N_JDS = 3000
VAL_FRACTION = 0.10
SEED = 0
MANIFEST = DOCS_DATA / "manifests" / "weak-pairs.csv"
REPORT = DOCS_DATA / "manifests" / "weak-pairs-report.json"
EXP = corpus.EXP_ORDER


def split_ids(ids: pd.Series, fraction: float, seed: int) -> set[str]:
    """A seeded validation subset of `ids`, independent of their order."""
    ordered = np.array(sorted(ids.unique()))
    rng = np.random.default_rng(seed)
    return set(ordered[rng.random(len(ordered)) < fraction])


def draw(jd: pd.DataFrame, cv: pd.DataFrame, n_jds: int, seed: int) -> pd.DataFrame:
    """One positive, one hard negative and one easy negative per JD, where each exists."""
    rng = np.random.default_rng(seed)
    cv = cv.assign(band=cv.exp_band.map(EXP)).dropna(subset=["band", "Primary Keyword"])
    jd = jd.assign(band=jd.exp_band.map(EXP)).dropna(subset=["band", "Primary Keyword"])
    by_kw = {k: g.sort_values("id") for k, g in cv.groupby("Primary Keyword")}
    jd = jd[jd["Primary Keyword"].isin(by_kw)].sort_values("id")
    jd = jd.iloc[rng.permutation(len(jd))[:n_jds]]
    rows = []
    for _, j in jd.iterrows():
        same = by_kw[j["Primary Keyword"]]
        pools = {
            "positive": same[same.band >= j.band],
            "hard_negative": same[same.band < j.band],
            "easy_negative": cv[cv["Primary Keyword"] != j["Primary Keyword"]],
        }
        for kind, pool in pools.items():
            if len(pool):
                pick = pool.iloc[int(rng.integers(len(pool)))]
                rows.append((f"{j['id']}__{pick['id']}", j["id"], pick["id"], kind,
                             int(kind == "positive")))
    return pd.DataFrame(rows, columns=["pair_id", "jd_id", "cv_id", "kind", "label"])


def build(n_jds: int = N_JDS, seed: int = SEED) -> pd.DataFrame:
    jd, cv = corpus.a2_train_documents()
    val_jd, val_cv = split_ids(jd["id"], VAL_FRACTION, seed), split_ids(cv["id"], VAL_FRACTION, seed + 1)
    parts = []
    for split, jds, cvs in (("train", jd[~jd["id"].isin(val_jd)], cv[~cv["id"].isin(val_cv)]),
                            ("validation", jd[jd["id"].isin(val_jd)], cv[cv["id"].isin(val_cv)])):
        n = round(n_jds * (VAL_FRACTION if split == "validation" else 1 - VAL_FRACTION))
        parts.append(draw(jds, cvs, n, seed + (split == "validation")).assign(split=split))
    return pd.concat(parts, ignore_index=True).drop_duplicates("pair_id")


def report(frame: pd.DataFrame) -> dict:
    return {
        "n_pairs": int(len(frame)),
        "by_split_and_kind": {s: g.kind.value_counts().sort_index().to_dict()
                              for s, g in frame.groupby("split")},
        "positive_rate": round(float(frame.label.mean()), 6),
        "note": ("Labels come from metadata, not from people: positive = same keyword and "
                 "experience meets the JD's band. Train and validation share no JD and no CV."),
    }


def load() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def with_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach text (and metadata) from the train-region sample."""
    jd, cv = corpus.a2_train_documents()
    jd, cv = jd.set_index("id"), cv.set_index("id")
    return frame.assign(
        jd_text=frame.jd_id.map(jd.jd_text), cv_text=frame.cv_id.map(cv.cv_text),
        jd_keyword=frame.jd_id.map(jd["Primary Keyword"]),
        cv_keyword=frame.cv_id.map(cv["Primary Keyword"]),
        jd_exp_band=frame.jd_id.map(jd.exp_band), cv_exp_band=frame.cv_id.map(cv.exp_band))


def check() -> list[tuple[bool, str]]:
    frame = load()
    part = pd.read_csv(PARTITION_MANIFEST)
    eval_ids = set(part[part.region == "eval"].doc_id)
    out = [(not (set(frame.jd_id) | set(frame.cv_id)) & eval_ids,
            "no weak pair touches an A2 eval-region document (D19)")]
    tr, va = frame[frame.split == "train"], frame[frame.split == "validation"]
    out.append((not (set(tr.jd_id) & set(va.jd_id)) and not (set(tr.cv_id) & set(va.cv_id)),
                "train and validation share no JD and no CV"))
    rebuilt = build()
    out.append((rebuilt.to_csv(index=False, lineterminator="\n")
                == MANIFEST.read_text(encoding="utf-8"),
                "determinism: rebuild reproduces weak-pairs.csv"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        frame = build()
        MANIFEST.write_text(frame.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
        REPORT.write_text(json.dumps(report(frame), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report(frame), indent=2))
    if args.check:
        results = check()
        for ok, message in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        return 0 if all(ok for ok, _ in results) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
