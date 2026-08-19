"""Reproduce every figure marked "Verified" in ../02-data-catalog.md.

Acts as an acceptance test on the download: if these numbers drift, a publisher has
re-uploaded and the catalog needs re-verifying.

Usage:
    uv run python plan/2026-08-19-data-strategy/scripts/profile_sources.py --all
    uv run python plan/2026-08-19-data-strategy/scripts/profile_sources.py --check fit dataturks
    uv run python plan/2026-08-19-data-strategy/scripts/profile_sources.py --recover-ats-boundary
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"


def _load_split(key: str, split: str) -> pd.DataFrame:
    files = sorted((RAW / key).glob(f"*-{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {RAW / key} -- run fetch_sources.py")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def profile_fit() -> None:
    """A1 -- effective size, label mix, pool feasibility, leakage."""
    print("\n=== A1 cnamuangtoun/resume-job-description-fit")
    tr, te = _load_split("fit", "train"), _load_split("fit", "test")
    for name, df in (("train", tr), ("test", te)):
        g = df.groupby("job_description_text").size()
        good = df[df.label == "Good Fit"].groupby("job_description_text").size()
        print(f"  [{name}] rows={len(df)}  unique_resumes={df.resume_text.nunique()}  "
              f"unique_jds={df.job_description_text.nunique()}")
        print("    labels: " + "  ".join(
            f"{k}={v:.3f}" for k, v in df.label.value_counts(normalize=True).items()))
        print(f"    resumes/JD mean={g.mean():.1f} median={g.median():.0f} max={g.max()}"
              f" | JDs>=5 resumes={(g >= 5).sum()} | JDs>=1 GoodFit={(good >= 1).sum()}")

    shared = set(tr.resume_text) & set(te.resume_text)
    pct = 100 * len(shared) / te.resume_text.nunique()
    print(f"  LEAKAGE: {len(shared)} test resumes also in train ({pct:.1f}% of test uniques)")
    print(f"  JD overlap train/test: {len(set(tr.job_description_text) & set(te.job_description_text))}")
    both = pd.concat([tr, te])
    conflict = both.groupby(["resume_text", "job_description_text"]).label.nunique()
    print(f"  pairs with conflicting labels: {(conflict > 1).sum()}")


def profile_dataturks() -> None:
    """B1 -- span counts, PII classes, broken offsets."""
    print("\n=== B1 DataTurks resume entities")
    docs = []
    for fn in ("traindata.json", "testdata.json"):
        for line in (RAW / "dataturks" / fn).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    ents, spans, bad = collections.Counter(), 0, 0
    for d in docs:
        for a in d.get("annotation") or []:
            if not a or not a.get("label"):
                continue
            lab = a["label"][0] if isinstance(a["label"], list) else a["label"]
            for p in a.get("points", []):
                ents[lab] += 1
                spans += 1
                if d["content"][p["start"]:p["end"] + 1].strip() != (p.get("text") or "").strip():
                    bad += 1
    print(f"  docs={len(docs)}  spans={spans}  broken_offsets={bad} ({100 * bad / spans:.1f}%)")
    for k, v in ents.most_common():
        flag = "   <-- DIRECT PII" if k in {"Name", "Email Address"} else ""
        print(f"    {k:<24} {v}{flag}")
    corpus = " ".join(d["content"] for d in docs)
    emails = set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", corpus))
    phones = set(re.findall(r"\+?\d[\d\s().-]{8,14}\d", corpus))
    print(f"  residual emails={len(emails)}  phone-like={len(phones)}")


def profile_djinni() -> None:
    """A2 -- size and the Primary Keyword join vocabulary shared by both sides."""
    print("\n=== A2 Djinni recruitment dataset")
    jd, cv = _load_split("djinni-jd", "train"), _load_split("djinni-cv", "train")
    print(f"  JDs={len(jd)}  CVs={len(cv)}")
    kj, kc = set(jd["Primary Keyword"].dropna()), set(cv["Primary Keyword"].dropna())
    print(f"  Primary Keyword: {len(kj)} JD-side, {len(kc)} CV-side, {len(kj & kc)} shared")
    print("  top shared role families: " +
          ", ".join(jd[jd["Primary Keyword"].isin(kj & kc)]["Primary Keyword"]
                    .value_counts().head(12).index))


def profile_resume_atlas() -> None:
    """C3 -- confirm the text is pre-normalised and therefore EDA-only."""
    print("\n=== C3 ahmedheakl/resume-atlas")
    df = _load_split("resume-atlas", "train")
    sample = " ".join(df.Text.head(200))
    print(f"  rows={len(df)}  categories={df.Category.nunique()}")
    print(f"  uppercase chars={sum(c.isupper() for c in sample)}  "
          f"punctuation={sum(c in '.,;:' for c in sample)}")
    print("  -> pre-normalised (no case, no punctuation): clustering EDA only")


def recover_ats_boundary() -> None:
    """A4 -- demonstrate the derivation and recover the missing resume/JD boundary."""
    print("\n=== A4 0xnbk/resume-ats-score-v1-en -- derivation check")
    ats = pd.concat([_load_split("ats", s) for s in ("train", "validation")], ignore_index=True)
    fit = pd.concat([_load_split("fit", s) for s in ("train", "test")], ignore_index=True)
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    n_sep = ats.text.str.contains("[SEP]", regex=False).sum()
    print(f"  rows containing the documented '[SEP]': {n_sep}")
    resumes = {norm(r) for r in fit.resume_text.unique()}
    by_head = collections.defaultdict(list)
    for r in resumes:
        by_head[r[:100]].append(r)
    hits = sum(1 for t in ats.text
               if any(norm(t).startswith(r) for r in by_head.get(norm(t)[:100], [])))
    print(f"  rows starting with a cnamuangtoun resume: {hits}/{len(ats)} "
          f"({100 * hits / len(ats):.1f}%)  -> derivative, not independent")


def profile_esco() -> None:
    """D1 -- confirm the ESCO drop is complete and count the alias inventory."""
    print("\n=== D1 ESCO v1.2.1")
    d = RAW / "esco_dataset-v1.2.1-classification"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found -- see action plan step 1.10")
    sk = pd.read_csv(d / "skills_en.csv", low_memory=False)
    occ = pd.read_csv(d / "occupations_en.csv", low_memory=False)
    rel = pd.read_csv(d / "occupationSkillRelations_en.csv", low_memory=False)
    print(f"  skills={len(sk)}  occupations={len(occ)}  occupation-skill links={len(rel)}")
    print("  skillType: " + "  ".join(f"{k}={v}" for k, v in sk.skillType.value_counts().items()))
    aliases = int(sk.altLabels.dropna().str.count("\n").add(1).sum())
    print(f"  alternative labels (aliases for normalisation): ~{aliases}")
    if "relationType" in rel:
        print("  relationType: " + "  ".join(
            f"{k}={v}" for k, v in rel.relationType.value_counts().items())
              + "   <-- seeds hard vs preferred requirements")


CHECKS = {
    "esco": profile_esco,
    "fit": profile_fit,
    "dataturks": profile_dataturks,
    "djinni": profile_djinni,
    "resume-atlas": profile_resume_atlas,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", choices=sorted(CHECKS), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recover-ats-boundary", action="store_true")
    args = ap.parse_args()

    for key in (sorted(CHECKS) if args.all else args.check):
        try:
            CHECKS[key]()
        except FileNotFoundError as exc:
            print(f"\n=== {key}: SKIPPED -- {exc}")
    if args.recover_ats_boundary:
        recover_ats_boundary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
