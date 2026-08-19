"""Reproduce every figure marked **Verified** in `docs/data/data-catalog.md`.

`verify` answers "are the right bytes on disk?"; this answers "do they still say
what the catalog claims they say?" — effective sample size, label mix, leakage,
pool feasibility, PII inventory and the defects each source is known to carry.
Each check returns its metrics, and `--all` writes them to
`docs/data/profile-metrics.json` so the dataset cards and notebooks cite measured
numbers rather than restating prose.

Usage:
    uv run python -m candidate_screener.data.profile --all
    uv run python -m candidate_screener.data.profile --check fit dataturks
    uv run python -m candidate_screener.data.profile --recover-ats-boundary
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from candidate_screener.config import INTERIM, PROFILE_METRICS, RAW

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(s: object) -> str:
    return _NORM.sub(" ", str(s).lower()).strip()


def _shingles(text: str, n: int = 8) -> set[int]:
    """Hashed word n-grams — the unit of the near-duplicate check."""
    w = _norm(text).split()
    return {hash(tuple(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def _best_containment(texts: list[str], reference: list[str]) -> list[tuple[float, int]]:
    """For each text, its highest shingle containment against any reference text.

    Containment rather than Jaccard, because a resume reformatted or truncated by a
    different scrape is still the same person's document even when lengths differ.
    """
    ref = [_shingles(t) for t in reference]
    index: dict[int, list[int]] = collections.defaultdict(list)
    for i, s in enumerate(ref):
        for h in s:
            index[h].append(i)
    out = []
    for t in texts:
        s = _shingles(t)
        hits = collections.Counter(i for h in s if h in index for i in index[h])
        if not s or not hits:
            out.append((0.0, -1))
            continue
        i, n = hits.most_common(1)[0]
        out.append((max(n / len(s), n / len(ref[i])), i))
    return out


def load_split(key: str, split: str) -> pd.DataFrame:
    """Concatenate the parquet shards of one split of a raw source."""
    files = sorted((RAW / key).glob(f"*-{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {RAW / key} — run fetch")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --- A1 -------------------------------------------------------------------

def profile_fit() -> dict:
    """Effective size, label mix, pool feasibility, leakage, label conflicts."""
    print("\n=== A1 cnamuangtoun/resume-job-description-fit")
    tr, te = load_split("fit", "train"), load_split("fit", "test")
    out: dict = {"splits": {}}
    for name, df in (("train", tr), ("test", te)):
        per_jd = df.groupby("job_description_text").size()
        good = df[df.label == "Good Fit"].groupby("job_description_text").size()
        out["splits"][name] = {
            "rows": len(df),
            "unique_resumes": int(df.resume_text.nunique()),
            "unique_jds": int(df.job_description_text.nunique()),
            "label_share": {k: round(float(v), 4)
                            for k, v in df.label.value_counts(normalize=True).items()},
            "resumes_per_jd": {"mean": round(float(per_jd.mean()), 1),
                               "median": int(per_jd.median()), "max": int(per_jd.max())},
            "jds_with_5plus_resumes": int((per_jd >= 5).sum()),
            "jds_with_good_fit": int((good >= 1).sum()),
            "good_per_jd_median": int(good.median()),
            "resume_chars_median": int(df.resume_text.str.len().median()),
            "jd_chars_median": int(df.job_description_text.str.len().median()),
        }
        s = out["splits"][name]
        print(f"  [{name}] rows={s['rows']}  unique_resumes={s['unique_resumes']}  "
              f"unique_jds={s['unique_jds']}")
        print("    labels: " + "  ".join(f"{k}={v:.3f}" for k, v in s["label_share"].items()))
        print(f"    resumes/JD mean={s['resumes_per_jd']['mean']} "
              f"median={s['resumes_per_jd']['median']} max={s['resumes_per_jd']['max']}"
              f" | JDs>=5 resumes={s['jds_with_5plus_resumes']}"
              f" | JDs>=1 GoodFit={s['jds_with_good_fit']}")

    shared = set(tr.resume_text) & set(te.resume_text)
    out["resume_leakage_test_in_train"] = len(shared)
    out["resume_leakage_pct"] = round(100 * len(shared) / te.resume_text.nunique(), 1)
    out["jd_overlap"] = len(set(tr.job_description_text) & set(te.job_description_text))
    both = pd.concat([tr, te])
    out["conflicting_label_pairs"] = int(
        (both.groupby(["resume_text", "job_description_text"]).label.nunique() > 1).sum())
    print(f"  LEAKAGE: {out['resume_leakage_test_in_train']} test resumes also in train "
          f"({out['resume_leakage_pct']}% of test uniques)  <-- do NOT use the shipped split")
    print(f"  JD overlap train/test: {out['jd_overlap']}")
    print(f"  pairs with conflicting labels: {out['conflicting_label_pairs']}")

    # Relevance density decides which retrieval metrics are attainable at all.
    good_te = te[te.label == "Good Fit"].groupby("job_description_text").size()
    q = good_te.quantile([0.25, 0.5, 0.75])
    out["good_fit_per_test_jd"] = {"n_queries": int(len(good_te)), "min": int(good_te.min()),
                                   "q1": int(q[0.25]), "median": int(q[0.5]),
                                   "q3": int(q[0.75]), "max": int(good_te.max()),
                                   "mean": round(float(good_te.mean()), 1)}
    for k in (10, 20, 50):
        out[f"pct_queries_recall{k}_reachable_at_0.9"] = round(
            100 * float((good_te <= k / 0.9).mean()), 1)
    d = out["good_fit_per_test_jd"]
    print(f"  Good Fit per test JD (n={d['n_queries']}): median={d['median']} "
          f"Q1={d['q1']} Q3={d['q3']} max={d['max']}")
    print("  Recall@k >= 0.90 reachable for: " + "  ".join(
        f"@{k}={out[f'pct_queries_recall{k}_reachable_at_0.9']}%" for k in (10, 20, 50)))
    return out


# --- A2 -------------------------------------------------------------------

def profile_djinni() -> dict:
    """Size and the `Primary Keyword` join vocabulary shared by both sides."""
    print("\n=== A2 Djinni recruitment dataset")
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    kj = set(jd["Primary Keyword"].dropna())
    kc = set(cv["Primary Keyword"].dropna())
    shared = kj & kc
    top = (jd[jd["Primary Keyword"].isin(shared)]["Primary Keyword"]
           .value_counts().head(12))
    out = {
        "jds": len(jd), "cvs": len(cv),
        "primary_keyword": {"jd_side": len(kj), "cv_side": len(kc), "shared": len(shared)},
        "top_shared_families": {k: int(v) for k, v in top.items()},
        "cv_side_family_counts": {k: int(v) for k, v in
                                  cv["Primary Keyword"].value_counts().head(12).items()},
        "english_level": {k: int(v) for k, v in cv["English Level"].value_counts().items()},
        "jd_chars_median": int(jd["Long Description"].astype(str).str.len().median()),
        "cv_chars_median": int(cv["CV"].astype(str).str.len().median()),
    }
    print(f"  JDs={out['jds']}  CVs={out['cvs']}")
    print(f"  Primary Keyword: {len(kj)} JD-side, {len(kc)} CV-side, {len(shared)} shared")
    print("  top shared role families: " + ", ".join(top.index))
    print(f"  median chars: JD={out['jd_chars_median']}  CV={out['cv_chars_median']}")
    return out


# --- B1 -------------------------------------------------------------------

def _load_dataturks() -> list[dict]:
    docs = []
    for fn in ("traindata.json", "testdata.json"):
        for line in (RAW / "dataturks" / fn).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a handful of malformed lines in the upstream mirror
    return docs


def profile_dataturks() -> dict:
    """Span counts, PII classes and the broken-offset defect that blocks BIO conversion."""
    print("\n=== B1 DataTurks resume entities")
    docs = _load_dataturks()
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
    corpus = " ".join(d["content"] for d in docs)
    out = {
        "docs": len(docs), "spans": spans,
        "broken_offsets": bad, "broken_offsets_pct": round(100 * bad / spans, 1),
        "entities": dict(ents.most_common()),
        "residual_emails": len(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", corpus))),
        "residual_phone_like": len(set(re.findall(r"\+?\d[\d\s().-]{8,14}\d", corpus))),
        "doc_chars_median": int(pd.Series([len(d["content"]) for d in docs]).median()),
    }
    print(f"  docs={out['docs']}  spans={out['spans']}  "
          f"broken_offsets={bad} ({out['broken_offsets_pct']}%)")
    for k, v in ents.most_common():
        print(f"    {k:<24} {v}" + ("   <-- DIRECT PII" if k in {"Name", "Email Address"} else ""))
    print(f"  residual emails={out['residual_emails']}  "
          f"phone-like={out['residual_phone_like']}   <-- de-identify before use (D5)")
    return out


# --- B2 -------------------------------------------------------------------

def profile_skillspan() -> dict:
    """Tag inventory of the two span layers, which the project label scheme must map onto."""
    print("\n=== B2 jjzha/skillspan")
    out: dict = {"splits": {}}
    for split in ("train", "validation", "test"):
        df = load_split("skillspan", split)
        layers = {}
        for col in ("tags_skill", "tags_knowledge"):
            tags = collections.Counter(t for row in df[col] for t in row)
            layers[col] = {"B": tags.get("B", 0), "I": tags.get("I", 0), "O": tags.get("O", 0),
                           "spans": tags.get("B", 0)}
        out["splits"][split] = {
            "sentences": len(df),
            "tokens": int(sum(len(t) for t in df.tokens)),
            "layers": layers,
            "sources": {k: int(v) for k, v in df.source.value_counts().items()}
            if "source" in df else {},
        }
        s = out["splits"][split]
        print(f"  [{split}] sentences={s['sentences']} tokens={s['tokens']} "
              f"skill_spans={layers['tags_skill']['spans']} "
              f"knowledge_spans={layers['tags_knowledge']['spans']}")
    total_skill = sum(v["layers"]["tags_skill"]["spans"] for v in out["splits"].values())
    total_know = sum(v["layers"]["tags_knowledge"]["spans"] for v in out["splits"].values())
    out["total_skill_spans"], out["total_knowledge_spans"] = total_skill, total_know
    print(f"  TOTAL skill spans={total_skill}  knowledge spans={total_know}  "
          f"(vs 472 Skills spans in B1)")
    return out


# --- C1 / C2 --------------------------------------------------------------

def profile_livecareer() -> dict:
    """PDF/HTML corpus: format inventory, category mix and contamination against A1."""
    print("\n=== C1/C2 livecareer resume corpus (PDF + HTML)")
    root = RAW / "snehaanbhawal-resume-dataset"
    csv = pd.read_csv(root / "Resume.csv")
    pdfs = list(root.rglob("*.pdf"))
    html = load_split("livecareer", "train")
    out = {
        "csv_rows": len(csv), "pdf_files": len(pdfs),
        "pdf_categories": len({p.parent.name for p in pdfs}),
        "categories": {k: int(v) for k, v in csv.Category.value_counts().head(8).items()},
        "html_chars_median": int(csv.Resume_html.astype(str).str.len().median()),
        "text_chars_median": int(csv.Resume_str.astype(str).str.len().median()),
        "hf_mirror_rows": len(html),
        "pdf_bytes_median": int(pd.Series([p.stat().st_size for p in pdfs]).median()),
    }
    print(f"  Resume.csv rows={out['csv_rows']}  PDFs={out['pdf_files']} across "
          f"{out['pdf_categories']} category folders  HF mirror rows={out['hf_mirror_rows']}")
    print(f"  median chars: html={out['html_chars_median']} text={out['text_chars_median']}  "
          f"median PDF={out['pdf_bytes_median'] / 1024:.0f} KB")

    # Contamination check: a resume shared with A1 leaks the matching benchmark if this
    # corpus is used for NER training or as a distractor pool. Exact matching finds
    # nothing (the two scrapes format differently), so match on 8-gram containment.
    fit = pd.concat([load_split("fit", s) for s in ("train", "test")], ignore_index=True)
    a1 = list(fit.resume_text.unique())
    matches = _best_containment(list(csv.Resume_str.astype(str)), a1)
    scores = pd.Series([m[0] for m in matches])
    out["a1_uniques"] = len(a1)
    out["a1_exact_matches"] = int((scores >= 1.0).sum())
    out["a1_overlap_by_threshold"] = {str(t): int((scores >= t).sum()) for t in (0.9, 0.7, 0.5)}
    out["contamination_threshold"] = 0.7
    contaminated = csv.loc[scores >= 0.7, ["ID", "Category"]].copy()
    contaminated["containment"] = scores[scores >= 0.7].round(3).values
    out["contaminated_resumes"] = len(contaminated)
    out["contaminated_pct_of_a1"] = round(100 * len(contaminated) / len(a1), 1)

    INTERIM.mkdir(parents=True, exist_ok=True)
    exclusion = INTERIM / "c1_a1_contamination.csv"
    contaminated.sort_values("containment", ascending=False).to_csv(exclusion, index=False)
    out["exclusion_list"] = str(exclusion.relative_to(RAW.parent.parent))
    print(f"  overlap with A1 (8-gram containment, {len(a1)} A1 uniques): "
          + "  ".join(f">={t}: {n}" for t, n in out["a1_overlap_by_threshold"].items()))
    print(f"  -> {len(contaminated)} resumes ({out['contaminated_pct_of_a1']}% of A1) flagged at "
          f">=0.70; exclusion list -> {out['exclusion_list']}")
    return out


# --- C3 -------------------------------------------------------------------

def profile_resume_atlas() -> dict:
    """Confirm the text is pre-normalised and therefore clustering-EDA only."""
    print("\n=== C3 ahmedheakl/resume-atlas")
    df = load_split("resume-atlas", "train")
    sample = " ".join(df.Text.head(200))
    out = {
        "rows": len(df), "categories": int(df.Category.nunique()),
        "uppercase_chars_in_sample": sum(c.isupper() for c in sample),
        "punctuation_chars_in_sample": sum(c in ".,;:" for c in sample),
        "top_categories": {k: int(v) for k, v in df.Category.value_counts().head(8).items()},
    }
    print(f"  rows={out['rows']}  categories={out['categories']}")
    print(f"  uppercase chars={out['uppercase_chars_in_sample']}  "
          f"punctuation={out['punctuation_chars_in_sample']}")
    print("  -> pre-normalised (no case, no punctuation): clustering EDA only")
    return out


# --- D1 -------------------------------------------------------------------

def profile_esco() -> dict:
    """Alias inventory and the essential/optional split that seeds hard requirements."""
    print("\n=== D1 ESCO v1.2.1")
    d = RAW / "esco_dataset-v1.2.1-classification"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found — see data/README.md §3")
    sk = pd.read_csv(d / "skills_en.csv", low_memory=False)
    occ = pd.read_csv(d / "occupations_en.csv", low_memory=False)
    rel = pd.read_csv(d / "occupationSkillRelations_en.csv", low_memory=False)
    out = {
        "skills": len(sk), "occupations": len(occ), "occupation_skill_links": len(rel),
        "skill_type": {k: int(v) for k, v in sk.skillType.value_counts().items()},
        "alt_labels": int(sk.altLabels.dropna().str.count("\n").add(1).sum()),
        "relation_type": {k: int(v) for k, v in rel.relationType.value_counts().items()}
        if "relationType" in rel else {},
        "digital_skills": len(pd.read_csv(d / "digCompSkillsCollection_en.csv", low_memory=False)),
    }
    print(f"  skills={out['skills']}  occupations={out['occupations']}  "
          f"occupation-skill links={out['occupation_skill_links']}")
    print("  skillType: " + "  ".join(f"{k}={v}" for k, v in out["skill_type"].items()))
    print(f"  alternative labels (aliases for normalisation): ~{out['alt_labels']}")
    print("  relationType: " + "  ".join(f"{k}={v}" for k, v in out["relation_type"].items())
          + "   <-- seeds hard vs preferred requirements")
    return out


# --- D2 -------------------------------------------------------------------

def parse_skill_cell(cell: object) -> list[str]:
    """`job_skills` ships as a *stringified* Python list, not a list — parse it as one.

    Iterating the raw cell yields characters and silently produces a vocabulary of
    punctuation, so every consumer must go through here.
    """
    if isinstance(cell, str):
        try:
            return [s for s in json.loads(cell.replace("'", '"')) if isinstance(s, str)]
        except json.JSONDecodeError:
            return []
    if isinstance(cell, (list, tuple)) or hasattr(cell, "tolist"):
        return [s for s in list(cell) if isinstance(s, str)]
    return []


def skill_frequencies(df: pd.DataFrame) -> collections.Counter:
    """Posting counts per skill over the `data_jobs` corpus."""
    freq: collections.Counter = collections.Counter()
    for cell in df.job_skills.dropna():
        freq.update(parse_skill_cell(cell))
    return freq


def profile_data_jobs() -> dict:
    """Confirm there is no JD text, and extract the tech skill frequencies we do want."""
    print("\n=== D2 lukebarousse/data_jobs")
    df = load_split("data-jobs", "train")
    text_cols = [c for c in df.columns
                 if "description" in c.lower() and df[c].astype(str).str.len().median() > 200]
    freq = skill_frequencies(df)
    out = {
        "rows": len(df), "columns": list(df.columns),
        "description_text_columns": text_cols,
        "distinct_skills": len(freq),
        "top_skills": dict(freq.most_common(20)),
        "top_titles": {k: int(v) for k, v in df.job_title_short.value_counts().head(8).items()},
    }
    print(f"  rows={out['rows']}  columns={len(df.columns)}")
    print(f"  columns carrying JD text: {text_cols or 'NONE'}   <-- not a JD corpus")
    print(f"  distinct skills={out['distinct_skills']}; top 12: " +
          ", ".join(f"{k}({v})" for k, v in list(freq.most_common(12))))
    return out


# --- A4 (rejected) --------------------------------------------------------

def recover_ats_boundary() -> dict:
    """Reproduce the evidence for rejecting A4: derivative, and the `[SEP]` claim is false."""
    print("\n=== A4 0xnbk/resume-ats-score-v1-en — rejection evidence")
    ats = pd.concat([load_split("ats", s) for s in ("train", "validation")], ignore_index=True)
    fit = pd.concat([load_split("fit", s) for s in ("train", "test")], ignore_index=True)
    n_sep = int(ats.text.str.contains("[SEP]", regex=False).sum())
    by_head: dict[str, list[str]] = collections.defaultdict(list)
    for r in {_norm(r) for r in fit.resume_text.unique()}:
        by_head[r[:100]].append(r)
    hits = sum(1 for t in ats.text
               if any(_norm(t).startswith(r) for r in by_head.get(_norm(t)[:100], [])))
    out = {"rows": len(ats), "rows_with_documented_sep": n_sep, "rows_starting_with_a1_resume": hits,
           "derivative_pct": round(100 * hits / len(ats), 1)}
    print(f"  rows containing the documented '[SEP]': {n_sep}")
    print(f"  rows starting with a cnamuangtoun resume: {hits}/{len(ats)} "
          f"({out['derivative_pct']}%)  -> derivative, not independent")
    return out


CHECKS = {
    "fit": profile_fit,
    "djinni": profile_djinni,
    "dataturks": profile_dataturks,
    "skillspan": profile_skillspan,
    "livecareer": profile_livecareer,
    "resume-atlas": profile_resume_atlas,
    "esco": profile_esco,
    "data-jobs": profile_data_jobs,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+", choices=sorted(CHECKS), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recover-ats-boundary", action="store_true",
                    help="rejected source A4; needs `fetch --source ats` first")
    ap.add_argument("--metrics", type=Path, default=PROFILE_METRICS,
                    help="where to write the machine-readable figures")
    args = ap.parse_args()

    keys = list(CHECKS) if args.all else args.check
    metrics, skipped = {}, []
    for key in keys:
        try:
            metrics[key] = CHECKS[key]()
        except FileNotFoundError as exc:
            print(f"\n=== {key}: SKIPPED — {exc}")
            skipped.append(key)
    if args.recover_ats_boundary:
        try:
            metrics["ats"] = recover_ats_boundary()
        except FileNotFoundError as exc:
            print(f"\n=== ats: SKIPPED — {exc}")

    if metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(
            {"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
             "checks": metrics}, indent=2) + "\n", encoding="utf-8")
        print(f"\nmetrics -> {args.metrics}")
    if skipped:
        print(f"skipped (not downloaded): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
