# `data/` — acquisition guide

Nothing in this directory is committed except this file. Everything below reconstructs the
**676 MB** working set. See [`docs/data/data-catalog.md`](../docs/data/data-catalog.md) for
what each source is, its licence and its known defects, and
[`docs/data/cards/`](../docs/data/cards/) for the per-source detail.

## 1. Automated (no credentials)

```bash
uv run python -m candidate_screener.data.fetch --all
```

Pulls A1 (fit pairs), A2 (Djinni JDs + CVs), B1 (DataTurks), B2 (SkillSpan), B3 (green),
C2 (livecareer HTML mirror), C3 (ResumeAtlas) and D2 (data_jobs) into `data/raw/`.

## 2. Manual — Kaggle PDFs (source C1)

Needed for the §5.2 PDF ingestion component. Requires a Kaggle account and API token.

```bash
kaggle datasets download -d snehaanbhawal/resume-dataset \
  -p data/raw/snehaanbhawal-resume-dataset --unzip
```

Expected: `Resume.csv` (2,484 rows) **and** `data/<CATEGORY>/<ID>.pdf` — **2,484 PDFs across 24
category folders**. A browser download works too, but extract the *whole* archive; taking only the
CSV leaves you without the PDFs, which are the entire point of this source.

On WSL, a browser download leaves one `*:Zone.Identifier` NTFS stream per file. Remove them:

```bash
find data -type f -name '*:Zone.Identifier' -delete
```

## 3. Manual — ESCO v1.2.1 (source D1)

1. Go to https://esco.ec.europa.eu/en/use-esco/download
2. Register (free), accept the terms, select **CSV**, language **English**
3. Extract to `data/raw/esco_dataset-v1.2.1-classification/`

Use 7-Zip on Windows — the site warns that the built-in extractor produces empty files.

Expected: `skills_en.csv` (13,960 rows), `occupations_en.csv` (3,043),
`occupationSkillRelations_en.csv` (126,051).

## 4. Verify

```bash
uv run python -m candidate_screener.data.verify  --all   # files, row counts, SHA-256 digests
uv run python -m candidate_screener.data.profile --all   # the Verified figures in the catalog
```

`verify` is an acceptance test: it fails if a row count drifts from the figure recorded in
`candidate_screener.data.sources`, which means a publisher has re-uploaded and every number in
the catalog needs re-checking before it is trusted. It writes
`docs/data/acquisition-manifest.json`; `profile` writes `docs/data/profile-metrics.json`.

## Expected layout

Verified 19 Aug 2026 — every figure below is asserted by `verify --all`:

```
data/
  raw/
    fit/                                  12.3 MB   A1  8,000 resume–JD fit pairs
    djinni-jd/                           145.9 MB   A2  141,897 IT job descriptions
    djinni-cv/                           237.4 MB   A2  210,250 anonymized CVs
    dataturks/                             1.2 MB   B1  220 resumes, 3,556 entity spans
    skillspan/                             0.7 MB   B2  11,543 sentences, 9,617 spans
    green/                                 1.1 MB   B3  9,968 sentences (held in reserve)
    snehaanbhawal-resume-dataset/        118.3 MB   C1  2,484 resumes + 2,484 PDFs (manual)
    livecareer/                           20.0 MB   C2  2,484 resumes as HTML
    resume-atlas/                         23.6 MB   C3  13,389 resumes (pre-normalised)
    esco_dataset-v1.2.1-classification/   40.3 MB   D1  13,960 skills (manual)
    data-jobs/                            75.3 MB   D2  785,741 postings, skills only
  interim/     parsed / de-identified / repaired
  processed/   model-ready splits and pools
  vocab/       ESCO + skill-frequency extracts
```

`interim/`, `processed/` and `vocab/` are populated in Phase 3 and are empty apart from
`interim/c1_a1_contamination.csv`, written by `profile --check livecareer`.

## Handling notes

- **DataTurks contains direct PII** (`Name`, `Email Address` are annotated classes, plus residual
  emails and phone numbers in the raw text). De-identify before use — see action plan §3.1.
- **40 of the C1 resumes (6.2%) also appear in the A1 benchmark.** Exclude them from any NER
  training set or distractor pool built off C1, or they leak into the matching evaluation.
- **Do not use the A1 shipped train/test split** — 99.8% of its test resumes appear in train.
  Build the doubly-disjoint split described in action plan §3.2.
