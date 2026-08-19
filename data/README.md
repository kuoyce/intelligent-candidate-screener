# `data/` — acquisition guide

Nothing in this directory is committed except this file. Everything below reconstructs the
~483 MB working set. See `plan/2026-08-19-data-strategy/02-data-catalog.md` for what each source
is, its licence and its known defects.

## 1. Automated (no credentials)

```bash
uv run python plan/2026-08-19-data-strategy/scripts/fetch_sources.py --all
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
uv run python plan/2026-08-19-data-strategy/scripts/profile_sources.py --all
```

This reproduces every figure marked **Verified** in the data catalog and acts as an acceptance
test on the download. If the numbers drift, a publisher has re-uploaded and the catalog needs
re-verifying before the numbers are trusted.

## Expected layout

```
data/raw/
  fit/                                    12 MB   A1  resume–JD fit pairs
  djinni-jd/                             140 MB   A2  141,897 IT job descriptions
  djinni-cv/                             227 MB   A2  210,250 anonymized CVs
  skillspan/                             648 KB   B2  skill/knowledge spans
  dataturks/                             1.2 MB   B1  220 resumes, entity spans
  esco_dataset-v1.2.1-classification/     50 MB   D1  skills taxonomy (manual)
  snehaanbhawal-resume-dataset/           54 MB   C1  2,484 resumes + PDFs (manual)
```

## Handling notes

- **DataTurks contains direct PII** (`Name`, `Email Address` are annotated classes, plus residual
  emails and phone numbers in the raw text). De-identify before use — see action plan §3.1.
- **40 of the C1 resumes (6.2%) also appear in the A1 benchmark.** Exclude them from any NER
  training set or distractor pool built off C1, or they leak into the matching evaluation.
- **Do not use the A1 shipped train/test split** — 99.8% of its test resumes appear in train.
  Build the doubly-disjoint split described in action plan §3.2.
