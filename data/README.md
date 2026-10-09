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

## Local entity extraction artifacts

Notebooks 10–12 call `candidate_screener.extraction` to rebuild local extraction
artifacts under `data/processed/extraction/` from the Djinni raw parquet files and
the committed `docs/data/manifests/a2-partition.csv`. Only training-region documents
are eligible. The default sample is 1,000 CVs and 1,000 JDs at seed 0; the package
supports configurable sample sizes. Source fields stay immutable and offsets are
relative to their original field text. Evidence surfaces and sample documents are
local artifacts and must not be committed or printed in notebook outputs.

Run provenance records input/partition hashes, seed, sampled document IDs, fields
and extractor/vocabulary versions. A seed vocabulary is used when ESCO is unavailable;
its identifiers are not ESCO identifiers. These unlabelled documents support coverage
and integrity checks only. Missing DataTurks/SkillSpan sources block gold accuracy
evaluation, which requires separate adapters and frozen evaluation protocols.

See [the approved extraction plan](../plan/2026-10-05-entity-extraction-notebooks/README.md)
for implementation scope and [notebook 10](../notebooks/10-entity-extraction-inputs.ipynb)
for reproducible input preparation.

## Handling notes

### Transformer extraction artifacts (6 Oct 2026)

Install the isolated inference environment using `uv sync --project runtime/transformer`.
Run `uv run python -m candidate_screener.extraction.model_assets` to download the
pinned GLiNER small-v2.1 checkpoint plus DeBERTa tokenizer/config artifacts into
`data/processed/extraction/models/gliner-small-v2.1/`. The download records both
revisions and per-file SHA-256. Runtime dependency versions are recorded separately
in `runtime-versions.json`; the inference worker checks them against its live versions.
Recreate that file after syncing the runtime with:

```bash
uv run --project runtime/transformer --no-sync python -c 'import json; from pathlib import Path; from importlib.metadata import version; Path("data/processed/extraction/models/gliner-small-v2.1/runtime-versions.json").write_text(json.dumps({n:version(n) for n in ("torch","transformers","gliner","tokenizers","numpy","huggingface-hub")}, indent=2, sort_keys=True)+"\n")'
```

`uv run python -m candidate_screener.extraction.transformer_run --sample-size 100`
performs local inference and saves spans, sampled IDs/provenance and field coverage
in `data/processed/extraction/transformer-runs/`. Matching compatible reruns reload
validated artifacts; they do not repeat inference or overwrite rule results.
`--check` validates hashes, original field offsets and training membership without
requiring bitwise score replay across devices. A timed-out run is explicitly partial.
There is no remote document inference or labelled corpus accuracy in this phase.

## Additional handling notes

### Extraction EDA

Analyze the generated transformer run with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/14-entity-extraction-eda.ipynb
```

It reads `output/extraction/transformer-metrics.json`, validates the corresponding
local artifacts, and reconstructs rules on precisely the same sampled training
documents. No additional neural inference is performed. Figures and descriptive
aggregate tables are exported under `output/extraction/eda/`. The notebook displays
bounded real entity/context examples; it does not export complete source documents.

- **DataTurks contains direct PII** (`Name`, `Email Address` are annotated classes, plus residual
  emails and phone numbers in the raw text). De-identify before use — see action plan §3.1.
- **40 of the C1 resumes (6.2%) also appear in the A1 benchmark.** Exclude them from any NER
  training set or distractor pool built off C1, or they leak into the matching evaluation.
- **Do not use the A1 shipped train/test split** — 99.8% of its test resumes appear in train.
  Build the doubly-disjoint split described in action plan §3.2.
