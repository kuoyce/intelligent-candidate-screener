# `data/` — acquisition guide

Hiring Desk v1 additionally stores private operational application state under
`processed/hr-portal/`: reviewer accounts, JD revisions, review/ranking history
and locally captured demo emails. Empty schemas regenerate at startup; actual
reviewer-authored drafts and history require a private backup and are not
reconstructible from dataset fetch/build commands. These records are excluded
from benchmark labels/training. See the
[operational workspace card](../docs/data/cards/HR-operational-workspace.md).

The v1 `processed/hr-portal/entity-cache/` is rebuildable from authorized
redacted JD/CV sources using the installed pinned GLiNER-medium assets and
`runtime/gliner-medium`. Cache keys include owner/workspace, source digest and
model/label/chunking configuration; the application rebuilds entries on demand.
RoBERTa-SQuAD2 uses its existing `processed/chatbot/qa/extractive/` checkpoint in
`runtime/transformer`. No model downloads or training run at app startup, and
these model outputs never become benchmark or human entity labels.

`data/datapack/` is the Docker app's data folder (D68): the 24 project files the Hiring Desk
reads, copied from this directory, plus the Docker deployment's own workspace
(`datapack/processed/hr-portal/`) and caches. It is made with
`uv run python -m candidate_screener.datapack create` (`--with-workspace` also copies the local
`processed/hr-portal/`), checked with `verify` and passed on with `handover`, which leaves the
workspace and caches out. Like the rest of `data/` it is git-ignored and never committed; keep it,
and above all its workspace, private. See the [Docker guide](../docs/docker.md).

Nothing in this directory is committed except this file. Everything below reconstructs the
**676 MB** working set. See [`docs/data/data-catalog.md`](../docs/data/data-catalog.md) for
what each source is, its licence and its known defects, and
[`docs/data/cards/`](../docs/data/cards/) for the per-source detail.

## 1. Automated (no credentials)

### Guided HR portal: user-supplied derived exports

The local reviewer consumes existing files under `data/processed/phase2-rerank-output/`:
`a2-cv-extraction.csv` (600CV rows), `a2-jd-extraction.csv` (60JD rows), and
`a2-x2-scores.csv` (600ranked pairs,10perJD). These are user-provided derived model
exports, not new raw sources or gold. The portal never regenerates them, trains a
replacement or exposes `final_label`. A producer rebuild command/run timestamp was
not available in this repository inspection; restore the supplied exports with
their documented digests when absent. This limitation is separate from raw corpus
acquisition below. See the [portal record](../src/llm-plan/2026-10-19-hr-conversation-design/05-implementation.md).

Text previews use only ID/text columns from the existing cache
`data/processed/matching/a2-pairs-v2-539ddf32117c.parquet`. Missing text leaves
previews/packaging unavailable. Model table mentions lack verified source offsets;
knowledge, degree and CV titles are not fabricated from role-family predictions.

Local account hashes, SQLite sessions and redacted report artifacts live under
ignored `data/processed/hr-portal/` with private permissions. Tokens never enter
saved sessions. Synthetic demo storage is separate and contains invented documents.
These caches do not replace company identity, retention, encryption or mail services.

Historical Qwen prototype assets are separate from these corpus downloads. Run
`uv run python -m candidate_screener.llm.qwen_assets` to provision the workshop's
`Qwen/Qwen2.5-0.5B-Instruct` under `data/processed/chatbot/qwen/`. Its manifest
pins the model commit, file digest and local inference configuration. Repeated
provisioning reuses that exact revision. These are pretrained inference weights,
not a new annotated corpus. For recorded R2 reranker weights, restore the selected
training cache first, then run `uv run python -m candidate_screener.app.rerank_assets`;
manifest export cannot recreate missing trained weights. See the chatbot setup
in [README](../README.md#streamlit-recruiter-chatbot).

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

As of 9 Oct 2026, notebook 13 requests a fifth GLiNER label,
`EXPERIENCE_YEARS`, for explicit professional durations. Changed labels and adapter
version produce separate cached runs; historical four-label artifacts are retained.
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

### Human entity gold (D56)

This is a separate human span task, not reopened Fit annotation. Confirm the
source, four categories, batch cap and annotator IDs before selecting real data.
The following is the proposed configuration, not an already-created campaign:

```bash
uv run python -m candidate_screener.entity_annotation.create --source a1 --labels KNOWLEDGE SKILL TITLE DEGREE --counts 80 20 20 --annotators reviewer
```

Use `--source a2-train` only if that population is selected. Creation freezes
original-source hashes, redacted display mappings, duplicate groups, document
splits, guideline and assignments under `data/processed/entity-annotation/campaigns/`.
It does not extract entities, train a model, or infer gold from existing Fit labels.
Start the UI using the printed manifest path and registered annotator ID:

```bash
uv run python -m candidate_screener.entity_annotation.ui --campaign PATH_TO_MANIFEST --annotator reviewer
uv run python -m candidate_screener.entity_annotation.export --campaign PATH_TO_MANIFEST --annotator reviewer --destination data/processed/entity-annotation/exports/REVISION_NAME
```

Only explicitly reviewed documents export as gold; missing, draft, skipped and
uncertain annotations are not negatives. Every saved review is append-only and
revision-checked. Export directories are immutable and include JSONL/Parquet
digests. No raw source text or review files belong in git. Back up the manifest,
review journal and revision exports: human decisions cannot be reconstructed by
rerunning inference or fetching raw documents. For a safe synthetic-only demo,
run `uv run python -m candidate_screener.entity_annotation.ui --demo --annotator reviewer`.

### Medium-v2.5 sentence-aware extraction

This optional experiment has a separate locked runtime and pinned assets. Recreate
them with the following commands; model weights and source evidence stay ignored:

```bash
uv sync --project runtime/gliner-medium
uv run python -m candidate_screener.extraction.medium_assets
PYTHONPATH=src uv run --project runtime/gliner-medium --no-sync python -c 'import json; from candidate_screener.extraction.medium_model import runtime_versions; from candidate_screener.extraction.medium_assets import MODEL_DIR; (MODEL_DIR/"runtime-versions.json").write_text(json.dumps(runtime_versions(),indent=2,sort_keys=True)+"\n")'
uv run python -m candidate_screener.extraction.medium_pilot
uv run python -m candidate_screener.extraction.medium_run
```

The model ID is `gliner-community/gliner_medium-v2.5`, pinned to revision
`88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da`. Its checkpoint-supplied tokenizer
and backbone configuration are pinned and hashed. The loader uses the checkpoint's
128003-token vocabulary rather than the backbone's reserved 128100 embeddings.
Inference stays offline. Sentence boundaries use spaCy's rule-only sentencizer;
no language model is downloaded. Original CV/JD text is unchanged.

Results live in `data/processed/extraction/medium-sentence-campaigns/`. Import
`load_campaign` from `candidate_screener.extraction.medium_run` for this profile;
pass an explicit campaign ID to inspect partial results. Its `latest.json` appears
only after every document has a terminal processing state. Failed and incomplete
states remain explicit; terminal does not mean successful. Notebook 19 writes
aggregate comparison records under `output/extraction/medium-sentence-comparison/`.

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
