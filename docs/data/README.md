# `docs/data` — the data documentation set

Everything the project knows about its data: what was evaluated, what was adopted, what is
on disk right now, and what each source is known to get wrong.

| File | What it is |
|---|---|
| [`data-catalog.md`](data-catalog.md) | The catalog — every source evaluated, tiered adopt / adopt-with-caveats / reject, with licence, PII status and verified figures |
| [`cards/`](cards/) | One dataset card per adopted source: provenance, schema, verified counts, known defects, and how it must be used |
| [`acquisition-manifest.json`](acquisition-manifest.json) | Machine-readable record of the download — per-file sizes, row counts and SHA-256 digests. Written by `verify` |
| [`profile-metrics.json`](profile-metrics.json) | Machine-readable **Verified** figures. Written by `profile`; the cards and notebooks cite it rather than restating prose |

`data/` itself is git-ignored, so these two JSON files are how a result stays traceable to
the exact bytes it was computed on.

## Cards

| Card | Catalog | Tier | Licence |
|---|---|---|---|
| [A1 · resume–job-description fit](cards/A1-resume-job-description-fit.md) | A1 | 1 | None declared |
| [A2 · Djinni recruitment dataset](cards/A2-djinni.md) | A2 | 1 | MIT |
| [B1 · DataTurks resume entities](cards/B1-dataturks.md) | B1 | 2 | None declared |
| [B2 · SkillSpan](cards/B2-skillspan.md) | B2 | 1 | CC-BY-4.0 |
| [B3 · green](cards/B3-green.md) | B3 | 2 | CC-BY-4.0 |
| [C1 · livecareer PDFs](cards/C1-livecareer-pdf.md) | C1 | 1 | Publisher terms |
| [C2 · livecareer HTML](cards/C2-livecareer-html.md) | C2 | 1 | None declared |
| [C3 · ResumeAtlas](cards/C3-resume-atlas.md) | C3 | 2 | MIT |
| [D1 · ESCO v1.2.1](cards/D1-esco.md) | D1 | 1 | EC ESCO terms |
| [D2 · data_jobs](cards/D2-data-jobs.md) | D2 | 2 | Apache-2.0 |

Rejected sources (A4, B4, D3, D4) are documented in the catalog only — the rejection and its
evidence are the record.

## Reproducing all of it

```bash
uv run python -m candidate_screener.data.fetch   --all   # download (~676 MB)
uv run python -m candidate_screener.data.verify  --all   # bytes: files, rows, digests
uv run python -m candidate_screener.data.profile --all   # meaning: the Verified figures
```

`verify` fails if a row count drifts from the figure recorded in the registry, which means a
publisher has re-uploaded and every number here needs re-checking before it is trusted.

## Two rules that outlive this phase

1. **B1 carries direct PII** (`Name` and `Email Address` are annotated classes). De-identify
   before use, and never commit un-redacted content — including notebook outputs.
2. **44 C1/C2 resumes overlap the A1 benchmark.** Exclude
   `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built
   off that corpus.
