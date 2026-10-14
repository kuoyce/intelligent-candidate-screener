# Five-label GLiNER extraction across all fit pairs

Status: user approved 14 October 2026; expanded explicitly to all pairs in both train and test parquet files.

## Population and configuration

Measured inventory: train 3,990 pairs (446 CVs, 245 JDs); test 659 pairs (158 CVs, 100 JDs). Combined 949 unique documents, 4,551,699 characters, zero conflicting texts per ID. Extract each unique (side, document ID) once, attach results to every original pair, retain zero documents in denominators.

Freeze installed pinned GLiNER, five current prompt categories KNOWLEDGE, SKILL, TITLE, DEGREE, EXPERIENCE_YEARS, threshold 0.5, CPU four threads, existing chunking. No fitting, prompt/threshold tuning or labels/metadata at inference. Five categories does not force five mentions.

## Deliverables and design

Package extraction.fit_run: strict source inventory, resumable per-document inference with atomic local checkpoints, exact source offsets and coverage, source/model/runtime/code hashes. Completed-zero, incomplete, failed and pending states remain distinct. Per-document budget 120 seconds. Resume only identical source/config/code; no blending retries. Real source text, IDs and evidence stay under ignored data. Aggregate reports/config may be committed.

Notebook 18-fit-gliner-comparison.ipynb: both splits and all five categories, total/unique mentions, zero-document rates, raw scores, completion and timing; distributions/charts like notebook 15. Include the previous real CV/JD pair with bounded redacted entity/context examples and comparison to earlier15-CV/zero-JD run; previous limited provenance is not strict replay gold. Previews max5rows/2000chars, no full documents. Exploratory Fit-conditioned summaries state A1 labels do not reproduce (A13). Fit labels do not supply entity precision/recall/F1. No new matching accuracy claim or final-set tuning. Update docs and implementation record.

## Workflow and verification

Dedicated branch, commit approved plan alone; preserve unrelated work. Delegate campaign/notebook implementation and independent review. Time deterministic pilot using exact config, estimate all949docs. State estimate before launch if over one hour. Jobs over5minutes start once in background with saved log, exit marker and peak RSS; end turn while running, no polling/babysitting. Resume campaign by identity.

Synthetic tests for conflicting identities, source corruption, zero denominators, incomplete/failed states, exact offsets, resume invalidation; demonstrate guarded regression. Execute notebook, full suite once with saved log, runner checks for affected records (cold before complete). Register new --check if introduced. No training/download/annotation/old record refresh.

## Open questions

FP1 resolved: all train/test pairs userapproved. FP2 scope: extraction operational/distribution comparisons, not gold entity or matching accuracy. FP3 runtime pending timed pilot; proceed with measured estimate.
