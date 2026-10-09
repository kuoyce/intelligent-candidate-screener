# Implementation and findings

6 October 2026. The user approved TE1–TE4. Branch
`feat/transformer-entity-extraction-2026-10-06`; approved plan commit `31a8398`
preceded implementation. Work was delegated across adapter/runtime, pipeline and
notebook/display agents, with independent review before final inference.

## Delivered output and model

Notebook 11 now describes all 15 stored fields and displays five actual reviewed
seed-rule entity mentions, contact-redacted inside the notebook. It preserves
the existing notebook source cells and adds the requested view. Whole source
documents and unreviewed neural surfaces are not displayed.

Notebook 13 runs/caches GLiNER inference, reports completion and per-type counts,
compares rules on identical completed documents, validates artifacts, and displays
a redacted synthetic neural example. Actual neural evidence is local parquet.

Model: `urchade/gliner_small-v2.1`, pinned revision
`4e091416cf7c3481db542c2a3d26156916f3a47f`; encoder tokenizer/config:
`microsoft/deberta-v3-small`, revision
`a36c739020e01763fe789b4b85e2df55d6180012`. Model/tokenizer/config artifacts
are SHA-256 recorded and git-ignored. Downloads are reconstructed through
`extraction.model_assets`. Inference operates offline using local assets.

Four custom model descriptions map to KNOWLEDGE, SKILL, TITLE and DEGREE. Scores
are finite bounded raw values, not calibrated probabilities. Canonical IDs remain
null; the model does not supply ESCO mappings. Negation/learning context still uses
a separately versioned rule heuristic: this is neural span extraction with hybrid
context handling, not an end-to-end neural screening decision.

The adapter budgets the exact GLiNER label prompt plus subword tokens, windows at
original word boundaries, and retains overlapping source intervals. Actual word
and token limits are 384; overlap is 32 words. Exact repeated span/label predictions
keep their maximum score; other overlaps remain explicit. Invalid predictions or
oversized words are quarantined and incomplete/time-limited fields are reported.

## Compatibility deviation

The proposed optional stack could not resolve on this Intel Mac: current PyTorch
has no macOS x86 wheel, while compatible Transformers requires Hugging Face Hub
below 1.0 and the main project requires >=1.28. No core dependencies were downgraded.
Instead, `runtime/transformer/` contains a separate uv project/lock with PyTorch
2.2.2, Transformers 4.48.3, GLiNER 0.2.21 and NumPy 1.26.4. Required-environment
markers resolve Intel-compatible wheels, including ONNX Runtime 1.23.2.

A persistent local worker exchanges documents/predictions over private local pipes,
loads only pinned assets, suppresses raw third-party output and uses four CPU
threads. Actual worker versions are checked against the runtime manifest and
recorded. Main-environment dependency files are unchanged. This compatibility
choice stays within the approved dependency/model installation scope. The worker
is explicitly closed after inference and after the notebook's synthetic example.

## Real runs

Smoke: five training CVs and five training JDs, **95 spans**, all 30 fields complete,
88.93 seconds including initialization. It used the earlier smoke pipeline digest;
subsequent finite-budget/runtime guards are part of the final run below.

Final run: `2c591cc8bff7fc9e7189-20261005T164947399919Z`, seed 0, threshold 0.5,
CPU, 30-minute budget. Completed **200/200 documents and 600/600 fields** in
**183.81 seconds**. No quarantined predictions or incomplete fields.

| Type | GLiNER spans | Rule spans on same IDs | Exact shared | Documents with neural mentions |
|---|---:|---:|---:|---:|
| 100 CVs | 1,077 | 404 | 243 | 100 |
| 100 JDs | 755 | 327 | 191 | 99 |

All **1,832** neural records have original field-relative exact offsets and
unique deterministic IDs for their returned values. Higher mention counts and
disagreement cannot establish superiority: the additional spans may be useful
evidence or false positives. No extraction gold precision/recall/F1, canonical
accuracy or ranking benefit is reported. No BERT fine-tuning was performed.

The raw score range is 0.5000–0.9859 for CVs and 0.5004–0.9687 for JDs. The 0.5
threshold is an exploratory setting, not an independently validated cut-off.
Memory observation is explicitly the parent process peak RSS only and excludes
the isolated worker; it is not a total inference-memory benchmark.

Real records: `data/processed/extraction/transformer-runs/<run_id>/entities.parquet`.
Adjacent manifests record sampled IDs and provenance; `field-coverage.json` records
processed intervals. Committed `output/extraction/transformer-metrics.json` contains
aggregates/config/hashes only and omits real text, sampled IDs and local model path.
Earlier rule outputs and matching/annotation manifests remain unchanged.

## Validation and remaining gates

The full main-environment suite reports **246 passed, 2 failed** (126.52 seconds).
Both failures remain the existing `test_baseline_golden.py` reproduction/environment
tests: vocabulary SHA-256 and 19 model fields differ. `baselines.run --check`
reproduced split/config/floor, failed the same vocabulary/model fields, and reported
unchanged environment. `data.verify --derived` retains its existing queue-resume
failure (200 already-judged pairs); available other checks passed with absent
derivatives explicitly skipped. No baseline record or queue was refreshed.

Regression demonstrations used controlled in-memory defects: bad original offsets,
unbounded/invalid model scores, lost window tails, and disabled artifact corruption
validation each caused a named test failure. Restored tests pass without downloaded
weights. Additional guards check finite budgets and live runtime versions.
Independent review found no train leakage or synthetic-as-gold claims and requested
the finite-budget/live-version guards, which were implemented before the final run.

Final notebook checks: notebook 11 executed seven code cells and notebook 13 nine;
nbformat schema validation and saved-output redaction checks passed. Maximum
per-cell outputs are 727 and 587 characters respectively; previews are capped at
five rows. Combined extraction/neural/notebook tests report **55 passed** after
execution. The final transformer CLI `--sample-size 100 --check` passed all
checkpoint/artifact hashes, training membership, field coverage, original offsets
and unique IDs for 1,832 spans. It intentionally does not demand cross-device
bitwise score replay. `git diff --check` passed.

TE1–TE4 are resolved. TE5 and original EE2–EE5 remain future gates: restore and
validate labelled extraction data, agree supervised training budget, independently
evaluate mappings/relations, and decide role/date-linking policy. No new human
annotation or deferred A1 judge output was used.
