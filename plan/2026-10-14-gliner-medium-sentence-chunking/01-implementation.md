# Implementation and validation record

Status: implementation and pilot validated; full campaign and final reporting pending.
Approved plan commit: eeb342e. Branch:
feat/gliner-medium-sentence-chunking-2026-10-14.

## Architecture and provenance

Delegated sentence planner, model acquisition/loader/runtime, reporting and
independent review. New medium modules leave the baseline inference code unchanged.
The pinned model is gliner-community/gliner_medium-v2.5 revision
88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da. Its supplied tokenizer and pinned backbone
configuration are hashed with the selected float32 weights. The loader strictly
loads all weights offline, using the checkpoint's 128003 embeddings instead of
the backbone configuration's reserved 128100. A regression demonstrates that
omitting this compatibility correction fails.

Sentence planning uses original contiguous substrings, sentence/line boundaries,
actual GLiNER prompt preparation, 16-token headroom and word limits. Oversized
sentences use overlapping whole-word windows; unrepresentable words are quarantined.
Medium campaign manifests include strategy/code/runtime/tokenizer/source identity.
Separate checkpoints and latest pointer prevent mixing with small-model results.
Partial pilots do not publish a completed latest pointer. Evidence remains exact
original-source slices; source coverage is validated before checkpoint publication.

Independent review identified absent runtime manifest fallback and hidden
initialization diagnostics. Both are corrected: complete runtime version records
are mandatory, and private initialization logs retain tracebacks. Inference
diagnostics retain only stack locations and exception classes, without source text.

## Baseline and operational validation

The original campaign 901a26c9e785920223b07c9b finished with exit0: all949documents,
913 complete with entities, 36 complete-zero, 13,946 spans; inference sum2012.72s.
Resumed invocation maximum RSS1,556,242,432bytes. Its elapsed wall time includes
session suspensions and must not be treated as an uninterrupted speed benchmark.
Log: logs/fit-gliner-resume.log. Existing baseline records/checkpoints are retained.

Focused tests initially30passed/1optional-spaCy skipped in the base environment.
Logs: logs/medium-focused-tests.log, medium-assets-tests.log,
medium-model-tests.log, sentence-chunking-tests.log, medium-report-tests.log.
Disabled guards demonstrably fail: medium-assets-regression.log,
medium-model-regression.log, sentence-chunking-regression.log,
medium-report-regression.log and medium-coverage-regression.log.
Cold transformer_inspection through the registry runner passed in19.2s,
validating hashes, exact source offsets, coverage and training membership.
Logs: logs/checks/transformer_inspection.log and summary.tsv.

## Pilot compatibility findings and remaining work

Offline medium model smoke passed. First actual sentence-aware pilot failed
before predictions because spaCy's CLI imports click, absent from the transitive
dependency set. Pin click explicitly in the separate medium runtime, resync and
refresh its runtime manifest before rerunning the pilot. Preserve failed logs
logs/medium-pilot.log and logs/medium-pilot-retry.log. This is a runtime compatibility
correction, not a change to preprocessing, thresholds, model or experimental scope.

The explicit click8.1.8 pin and runtime resync resolved the spaCy import failure.
The real sentence fixture passed and preserves all eight technical mentions.
The corrected pilot processed four deterministic length-quartile documents:
two CVs and two JDs, 16,358characters, 96entities, all complete. Initialization12.11s,
inference17.95s; length-scaled estimate5,007s (~83min), communicated before launch.
Maximum RSS2,018,185,216bytes. Logs: logs/medium-pilot-final.log and medium-pilot.json.
This estimate is not a bound; sentence fallback and the120s soft budget are frozen.

Full suite:1,035passed,9skipped,3failed in152.21s, logs/medium-full-pytest.log.
Two failures are frozen TF-IDF baseline reproduction: vocabulary digest differs,
and19model fields differ; the baseline source files are unchanged by this work.
Do not rewrite frozen scores or suppress their tests. The third failure is an
error saved in notebook18 by execution before the baseline latest pointer existed;
regenerate that notebook now that baseline inference has completed.

Next: launch the one authorized full medium campaign; execute notebook19 and audit
aggregate records; report the unresolved frozen-baseline reproduction failures.
Do not mark
complete before all validation and reporting finishes. No training, new annotation,
final-set scores or entity-accuracy claims are introduced.
