# Implementation and validation

5 October 2026. The user approved EN1–EN4. Branch:
`feat/entity-extraction-notebooks-2026-10-05`; plan commit `835ddd9` preceded code work.

## Delivered behavior

Notebooks 10, 11 and 12 now prepare local A2 training inputs, perform actual
rule-based extraction, and validate artifacts with explicit supervised-evaluation
gates. Package `candidate_screener.extraction` owns the schema, input selection,
seed vocabulary, context rules, exact scorer and artifact CLI. No dependencies
were added. The frozen baseline helper, corpus bytes, labels and split manifests
were not changed. Existing notebook 08/09 and earlier plan files remain untouched.

Input selection validates complete raw/partition ID coverage, uniqueness and
train-only membership before deterministic sampling. Original fields remain
independent; null text becomes empty, and offsets remain relative to original
non-null strings. A run records input/partition/code/vocabulary digests, sample
IDs, configuration and artifact hashes. Reruns refuse incompatible or corrupted
artifacts rather than overwriting them.

The extractor recognizes a limited set of tool/knowledge, competence, title and
qualification phrases with a seed namespace. It preserves exact mention spans,
supports punctuation-sensitive aliases and conservative Go/R context, and records
negated/learning/unqualified mention status. It never labels a mention as proof
of applied experience or proficiency. Qualifications remain phrase matches rather
than full degree/field-of-study relations. Context rules are English heuristics;
unseen vocabulary, negation scope, ambiguous aliases and credential variants still
require labelled evaluation. Rules do not emit probability confidence.

## Measured default run

Run ID: `13dc70f74d5f872c8e30`, seed 0; 1,000 training CVs and 1,000 training JDs.

| Type | Documents with a mention | Descriptive coverage | Evidence spans |
|---|---:|---:|---:|
| CV | 680 / 1,000 | 68.0% | 2,994 |
| JD | 754 / 1,000 | 75.4% | 3,604 |

All 6,598 spans have exact original-field offsets and unique deterministic entity
IDs. Per-label entity/document counts and context counts are in
`output/extraction/extraction-metrics.json` and notebook 11. These are descriptive
counts, not extraction accuracy, normalisation accuracy or ranking performance.

Local records live in `data/processed/extraction/<run_id>/`: `entities.parquet`,
`sample-manifest.json`, `metadata.json`. Committed output contains configuration,
hashes and aggregates only, omitting sampled IDs and real text/surfaces.
Development runs generated earlier code-digest directories, which remain local;
the run named above is the final implementation. No local artifact is committed.

## Checks and review

- Three notebooks executed and passed nbformat validation: notebook 10 has four
  executed code cells, 11 has six, and 12 has seven.
- Saved-output review and `assert_clean` passed. Outputs contain aggregates or
  redacted synthetic examples only; maximum per-cell lengths are 661, 629 and
  661 characters respectively. Real document text and entity surfaces are never
  displayed. Synthetic tables are capped at five records.
- Extraction/new notebook tests: **25 passed** after final execution.
- Full suite: **216 passed, 2 failed** (111.86 seconds). The two failures are
  `test_baseline_golden.py::test_baseline_reproduces_the_committed_record` and
  `::test_environment_drift_alone_does_not_fail_the_check`. Before implementation,
  the suite had the same failures with 191 passing tests.
- `data.verify --derived`: existing queue resume failure, **200 already-judged
  pairs re-dispatched**. Available other checks passed; absent local derivative
  checks were explicitly skipped. No queue was regenerated.
- `baselines.run --check`: split/config/floor reproduced; vocabulary hash and
  19 model fields differ, with environment unchanged. This is the existing
  baseline mismatch recorded in September, not a changed extraction baseline.
- Notebook 12 validated the final run's source/code/artifact provenance, all
  field offsets and deterministic reproduced predictions successfully.
- `extraction.run --sample-size 1000 --seed 0 --check` passed on the final run:
  all 6,598 entities checked and reproduced, with artifact hashes verified.
- Independent agent review found no blocking leakage, privacy or inference
  defects. Its per-label document-coverage reporting request was implemented.
- `git diff --check` passed.

Named regression tests were demonstrated to fail against controlled defects,
then pass with the implementation: punctuation aliases removed, Go/R ambiguity
guard disabled, context statuses forced to unqualified, surface/offset validation
disabled, and a selector including evaluation document `b`. These demonstrations
used in-memory monkeypatches and did not leave defective source edits. The tests
use inline synthetic data and require no downloaded corpus. Artifact corruption
is additionally rejected by a synthetic manifest-tampering test.

## Deviations and remaining gates

The seed-rule approach without spaCy/ESCO was explicitly approved as EN3; the
original ESCO-backed and supervised stages remain pending. No further scope
deviation was needed. Adapters for absent B1/B2 were deliberately not implemented
without local data to validate them; their contracts and repair/split/privacy
requirements are documented in notebook 12. It reports blocked evaluation even
if files later become present until validated adapters exist.

EN1–EN4 are resolved. EE2's source restoration/use questions remain open for the
supervised phase, EE3 for BERT, EE4 for relation/theme annotation, and EE5 for
role/date linking. No gold F1, canonical mapping accuracy, duration totals, theme
claims or downstream ranking gains are reported. The original one-session
annotation budget and A13 caveat remain unchanged.
