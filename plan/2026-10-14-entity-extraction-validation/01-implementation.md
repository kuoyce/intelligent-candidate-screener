# Saved extraction validation

14 October 2026. User approved the plan. Dedicated branch:
`feat/entity-extraction-validation-2026-10-14`; plan commit `d0f1756` preceded
implementation. Helpers/tests and notebook/reporting were delegated separately,
with an independent read-only reviewer. Unrelated chatbot working changes remain
untouched and are not part of this delivery.

## Result and interpretation

The historical notebook 13 run
`2c591cc8bff7fc9e7189-20261005T164947399919Z` passes source-integrity validation:
**1,832 entities, 200 documents (100 CVs / 100 JDs), 600 fields, zero audit failures**.
The entities comprise 1,330 KNOWLEDGE, 351 TITLE, 109 SKILL and 42 DEGREE mentions;
1,077 are CV-side and 755 JD-side. These counts come from the saved entities,
not regenerated GLiNER predictions. Its original score threshold is 0.5.

The separately audited experience-enabled run
`4fde336ec50802f910d2-20261009T161236977778Z` also passes: **1,847 entities,
200 documents, 600 fields**, including 16 EXPERIENCE_YEARS spans. It is not pooled
with the original run. Both final audits are logged in
`logs/extraction-source-validation-final-audit.log`.

The audit reconstructs each run's historical fields, checks source/partition and
artifact hashes, deterministic sampled training IDs, surface slices and bounds,
saved prompt labels, finite scores, duplicate evidence and processing coverage.
It retains historical code provenance without claiming a replay under current
code. This is essential because today's input defaults include fields that the
original run did not use. Current rules on historical fields are explicitly an
inspection comparator, never gold.

Notebook 17 displays saved entities alongside bounded redacted source snippets
for each of the four labels and both document types. It also shows shared and
disagreeing rule predictions, and a heuristic inspection queue. An inspection
flag is not an adjudicated error. Examples do not estimate precision or recall.
The aggregate source-audit JSON is saved under ignored
`data/processed/extraction/validation/<run_id>.json`.

Local filesystem inventory confirms DataTurks and SkillSpan raw directories are
absent. The existing `output/extraction/extraction-metrics.json` evaluates a
different learned tagger and rules; its accuracy figures cannot be assigned to
GLiNER. No gold GLiNER precision/recall/F1 is reported. Djinni Fit labels cannot
establish entity correctness or missing-entity recall. This limitation remains
explicit, rather than reopening annotation collection under D42.

## Validation

Focused source-audit tests: **13 passed** (1.93 seconds). The full suite ran once:
**983 passed, 2 failed, 8 skipped** (137.33 seconds), recorded in
`logs/entity-validation-pytest.log`. Both failures are the existing baseline golden
reproduction/environment checks, with vocabulary/model differences; no baseline
record was overwritten. Optional model/data-dependent tests skipped explicitly.
Warm and cold affected-record runner invocations selected **zero existing checks**
before documentation edits: the new audit is not imported by an existing model
record, and no model/training or scoring implementation changed. Logs are
`logs/entity-validation-check-selection.log` and `logs/entity-validation-cold.log`.
Documentation edits describe the audit and do not change record inputs; the
runner's broad documentation-prefix selection is not a reason to retrain
unaffected models. Independent code review approved the historical-source policy,
mandatory artifact hashes and malformed-evidence reporting. `git diff --check`
passed.
Notebook 17 executed successfully, with **18 executed code cells**, no error
outputs and a maximum text-cell output of **1,036 characters**. Its three notebook
tests passed after execution (`logs/entity-validation-notebook-tests.log`). A
controlled in-memory disabled-audit defect made the uncovered-tail regression
test fail; production code was not edited for the demonstration
(`logs/extraction-source-validation-regression.log`).
No model inference, training, downloads, annotation dispatch, threshold tuning,
collected-label edits or existing model-record refresh occurred.

## Deviations and remaining questions

The aggregate report lives under ignored data rather than introducing another
model record in output: this audit inspects historical artifacts and is not a new
model fit or inference campaign. Its reproducible findings are recorded here and
in the executed notebook. Gold evaluation is unavailable due to missing local
assets, as allowed by the approved plan. Establishing Djinni semantic accuracy
remains a separate decision requiring compatible gold; no accuracy claim rests
on the present audit.
