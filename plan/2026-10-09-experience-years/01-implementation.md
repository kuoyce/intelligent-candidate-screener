# Implementation and validation

9 October 2026. User approved the plan; branch `feat/experience-years`, plan
commit `53e1ec9` preceded implementation. Adapter/schema and analysis changes
were delegated, then reviewed against the approved scope.

## Delivered behavior

GLiNER adapter v2 requests `EXPERIENCE_YEARS` using the description
`years of experience`. Shared entity
validation accepts this label with exact source offsets and raw bounded confidence.
No numeric normalization, date arithmetic or fit classifier is introduced.
Notebook 13 documents the fifth type and uses an authored experience-duration
example. README, A2 card, catalog and reconstruction instructions are updated.

EDA uses the saved model label scope and observed labels instead of assuming four
types. Rules retain their original scope; an unsupported rule label is omitted
instead of drawn as zero predictions. Historical records/charts are retained.
The new aggregate record is `output/extraction/transformer-experience-years-metrics.json`.

## Validation

Three synthetic CV/JD cases (exact duration, range, minimum) failed against the
old adapter, then passed with the change. They check exact offsets with Unicode
prefixes, unchanged wording, document type, confidence and coverage. Adapter/EDA
tests: 25 passed, including historical scope and five-label chart generation.

Full suite: **266 passed, 2 failed**. Both failures are the previously documented
baseline golden/environment tests. `baselines.run --check` again reports the
existing vocabulary digest and 19 model-field differences; it passes split/config/
floor. `data.verify --derived` retains its existing resume failure: 200 already-
judged pairs re-dispatched. No baseline or annotation records were refreshed.
Notebook structure/safety tests: 12 passed. `git diff --check` passes.

Initial smoke: five CVs and five JDs, **10/10 documents and 30/30 fields complete**,
94 spans in 46.20 seconds, no quarantined predictions. The initial long prompt
returned no duration spans, including in the subsequent 200-document run (1,798
spans, 202.30 seconds) and authored CV example. Those local experimental artifacts
are retained, but are not the delivered five-label aggregate record.

The prompt was shortened to `years of experience` after local authored probes.
With all five prompts at the unchanged 0.5 threshold, it extracted `5 years of
professional experience` at 0.6224; the initial long prompt extracted none.
The shorter prompt missed authored JD minimum/range mentions, and neither prompt
produced duration labels for the authored age/calendar-year/course examples.
These probes guide prompt choice, not an accuracy estimate or proof of exclusion.
Final-prompt smoke completed 10/10 documents and 30/30 fields, with 98 spans in
25.80 seconds; no duration mentions were returned in this small sample.

Review also found a four-label assumption in document coverage summaries. A
regression with two duration spans in one CV and a JD with none failed against
the old code, then passed after using the model label scope for both span and
document counts, including zeros. All 14 pipeline tests pass. After the prompt
and summary changes, all **51** adapter/pipeline/EDA/notebook tests pass.

Final run: `4fde336ec50802f910d2-20261009T161236977778Z`, threshold 0.5,
seed 0, CPU. **200/200 documents and 600/600 fields complete** in **174.46
seconds**, with **1,847** spans and no quarantined predictions. EXPERIENCE_YEARS
contributes **11 spans in 9 CVs** and **5 spans in 5 JDs**. These are predicted
mentions, not verified duration values or accuracy measurements. The synthetic
notebook preview includes `5 years of professional experience`; its presence is
asserted during execution. Post-execution notebook tests: 12 passed.
Saved-run validation passes checkpoint/artifact hashes, training-only membership,
all original field offsets, unique IDs and full field coverage for all 1,847
records. Notebook schema, execution and the 2,000-character output cap pass.

Jupyter execution required escalation for local kernel sockets, after sandbox
execution failed with `Operation not permitted`. Inference remains local.

## Open questions

Plan scope is resolved by approval. Numeric normalization and explicit linkage
to overall versus skill-specific experience remain deferred; independent span
accuracy and classification benefit remain unmeasured.
