# Merge repair implementation record

11 October 2026. User authorized "resolve all conflicts" during an active merge
of `feat/stage2-rerank` (5fc2856) into `feat/experience-years` (0709709). All five
conflicts were resolved without discarding either workflow. The plan was written
before repair; creating a new branch or a separate plan commit would interrupt
the user-started merge, so the repair record travels with its merge commit.

## Resolution

- Combined `rules.py` retains the seed document extractor/context helpers and
  the incoming skill matcher. Original test cases from both branches are retained;
  an added compatibility case exercises both interfaces.
- Combined `metrics.py` retains exact source/field/label/layer scoring and the
  incoming strict/partial unit scorer and bootstrap functions. Package exports
  retain the local workflow alongside the new modules.
- Incoming evaluated skill-extraction metrics retain the original filename.
  The local seed-rule aggregate is preserved byte-for-byte under
  `output/extraction/rule-inspection-metrics.json`. No figure is pooled or rewritten.
- Registered both local extraction check commands in the incoming runner.
- Repaired the runner's child-process lifetime: retain Popen objects until wait4
  reaps them, set their return codes, and use macOS/Linux RSS units correctly.
  The scheduler test failed before this fix with ChildProcessError; all nine
  runner tests pass after it.
- Removed a runtime dependency collision: importing pandas at rules module load
  broke the isolated GLiNER environment. The isolated adapter import failed with
  ModuleNotFoundError before the fix and passes after lazy pandas loading in
  vocabulary I/O. Matching/extraction policy is unchanged.
- Fixed the incoming optional NER test to check cached-base dependency availability
  before importing transformers. The base environment correctly skips it.
- Removed sampled document IDs from the five-label public aggregate record;
  sampled IDs remain in the ignored local manifest. Metrics and hashes are unchanged.

## Validation and limits

AST comparisons established that both original rules/metrics APIs retained their
function/class definitions; lazy pandas loading is the documented subsequent
runtime adjustment. Both conflicting metric JSON records were preserved exactly.
No unresolved Git paths or conflict markers remain; diff whitespace checks pass.

Combined targeted rules/adapter/evaluate/normalise/evidence suites: **64 passed**.
The full suite was run once and logged at `/tmp/merge-pair-pytest.log`:
**705 passed, 8 skipped, 3 failed**. Two failures are the pre-existing frozen
baseline vocabulary/model mismatch. The third was the missing optional-transformers
test; it is fixed, and subsequent rules/NER tests report **31 passed, 2 skipped**.
No claim that the full suite is entirely green is made.

Cold checks ran through the registry. Rule inspection, derived-data verification,
and experience evaluation pass. The initial transformer check exposed the pandas
collision; after repair its cold rerun passes in **184.9 seconds**, validating
**1,847** spans, exact offsets, checkpoint/artifact hashes, training membership,
and complete field coverage. Evaluated extraction remains blocked by absent ESCO
raw data. Baseline check retains its known vocabulary and 19 model-field differences.
Logs: `/tmp/merge-pair-checks/` and `/tmp/merge-pair-transformer-fixed/`.

Incoming neural training grids were not retrained: no training logic was changed,
and the optional main ML environment/models are unavailable locally. The incoming
stage-2 record's own cold-check limitation remains historical, not newly validated.

## Continued work

The expanded user request is a distribution EDA plus a new entity-weighted model.
A read-only frozen 200-document plot preview was generated with saved hashes and
complete coverage checked, anonymous document indices and no raw text display.
The implementation proposal is in `plan/2026-10-11-entity-weighted-matching/`;
training/selection has not begun and no new final-set scores have been computed.
