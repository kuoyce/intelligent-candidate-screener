# Chat-3 integration

Authorized by the owner's request to implement the proposed merge on 24 October 2026.

## Design and sequence

1. Preserve all tracked and untracked local work in a named Git stash and retain it
   until validation succeeds. Record the original branch and commit.
2. Create an integration branch, commit this plan, and merge
   `origin/exp/chat3-holdout` preserving both histories.
3. Restore local work; resolve shared documentation by retaining both contributions.
   Qualify independently allocated decision IDs rather than silently overwriting decisions.
4. Keep distinct notebooks intact; document duplicate numerical prefixes before deciding
   whether a rename is needed. Avoid changing historical notebook paths unnecessarily.
5. Review conflict resolutions with a sub-agent, run the full pytest suite once with a
   persistent log, inspect affected checks and their cost, then run required validation.
6. Fast-forward the original branch to the validated integration result. The owner will
   release and push; do not publish unvalidated work.

## Constraints

Preserve local changes, ignored datasets, model weights, benchmark labels and both histories.
Do not include `.DS_Store` or temporary notes in release commits. No model retraining or
new final-set scoring is part of conflict resolution. Long checks require owner confirmation
under the incoming branch's D58 policy; record pending validation explicitly.

## Open questions

- Which required checks need a long run? Inspect runner selection and estimates after merge.
- Decision IDs D56–D61 were independently allocated. Preserve provenance and qualify citations.

## State

Original branch: `feat/hiring-desk-v1-2026-10-23` at `a04dbfe`.
Incoming reference: `origin/exp/chat3-holdout` at `b300943`.
Merged history commit: `c5f063c`; the original branch now points to this commit.
All local work was restored from the retained stash named
`pre-chat3-integration local work 2026-10-24`.

## Implementation record

- Resolved decision-log and restored-work conflicts without discarding either history.
  D56–D61 now have explicit `(research)` and `(entity/app)` provenance; historical
  references retain their meaning through the linked plan.
- Combined AGENTS rules, retaining app safeguards, research train/test boundaries,
  and the incoming long-check approval requirement.
- Renamed local notebooks 17/18/19 to 27/28/29 and updated references. This avoids
  changing the incoming research sequence 17–26. A sub-agent reviewed resolutions
  and references and audited release artifacts for obvious credentials/PII patterns.
- Excluded `.DS_Store` and the existing `tmp.txt` scratch note from release staging.
- Full pytest log: `logs/chat3-integration-pytest.log`: 1,678 passed, 11 skipped,
  2 failed in 209.56 seconds. Both failures are in `test_baseline_golden.py`:
  local rebuilt vocabulary digest and TF-IDF results differ from the frozen record.
  Baseline code, its test, data-layer code and frozen baseline record are unchanged
  from `a04dbfe`; the root cause is not yet established. Do not overwrite the
  frozen record or weaken the tests to hide this mismatch.
- Notebook 29 was executed successfully against existing local campaigns; saved
  output log: `logs/chat3-notebook29.log`. Notebooks 27 and 28 were already executed.
- Check selection/estimates: `logs/chat3-integration-check-estimates.log`.
  The affected cold selection includes rerank_chat (23,400 s), stage2 (22,389.5 s),
  c2_validation (15,608 s), and other checks; owner approval requested before launch.
  `--explain` also executes checks in this runner; that inspection command was
  interrupted during import-closure analysis before any checks were launched.
- The owner explicitly requested committing local changes after being informed
  of the two failed tests and pending cold checks. This authorizes the snapshot
  commit without claiming validation completion; the failures and cold checks
  remain outstanding. The plan is not complete.
- Merged history through `c5f063c` was pushed to
  `origin/feat/hiring-desk-v1-2026-10-23` at the owner's request. The subsequent
  local-work snapshot is a separate commit.
