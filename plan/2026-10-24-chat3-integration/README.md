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
Implementation and validation pending.
