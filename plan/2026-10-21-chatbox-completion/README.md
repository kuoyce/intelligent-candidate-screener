# Complete portal chat while retaining dropdowns

21 October 2026. Status: approved by the user; implementation in progress.

## Requested outcome

Finish typed HR conversation in the primary portal and retain all existing
dropdown navigation and review actions. Preserve unrelated working-tree changes.

## Audit findings

The primary entry point delegates to portal_ui.py. It already renders chat input,
transcripts and citation expanders alongside the Review action dropdown.
portal_controller.py persists bounded, authorized, snapshot-scoped QA turns and
portal_qa.py retrieves evidence with optional validated Qwen excerpt selection.
This is a partial implementation, despite documentation saying it is unstarted.

Confirmed integration gaps:

- Dropdown candidate/category selection is stored separately from qa_context;
  chat currently receives only qa_context, so dropdown selection does not establish
  the subject of a subsequent typed follow-up.
- RevisionConflict is not handled by the UI's recovery handlers. Concurrent review
  changes can surface an exception instead of an actionable recovery message.
- Chat creates a fresh submission ID on every submit, without a retained pending
  request for safe retry after storage/concurrency failures.
- Model availability and fallback provenance are only partially exposed. Successful
  Qwen output is validated excerpt selection, not free-form factual synthesis.
- Documentation and acceptance evidence do not match the existing implementation.

## Design and implementation sequence

1. After approval, create a scoped branch and commit only this finalized plan.
   Do not stage unrelated existing edits or untracked implementation files wholesale.
2. Delegate scoped implementation and independent review with explicit ownership.
3. Define precedence: explicit candidate/category in the question wins; a newly
   selected dropdown subject establishes the next follow-up; otherwise preserve
   valid conversation context. Reject ambiguous references and stale snapshots.
4. Add recoverable submission state with a stable turn ID, bounded retries and
   honest save/conflict feedback. Preserve dropdown actions during failures.
5. Keep source-grounded QA, citations and deterministic fallback; expose actual
   inference mode/model availability. Never invent ranks, credentials or Fit calls.
6. Test typed questions, dropdown-to-chat follow-ups, comparisons, resume, duplicate
   retry, conflicts, source changes and unavailable inference using synthetic data.
   Show regression tests fail against the prior implementation.
7. Independently review; run focused tests, full pytest once stable, and affected
   registered checks with retained logs. Apply D55 before marking a plan complete.
8. Update README and implementation records with measured results and remaining
   runtime/integration limits. Do not train models or deploy external services.

## Acceptance criteria

Chat and dropdowns coexist in one authorized review. Dropdown context is honored
without overriding explicit typed references. Transcript survives reruns/resume;
retries do not duplicate saved turns. Failures show recovery instructions rather
than an uncaught exception. Citations resolve to the current authorized source.
No action is executed merely because chat requests it. Missing inference remains
an explicit usable evidence fallback, not a claimed successful LLM answer.

## Open questions

- Plan approval: granted by the user on 21 October 2026.
- Actual configured local Qwen availability remains to be verified; synthetic
  provider tests alone will not establish a working live model.
- Corporate integrations remain outside this chatbox completion scope.
