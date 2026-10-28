# Preserve checked comparisons in v3

## Authorization and design

The owner approved the quick fix: when v3 rejects Qwen prose or generation is
unavailable, retain the existing v2 checked comparison instead of replacing it
with v3's reduced summary. Accepted model prose remains unchanged.

Use the existing v2 source-checked renderer, with the current snapshot and
reauthorized comparison facts, for turns comparing multiple selected candidates.
Include broad comparisons, criterion comparisons, directed explanations and the
compare action. Do not reuse unvalidated model text or stale source quotes.
Keep v3 mode, qwen_fallback, the original fallback reason, model=None and empty
validated model claims. Preserve table behavior, storage bounds, single-candidate
fallbacks and the JD quotation fix. Never instantiate a v2 Qwen service.

## Implementation and validation

The owner explicitly requested implementation on the existing branch; do not
switch branches. Delegate code and regression tests. Demonstrate the reported
“Compare candidate 1 and 9.” invariant fails
before the fix using invented sources, then verify it matches v2's checked
comparison after rejected/unavailable generation. Cover compare action, source
tampering, accepted prose, directed comparisons and existing fallback behavior.
Run focused tests and affected check selection, review the diff and update docs.
The last full suite took 289.68 seconds with existing failures; obtain owner
confirmation before repeating a run of that length under Testing rule 4.

## Open questions

Only full-suite timing approval remains; implementation scope is authorized.

## Approved comparison-table follow-up

The owner requires v3 to retain v1's comparison tables. Explicit comparison
requests already set the table flag, but comparison follow-ups without the word
"compare" can hide the table. Set table visibility for the resolved comparison
intent with multiple participants, preserving expandable directed explanations.
Verify the real Streamlit transport renders tables independently of supporting
source visibility for both the initial question and a comparison follow-up.
Keep the owner's unrelated source/answer-details presentation edits intact.
Use the same branch, delegate implementation and demonstrate the regression
fails before the change. Full-suite timing approval is still pending.

## Implementation and validation record

Implemented on `fix/jd-quotation-fallback` as explicitly requested. V3 rebuilds
the current source-owned facts and invokes the existing v2 renderer for
multi-candidate fallbacks. It retains v3 fallback provenance, the original failure
reason and no model-authored claim mappings. Accepted prose, single-candidate
answers, tables and the storage guard retain their existing paths.

Delegated invariant tests failed nine times before the fix
(`logs/v3-comparison-fallback-before.log`). Focused verification passed all 143
tests (`logs/v3-comparison-fallback-focused-final.log`), covering v2/v3 answer
equality for the reported ranks, criterion questions, directed explanations and
the compare action, with rejected and unavailable generation. Tampered source
tests verify stale text is not reused. Existing tests were updated to assert
v2's full qualification and counterevidence wording. The earlier JD qualifier
test was aligned with the owner's raw-source presentation edit.

Review confirmed fresh citation authorization, participant coverage, preserved
qualification context, no additional model call and unchanged provenance. README
documents the fallback. The affected check runner selected no records
(`logs/v3-comparison-fallback-checks.log`); `git diff --check` passes.
The full suite was not repeated: its preceding 289.68-second run has documented
existing failures, and the owner timing confirmation requested under Testing
rule 4 is pending. No branch switch, commit or running-service change was made.

### Comparison table verification

V3 now sets table visibility from the resolved comparison intent as well as an
explicit comparison request, retaining the multiple-participant guard and
expandable directed explanations. Four new regression cases failed before this
change (`logs/v3-comparison-table-before.log`): accepted, rejected and unavailable
generation follow-ups, plus the real Streamlit transport. All 58 v3 service/UI
tests pass afterward (`logs/v3-comparison-table-focused.log`). The UI test verifies
both initial and follow-up tables render with friendly participant labels while
both supporting-source controls remain unchecked. No changes to `portal_ui.py`
were needed; the owner's presentation edits remain intact. Full-suite timing
approval is still pending. No branch or running-service change was made.
