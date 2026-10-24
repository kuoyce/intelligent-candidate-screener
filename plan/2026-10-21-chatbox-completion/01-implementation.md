# Chatbox completion implementation record

21 October 2026. User approved the completion plan. Branch:
feat/chatbox-completion-2026-10-21; approved plan commit: 7b8ca41.
Status: local implementation and independent review complete; validation recorded
below. This does not establish production readiness or uniform Qwen answer quality.

## Audit and implementation

The starting portal already rendered chat beside dropdowns. Its 25 focused tests
passed (logs/chatbox-audit-focused.log), but dropdown context did not reach chat,
save conflicts lacked recovery, and documentation still called chat unstarted.

Separate agents implemented controller/QA context integration and UI recovery;
a third agent reviewed the changes independently. Existing unrelated dirty files
were preserved. No training, label collection, ranking changes or email occurred.

Dropdown candidate/category selections establish the next conversational scope.
Explicit typed references take precedence; subsequent questions keep their chat
context until a new dropdown selection. Scope includes the JD and snapshot;
foreign candidates and stale context are rejected. All five existing dropdown
categories remain supported, including KNOWLEDGE.

The chat is rendered after dropdown selections have been persisted, so selections
on the same rerun apply to the question. Failed submissions retain a redacted
prompt and stable turn ID, with explicit Retry/Discard controls. Retries are
not automatic; changed snapshot or candidate context prevents silently retargeting
a question. Dropdown navigation remains usable. Inference provenance distinguishes
validated model-selected excerpts, failed inference and deterministic evidence.

## Verification

- Context regression tests failed before implementation; all 24 focused context/QA
  tests passed afterward (logs/chatbox-context-before.log and chatbox-context-after.log).
- UI recovery regressions failed before implementation
  (logs/chatbox-ui-before.log); all 15 final UI tests passed
  (logs/chatbox-ui-verified.log), covering storage errors, revision conflicts,
  lost save acknowledgements, context changes and dropdown follow-ups.
- Review identified numeric citation IDs and source owners being altered by prose
  redaction during persistence/display. Narrowly validated identity fields now
  survive exactly; arbitrary prose still passes redaction. Both store regressions
  failed before changes; all 13 store tests passed (logs/chat-citation-store-after.log).
- Scoped cold registered checks selected no affected research records across the
  four changed portal modules (logs/chatbox-scoped-checks.log). Unrelated dirty
  model/training files were excluded from this task's scope.
- Full pytest: 1,276 passed, 10 skipped, two failed in 185.36 seconds
  (logs/chatbox-full-pytest.log). Both failures are in test_baseline_golden.py:
  test_baseline_reproduces_the_committed_record and
  test_environment_drift_alone_does_not_fail_the_check. They report a vocabulary
  digest mismatch and consequent TF-IDF metric/coefficient differences. The
  earlier logs/portal-full-pytest.log contains both identical failures and the same
  vocabulary digest mismatch. Baseline code/tests/frozen records were not changed
  by this task; those failures remain unresolved outside the chatbox scope.

## Actual local inference

The existing pinned Qwen service was initially unavailable. Started the already
provisioned loopback runtime with its existing manifest; no downloads or training.
Model: Qwen/Qwen2.5-0.5B-Instruct, revision
7ae557604adf67be50417f59c2c2f167def9a775; float32 CPU, four threads.
Service startup: 11.37 seconds, peak RSS 2,893,250,560 bytes
(logs/chatbox-qwen-service.log).

Three synthetic questions were exercised (logs/chatbox-live-smoke.log):

| Question | Observed result | Seconds |
|---|---|---:|
| What does this role require? | Validated model-selected excerpt, one citation | 11.45 |
| Compare candidates 1 and 2 on Python | Model output rejected; deterministic evidence fallback, three citations | 30.42 |
| What degree does candidate 2 have? | No relevant source evidence; explicit insufficient-evidence response, no inference | <0.01 |

These checks establish local integration, not broad conversational quality or
production accuracy. The small checkpoint did not produce valid output for every
question. The UI retains evidence fallback and must not claim all answers are
generated successfully by Qwen. Model factual output remains exact validated
source excerpts rather than unchecked paraphrased synthesis. Real company SSO,
events, original PDFs, directory/mail gateway and hosting remain outside scope.

The initial smoke exposed a provenance display issue: QwenProvider keeps its
checkpoint name under config.model. A named regression demonstrated the class-name
fallback before the fix; QA now reports the configured checkpoint name correctly.
The inference results above were not rerun solely to update that metadata field.

## Open questions

User acceptance correction: the first delivery only pointed to CV passages,
which did not fulfill question answering. The
[answer-first acceptance correction](02-answer-acceptance.md) supersedes that
product-completeness claim. Corrected extraction and validation are now complete:
48 focused QA/context/UI tests passed; final full suite passed 1,285 tests with
10 skips and the same two baseline reproduction failures. See the acceptance
correction for exact logs, live inference results and remaining retrieval limits.
Baseline reproducibility failures, small-model output reliability and corporate
dependencies remain explicit limits.
Corporate integrations remain as recorded in the original portal design; none
is represented as delivered. The local Qwen service was left running on loopback
port 8000 for the existing portal configuration.
