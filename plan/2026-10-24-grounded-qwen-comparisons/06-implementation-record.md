# Extractive v1 implementation record — 24 October 2026

Current branch: `feat/hiring-desk-v1-2026-10-23`. The finalized revised scope was
committed as `a04dbfe` before implementation. Earlier generator/LoRA/larger-Qwen
proposals are superseded by `05-extractive-implementation.md` and D61. No new
branch, checkpoint download, fine-tuning or annotation was started.

## Delivered application path

- `portal_workspace.attach_services` attaches a shared lazy `QaClient` and a
  principal/workspace-scoped `PortalEntityService` to the primary controller.
  The primary UI no longer creates or invokes a Qwen provider. Historical Qwen
  turns and the historical controller API retain their provenance/compatibility.
- `portal_entities` serializes GLiNER-medium work outside the UI thread, using
  the existing pinned isolated runtime. It queues only the requested JD/CVs,
  deduplicates concurrent reruns and validates exact spans/coverage before
  private cache writes. Source/model/label/chunking/redaction changes invalidate
  entries. Every read and post-inference write rechecks authorization. Failed,
  partial, unavailable and completed-empty results remain distinct.
- `portal_extractive` validates RoBERTa start/end predictions independently for
  each requested document. Citations identify owner, snapshot, source digest and
  exact redacted-preview offsets. Natural questions remain separate from lexical
  retrieval terms. No answer is a source-backed absence claim.
- Broad comparisons return six application-built evidence rows immediately,
  without starting QA. They use GLiNER's five configured entity categories plus
  text retrieval for certifications. Focused comparisons extract each document's
  answer independently. Comparison actions and chat share this service.
- Degree status, JD negation and validity dates remain visible beside extracted
  answers, including continuation sentences such as `Status: in progress.` and
  `Valid until 2021.`. Explicit duration relationships are required for
  skill-specific years; a nearby job title and total experience are insufficient.
- Worker writes and partial response reads have bounded deadlines and cleanup.
  Source fallback is labelled, with unavailable/no-answer reasons. Cached results
  never change applicant order, scores, Fit calls or action permissions.

Controller post-inference checks, idempotent saved chat, job switching, 40-day
reviewer access, new-JD/live-ranking boundary, alerts and captured mail remain in
the v1 flow. Implementation changes are left reviewable in the existing working
tree, alongside prior uncommitted work; model code has not been committed and
this historical plan is not marked complete under D55.

## Review and validation

Scoped sub-agents implemented entity caching and extractive QA while the parent
integrated controller/UI/state and documentation. An independent review found
five reproducible qualifier/retrieval/duration/timeout issues; all were fixed
with fail-before regression proofs. Final static review found no material
remaining issue.

Retained logs under git-ignored `logs/`:

| Check | Result | Log |
|---|---|---|
| Entity cache/worker tests | 30 passed | `portal-entities-sandbox-tests.log` |
| Extractive service after review | 62 passed; 9 new failures demonstrated before fix | `portal-extractive-review-focused.log`, `portal-extractive-review-fail-before.log` |
| Adjacent source qualifiers | 53 passed; 7 regressions failed before fix | `portal-qa-adjacent-qualifiers-tests.log`, `portal-qa-adjacent-qualifiers-fail-before.log` |
| QA worker deadlines | 13 passed; 4 regressions failed before fix | `qa-timeout-final.log`, `qa-timeout-before.log` |
| Primary UI/controller | 15 passed; previous table projection failed both qualifier examples | `v1-extractive-primary-final.log`, `v1-extractive-ui-context-before.log` |
| Final UI presentation | 32 passed after category labels, larger rows and removal of repeated factual-answer tables | `v1-extractive-ui-presentation.log` |
| Full suite, run once after semantic fixes | 1,546 passed, 10 skipped, 2 existing baseline failures; 195.25 seconds | `v1-extractive-pytest.log` |
| Registered affected checks, scoped to this amendment | No affected research records; runner exit 0 | `v1-extractive-checks.log` |
| Supplied CSV exports | All 60 jobs retain original ten candidates, order and scores; exact comparison source slices checked for each job | `v1-extractive-csv.log` |
| Native installed models, invented sources | Three documents extracted; six comparison rows; repeated comparison added zero entity inferences; positive/no-answer/context/tenure checks passed | `v1-extractive-final-smoke.log` |
| Desktop/mobile browser | Comparison action/chat, source fallback provenance and job switching passed; no mobile document overflow | `v1-extractive-browser-check-final.log` |
| Restarted local app | Reviewer sign-in required, no demo-workspace switch; port 8503 | `v1-extractive-launch-check.log` |

The full-suite failures are the same two in
`hiring-desk-chat-pytest-final.log` before this amendment: baseline vocabulary
digest and TF-IDF record drift. Their hashes and model discrepancies match.
Baseline code/records were not changed. Whole-tree `git diff --check` also finds
previous trailing whitespace in `extraction/entity_distributions.py`, outside
this change.

The final native smoke completed in 28.64 seconds, including model loads. First
QA took 9.26 seconds, subsequent factual requests 0.07–0.22 seconds in this tiny
invented workload. `RUSAGE_CHILDREN.ru_maxrss` reported 596,996,096 bytes; this is
not a peak measurement of the entire model process tree. These timings establish
local plumbing only, not realistic throughput or domain QA accuracy. RoBERTa
declined the invented certification/JD-negation questions and the UI retained
their complete labelled source excerpts. Completed and in-progress degree
answers, a positive Python duration, and an unsupported license were handled
without Qwen. Every returned citation sliced its claimed source exactly.

## Remaining questions and limits

- The teammate's actual live-ranking interface is still pending; its existing
  adapter and CSV-independent behavior are preserved. No new model interface is
  assumed here.
- No measured claim is made that GLiNER improves QA accuracy. Entity enrichment
  organizes source mentions; RoBERTa remains a general pretrained QA model.
  Long/unsearched/omitted evidence is explicitly partial or not found.
- Extraction refresh is explicit. Failed jobs retain their failure state for
  the current source/configuration; Refresh does not silently relaunch them.
- Domain QA training/selection and an independent quality study would require a
  separate scoped amendment. Qwen fine-tuning is not pending in this flow.

No unresolved implementation question blocks this approved extractive scope.
