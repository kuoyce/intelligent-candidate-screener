# V3 implementation and feasibility record

26 October 2026. Owner approved execution. Plan committed as `d35d632` on
`feat/hiring-desk-v3-inference-prompt`. Existing uncommitted v2, ONNX ranking,
reports and owner files were retained. Application changes remain under review;
no remote push, model training, model download or service restart was performed.

## Delivered implementation

- A packaged inference-only system prompt, separate plain-text Qwen adapter and
  isolated worker. V2's relation-ID parser remains unchanged.
- Source-owned context, bounded same-scope history, explicit unknown states,
  original question retained during retrieval normalization, and conservative
  registered-paraphrase support checks. These are not a general entailment model.
- V3 controller/UI integration behind `HR_PORTAL_CHAT_VERSION=v3`; explicit
  comparisons retain tables, “why” has an expandable table, sources are on demand.
- Authorized literal JD-requirement search independent of ranking, including
  required/preferred/negative/uncertain statuses and bounded coverage. Saved
  cross-JD results are reauthorized on display and duplicate submissions.
- Published unranked jobs accept standalone JD questions. This initial surface
  explicitly has no saved multi-turn memory; ranked candidate review has bounded
  saved memory. Full persisted JD-only dialogue is deferred, not implied complete.

## Feasibility findings and development refinements

The initial integrated real-Qwen development walkthrough accepted **0 of 4 factual
turns**. One timed out; the JSON-heavy payload caused the model to repeat the
earlier greeting. The compact revision removes internal hashes and duplicate
fact structures, keeps complete owner-qualified contexts and checked differences,
and puts the current question last. Model weights and service limits are unchanged.

The second real walkthrough also accepted **0 of 4** before validator refinement:
two faithful degree answers used unregistered grammatical variants; two answers
made unsupported claims. In particular, the model claimed an ongoing master's
was completed and invented a bachelor's superiority. Neither was displayed.
These observations motivated narrow grammatical variants, not weaker factual
checks. All of these are seen development cases from one invented source family;
they do not establish general accuracy or satisfy the held-out quality gate.

Development logs (git-ignored): `logs/v3-real-walkthrough.json`,
`logs/v3-real-compact-walkthrough.json`, `logs/v3-prose-live-synthetic.log`.
The latter is transport-only output and contains an unsupported claim; it is not
an accepted factual answer.

## Regression and review evidence

- V2 rejects natural prose as invalid JSON; it does not receive saved history.
  Before-fix logs: `v3-prose-before.log`, `v3-transport-before.log`.
- JD punctuation and negation failures reproduced before correction:
  `v3-jd-search-edge-before.log`.
- Changed cross-JD sources previously remained visible: the named regression
  failed before the display reauthorization fix (`v3-source-reopen-before.log`).
- Independent review caught cross-JD `context=None` history, stale source display,
  and missing explanation tables; each has a regression test.
- Qualification-confirmation retrieval initially treated the user's assertion as
  a filter; the failing test is in `v3-confirmation-before.log`. Retrieval now asks
  neutrally for qualifications while Qwen receives the original question.
- The central check registry selects no committed model/research records for the
  v3 source paths (`logs/v3-affected-checks.log`: `{}`). Unrelated pending model
  changes were excluded from this scope; no cold research retraining was launched.

## Rollout decision

### Owner-requested activation, 26 October 2026

The owner explicitly requested v3 running and support for ownership/05. The local
portal was restarted with `HR_PORTAL_CHAT_VERSION=v3` and
`SCREENER_QWEN_DISABLED=false` at port 8503; health returned 200. Launch details
are in `logs/hiring-desk-v3-launch.json`, service log in
`logs/hiring-desk-v3-portal.log`. This is local experimental activation, not a
claim that the general quality gates below passed.

Fixed owner-scenario routing: single-candidate good-fit summaries now use checked
JD criteria without creating Fit decisions; relevant job experience no longer
forces tenure-only extraction; the owner's `canddiate` typo is normalized for
retrieval; JD skill searches accept skill-label prefixes and suffixes. Long
source contexts have source-state summary fallbacks while preserving full
qualifiers in supporting sources. The independent review found no new blocking
qualification/negation strengthening.

Focused verification after these changes: **68 tests passed** in
`logs/v3-owner-flows.log`, including a real Streamlit AppTest walkthrough of all
owner happy and negative cases with synthetic sources and offline model stubs.
This checks routing, useful evidence availability, tables and negative redirects;
it does not assert natural freeform Qwen success on every unseen CV.

Final implementation verification: full `uv run pytest` completed in 263 seconds:
**2,026 passed, 56 skipped, six failed, three errors**. All new v3 tests passed.
The six failures are the previously recorded two baseline-golden drifts and four
legacy UI tests expecting removed dropdown actions. The three errors are local
socket permission errors in entity annotation UI fixtures, unrelated to v3.
Full log: `logs/v3-full-pytest.log`; the socket-only retry passed all **28 tests**
outside the sandbox (`logs/v3-socket-tests.log`), resolving those three errors.
Whitespace validation passed. The final narrow degree paraphrase and error-reason
fixes passed in the full suite; they were shown to fail before the fixes in
`v3-qualification-alias-before.log` and `v3-provenance-before.log`.
No further live prompt experiment was run after these grammar-only refinements;
the recorded live acceptance remains the pre-refinement development result.

Keep v3 experimental despite local activation at the owner's request. Do not claim
completion of general quality acceptance. The proposed 32-family dataset and two human
reviewers are not yet established; no held-out benchmark is claimed. Continue
only after recording acceptable model-authored answer coverage and human quality
review. V1/v2 and existing ranking configuration remain available unchanged.
