# Chat comparison correction — 24 October 2026

Scope: repair the approved preserved-chat functionality after the user reported
that comparisons fail, and verify/restore the already provisioned local Qwen
service. This is remediation within the approved v1 scope, not new training or
a change to backend ranking.

Initial findings: rank parsing requires 'candidate'/'applicant'/'rank', so
'compare 1 and 2' cannot select participants. Even 'compare candidates 1 and 2'
has no lexical topic terms, so retrieval returns no evidence and does not invoke
Qwen. No Qwen inference service process was running during diagnosis; the app
can construct a configured client without establishing service availability.

Repair: accept explicit comparison rank syntax without mistaking years for
candidate ranks; provide exact source excerpts for broad comparisons while
retaining strict topic/credential/negation checks for specific questions. Show
actual service availability separately from configuration. Restore the existing
pinned local runtime, then exercise comparison and a specific question through
the real Qwen service using invented documents. Keep validated evidence fallback.

Verification: fail-before regression tests, focused QA/context/UI tests, full
suite and scoped affected-record selection. Record real model identity and
validated/fallback outcomes; never claim model output was accepted if rejected.

Open dependency: actual Qwen generation reliability must be measured; successful
service startup alone does not prove a valid answer. No fresh model download,
fine-tuning, candidate-ranking change or annotation dispatch is authorized here.

## Implemented correction

Bare numeric comparisons, prefixed ranks, hashes, compact references, explicit
candidate IDs and rank ranges now resolve independently of lexical retrieval.
Reference syntax and 'against JD' wording are removed from topic constraints.
Experience/count units terminate reference lists so 'candidate 1 with 12 years'
does not refer to candidate 12. New topic words reset an unrelated previous
category; bare follow-ups retain it. Broad comparisons use JD wording to prefer
relevant CV excerpts. Specific credential questions keep relevance/qualification
checks, and JD requirements cannot substitute for candidate credentials.

Qwen's prompt now uses short request-local IDs (`e1`, `e2`, etc.) and provides the
actual candidate rank for each excerpt. One excerpt per source is ordered before
extras. Responses are mapped back to canonical snapshot/document/offset hashes;
duplicate, foreign, altered or missing-participant evidence remains rejected.
The small model initially returned quote arrays instead of the required object;
the simplified prompt resolved both measured comparison cases without relaxing
validation. No checkpoint or inference budget was changed.

The UI probes service metadata/tokenization using an invented status message and
a short deadline, cached for ten seconds. A locally loaded tokenizer without
service attestation is labelled configured/unverified rather than connected.
Per-answer labels distinguish validated Qwen selection, deterministic source
retrieval, model failure/fallback and no matching evidence/no inference.

## Measured runtime results

Restored the existing loopback runtime at port 8000 using already provisioned
assets: `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, float32 CPU/four threads.
This is a pretrained instruction model used for source selection, not a Qwen
fine-tuning run performed by this portal. The separate extractive-QA experiments
are not the inference path used by Hiring Desk.

Final synthetic live requests both returned `llm_validated_excerpts`, with two
candidate owners and two validated citations:

| Question | Measured seconds |
|---|---:|
| Compare 1 and 2 | 3.66 |
| Compare candidates 1 and 2 on Python | 3.48 |

Logs: `logs/hiring-desk-chat-live-final.log` and
`logs/hiring-desk-qwen-service.log`. These two integration checks are not an
accuracy benchmark or a guarantee that every future Qwen output is valid.
Fallback remains necessary for unavailable services or invalid responses.

A real-service AppTest comparison also passed through the UI/controller/save
path using invented CVs (`logs/hiring-desk-chat-ui-live.log`). A read-only
deterministic check across all 60 CSV JDs found both candidate sources for every
broad comparison, logging only aggregate counts
(`logs/hiring-desk-chat-real-csv.log`). No applicant text was logged or committed.

## Regression evidence and verification

New tests failed before repairs for bare comparisons, empty provenance labels,
configured-only availability, measurement parsing and structural reference terms.
Logs: `logs/hiring-desk-chat-before.log`, `hiring-desk-chat-status-before.log`,
`hiring-desk-chat-offline-before.log`, `hiring-desk-chat-reference-before.log`.
Focused QA/context/UI tests passed after the changes, including forged claims,
negative qualifications, short-ID canonical restoration and owner coverage.

Full-suite and affected-record runner logs are
`logs/hiring-desk-chat-pytest-final.log` and `logs/hiring-desk-chat-checks.log`.
Final suite: **1,431 passed, 10 skipped, two previously observed baseline golden
failures**; no chat/UI test failures. Focused QA/context/UI verification after
the reference corrections passed 79 tests; the additional local-tokenizer
attestation regression passed in the final full suite.
The runner selects no research records for these app-code changes; unrelated
dirty files were excluded. Baseline vocabulary/model drift remains outside this
repair. No model training, cold scoring or final-set evaluation was performed.

Hiring Desk was restarted on its existing loopback port 8503 with Qwen left
running separately on 8000. The production sign-in page was verified after reload
(`logs/hiring-desk-chat-launch-check.log`). Existing reviewer credentials, saved
reviews, CSV ranks and the 40-day authentication policy were not changed.
