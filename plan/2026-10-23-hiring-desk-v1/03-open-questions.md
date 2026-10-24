# Decisions and open questions

Additional confirmed requirement: retain the current CSV-backed JD/CV review
flow alongside new live-ranked JDs. Existing exports must remain reviewable with
the model service offline. This is mandatory acceptance, not an optional fallback.

Confirmed by the user on 23 October: new JDs must support candidate ranking via a
teammate's live-model integration. Email only needs to demonstrate procedure and
flow; implementation choice is delegated. Use local MIME-message capture.

| ID | Decision | Proposed v1 choice | Status |
|---|---|---|---|
| Q1 | Do newly created JDs rank candidates? | Yes; request/status/result adapter described in `05-live-ranking-contract.md`. Teammate owns inference and candidate pool. | Scope confirmed; boundary proposed |
| Q2 | Which email setup? | Local `.eml` capture, preview, explicit confirmation and download; example Tech/Business addresses. | Choice delegated; specified in design |
| Q3 | Are real addresses/mail credentials needed? | No; reserved example addresses and local capture meet the demonstration scope. | Resolved by reduced scope |
| Q4 | What should trigger alerts? | JD publication, ranking ready/failed, email captured/failed. No applicant-arrival alerts without a source. | Approved; implemented |
| Q5 | What does top ten include? | All ten ranked redacted CV text files and profiles plus comparison report; replace new top-eight handoffs using schema v2. | Proposed from user's ten-applicant requirement |
| Q6 | Is Impeccable available elsewhere? | No installed skill found in available catalog or local skill/plugin directories. Use a documented manual audit unless supplied. | Tool unavailable; alternative disclosed |
| Q7 | May app data be persisted? | Owner-authorized operational JD drafts/published revisions in local SQLite, separate from benchmark data, with versioned migrations. | Approved; implemented |

The teammate's interface is not yet supplied. Obtain its invocation method
(Python or HTTP), request/result schema, status behavior, configured candidate
pool identity/eligibility and model/evidence provenance. These are dependencies
for binding and validating the live adapter, not for starting independent UI,
authoring, alert or email work after approval. Never silently choose benchmark
CVs. Use interface documentation supplied in the workspace or by the user;
this plan does not authorize contacting the teammate.

The user approved the revised plan on 23 October 2026, including these proposed
choices. Only the teammate's actual modeling interface/pool remains an external
integration dependency. Independent implementation is authorized.
