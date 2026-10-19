# Requirement traceability, acceptance and unresolved choices

## Requirement matrix

| ID | User requirement | Design / verification |
|---|---|---|
| C01 | Hiring-team HR users; critical, formal review of JD/CV requirements | Narrow task menus, neutral evidence language, source-linked comparisons |
| C02 | Identify each HR through login ID/token | Separate company sign-in, verified identity, role/department-scoped access |
| C03 | Alert HR to new jobs and applicants | Authorized event inbox, deduplication, per-user read state and stale-review notices |
| C04 | Top10 for each JD from ranking-results database | Consistent versioned ranking adapter; no LLM-generated ranks |
| C05 | Skills, knowledge, title, degree, years of experience | Five-type display/provenance; four-label human gold remains unchanged |
| C06 | Preview each top10 CV as PDF/text in a popup | Read-only authorized redacted derivative/text dialog; safe unavailable fallback |
| C07 | Session report based on top8 | Versioned report identifies ranks1–8 and documents session/snapshot identity |
| C08 | Email eight shortlisted profiles/CVs and comparison of10 to department | Complete package manifest, directory recipient, confirmation dropdown, duplicate-safe outbox |
| C09 | No candidate email/calls | No such intent, recipient option or backend capability |
| C10 | No direct appointment scheduling | No scheduling/calendar operation or department-head booking |
| C11 | Store conversations | Server-side owner-scoped session storage and reauthenticated resume |
| C12 | Dropdown-only, no images/video/files | No chat text entry/upload/media widgets in HR workspace; portal supplies documents |
| C13 | Warm, on-brand, memorable | Proposed Hiring Desk persona; short formal copy and consistent recovery vocabulary |
| C14 | Brief, relevant, narrow-focus turns | One decision/question per turn; detailed content in tables/previews/reports |
| C15 | Happy paths and error flows | Mermaid state flow and scripted recovery scenarios in conversation design |

## Hard acceptance invariants

- Authorization is enforced on every backend read/action, including resumed sessions
  and preview links. No unauthorized cross-HR/department disclosure in negative tests.
- Rankings, evidence, report and attachment identities all belong to one immutable
  JD/run snapshot. Missing rows, stale data and unavailable models are explicit.
- Exactly eight intended profiles/CVs and the correct comparison report are
  validated before a complete department package is sent. No silent replacement,
  rank fabrication or incomplete eight-person success message.
- Candidate-contact and scheduling operations are absent from both UI and services.
- Every send has a separate confirmation, valid permissions, a permitted directory
  recipient and an idempotent delivery record. Unknown is never displayed as sent.
- Every displayed/exported human-facing document follows the redaction contract.
  PDF rendering cannot bypass it. Citation offsets refer to their stated coordinate
  space; missing page information is not invented.
- All HR conversation decisions, confirmations and recovery paths are dropdowns.
  Authentication is separate; tokens/passwords are not chat messages or history.
- Session writes and gateway acknowledgements determine “saved” and “sent”. LLM
  prose, a button/dropdown rerun or optimistic UI state cannot establish either.

## Scenario-based evaluation

Freeze synthetic scripts before integration tests. Cover normal review/send,
one-candidate preview, requirement comparison, saved-session resume, new-JD alert,
new-application alert, access denied, token expiry, no jobs, no applicants,
fewer-than10/fewer-than8, ties/duplicate rankings, unavailable evidence/PDF,
database outage, stale snapshot, save failure/conflict, recipient mismatch,
gateway reject/timeout and duplicate confirmation.

For each script record expected menus/state, candidate/run identities, artifacts,
allowed service calls and recovery. Measure completion, turns, unnecessary
clarifications, recoveries, preview/report correctness, unauthorized disclosures,
duplicate sends and status accuracy, with denominators. Hard invariants require
zero violations in deterministic tests. Language-quality/task-completion targets
and alert/response SLAs are set before the final walkthrough once infrastructure
and review method are confirmed; none is claimed as already measured.

## Persistent open questions

| ID | Decision needed | Proposed direction / blocking dependency |
|---|---|---|
| O01 | Which company identity provider and token/login flow exist? | Reuse company SSO; login ID alone is insufficient. Blocks real authentication. |
| O02 | Which HR may access which jobs, departments and shared sessions? | Least scope from portal permissions; sharing disabled until specified. Blocks access policy. |
| O03 | Ranking database schema, API, run completeness, tie order and refresh contract? | Read-only versioned adapter; no silent recomputation. Blocks real top10. |
| O04 | Portal notification mechanism, recipients and delivery SLA? | In-app alerts proposed; event webhook/queue/polling undecided. No external alerts assumed. |
| O05 | Where are authoritative CV PDFs/text and redacted derivatives? | Authorized portal document store; safe text fallback. Blocks real previews/attachments. |
| O06 | Session/report store, retention/deletion and encryption policy? | Company-approved backend; private per HR, explicit sharing. Blocks durable rollout. |
| O07 | Report format and treatment of fewer-than10/fewer-than8? | PDF plus structured comparison proposed; partial reviews allowed, complete handoff blocked pending policy. |
| O08 | Can HR replace one of ranks1–8 with rank9/10? | No substitution in initial design: user specified top8 ranked candidates. Additional curation needs a separate explicit rule. |
| O09 | Hiring-department directory, mail gateway, size limits and attachment policy? | Recipient selected from job-linked permitted directory; package contains redacted artifacts. Blocks real sending. |
| O10 | Brand name and approved voice examples? | Hiring Desk is a working name; warm, formal, evidence-conscious copy. |
| O11 | Does dropdown-only permit dedicated authentication fields outside chat? | Proposed yes, through the portal/IdP screen; no credentials entered into conversation. |
| O12 | What production deployment, concurrency and response expectations apply? | Existing local prototype is not a production identity/storage/mail service. Measure before release. |

These are design/integration questions, not requests to approve individual emails.
This task produces reviewable documentation only. Future interactive sends require
the HR user to confirm the specific package and recipient in the application.
