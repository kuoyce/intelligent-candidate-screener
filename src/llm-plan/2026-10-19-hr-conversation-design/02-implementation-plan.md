# Implementation plan: guided HR portal assistant

Status: proposed implementation; the earlier local prototype does not establish
that these portal integrations exist. No email is authorized or sent by this plan.

## Architecture

```text
Company identity provider -> authenticated HR principal + department permissions
Company portal events -> validated event inbox -> scoped HR alert dropdown
Ranking database -> authorized JD/run selector -> immutable top10 snapshot
Document store -> authorized redacted PDF/text preview
Extraction/evidence services -> five-type display + exact source citations
Dropdown action -> typed dialogue controller -> bounded application service
Validated evidence -> optional LLM explanation -> claim/citation validator
Session/report store -> resumable review + immutable report/package versions
Confirmed department handoff -> outbox -> mail gateway -> delivery receipt
```

The LLM is optional for completion of the guided workflow. Menus, identity,
permissions, ranking order, selected attachments and mail actions are deterministic
application responsibilities. Generated text cannot create an action, recipient,
candidate ID or score. Missing inference falls back to a short evidence-based
message without losing the current review. Source documents are untrusted data.

## Phase 0 — Approve conversation contracts

Review the scripts, state transitions, traceability and open questions in this
folder. Resolve identity/database/event/storage/mail dependencies before dependent
implementation. Confirm defaults: dropdown-only HR workspace, database top10,
unchanged ranks1–8 for the package, in-app alerts and directory-based recipients.
Retain an explicit unresolved list rather than selecting an infrastructure vendor
or assuming that existing local demo services are production integrations.

Deliverables: versioned action/menu schemas, sample dialogues, error copy,
integration contracts, synthetic walkthrough scenarios and acceptance criteria.
After approval, create a scoped branch and commit the finalized plan. Delegate
identity/persistence, ranking/evidence/reporting and controller/UI work with clear
file ownership; integrate through reviewed contracts.

## Phase 1 — Identity and authorization

Integrate the chosen portal identity provider. A login ID identifies a user; it is
not authentication on its own. Verify the provider-issued session/token using its
documented mechanism and derive HR subject, role, department scope and expiration.
Never accept a pasted token through chat or record tokens in saved conversations.
The precise login/token interface remains dependent on the existing company IdP.

Check authorization on every job list, ranking result, CV preview, saved session,
report, notification and email operation, not only at menu rendering. Revalidate
before sending and after resume. Handle expired/revoked access without exposing
the protected content. Support logout and authenticated return to an authorized
saved review; browser session state alone is not an identity system.

Acceptance: invalid/expired identity cannot read applicant data; another HR cannot
open an unshared session by guessing an ID; cross-department access is denied;
unauthorized workspaces reveal no candidate counts or names.

## Phase 2 — Ranking and evidence adapters

Read existing ranking results for the selected authorized JD. Do not retrain or
recalculate rankings inside the chatbot. Require JD/version, ranking-run ID,
candidate/document versions, rank, score semantics, model identity, completion
and timestamps. Read results consistently from one run, retaining its tie order;
if ties are not resolved by the producer, require that contract before claiming
an exact top10/top8. Never blend rows from different runs or silently fill holes.

Create immutable review snapshots. Default top10 must be the first ten rows of
a validated complete ordered run. Invalid rows or holes block that review; do not
skip them and backfill from lower ranks. A genuinely smaller pool is visibly partial.
Ranks1–8 form the proposed package. A complete eight-person package cannot be
produced with fewer than eight candidates or missing required documents. A pool
of eight/nine can have an eight-person draft but its all10 comparison is partial;
department handoff stays blocked until the short-pool policy is explicitly agreed.

Expose KNOWLEDGE, SKILL, TITLE, DEGREE and EXPERIENCE_YEARS with extraction status,
source/model identity and original/display offsets. Keep unknown, not-found,
contradicted, expired and incomplete states distinct. Do not turn a reranking
score into a Fit probability or assert that every entity mention meets a required
qualification. This adds no entity-gold category and changes no benchmark labels.

Acceptance: top10/top8 IDs agree with the same database snapshot; omitted/duplicate
rows fail explicitly; all evidence belongs to the selected candidate/JD/version;
new applications make prior snapshots stale without editing historical reports.

## Phase 3 — Guided controller and dropdown UI

Replace free-form chat with the reviewed state machine in the HR workspace.
Use fixed action IDs and server-validated dropdown values. Render one narrow
question at a time; candidates and requirements appear only when authorized and
valid for the current state. Streamlit remains a thin transport over tested
services. The legacy upload/interview demonstration is isolated and not reachable
as an HR workspace action.

Use read-only dialogs for PDF/text previews. Display a redacted derivative using
the project redaction contract; a raw PDF iframe is not an acceptable redaction
method. Preserve parser/page provenance where reliable. If a safe PDF cannot be
produced, offer redacted text and disclose lost layout/page information; if neither
is available, show an unavailable state. Do not accept files through these dialogs.
Candidate selection never changes ranking order.

Acceptance: no free-text chat/uploader/camera/microphone controls in this workspace;
tampered action IDs and stale dropdown options are rejected server-side; previews
cannot expose another candidate; one failed preview does not lose the job context.

## Phase 4 — Sessions and event notifications

Persist sessions in the selected server-side store: owner/authorized collaborators,
JD/run/snapshot references, menu state, selected requirement/candidates, validated
answer history, drafts and report/package/delivery IDs. Keep operational logs free
of document text and credentials. Define encryption, access, retention and deletion
with the company policy rather than inventing a duration. Streamlit session state
can mirror this record but cannot be its only storage.

On resume, recheck permissions and compare current portal/ranking versions. Offer
refresh or read-only historical view for stale reviews; sending from a stale
snapshot requires refreshing and reviewing the package again. Save failures remain
visible; never say “saved” until the store confirms the write. Use revision checks
for conflicting concurrent edits.

Consume validated new-JD and new-application events through the portal-supported
mechanism (webhook, queue or polling: unresolved). Store event IDs for deduplication
and per-HR read state. Route only to HR authorized for that job/department. Alerts
appear in the home dropdown with counts; permission is checked again when opened.
Do not expose candidate names in notifications outside the authenticated workspace.
Provide an unavailable/retry state if the event feed fails. Alert ordering and
polling frequency require an agreed operational contract.

Acceptance: resume after reconnect works; duplicate events produce one logical
alert; an unrelated department sees none; a new application marks the relevant
review stale; event failures do not imply that no new applications exist.

## Phase 5 — Reports and department handoff

Build a versioned session report from the current snapshot. Include JD/run/model
identity, generation time, reviewed requirements, ordered comparison of all10,
the eight proposed candidates, cited strengths and uncertainties, extraction
coverage and explicit unavailable classifications. Keep detailed facts in tables
and short summaries. No fabricated entities, scores or hiring commitments.

Create a package manifest: session/JD/snapshot/report version, exactly eight
candidate IDs and corresponding authorized CV/document digests, profiles,
department recipient ID, attachment checksums and completeness. Report format,
attachment redaction policy and mail size limits remain integration decisions.
Every human-facing artifact must meet project redaction rules; do not attach raw
PII-bearing PDFs by default. Missing documents never cause silent substitution
with rank9/rank10. Email includes eight CVs/profiles and the comparison of all10,
not eight attachments plus a report comparing only eight.

Resolve recipients through the authorized hiring-department directory associated
with the job. No arbitrary email input and no candidate contact action. Show the
resolved department/address, eight candidate IDs and report version before a
separate Confirm send / Cancel dropdown. Confirmation expires when the snapshot,
recipient, documents or package changes, or when authentication expires.

Use an outbox/delivery service with an idempotency key tied to the reviewed package
and send action. A Streamlit rerun/double submission must not send a duplicate.
Store queued, sent handoff, failed and unknown separately. If a gateway timeout
leaves status uncertain, query the same request before retrying; do not submit a
new send blindly. Report only acknowledged status. Do not claim an email was read.

Acceptance: exactly eight correct documents and the all10 comparison are checked;
only the permitted department receives the package after confirmation; cancellation
sends nothing; repeated confirmation reuses one delivery request; no candidate
email/call, calendar booking or department-head scheduling is available.

## Phase 6 — Validation, walkthrough and release evidence

Use synthetic applicants, an identity-provider double, ranking-store fixtures,
event/store doubles and a fake mail gateway. Cover happy paths, expired identity,
forbidden scope, zero/fewer applicants, stale results, missing entities/documents,
save conflicts, notification replay, gateway rejection and unknown delivery status.
Demonstrate regression tests fail when their guards are removed. Run focused tests
while iterating, then the full suite with retained logs and the affected checks
through the registered runner; cold checks apply where D55 requires them.

Conduct a scripted conversation walkthrough using the design process. Record
task completion and error recovery as measured results with counts; do not call
synthetic checks production HR validation. Review authorization, redaction,
recipient selection and irreversible-action behavior independently of language
quality. Update implementation records with actual integrations and limitations.

Release requires the hard invariants in the acceptance document, resolved critical
integration choices and a concrete reviewed deployment configuration. Production
hosting is not performed by this documentation task. No training, data collection,
outbound email or notification dispatch is started by writing these plans.
