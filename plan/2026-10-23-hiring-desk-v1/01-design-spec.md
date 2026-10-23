# Design specification

## 1. Application shell and navigation

After local sign-in, show one production review workspace. Remove the Workspace
selector and user-accessible synthetic mode. Synthetic fixtures remain available
only to tests and development entrypoints, isolated from operational storage.

Left drawer: Review jobs, Create new JD, Alerts with unread count, Saved reviews.
Keep signed-in identity concise and Log out at the bottom. Put connection and
source details in a compact service-details panel instead of repeated banners.
Critical actionable errors remain visible beside the affected action.

## 2. JD-first review screen

Keep existing CSV-backed jobs in the default job browser. Load their JD and CV
extraction tables and `a2-x2-scores.csv` through the current validated adapter;
use the existing source-text join for full redacted JD/CV previews. A new model
service is not a prerequisite for reviewing these jobs. Preserve their scores,
ranks, evidence, saved-review access and chat behavior. Show a concise source
label distinguishing CSV export from live ranking, without adding a workspace
selector. Route each review by its stored source/snapshot identity, not by title.

At the top, show a horizontal group of selectable job cards with Previous and
Next controls. Each card includes title, department when known, a short faithful
summary and screening availability. The selected card is visibly distinct.
Show position/count, use stable JD IDs, disable boundary controls and retain
selection on reruns. Controls must work by keyboard and adapt to narrow screens.
Do not auto-advance or depend on horizontal scrolling alone. For the existing
60-job export, add title search and pagination rather than rendering 60 cards.

Selecting a card opens that JD's complete redacted source text on the same page.
Above it, show a compact requirements summary with skills/knowledge, experience
and qualifications only when actually present. Start with deterministic source
excerpts/validated extracted mentions; label missing fields as unspecified.
Any model-generated paraphrase is optional, traceable and cannot add criteria.
Full JD text remains accessible and readable, not replaced by the summary.
Preserve original source headings; do not pretend an inferred heading was original.
If the full source is unavailable, explain that state and never substitute table
mentions while labelling them the whole JD.

Below the JD: candidate overview, existing Review action dropdown, selected
action panel, and the existing Ask Hiring Desk chat. Retain source citations,
clarification behavior, retry handling and backend ranks. Label scores accurately.
Use clear tables and progressive detail for dense entity comparisons.

Keep the job browser available inside every active review and provide Review
another job. Switching jobs saves/resumes a compatible owned review, then clears
job-specific candidate selections, action widgets, explanation caches, pending
questions and evidence references. History remains with its original review.
Never transfer a pending question or candidate selection into another JD.
Unsaved JD creation stays separately recoverable when navigating away.

## 3. Guided Create new JD page

Dedicated page with a requirements checklist, chat conversation and editable JD
preview. Ask one focused question at a time: title, department, responsibilities,
skills, domain knowledge, minimum/maximum experience, qualifications and optional
location/work arrangement. Distinguish required from preferred requirements.
Allow back/edit/skip for optional fields and explicit 'not specified'; validate
numeric bounds and contradictions. Never infer missing years or requirements.

Use a deterministic guided flow as the baseline. A configured LLM may phrase
follow-ups and draft prose from confirmed inputs, with structured validation.
Provider unavailability must preserve the draft and allow completion. Drafting
chat is separate from candidate-review QA and cannot execute save/publish/send.

Lifecycle: draft -> reviewed -> published; edits create a new revision and require
review again. Save/resume is explicit and owner-authorized. Publish requires a
final preview and explicit click, and creates one new-JD alert. Published means
visible in this local application, not advertised on an external jobs portal.
Store job UUID, revision, owner, authorization workspace, hiring department,
structured requirements, redacted text, timestamps and provenance. Operational
jobs do not enter benchmark exports, gold labels or training sets.

New-JD ranking is in v1 scope, as confirmed by the user. After publication, offer
Rank candidates; if the live adapter is available, submit the confirmed JD
revision to the teammate's model and show progress. On success, open the same
review actions/chat flow using the returned ranking snapshot. Offer Re-rank for
an explicit fresh run. Edits to a JD require new review/publication and a new
ranking; old snapshots remain historical. Never call inference on every rerun.

Separate publication status from ranking status: not requested, queued, running,
ready, failed, or service unavailable. A pending job remains visible in cards.
Provide an explicit retry after failure. If the teammate's service is not ready,
preserve the JD and show that dependency accurately; do not populate unrelated
benchmark rankings. The proposed request/result boundary is specified in
`05-live-ranking-contract.md`; the model and candidate-pool implementation remain
owned by the teammate. This upgrade must support real results, not just fixtures.

## 4. Alerts

Implement a useful in-app inbox from actual application events: JD publication,
ranking ready/failed, and email captured/failed. Each alert
has an event ID, timestamp, concise message, related JD/package link and unread
state per user. Read/unread must be explicit; merely rendering a page does not
silently mark all alerts read. Guard every deep link by owner/workspace access.

Write business changes and their alerts transactionally; deduplicate by source
event identity. Define visibility explicitly: JD publication for its authorized
workspace; delivery events for the initiating reviewer/authorized recipients,
not every user who happens to share a department label. Preserve existing event
types and historical read states through a schema migration.

There is currently no applicant ingestion feed. Do not manufacture 'new
applicant' alerts. Add those only when a genuine source adapter exists. On-screen
refresh and navigation update the count; no push-notification promise in v1.

## 5. Top-ten department email — procedure demonstration

The user accepts a lightweight implementation to demonstrate procedure and flow.
Use Tech Department and Business Department with reserved example addresses
`tech@example.invalid` and `business@example.invalid`. Separate these routing
labels from authorization workspace access (currently benchmark). Let the reviewer
choose the department explicitly rather than infer it from the JD title.

Introduce handoff schema v2 containing the unchanged backend ranks 1–10, all ten
redacted CV text derivatives/profiles, JD context, comparison report and digests.
Block a complete ten-CV handoff when any required source is missing. Explain
precisely what is unavailable. These are redacted text files, not original PDFs.
Keep old eight-CV schema v1 packages readable and immutable. Version manifests,
store validation and UI labels together; do not reinterpret `top8_ids` as ten.

Flow: choose department -> build current snapshot package -> preview
recipient, subject, body, ten candidates and attachments -> explicit Send button
-> persist a valid MIME `.eml` message with the package -> recorded result and alert.
Changing recipient, source snapshot or package invalidates the prior preview.
Label the panel 'Email demonstration — captured locally' and the confirmation
'Send to demo outbox'. Show success as 'Email captured in demo outbox'; never
claim a real mailbox received it. Offer an owner-authorized download of the
captured `.eml` and preserve the send record across app reloads.

Build the MIME message with the standard library and store it privately in
git-ignored application data. Use the existing outbox with a versioned captured
status, a stable request identity and an atomic artifact write so repeated clicks
do not create duplicate messages. Recover an interrupted capture using the same
request identity. A write failure produces a retryable error and failed alert.
Validate the snapshot, attachments and current access before capture. Keep a small
transport boundary for future SMTP; no production gateway, worker, credentials,
delivery callbacks or mail-service deployment is required for v1.

## 6. Readability and visual system

Use a calm, consistent surface/background palette, a clear title/body/caption
hierarchy, comfortable body size, bounded reading width, generous section gaps
and visible selected/focus states. Avoid dense unformatted JSON as primary UI.
Keep action labels short and specific. Put long evidence in labelled panels;
do not bury essential JD content or blocking errors in collapsed details.

Audit contrast, keyboard order, focus visibility, accessible names, colour-only
signals, touch targets and overflow at desktop/tablet/mobile widths and 200%
zoom. Target WCAG AA contrast (4.5:1 normal text, 3:1 large text). Verify visually
in-browser because Streamlit AppTest does not verify layout or computed CSS.
Constrain CSS to app components and minimise reliance on internal DOM selectors.
Retain the current Streamlit transport unless prototyping proves a specific
interaction impossible; any frontend migration needs a separate scope decision.
