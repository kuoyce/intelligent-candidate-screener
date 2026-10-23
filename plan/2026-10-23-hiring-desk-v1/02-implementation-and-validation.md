# Execution sequence and acceptance

## Phase 0 — approval and baseline

Resolve the high-impact questions, approve the design, create a branch and commit
the plan. Capture current UI screenshots using synthetic fixtures only. Record
current test failures before editing; the prior auth task observed two baseline
golden failures from vocabulary/model drift. Preserve unrelated working changes.
Confirm whether Impeccable becomes available; otherwise run the documented audit
manually and identify it accurately. Never claim an unavailable tool was used.

## Phase 1 — screen structure and JD navigation

Required regression acceptance: with the live adapter unconfigured or failing,
existing CSV-backed jobs still open, preserve all ten exported ranks/scores and
support JD/CV source previews, review actions and grounded chat. Exercise CSV
job -> live-ranked job -> CSV job switching and saved CSV-review restoration;
verify source identity, candidate selections and history remain separate. Do not
alter the export adapter's schema checks to accommodate live-result formats.
Use synthetic CSV/source fixtures so these checks require no downloaded data.

UI sub-agent implements the shell, removes public demo switching, adds accessible
job cards, search/pagination, full-JD view and summaries. Controller/state work
adds explicit job switching and safe resume. Review actions and chat remain below
JD context. Validate A -> B -> A, including pending chat, selected CVs, stale
snapshots, saved review restore and missing full-source text.

Acceptance: users can select and read a whole JD, review it, and select another
JD without logging out or losing the previous review's saved context. No history
or evidence appears under a different job. No production demo selector exists.

## Phase 2 — guided JD creation, persistence and live ranking

Service sub-agent adds versioned migrations and a JD draft service; UI sub-agent
adds the guided page and editable preview. Use existing owner/revocation guards.
Drafting functions remain independently testable with synthetic input.
Map the proposed live request/result contract to the teammate's actual interface
when available. Implement an injected ranking adapter, durable request status,
result validation and conversion into review snapshots. Keep transport-specific
mapping in the adapter. Preserve existing export reviews.

Acceptance: complete, edit, save, reload and publish a JD; required/preferred and
experience constraints are preserved. Duplicate publish creates one job revision
and one alert. Missing model service still allows guided completion. Another
account cannot load or edit a private draft. New jobs have no fabricated ranks.
An explicit ranking request leads to real adapter results entering the normal
review/chat flow. Test stale responses after JD edits, source changes, duplicate
submissions, adapter failure/unavailability and job switching during inference.
A validated run with the teammate's service is required to claim live integration
complete; fixture tests prove only the application boundary.

## Phase 3 — alerts

Extend event kinds and visibility; attach publication, ranking and email-capture producers.
Add unread badge, filters, explicit read controls and authorized deep links.

Acceptance: publication creates an alert once; read state survives reload and is
independent per user. Unauthorized users cannot read another user's delivery
details. No synthetic alerts appear in the real workspace. New-applicant alerts
remain unavailable until an ingestion adapter is supplied.

## Phase 4 — ten-applicant package and email

Delivery sub-agent versions package generation and store validation, adds the
example department directory and local `.eml` capture adapter. Replace the send
placeholder with a preview/confirm/demo-outbox flow and persisted download.
Reuse standard-library MIME handling; avoid mail-server and background-worker
dependencies for this demonstration.

Acceptance: exactly ten current ranked candidates in the v2 package; ten redacted
CVs and profiles, comparison and manifest match the selected snapshot. Old v1
packages remain readable. The `.eml` parses correctly and its recipients, body
and attachment match the preview. Repeated clicks/reloads do not duplicate
capture. Missing source, revoked access, stale snapshots and write failures
produce accurate states and alerts. UI clearly states local demonstration capture.

## Phase 5 — audit, review and validation

Independent review covers cross-job state, migration compatibility, auth guards,
department routing, email state transitions and accessibility. Audit all pages:
login, job browser, review actions, chat, JD creation, alerts, saved reviews and
send preview, including loading/empty/error states. Capture before/after images
from synthetic fixtures and fix material contrast, clipping and navigation defects.

Run focused tests during changes and prove regression tests fail against the
behavior they guard. Then run the full `uv run pytest` once, keeping its log.
Use the checks runner for affected records; first inspect `--explain` so unrelated
dirty files do not accidentally trigger a large evaluation/training campaign.
No final-set scoring is needed. Follow D55 if any cold requirement is triggered,
including before declaring the plan complete; schedule long checks separately
and report environmental blockers instead of claiming a clean result.

Document migrations, demo-outbox use, ranking configuration, recovery, JD lifecycle
and source limitations in README and the implementation record. Update the
decision log for the top-ten contract and approved operational data storage;
update data catalogue/cards only if the benchmark/source data facts change.

## Likely code locations

Extend `app/portal_ui.py`, `portal_controller.py`, `portal_data.py`,
`portal_store.py` and `portal_reports.py`; keep Streamlit thin by introducing
small services such as `portal_jobs.py`, `portal_jd_drafting.py`,
`portal_notifications.py`, `portal_ranking.py` and `portal_mail.py`. Add targeted
tests under `tests/`; keep the local mail capture synchronous and small.
Use versioned SQLite migrations/backups rather than destructive table recreation.

The phases are dependency-ordered; UI, drafting and delivery sub-agents may work
in parallel only after their shared state/storage contracts are agreed. Benchmark
model code, frozen annotation instruments and research records are out of scope.
