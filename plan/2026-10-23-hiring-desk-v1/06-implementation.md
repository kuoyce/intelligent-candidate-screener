# Implementation — 23–24 October 2026

Application work is implemented. Real live-model binding remains pending the
teammate's adapter and authorized candidate pool; no model or rank is fabricated.
This record does not declare plan completion or cold research reproducibility.

## Delivered behavior

- JD cards, previous/next controls, title search, grounded summaries and full
  redacted JD text above review actions and existing evidence chat.
- Independent CSV and operational ranking adapters; saved reviews retain their
  snapshot identity and bounded conversations when switching jobs. A read-only
  real-export check loaded all 60 JDs, each with ten existing ranks and full JD
  source text. No labels, training or final-set scoring were changed.
- Production-only sign-in/navigation, Create new JD, Alerts and Saved reviews;
  the previous 40-day account lifetime and bottom-sidebar logout remain.
- Deterministic guided JD questions with required/preferred fields, optional
  unspecified answers, edit/resume, optimistic revision checks, explicit preview
  confirmation/publication and a configurable ranking submit/status boundary.
- Private drafts; published JD content and publication alerts shared read-only
  within the authorized workspace. Ranking requests remain author-controlled;
  per-user publication read states do not expose private drafts or mail events.
- Versioned ten-CV handoff packages, historical eight-CV compatibility, backend
  score meaning in v2 reports/mail, explicit email preview/confirmation and
  private atomic `.eml` capture/download. Capture history survives review changes.
- Real publication/ranking/capture alerts with unread count, explicit read/unread
  controls and guarded links. No invented applicant-arrival notifications.

## Validation

- Final full suite: **1,406 passed, 10 skipped, two failures** in the previously
  observed baseline golden tests. They report the same vocabulary/model drift
  seen before this task; no baseline/training implementation was changed.
  Log: `logs/hiring-desk-v1-pytest-final.log`.
- Independent review then found one additional stale-saved-review refresh loop.
  A bounded regression reproduced a second unwanted `open_job` call; routing the
  single refreshed session into the job browser fixes it. Focused recovery and
  adjacent UI/controller/workspace tests passed (32 tests) after that change; see
  `logs/hiring-desk-v1-recovery-final.log`.
- Service-specific verification: 84 JD/ranking/CSV tests and 58 report/store/mail
  tests passed, including shared-publication migration, independent reads,
  ownership, stale responses, arbitrary candidate-ID rejection, score provenance,
  capture recovery and legacy v1 rendering. Logs:
  `logs/hiring-desk-workspace-publication-v2-tests.log` and
  `logs/hiring-desk-mail-review-focused.log`.
- Regression evidence includes old-behavior failures for browser switching,
  saved selection, stale recovery and requirement priority, plus membership,
  candidate identity, workspace publication and mail/schema invariants. Logs
  named `logs/hiring-desk-*-before.log` / `*-mutant.log` retain the results.
- The checks runner's affected-record selector, scoped to this task's app code,
  selects no research records: `logs/hiring-desk-v1-checks-final.log`. Existing
  unrelated working-tree changes and descriptive documentation were excluded
  from source dependency selection. No model/training/cold scoring was launched.
- Browser navigation, mobile width and root text scaling checks passed with
  synthetic fixtures: `logs/hiring-desk-browser-v1-final.log`. See the UI audit.
- The real CSV adapter check printed only aggregate counts:
  `logs/hiring-desk-v1-real-csv.log`. Local applicant text remains uncommitted.

## Integration and operational notes

Configure `HR_PORTAL_RANKING_ADAPTER=module:function` and
`HR_PORTAL_CANDIDATE_POOL=<authorized-pool>` when the teammate supplies the
interface. The callable returns promptly with queued/running or validated ready
results. The UI never trains models or silently borrows CSV candidates for a new
JD. A real end-to-end model run remains required before claiming that connection.

*Update 24 October 2026 (plan `2026-10-24-hiring-desk-inference-api`):* the
connection is bound and was run end to end in a browser on an isolated copy of
the workspace: `HR_PORTAL_RANKING_ADAPTER=candidate_screener.app.portal_inference:adapter`,
`HR_PORTAL_CANDIDATE_POOL=a2-final-cvs` (the 600 A2 final-set CVs, chosen
explicitly by the model owner, so no pool is chosen silently). Results, timings
and the app-side changes it needed (`ranking_system`, a parseable experience
line, "Why #k") are in that plan's implementation record.

Email uses `example.invalid` recipients and local MIME capture as requested;
production SMTP is intentionally outside this demonstration. Live inference
transport can be wrapped by the teammate's adapter; no endpoint was guessed.

Additive SQLite migrations preserve old sessions, packages and read states.
Before restarting the local app on port 8503, a private SQLite backup was created
at `data/processed/hr-portal/backups/workspace-pre-v1-20261024-002009.sqlite`.
The temporary synthetic preview server was closed. The production sign-in page
was checked in a browser after restart, without signing in as any reviewer.
Reviewer-authored application state requires private backup and is documented as
operational data, not a reconstructible acquired benchmark. Existing unrelated
working-tree changes were preserved; only the approved plan was committed.
