# Guided HR portal implementation record

19–20 October 2026. Status: local review workflow implemented; final broad
verification in progress. Corporate SSO, event feed, department directory/mail
gateway and original PDFs remain unconnected, not simulated as completed services.

## Approval and scope

User approved implementation and supplied the CV/JD extraction CSV paths.
Branch: feat/hr-guided-portal-2026-10-19. Approved plan commit:5becff6.
Data, state/outbox and reporting work were delegated; parent integrated identity,
controller, primary UI and independent review. Existing unrelated dirty work is
preserved. No model was trained, ranking recomputed, benchmark scored or email sent.

The primary streamlit_app.py now opens the dropdown-only portal. The earlier
free-form/upload/interview prototype was preserved as legacy_streamlit_app.py;
its existing behavior tests target that explicit legacy entry point.

## Actual input contract

Read-only audit found60JDs,600CVs,600ranked pairs and10ranks perJD. The supplied
X2 export is a benchmark pool, not a full company applicant search or verified R2
production endpoint. The adapter preserves x2_rank and x2_score and validates
contiguous ranks/unique identities/joins; it never reranks, backfills invalid rows
or reads final_label into its loaded data. Source preview uses only jd_id,cv_id,
jd_text,cv_text from the existing text_v2 cache, without metadata or labels.
All600CV and60JD text previews were available at audit time.

Permitted-column run identity: phase2-x2:8e85128fc6f47fd69ef110bf. This identifies
loaded content; producer timestamps and original model-run metadata are not known.

| Input | SHA256 |
|---|---|
| a2-cv-extraction.csv | af349203ff24b93ce60191fbf10fa0fb81570cc2a4bd046b88449ea2623e36df |
| a2-jd-extraction.csv | 0ec851a988766ac08b40866d23c4a6727bcd81f9560177d802738e0b0f590b0d |
| a2-x2-scores.csv | 1e5d107e49cb137c549391902b5e1e293b76db548ed484cb7201f0ab1893e4d5 |

Skills/qualified skills, experience numbers and predicted role-family aggregates
are shown with provenance. No knowledge/degree/CV-title entities are invented;
JD title is labelled table metadata, not a verified span. The four-category human
entity campaign is unchanged. EXPERIENCE_YEARS is a separately provenanced display
field. CSV mentions do not establish source offsets, active qualifications or gold
accuracy. Optional contextual excerpts use redacted-preview coordinates only.

## Delivered modules and behavior

- portal_data.py: whitelist CSV reads, validation, immutable/copy-return snapshots,
  mid-read file-change checks, redacted source previews and isolated in-memory demo.
- portal_auth.py: explicit local-token accounts with salted PBKDF2 hashes, private
  files, provisioning lock, expiry and revocation-version checks. A login ID alone
  never grants access. This is not company SSO. Credentials are not conversation
  state and are not printed by provisioning.
- portal_store.py: SQLite owner/department/authentication-realm isolation; revision
  checks; event dedup/read state; delivery drafts/outbox with idempotency. Real and
  demo subjects cannot collide even in a shared test database. No external service
  is called; an unconfigured gateway cannot queue/send or produce a fake receipt.
- portal_controller.py: rechecks identity and source freshness on protected actions;
  opens/resumes reviews; rejects foreign candidates; stores private immutable
  reports; prepares same-snapshot packages; supports explicit unavailable handoff.
- portal_reports.py: text/escaped HTML comparisons of10 with explicit ranks1–8
  shortlist; deterministic ZIP with8redacted CV text derivatives/profiles and
  artifact hashes. Missing source documents/partial pools block a complete package.
  Original PDFs/page layout are not fabricated. Numeric facts and generated opaque
  IDs remain exact; human-readable source strings are redacted.
- portal_explainer.py: fixed-category source retrieval with optional explicitly
  configured Qwen selecting a validated excerpt ID. No generated factual prose,
  ranks or actions are accepted. Absent/malformed/unavailable inference falls back
  to deterministic source selection. It does not fit or call an annotation model.
- portal_ui.py: narrow dropdown review menus, read-only text dialogs, comparison,
  source excerpts, session/report storage and local sign-in outside conversation.
  Artifact bytes are delivered inline to the authorized session, without creating
  a reusable unauthenticated shared media-file URL.

Configured local review is the default following the user's20October correction.
The explicitly selected synthetic demo has two invented jobs and no access to real files.
Configured local review loads actual exports only after local-token authentication
and benchmark-scope authorization. Demo storage is a separate private temporary
database; actual local sessions live under ignored data/processed/hr-portal/.
Retention/encryption/SSO policy still requires company configuration.

## Verification and review evidence

Synthetic tests cover guarded sign-in, expiry/revocation, cross-owner/realm access,
no pre-login real reads, tampered rankings, forbidden labels, race-during-load,
previews, numeric/negative context, report/attachment identity, missing documents,
session revisions, stale drafts, duplicate-safe delivery state and UI reruns.
No base test requires real data or model weights; Streamlit tests are optional.

Fail-before logs include portal-data-regression.log, portal-report-regression.log,
portal-report-decimal-regression.log, portal-report-identity-regression.log,
portal-auth-revocation-regression.log, portal-store-realm-regression.log,
portal-store-gateway-regression.log, portal-store-identifier-before.log and
portal-explainer-regression.log under logs/. Reviews found and fixed wrong
dataclass serialization, numeric/ID corruption by blanket redaction, authentication
realm collisions, account-file concurrency and gateway provenance overwrites.

Final focused verification passed73tests, including the portal and preserved
legacy UI/session behavior (logs/portal-final-focused.log). The scoped affected
record runner selected zero research checks for the nine changed portal and
entrypoint modules (logs/portal-scoped-checks.log); unrelated pre-existing dirty
model/training files were excluded from this task's scope. No model/training
code or final-set score is changed, and the production plan is not marked complete.
The full repository suite is running with output in logs/portal-full-pytest.log.
Read-only real-data report/package
validation built10comparison rows and8CV attachments without printing source text
or persisting document bytes (logs/portal-real-report-validation.log). No extraction
accuracy estimate is inferred from successful joins or report generation.

Local Streamlit service started on http://127.0.0.1:8503 with a successful health
response (logs/portal-streamlit.log). Configured local review is the initial
workspace and uses the supplied exports only after token
authentication. Provision a local account using the README command; no account
or token was invented for the user and no real personal text was put in logs.

## Remaining dependencies and limits

The local workflow is reviewable; this is not a completed production portal.
Company IdP/permissions, live ranking database/run producer metadata, authenticated
portal events, actual department recipients/mail gateway, original redacted PDF
derivatives, and retention/encryption/hosting policy remain unresolved in the open
questions record. Existing benchmark CSVs do not supply those systems. Alerts state
that the feed is disconnected rather than claim no applications exist. The mail
menu states delivery is unavailable and exposes no candidate-contact/scheduling
operation. No token, email recipient or external service was invented to bypass
these dependencies.

The underlying research winner, final labels, entity annotations and prior model
training jobs are unchanged by this portal task. The older QA experiment remains
separate; no interrupted job is restarted or declared complete here.
