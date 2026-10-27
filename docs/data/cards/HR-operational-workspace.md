# Hiring Desk operational workspace

Approved: 23 October 2026, [v1 plan](../../../plan/2026-10-23-hiring-desk-v1/README.md),
decision D60. This is application state, separate from acquired research sources.

- Location: git-ignored `data/processed/hr-portal/`.
- Contents: hashed accounts, authorized reviews, operational JD revisions,
  request/model/pool provenance, redacted ranking snapshots, reports, versioned
  ten-CV packages and privately captured demonstration MIME messages.
- Legacy CSV-backed export reviews and eight-CV handoff records remain supported.
  Corpus counts, labels, research results and annotation are unchanged.
- Operational JDs are not benchmark annotations or training input. Live-model
  connection awaits the teammate's actual interface.
- No population count is claimed: records depend on local reviewer actions.
- Added 24 October 2026 (D61): `entity-cache/` contains rebuildable, private
  GLiNER-medium inference results from authorized redacted JD/CV text, scoped by
  owner/workspace, source digest and pinned model/label/chunking configuration.
  Each result retains exact offsets, confidence and coverage/status; these are
  model suggestions, never human gold or new training data. QA spans retain
  source references in saved chat. Neither model changes applicant ranks.
- Empty schemas regenerate at app startup; synthetic fixtures recreate tests.
  Real reviewer-authored drafts/history require a private local backup, since
  fetch/build scripts cannot reconstruct user-authored application state.
- Only redacted display text is persisted. Opaque IDs/digests preserve joins.
  Mechanical redaction does not guarantee anonymity.
- Tech/Business example recipients use `example.invalid`; captured `.eml` files
  are not delivered to an external mailbox.
- Added 28 October 2026 (D68): the Hiring Desk in Docker keeps its own workspace in
  the data folder, at `data/datapack/processed/hr-portal/`, separate from the native
  `data/processed/hr-portal/` (`datapack create --with-workspace` copies the native one
  in once). `datapack create` keeps it, `datapack handover` excludes it (with the
  caches), and `datapack purge --yes` deletes it. Back it up before every upgrade with
  the commands in the [Docker guide](../../docker.md#9-backups-operator).

Migrations are additive. Never commit local accounts, documents or captured mail.
