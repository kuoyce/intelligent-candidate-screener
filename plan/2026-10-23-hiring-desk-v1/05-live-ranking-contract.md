# Proposed live-ranking integration boundary

This is an application-side proposal for the teammate's adapter, not a claim
that an endpoint or model schema already exists. Adapt to the actual interface
when supplied rather than requiring a particular transport.

## Ownership

Retain the existing CSV export adapter as an independent source of review
snapshots. Live requests apply to the new-JD path; unavailability cannot disable
existing export reviews. Source type and immutable snapshot identity determine
which adapter supplies data. Never overwrite CSV scores with live results or
attach an existing export ranking to a newly authored JD.

This upgrade owns confirmed JD inputs, request lifecycle, authorization, progress
and error UI, persisted review snapshots and validated result rendering. The
teammate owns candidate ingestion/pool eligibility, extraction, model loading and
inference. No training at app startup; no UI-generated ranks or Fit decisions.
Live modeling means inference integration, not online training on reviewer chat.

## Request and lifecycle

Application request: request ID, JD ID/revision/content digest, confirmed JD title
and text, configured candidate-pool reference and requested top-k=10. Use
inference text consistent with `text_v2` rules; structured draft fields do not
become new inference metadata/oracles. Derive authorization server-side; a
user-supplied owner cannot grant access. Secrets stay outside stored chat/JDs.

Normalize Python or HTTP adapters into submit/status/result operations. A
synchronous model may complete submit immediately; a long request returns a
handle. Persist requests so reruns/navigation do not resubmit. Support explicit
status refresh/retry without uncontrolled polling or automatic retraining.

## Result

Require matching request/JD revision/digest, run/model identifiers, candidate-pool
version, ordered unique candidate IDs, finite backend scores and their meaning,
candidate source references/text for authorized review, and available extraction/
evidence provenance. Normalize into review snapshots, retaining model order and
honest unavailable-evidence states. Labels/Fit calls come only from a validated
backend result with its decision provenance, never inferred by the UI or chat.

Validate authorization, candidate membership and contiguous ranks. Redact before
human display or committed artifacts. Do not persist unredacted CVs in generic
session storage; retain authorized source references and redacted views. The LLM
may explain validated evidence but cannot invent candidates, scores or ranks.

An older JD revision's result cannot replace a newer active ranking. Keep
snapshots immutable; re-ranking creates a new snapshot/history. Duplicate
completion signals create one ready alert. A failed run preserves historical
results without labelling them current.

## Incomplete and unavailable results

No adapter: service unavailable, published JD retained. No candidates: explicit
empty result. Fewer than ten eligible candidates: show the real count, permit
only supported review actions and block the complete ten-applicant package.
Adapt operational result handling without weakening exactly-ten validation for
benchmark exports. Missing evidence/source text stays unavailable; do not borrow
another job's or candidate's data to fill a screen or package.

## Completion evidence

The implemented adapter is selected by trusted server configuration
`HR_PORTAL_RANKING_ADAPTER=module:function` plus
`HR_PORTAL_CANDIDATE_POOL=<pool-reference>`. A Python wrapper may call the
teammate's HTTP service; no endpoint is guessed. Its callable accepts
`adapter(operation, payload)`, where operation is `submit` or `status`.

Every response echoes `request_id`, `jd_id`, `jd_revision`, `jd_digest`,
`workspace`, `authorization_scope` and `candidate_pool_ref`. Ready responses
also include `run_id`, `model_id`, `candidate_pool_version`, `score_semantics`,
`eligible_candidate_ids` and `candidates`. Candidates contain `candidate_id`,
contiguous `rank`, finite `score`, `source_ref` and optional `cv_text`/entities.
Use UUIDs, digest-based IDs or `Applicant-<number>` supported by the portal's
typed identity validator. Arbitrary display names are rejected before marking
results ready so chat/report/package storage remains valid. Full field bounds,
status behavior and evidence provenance are documented in
`src/candidate_screener/app/portal_ranking.py`; synthetic contract fixtures live
in `tests/test_portal_ranking.py`. These fixtures are never offered as real ranks.

Synthetic injected adapters test lifecycle and state isolation. Complete live
integration with a documented run through the teammate's actual interface and
authorized pool. Record provenance/timing without exposing applicant PII. Until
that succeeds, report the adapter/UI as implemented and live connection pending.
