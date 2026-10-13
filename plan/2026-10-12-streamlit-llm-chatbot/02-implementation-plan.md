# Implementation sequence and acceptance criteria

## Approval gate

Review this plan and resolve Q1–Q7 in the README. Approve the intended model role,
classification behavior, data boundary and MVP scope before dependent work starts.
Create the feature branch and commit the approved plan. Use sub-agent-driven
implementation as required by AGENTS.md: assign bounded backend, QA and UI/test
tasks with one owner for shared integration files. Review each contract before
dependent tasks proceed; avoid concurrent heavy inference/training jobs.

## Phase 1 — Artifact and contract audit

1. Inspect the approved model record and available weights without training.
2. Define snapshot/result schemas, classification availability, shortlist/tail
   behavior and score semantics. Document threshold lineage and truncation.
3. Define provider configuration and resolve the chosen Qwen runtime/checkpoint.
4. Record a small synthetic QA fixture set and proposed acceptance bars before
   real-provider trials. Keep QA evaluation separate from ranking evaluation.

Acceptance: each configured model has a loadable frozen manifest; unvalidated
classification is unavailable; no application loader reads benchmark labels or
starts training. Missing artifacts are reported as blockers, not successful checks.

## Phase 2 — Frozen screening backend

Implement inference-only loaders, explicit retrieval and shortlist reranking,
classification where validated, evidence integration and snapshot construction.
Reuse `System`/`Reranker` interfaces and current evidence/extraction modules.
Do not change model training, loss functions or research selection as part of
wrapper integration.

Acceptance: synthetic scorers prove score/ID alignment, deterministic ties,
shortlist bounds, full-pool accounting, explicit unscored tail and correct
threshold behavior. Missing weights produce a named backend failure. Modifying
inputs/configuration changes snapshot identity. Existing pipeline/intents behavior
remains supported through explicit compatibility adapters.

## Phase 3 — QA service and Qwen adapter

Implement bounded request interpretation, evidence context, provider adapter,
versioned prompts, structured responses, citation validation and template fallback.
Add stable candidate resolution and follow-up state tied to snapshot identity.
Use synthetic provider doubles first; no model downloads are required by tests.

Acceptance: malformed responses, fabricated IDs, incorrect quotes, stale snapshots,
unsupported operations and provider outages cannot generate a successful factual
answer. Backend numeric facts remain unchanged. Unknown qualifications produce
insufficient-evidence wording. Documents cannot request tool execution or alter
scores. Provider requests and displayed responses pass redaction.

## Phase 4 — Streamlit integration

Wire the existing UI to the screening and chat services. Add session state for
snapshot/history, progress/error states, model identity, reranking coverage,
classification availability and evidence citations. Reset or invalidate chat
when documents or configuration change. Retain existing reviewed-source lineage.

Acceptance: a user can load synthetic applicants, screen, ask a free-form question,
compare candidates and inspect supporting evidence. Follow-ups use the current
snapshot. Separate sessions cannot see one another's documents or answers.
QA outage falls back without losing rankings. Application startup never trains.

## Phase 5 — Validation and measured smoke trial

Write named invariant tests and demonstrate that each regression test fails against
the code mutation it guards. Cover the contracts above, including context budget
overflow, prompt injection, redaction changing citation offsets, history
invalidation, cross-session isolation and probability/percentile terminology.
Model-dependent smoke tests are optional-group guarded; base tests remain runnable
without data, `ml`, `app` or a model service.

Proposed synthetic QA matrix: at least 30 authored questions spanning the supported
intents and adversarial/error cases. Approval freezes fixture coverage and bars
before provider trials. Require all deterministic backend/citation invariants to
pass; report supported claims / audited claims, question completion counts,
fallback counts and failure examples. Human review checks whether each narrative
claim follows from its cited evidence. No new applicant labels are collected.

Run one real-provider smoke configuration on synthetic documents. Record exact
model/runtime revisions, quantization, prompt digest, decoding settings, cold
load time, warm screening/answer latency, context sizes and memory. Select runtime
limits from measurements and approved Q4 latency needs. Time one configuration
before considering more trials; no unbounded model grid is authorized.

Run narrow tests during development; then run `uv run pytest` once with its full
log in `logs/pytest.log`, read it, and run affected checks through
`uv run python -m candidate_screener.checks`. Inspect runner selection with
`--explain`. Register any new reproducibility check in the runner. D55 requires
cold checks before commits touching model/training code, before plan completion,
and before any final-set score. Do not claim completion while those checks are
deferred. The existing stage-2 cold reproduction may take hours: measure cost and
state it before launch. Jobs exceeding about five minutes run once in the
background with a log, exit marker and peak RSS; end the turn and resume on exit.

This plan adds no final-set experiment. If model adoption or new classification
calibration is requested, approve a separate registered protocol first: train/dev
only for tuning, one frozen final reference, generic/targeted/total together,
JD bootstrap CIs and paired deltas; within-family AUC, `llm_free` ordering and
the A1 reproducibility qualification wherever applicable.

## Phase 6 — Review and documentation

Review integration against the approved contracts and resolved questions. Update
README with `uv` installation/run instructions, provider configuration, actual
backend identity, unavailable-classification behavior and measured limitations.
Add `03-implementation.md` with files changed, test/check logs, QA results,
runtime measurements and deviations. Update linked P9 progress if this fulfills
its tasks. Change AGENTS.md or decisions only when approved rules actually change;
no source-card/catalog changes are needed unless a new data source is registered.

Delivery is complete only when approved scope works end to end, regression
invariants and required checks pass, model/threshold identity is traceable,
and every high-impact open question is resolved or explicitly deferred with the
dependent feature excluded. Deployment and publication require their own concrete
reviewable result and authorization; they are not included in this plan request.
