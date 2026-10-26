# Delivery, evaluation and open questions

## 1. Implementation sequence after approval

Create `feat/hiring-desk-v2-conversation`, commit the approved plan, and use
scoped sub-agents with distinct ownership. Preserve the current branch and
backup stash; do not implement over unrelated changes. Record the approved v2
scope in decisions with an unambiguous new ID after checking both histories.

| Phase | Deliverable and proposed ownership | Exit evidence |
|---|---|---|
| 0: baseline | Parent inventories assets/runtime and freezes v1 dialogues, failure taxonomy, evaluation manifest and targets | Reproducible v1 outcomes; one measured configuration and actual resource limits |
| 1: conversation | State/controller agent adds versioned DialogueFrame, pending-slot resolution, plural comparison context, participant edits and exact claim reopening | New regressions demonstrated failing against v1; saved/reloaded happy flows pass |
| 2: comparison facts | Evidence agent adds shared criterion matrix, coverage/qualification states, atomic QA and checked relations | Same facts through chat/dropdown/report; every requested owner represented; no naked-span polarity inference |
| 3: conversational baseline | Parent/UI integration adds direct application prose, friendly labels, selective tables and source controls | Q1 completes the frozen development walkthroughs without any generator |
| 4: frozen Qwen | Model agent introduces a separate v2 AnswerPlan adapter, bounded prompt, validation, labelled fallback and real local smoke | Q2 measured against Q1; actual availability and model acceptance recorded separately |
| 5: targeted refinement | Conditional reader calibration/adaptation and Qwen prompt/capacity/LoRA work from classified dev errors | One intervention at a time; timed pilot; reviewed data/resource configuration; no final-set access |
| 6: release review | Independent reviewer checks contracts, migrations, UI, source ownership, measured quality and rollback | Full tests, applicable registered checks, held-out v2 evaluation and implementation record |

Do not implement all model variants before discovering whether phases 1–4 solve
the complaint. The first shippable v2 can use frozen weights if it meets the
conversation contract. Training is conditional improvement, not a completion ritual.

## 2. Concrete code boundaries

Proposed modules, adjusted only after interface review:

- `app/conversation_schema.py`: frame/evidence/relations/answer versions and validators.
- `app/conversation_service.py`: resolve -> retrieve/read -> compare -> compose;
  wraps existing `PortalExtractiveService` rather than importing legacy UI.
- `app/comparison_facts.py`: explicit criterion cells and permitted relation derivation.
  Evidence-agent ownership includes polarity/qualification-state assignment and
  one validator plus positive/counterexample fixtures for each permitted relation.
- `app/conversation_render.py`: concise controlled prose shared by model/fallback modes.
- `llm/conversation_qwen.py`: v2 prompt and AnswerPlan mapping using existing bounded
  `QwenProvider`; preserve exact-quote validation in historical `portal_qa`.
- `app/portal_controller.py`, `portal_store.py`: frame migration and existing
  optimistic concurrency, per-turn identity, snapshot/access rechecks.
- `app/portal_ui.py`, `portal_workspace.py`: explicit v1/v2 feature flag, service
  binding, display labels, citation reopening and factual output only after validation.
- `llm/qa_train.py` plus a versioned dataset adapter: only conditional reader work.
  Model-agent ownership includes the missing dev-evaluation/checkpoint-selection
  loop, calibration record and tests proving held-out families are never read.
- Package-owned evaluation/training modules: runnable code stays under `src/`,
  never `plan/`; any new record `--check` enters the central registry.

Avoid modifying ranking systems to create chat recommendations. Existing evidence
caches remain private and read-authorized. Add frame/schema version to saved
sessions with backwards-compatible reading and an explicit v1 rollback path.

## 3. Proposed authored dataset

Before tuning, freeze 120 wholly invented scenario families: 72 train, 24 dev,
24 test, each with one JD, 2–3 CVs and approximately six linked turns. This is a
proposed collection size, not data that already exists or a power claim. Review
authoring effort and dev pilot variance before finalizing the held-out manifest.

Every family includes participant selection and a follow-up; balance broad and
focused comparisons, missing evidence, ambiguous references, contradictory or
negated statements, ongoing/expired qualifications, explicit skill tenure and
long-document evidence. Include greetings, acknowledgements and operational
guidance without polluting factual-answer targets.

Split by source family **and near-duplicate/template lineage**, not by question,
paraphrase, candidate name or turn. Hold out factual combinations and language
patterns; renaming candidates does not create independence. Persist source and
split digests before experiments. If a pattern is inspected during development,
its outcomes are exploratory rather than hidden test evidence.

One record contains source documents, scenario/lineage/split IDs, full dialogue,
expected frame changes, atomic per-owner questions, acceptable answer spans or
unanswerable status, semantic qualification states, allowed comparison relations,
expected clarification and allowed next actions. Alternative natural wordings
are accepted by rubric, not exact string match. Authored answer plans must be
human-reviewed against invented sources. No model-generated teacher answer is
treated as correct merely because it is fluent.

Derive RoBERTa atomic span examples and Qwen instruction examples from the same
family split, so one model cannot train on the other's held-out CV. Reuse existing
synthetic scenarios as seen development/regression material; do not relabel them
as unseen v2 test data. Keep teacher/model assistance provenance if used; this
does not authorize Claude dispatch or new relevance/entity annotation.

Training uses authored train families only. Do not use A2 final/eval documents,
Fit judgements, chat-1/chat-2 test labels, HR production transcripts or private
CVs for prompt/model tuning. Research chat-3 permission does not automatically
authorize conversational-model training. No real applicant-domain collection is
needed for this initial plan. Any later expansion needs its own data decision.

No dataset or acquisition claim changes during this planning task. When authored
artifacts are actually added, update `data/README.md`, catalog/card and provenance
in the same commit; model bytes stay ignored and reproducible manifests/metrics
live under a new `output/chatbot-v2/` namespace.

## 4. Experiments that identify the cause

| Experiment | Fixed | Changed | Question answered |
|---|---|---|---|
| Q0 -> Q1 | Base readers, source fixtures | State, retrieval, matrix, controlled prose | How much is application orchestration responsible? |
| Q1 -> Q2 | Frame, evidence matrix, readers | Frozen 0.5B composition | Does Qwen add usefulness beyond controlled comparison prose? |
| R1 -> R2 | Retrieval and base weights | Dev-selected no-answer threshold | Can calibration repair reader over/under-answering? |
| R2 -> R3, conditional | Evidence, question rewriting, generator disabled | Reader weights | Does domain span adaptation improve facts safely? |
| Q2 -> larger frozen config, conditional | Prompt/evidence/decoding where feasible | One capacity/runtime configuration | Is small-model capacity the remaining bottleneck? |
| Selected frozen Qwen -> LoRA, conditional | Same model, prompt, retrieval, decoding | Adapter weights | Does task training improve dialogue without evidence regressions? |

Select the release candidate on dev; freeze model, threshold, prompt, policy and
runtime manifests before final test. Test Q0, Q1 and the selected v2 on the same
held-out families once for paired reporting. Do not use the test to choose among
LoRA checkpoints, model sizes, prompts or safety policies. If test failures cause
further tuning, those results become exploratory and a new held-out set is needed.

Error attribution: resolution -> retrieval -> reader -> relation construction ->
composition -> presentation. Record the earliest failing stage. Test with oracle
passages and oracle evidence cells to isolate model errors. No training a reader
to fix a renderer failure, or increasing generator size to fix wrong owner IDs.

## 5. Metrics and proposed gates

Targets below are proposals for approval before evaluation, not observed results.
Report each metric with numerator/denominator and a 95% CI; report paired deltas
for variant comparisons. Bootstrap over independent scenario/JD families, keeping
all candidates/turns together, never treating turns as independent observations.
Report category breakdowns with the aggregate; synthetic results are not A2
generic/targeted benchmark scores. Any future A2 reporting must follow its own
stratum and CI rules and cannot be used to tune this flow.

| Measure | Definition / proposed release gate |
|---|---|
| Dialogue completion | Whole requested task plus follow-up completed without restating available context: >=90% and >=10 percentage points over Q0 on dev; report held-out result without retuning |
| Reference resolution | Correct candidate set, criterion and pending-slot recovery: >=95%; 100% on fixed critical reference regression cases |
| Balanced comparison | Every requested participant has a supported or explicit unknown/unavailable cell: 100% of accepted responses |
| Claim support | Human-audited clause matches owner, source meaning, negation and numeric linkage: >=98%; zero accepted critical fabrications in the frozen safety suite |
| Critical invariants | No unauthorized/stale evidence, invented rank/score/Fit, source injection action or changed numeric/negation/credential status: zero failures in the fixed suite |
| Reader quality | Exact match/token F1 on answerable atomic questions; separate unanswerable false-positive/false-negative and qualifier retention rates; R3 promotion needs >=3 pp dev F1 gain without increased unanswerable false-positive rate |
| Qwen benefit | Compared with Q1, >=0.5 increase on a 1–5 usefulness rubric and no support/ownership regression; otherwise ship Q1 or retain Qwen as experimental |
| Model acceptance | Validated generated plans / attempted model turns; aim >=90%, with fallback and timeout rates reported separately from total service success |
| Responsiveness | Proposed warm two-person end-to-end p95 <=15 s; progress indication promptly, hard shared inference deadline <=30 s; cold load measured separately |

If resource limits or dev pilot show these bars are unsuitable, amend them before
held-out evaluation with a reason. Wide CIs/insufficient evidence are reported
as such; a small set with zero observed incidents does not prove zero risk.
Do not hide a generator failure by reporting a successful template as a Qwen win.

Two reviewers independently score usefulness, clarity, continuity, evidence
faithfulness and unnecessary clarification on a blinded, shuffled set of full
dialogues. Record agreement and disagreements without an agreement target.
Critical factual failures require source-based adjudication. Review roles and
availability must be confirmed before collecting scored human evaluations;
these are synthetic conversational assessments, not applicant relevance labels.

## 6. Regression and real-runtime verification

Show named regressions failing against v1 for plural qualifications, remembered
criteria after clarification, add/remove candidate, ordinal references, exact
claim reopening and negated spans used as positive qualifications. Include
source revocation, changed ranking/JD, persisted session migration, concurrent
submission IDs, one missing participant, partial retrieval and model timeout.

Add UI walkthroughs for the happy flows in document 01. The happy-path test must
assess the answer's comparative conclusion, not just a table/citation existing.
Mock-provider tests verify schema/guards; bounded actual-model tests establish
local inference behavior and latency. Neither substitutes for the other.

Run focused tests during work and the full suite once stable; keep logs. Existing
merge validation is 1,678 passed / 11 skipped / two baseline golden failures;
track those failures separately rather than altering frozen records. New chat
failures must be resolved and the release record must show the actual suite result.
Use the check runner's import-closure selection and estimates; note `--explain`
also runs checks in the current implementation, so inspect selection through a
read-only listing/helper before launching a costly run.

Follow D55 for training/model commits or plan completion and D58 (research) for
owner approval before long/cold/all checks. Start one timed pilot before training
or model-size expansion, record wall time/peak process-tree memory, extrapolate,
and obtain resource/time confirmation before a long run. Do not run heavy model
jobs together. Jobs over five minutes run once in the background with an exit
marker/resource log and the turn ends. No hours-long job is implied by approval
of this draft. App-only changes should not trigger unrelated benchmark retraining
without inspecting their actual dependency impact.

## 7. Rollout and completion

Feature flag v2 per local configuration. Keep v1/table fallback and legacy turns
readable. Release only the dev-selected model/runtime after held-out review;
report limitations and actual modes. Roll back on owner/source violations,
material unsupported comparative claims, or unbounded latency. Model selection
for chat never changes the research-selected ranking system.

Completion requires approved contracts, implemented happy/recovery flows,
versioned manifests and rollback, measured runtime/model availability, documented
evaluation including fallback rates, and required checks. A working synthetic
demo alone is not completion. No external deployment/push is requested by this plan.

## 8. Persistent open questions and proposed defaults

| ID | Question / proposed default | Blocks |
|---|---|---|
| V2-01 | Resolved: owner approved implementation24Oct2026 and resumed25Oct2026; hybrid scope recorded as D65 | None |
| V2-02 | What hardware/RAM/device is available, and is a larger local checkpoint acceptable? Default existing 0.5B inference only until measured | Capacity experiment and training, not state/matrix work |
| V2-03 | If 0.5B underperforms, prioritize one 1.5B frozen trial or LoRA? Default trial the frozen configuration before training, subject to hardware | Optional model refinement |
| V2-04 | Who reviews synthetic targets and blinded conversations? Default two independent project reviewers; identities unassigned | Scored human evaluation/data acceptance |
| V2-05 | Accept proposed quality/latency gates and family counts? Freeze after dev authoring/resource pilot, before held-out evaluation | Final evaluation registration |
| V2-06 | Is controlled AnswerPlan composition sufficiently natural? Default release this first; allow paraphrase only after separate support review | Paraphrase experiment, not controlled v2 |

Resolved by this request: focus is v2 meaningful HR comparison conversation and
Qwen/RoBERTa refinement planning. No approval inferred for training, new labels,
hosted processing, downloads or changes to applicant ranking.
