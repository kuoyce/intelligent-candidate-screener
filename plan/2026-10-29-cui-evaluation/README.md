# CUI evaluation: implementation plan and study specification

29 October 2026. **Status: implementation approved by the owner on 29 October 2026.** Implementation may proceed; reviewer assignment, live configuration confirmation and long-run approval remain open. No live evaluation, training, deployment or scoring has started.

## Objective and boundaries

Evaluate the implemented v3 CUI on factual correctness, multi-turn continuity, the incremental usefulness of generation, and operational recovery. Compare v3 against its controlled answer path using identical authorized evidence. Keep ranking evaluation separate.

This is a preliminary synthetic study, not a claim of recruiter productivity, production reliability or general QA accuracy. The ten-family proposal below is smaller than the earlier 32-family v3 proposal and does not satisfy or replace its promotion gates. Results do not promote v3 to the default. No model training, downloads, real applicant processing, relevance annotation or changes to ranking are authorized by this plan.

Architecture reference: [CUI architecture and methodology](../../docs/conversational-qa-architecture.md). Existing acceptance contract: [v3 evaluation plan](../2026-10-25-hiring-desk-v3/02-implementation-and-evaluation.md).

## Step 1 — Approve scope and resolve study ownership

1. Obtain approval for this concrete protocol and its preliminary-study status.
2. Assign a scenario author and two independent human reviewers. Reviewers must not see system identities during rating; record any involvement in implementation or scenario authoring.
3. Confirm target hardware and the pinned Qwen3.5 runtime to evaluate.
4. After approval, create a dedicated branch and commit the approved plan. Preserve unrelated changes, including the report, portal UI and architecture document; do not stage them implicitly.

**Exit:** approved scope and recorded owners. Approval of implementation does not replace the separate permission required before a long live-model run.

## Step 2 — Define fixtures, expected outcomes and review rubrics

Create two development-only families for harness debugging and reviewer calibration, plus **ten new evaluation families**. A family contains an invented JD, two or three invented CVs, related questions and any dialogue turns. Related source templates and paraphrases remain within one family. Do not reuse previously tuned degree walkthroughs as held-out examples.

The evaluation set contains:

- **30 standalone factual questions:** three per evaluation family.
- **Eight dialogues of four to six turns:** distributed across eight evaluation families; additional to the standalone questions.
- **Five failure conditions:** unavailable generator, generation timeout, malformed output, missing evidence and changed/revoked source access. These form a separate recovery suite and do not enter ordinary generation-rate denominators.

Cover explicit skills, ongoing degrees, expired certification, negation, missing evidence, total versus skill-specific tenure, balanced comparisons and unsupported superiority premises. Dialogue tasks cover participant changes, criterion switches, pronouns, source reopening and job switching.

For each question or turn, specify:

- Expected intent, participants, criterion and clarification behaviour.
- Required atomic facts, their source owner and exact supporting spans.
- Mandatory qualifications/counterfacts and prohibited inferences.
- Whether an uncertainty response is required; allow partial answers when only part of the question is supported.
- Whether the turn is eligible for generation, defined from the task before observing outputs.
- Task completion criteria; do not prescribe a single answer string.

Two reviewers calibrate on development examples only. Define anchored 1–5 ratings for directness, clarity and usefulness (1: fails the task; 3: useful but materially incomplete or awkward; 5: direct, clear and complete). Record factual correctness independently of style.

**Exit:** reviewed synthetic fixtures, an explicit coverage matrix and rubric. No evaluation outputs have been generated.

## Step 3 — Implement the reusable evaluation harness

Proposed package: `src/candidate_screener/evaluation/cui/` with schema, fixtures, runner, review export and report modules. Entry point: `uv run python -m candidate_screener.evaluation.cui`. No runnable scripts under `plan/`.

Inspect and reuse `app/conversation_v3_smoke.py`, `app/conversation_smoke.py`, `ConversationV3Service`, `ConversationService` and the existing test doubles. Avoid copying application routing into a second implementation.

Proposed commands:

| Command | Purpose |
|---|---|
| `validate --fixtures PATH` | Check schema, source spans, family separation, task counts and expected outcomes |
| `run --split dev --provider stub` | Debug complete execution without a live model |
| `freeze --fixtures PATH --protocol PATH` | Write content digests and study/configuration identity before evaluation |
| `run --split eval --mode paired --manifest PATH` | Run both configurations against the frozen protocol |
| `recover --provider stub` | Execute deterministic injected-failure cases |
| `export-review --run-id ID` | Produce randomized, blinded review packets and a private mapping |
| `report --run-id ID --reviews PATH` | Validate reviewer coverage and produce results |

Treat command names as proposed interfaces until implementation review. A new `--check` command, if introduced, must be registered with `candidate_screener.checks`.

Each turn record stores family/dialogue/turn IDs, configuration identity, source and prompt digests, model/runtime identity, displayed answer, evidence references, generation mode, fallback reason, completion state and timings. Raw prompts/rejected model outputs are optional synthetic-only artifacts in ignored storage. Never enable applicant prompt logging in the live application.

**Exit:** offline stub execution produces complete, auditable records; invalid fixtures fail explicitly.

## Step 4 — Implement a fair controlled-versus-v3 comparison

Use two configurations:

1. **Controlled baseline:** the same v3 routing, extraction, context and comparison logic, with generation disabled and its existing controlled answer/fallback path used.
2. **V3:** the same pipeline with the pinned Qwen3.5 composer and current validation/fallback enabled.

The baseline is not a separately configured historical v1 UI or an independently selected retrieval pipeline. Preserve truthful generation-mode metadata for both arms.

For standalone quality comparisons, freeze the authorized evidence/frame input and reuse it across arms to isolate generation. For complete dialogues, start both arms from equivalent fresh snapshots and run each with its own history: prior answers may differ, so forcing identical histories would not measure realistic continuity. Label these two protocols separately.

Use isolated workspaces, deterministic scenario ordering/randomization seeds and identical source/model configurations. Alternate arm order where feasible; avoid concurrent inference on the same CPU service. Use an explicit cache policy and document which measurements include extraction or model initialization.

**Exit:** tests prove neither arm receives different sources, unauthorized history or a hidden model advantage.

## Step 5 — Add metric computation and independent review

| Dimension | Definition and denominator |
|---|---|
| Supported-claim rate | Human-audited supported factual claims divided by all factual claims in displayed answers; report claim-free answers separately |
| Complete-answer rate | Questions satisfying all required facts/caveats, or the correct uncertainty outcome, divided by all questions |
| Correct abstention | Appropriate uncertainty responses divided by questions requiring uncertainty; separately report unnecessary abstention on answerable questions |
| Critical errors | Counts of invented credentials/tenure/ranking decisions, incorrect source ownership and unauthorized disclosure; identify affected families |
| Reference resolution | Correct resolutions divided by predefined reference opportunities, including expected clarification |
| Dialogue completion | Dialogues satisfying every predefined task with no critical error divided by all dialogues |
| Unnecessary clarification | Count and rate per dialogue/turn using annotated context-sufficient opportunities |
| Accepted generation | Accepted model-authored answers divided by all predefined eligible factual turns in the v3 arm |
| Fallback | Fallback answers divided by the same eligible-turn denominator; report outages/skips and reasons explicitly |
| Naturalness/usefulness | Paired 1–5 directness, clarity and usefulness ratings; compare like-for-like outputs |
| Latency | End-to-end median and maximum, timeout counts, and p95 with sample size and small-sample caveat; separate cold/warm and generation/fallback |
| Recovery | Passed predefined assertions divided by injected cases, with individual failure descriptions |

Human reviewers independently segment factual clauses and judge their support against sources, then reconcile segmentation and label disagreements. Preserve both original ratings and adjudicated outcomes. Report raw agreement and an appropriate agreement statistic only where sample size and label variation make it meaningful.

Blinded packets include the question, relevant dialogue context, source evidence and randomly labelled displayed answers; omit model/fallback identity. Keep an internal identity map for analysis. Review rejected synthetic generations separately to distinguish unsupported generation from validator false rejection; do not mix these diagnostic ratings into displayed-answer quality.

Report denominators and family-level 95% bootstrap confidence intervals. Use paired resampling of the same families for arm differences; keep all related questions, claims and turns together. Freeze the resampling seed and number of replicates (proposed: 10,000). With ten families, characterize results as preliminary and avoid significance claims based on turns treated as independent samples. Report standalone questions and dialogues separately.

**Exit:** metric definitions and review forms are frozen before evaluation. No post-result threshold selection.

## Step 6 — Verify the harness and recovery boundaries

Add synthetic tests under `tests/test_cui_evaluation_*.py` covering:

- Wrong-owner spans, mismatched digests and invalid fixture expectations.
- Missing turns, duplicate run IDs, incomplete reviewer coverage and unexpected statuses.
- Denominator accounting, paired family resampling and separation of accepted generation from successful fallback.
- Blinding and mapping integrity.
- All five failure conditions, including no stale-source disclosure.
- Fake-clock deadline tests; avoid waiting 60 seconds in ordinary unit tests.

Demonstrate that named regression tests fail against the defect they guard. Iterate with focused tests, then run the full `uv run pytest` suite once and retain/read the log. Record unrelated failures without changing research outputs to make them pass. Standalone evaluation tooling does not require research cold checks unless implementation changes underlying model/data/scoring code. Inspect affected check selection if scope expands; obtain approval for long runs.

After approval, use scoped sub-agents for fixture/schema work, harness/metrics work and independent review, with disjoint file ownership. The parent integrates and verifies the protocol.

**Exit:** focused evaluation tests pass, broad test results are recorded and independent implementation review is complete.

## Step 7 — Freeze, time and execute the live study

1. Run the two development families first; fix harness defects and calibrate rubric only on development data.
2. Freeze evaluation fixture/protocol digests, code identity (including any patch digest), prompt hash, validator version, model manifest, generation parameters, tokenizer settings, hardware and cache policy. Do not silently modify a frozen artifact.
3. Time one representative development configuration before projecting total runtime. Distinguish cold initialization from warm factual turns.
4. Provide the owner the measured estimate and obtain confirmation before the full live run. There are 62–78 evaluation turns per arm (30 standalone plus 32–48 dialogue turns), or 124–156 total paired turns. At the 60-second v3 ceiling, the v3 arm alone represents up to 62–78 minutes of turn budgets; baseline, setup and recovery are additional. This is a scheduling bound, not a latency prediction.
5. Run the approved job once in the background with logs, exit marker and platform-appropriate peak-memory measurement. End the turn; do not poll or launch a waiter. Resume analysis when the job finishes.
6. Preserve failures and timeouts. Do not rerun only failed turns to improve the reported rate. If an infrastructure failure invalidates a run, preserve it and document the reason before creating a new complete run.

**Exit:** immutable run records with all expected turns accounted for. No tuning on evaluation outcomes.

## Step 8 — Review, analyze and write the report

1. Export blinded packets and collect both independent reviews.
2. Adjudicate disagreements without deleting original ratings.
3. Produce factual-quality, dialogue-completion, generation/fallback and latency tables, plus paired usefulness differences and uncertainty.
4. Categorize errors: retrieval omission, extraction error, reference/state error, unsupported generation, validator false rejection, runtime failure and presentation failure.
5. Update `docs/conversational-qa-architecture.md` with measured results and a clearly dated limitations section. Add a standalone `docs/cui-evaluation-results.md`; do not edit the Word report unless requested.
6. Record implementation deviations and open questions in this plan directory. Any improvement motivated by the evaluation makes those cases development data; a new independent set is needed for a further held-out claim.

Proposed artifact locations:

| Artifact | Location |
|---|---|
| Protocol, rubric, coverage and implementation record | `plan/2026-10-29-cui-evaluation/` |
| Compact synthetic fixtures and schema | Package resources under `src/candidate_screener/evaluation/cui/` |
| Synthetic execution/reviewer working artifacts | Ignored `data/processed/cui-evaluation/` |
| Execution logs | Ignored `logs/cui-evaluation/` |
| Aggregate metrics, configuration and digests | `output/cui-evaluation/` |
| Reader-facing findings | `docs/cui-evaluation-results.md` |

Evaluation fixtures are offline resources, not new production runtime dependencies. Do not add them to the Hiring Desk startup or require its private data folder for tests.

**Exit:** reproducible preliminary findings, explicit limitations and no unsupported promotion claim. If a cold check is required by repository rules before marking the plan complete, obtain approval and perform the applicable check first; do not silently waive that gate.

## Persistent open questions

| ID | Question / proposed choice | Blocks |
|---|---|---|
| CUI-01 | Resolved: owner approved the ten-family preliminary study. | None |
| CUI-02 | Who authors the independent evaluation scenarios and who are the two reviewers? | Fixture independence and human acceptance |
| CUI-03 | Confirm the final demonstration hardware and pinned Qwen3.5 configuration | Live configuration freeze |
| CUI-04 | Approve the measured live-run estimate after the development timing exercise | Full live execution |
| CUI-05 | If independent reviewers are unavailable, accept an explicitly labelled single-reviewer pilot? No independent-human-quality claim would be made. | Human evaluation/report interpretation |

## Completion criteria

- Frozen fixtures and protocol, tested runner, complete paired run records and explicit failure accounting.
- Independent reviews completed, or a clearly documented owner-approved reduction in study scope.
- Metrics reported with denominators, family-level uncertainty, configuration identity and limitations.
- Report distinguishes system correctness, accepted generation and fallback usefulness.
- No claim that this preliminary study passes the earlier v3 promotion gates or establishes real-recruiter productivity.
