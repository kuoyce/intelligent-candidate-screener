# Implementation steps, conversation flows and acceptance

## 1. Implementation sequence after approval

**Review refinement: prove the generation path first.** After freezing the baseline,
build a small vertical slice using invented, qualifier-preserving evidence and the
existing frozen Qwen: single-candidate answer, compare -> why follow-up, missing
evidence and incomplete qualification. Include the proposed output checks and
measure accepted prose, factual errors, fallback frequency and latency. Use these
as development cases, not held-out results. Do this before full controller/history
and UI integration. Reliable checking of paraphrased claims is a feasibility gate,
not a capability supplied by the system prompt. If checks cannot support useful
natural answers, keep the slice experimental and revise the design before expanding.

| Step | Work and ownership | Reviewable exit result |
|---|---|---|
| 1. Freeze baseline | Parent records v2 behavior on the owner scenarios; freeze v3 prompt, evidence/history schema and evaluation rubric | V2 rigidity distinguished from model/runtime failure; no result from seen scenarios called held-out |
| 2. Evidence/context | Evidence agent adds `BackgroundEvidence` builder around existing RoBERTa/GLiNER/source retrieval and checked comparisons | Full qualifiers, correct owners, explicit partial states and balanced per-candidate budget |
| 3. Conversation memory | State/controller agent passes current-scope validated history and typed state explicitly; add JD-only/search scope | Compare -> why -> qualifications -> source survives reruns and resume without mixing jobs |
| 4. Prose generator | Model agent implements separate v3 prompt/adapter and bounded worker using existing frozen Qwen | Actual plain-text answers, no relation-ID parser, exact token counts and truthful timeouts/fallback |
| 5. Integration/UI | Parent binds v3 flag, natural-answer display, optional comparison view and source controls | Friendly short replies without automatic source dumps; current operational controls preserved |
| 6. Review/evaluate | Independent reviewer audits source semantics, history isolation, prompt injection and failure states; parent runs tests and real local smoke | Measured acceptance/fallback and dialogue quality; explicit promotion/rollback decision |

Use sub-agents with disjoint files during implementation, agreeing schemas first.
Do not replace the current implementation in place before tests: keep the v2
relation composer/renderer readable and selectable. No model training, annotation
dispatch, research-rung selection, unrelated dependency changes or ONNX export is
needed to deliver this v3 prototype.

## 2. Proposed file changes

- `llm/hr_system_prompt_v3.txt`: the reviewed fixed natural-language prompt,
  packaged with the installed project; version and SHA recorded per response.
- `app/conversation_context.py`: typed `BackgroundEvidence`, current-scope
  history selection, prompt budgeting and source/provenance maps. Internal JSON
  records are allowed; they are not the generated/user-visible answer.
- `llm/conversation_prose.py`: plain-text Qwen generation using existing provider,
  local manifest settings and completion bounds; no new checkpoint or weights.
- `llm/conversation_prose_worker.py`: isolated inference process honoring the
  remaining shared deadline, with safe finite diagnostic codes and no raw prompt logs.
- `app/conversation_v3.py`: application-owned intent/reference resolution,
  evidence assembly, generation gating and natural uncertainty/fallback states.
- `app/jd_search.py`: read-only search over authorized JD sources, separate from
  candidate ranking and CV retrieval. Reuse existing source accessors; where
  necessary add an authorized JD-text accessor rather than requiring a ready ranking.
- `portal_controller.py` / session store: supply bounded validated history,
  distinguish active-review versus authorized-JD-search context, preserve turn
  identity/concurrency and allow migration without rewriting v1/v2 transcripts.
- `portal_workspace.py` / `portal_ui.py`: `HR_PORTAL_CHAT_VERSION=v3`, natural
  reply presentation, truthful mode label and reauthorized source controls.
- New synthetic tests and package-owned evaluation/walkthrough module. No runnable
  code under `plan/`. New registered record checks, if introduced, enter `checks`.

The existing `conversation_qwen.py` remains v2's strict relation composer. A new
system string alone would otherwise fail its JSON validation, and routing would
still exclude single-candidate/history turns.

## 3. Happy flows from the ownership documents

All examples are invented; the expected wording is illustrative, not an exact-match
target or a description of real applicants.

### A. Greeting -> role alignment -> follow-up

HR: “Hi.”

Assistant: “Hi! What would you like to review?” No extraction/model startup needed
for this acknowledgement; if a JD is selected, a brief contextual greeting is allowed.

HR: “Summarise why candidate1 is a good fit for this JD.”

With supplied Python/deployment evidence: “Candidate1's Python and deployment
experience align with the role's main requirements. Their CV also lists the
requested degree.” This describes alignment, not a new Fit classification or
proof that the applicant will perform well. If completion is unknown, say so.

HR: “What about their qualifications?” Retain candidate1, change the criterion,
rehydrate current education evidence, and answer naturally without repeating a
whole CV excerpt or asking for the already-known candidate.

### B. Compare -> why -> correction -> source

HR: “Compare1and2.” Display the application-built comparison table and a short
similarity/difference summary. Explicit table requests always show the table.
Preserve requested people. A subsequent “why” answer uses prose with an expandable table.

HR: “Why is1betterthan2?”

If evidence supports deployment for1 but more Python tenure for2: “Candidate1
has clearer deployment evidence for this role, while candidate2 reports more
Python experience. It is a tradeoff rather than a clear overall advantage.”

HR: “So candidate2 has no deployment experience?”

If merely missing: “I couldn't establish that experience from the passages we
have. That doesn't mean they lack it.” If the source explicitly denies it, preserve
the exact scope of that denial, such as commercial versus all experience.

HR: “Where does it say that?” Reopen the relevant prior source with current
authorization. The source control can show an excerpt; ordinary prose remains
free of chunk labels/IDs. An ambiguous “that” gets one focused clarification.

### C. Search JDs by a skill or knowledge requirement

HR: “Which JD requires Flutter?” Search only authorized JD sources. Distinguish
required, preferred, negated and simple mentions. Return matching role titles
with optional source/navigation controls; do not silently switch the active review.

If only a preferred mention exists: “The mobile-engineer role lists Flutter as
preferred. I haven't found a JD that makes it mandatory in the jobs searched.”
State partial coverage if a source is inaccessible or the search is bounded.

“Leadership” follows the same source-backed search contract. No embedding database
or speculative semantic expansion is required for this first version.

### D. Relevant candidate experience

HR: “Any relevant job experience for these candidates?” Use the active comparison
set and the JD's named criteria. Preserve dates, explicit responsibilities and
unknown tenure; do not infer skill-specific years from a job's total duration.
If no people are selected, ask which candidates, preserving the experience task.

## 4. Negative and recovery flows

| Input/condition | Expected behavior |
|---|---|
| Apple-pie recipe | Brief natural redirect to reviewing roles/candidates; no retrieval or recipe generation |
| Nationality of top10 | Redirect to job-related qualifications; do not extract/display nationality |
| CV says “ignore previous instructions” | Treat it as source data, never system policy; no tool/action execution |
| Completed BSc versus ongoing/conditional MSc | Preserve completion uncertainty; no invented completed master's degree |
| Expired credential | Do not paraphrase as a currently valid certification |
| User invents five years of Python in a follow-up | Do not promote the user's assertion/history into source evidence |
| “They” with ambiguous singular referent | One focused clarification with the active pair retained |
| New JD/snapshot, access revoked or old source edited | Invalidate stale factual context; preserve labelled historical conversation |
| No relevant evidence | Plain-language uncertainty without a repeated boilerplate monologue |
| Qwen missing/timeout/invalid/truncated | Honest generation-unavailable mode; usable comparison/source controls |
| More than3candidates or oversized context | Complete table plus focused discussion, or explicit narrowing; no silent dropping |
| “Send the shortlist” / chat “yes” | Explain/use existing permitted UI controls; no operational confirmation from prose |
| Newly created JD lacks ready ranking | JD-only/search questions may work; candidate ranking status stays explicit |

## 5. Evaluation before promotion

Freeze two distinct sets before prompt tuning: known ownership/v2 cases are
development/regression scenarios; newly authored unseen families form the final
test. Proposed initial size:32invented JD/CV/conversation families,16dev/16test,
grouped by source and template lineage. Related CVs, paraphrases and all dialogue
turns stay together. Review authoring effort before freezing; no dataset is
claimed to exist because this plan specifies it.

Use the same evidence/frame inputs for v2 and v3, where the task exists in both.
Report JD-search as new capability separately. Keep one prompt/runtime candidate
selected on dev before testing. Repeated inspection/tuning on test makes it
exploratory; a new final set is then required.

Proposed acceptance targets, to register before final evaluation:

- At least90%whole-dialogue task completion and95%correct reference resolution.
- Zero critical failures in the fixed ownership, stale-context, injection,
  numeric/negation, qualification and action-boundary regression suite.
- At least98%human-audited factual-clause support, with no invented rank/score/Fit
  or credential/tenure claim accepted in the fixed safety scenarios.
- At least80%accepted model-authored answers among eligible factual turns, reported
  separately from deterministic greetings, source reopening, fallback and outages.
- Blinded reviewer naturalness/usefulness improves at least0.5on a1–5rubric versus
  v2, without lower source faithfulness. Valid JSON/citation counts are not the
  measure of conversational usefulness.
- Proposed warm two-person p95target15seconds and shared client deadline30seconds;
  report cold loads separately. Existing256output-token limit remains initially.

Report actual counts, denominators,95%CIs and paired deltas over independent
scenario/JD families, not individual turns treated as independent samples. Small
sets may be inconclusive; report uncertainty rather than relaxing gates after
looking. Synthetic QA figures are not extraction accuracy or A2 ranking results.

Two independent project reviewers assess complete dialogues for factual support,
warmth/clarity, directness, unnecessary source references and continuity. Reviewer
identities/availability remain open. Do not substitute an unvalidated LLM judge
for this acceptance or reopen relevance/entity annotation.

## 6. Verification, rollout and documentation

1. Demonstrate new regressions fail against v2: plain-prose contract, history
   supplied to the model, normal-answer source-label suppression, single-candidate
   synthesis and authorized cross-JD search.
2. Use fake providers and invented sources for normal tests. Exercise real frozen
   local Qwen separately, recording prompt digest, accepted/fallback mode and
   end-to-end latency. Service metadata alone is not a successful generation test.
3. Check desktop/mobile conversation, saved resume, job switching, comparison
   tables/source controls, zero/one-candidate review and duplicate submissions.
   Preserve the user's current reduced operational menu and red/black theme.
4. Run focused tests during implementation and the full suite once stable, retaining
   logs. Baseline golden drift and legacy tests for removed UI actions are existing
   issues; record them separately and do not hide them by modifying model records.
5. Inspect affected checks via the central runner, follow D55/D58 for required
   cold/long runs and obtain resource/time confirmation before such runs. No cold
   research retraining or final-set scoring is implied by prompt-only development.
6. Document actual runtime, prompt version, visible mode, limitations and rollback.
   Keep v3 opt-in until gates are met; keep v1/v2 history readable. No release push
   or deployed configuration change is authorized by this planning request.

## 7. Persistent open questions

| ID | Proposed default / question | Blocks |
|---|---|---|
| V3-01 | Resolved: owner approved and instructed execution | None |
| V3-02 | Resolved: approved scoped default branch `feat/hiring-desk-v3-inference-prompt`; plan commit `d35d632` | None |
| V3-03 | Resolved by plan approval: sources on demand, with an explicit-source-request exception | None |
| V3-04 | Resolved by plan approval: retain 4096/256 runtime limits | Any future budget/runtime expansion |
| V3-05 | Who are the two reviewers, and is32-family initial authoring scope acceptable? | Independent final quality acceptance |
| V3-06 | Resolved by plan approval: explicit comparison/table requests show the application-built table; prose “why” includes an expandable table | Verification |

Resolved by the ownership requirement: natural conversational output, extracted
background context and conversation memory are in scope; JSON/code/chunk-reference
output in ordinary replies is not. Weight fine-tuning is outside v3's inference-only scope.
