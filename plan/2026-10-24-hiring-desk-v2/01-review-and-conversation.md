# Review and conversation specification

## 1. What changed between the amendment and v1

The older amendment specified an outcome: explain and compare, remember who is
being discussed, answer directly, then provide evidence and uncertainty. Its
architecture allowed typed cited generation. The latest [v1 scope](../2026-10-24-grounded-qwen-comparisons/05-extractive-implementation.md)
deliberately removed primary Qwen inference and cross-candidate generated
explanations. This reduced the interpretation surface but also removed the layer
that connects separate facts into a useful discussion.

| Concern | Earlier intended outcome | Current code | v2 change |
|---|---|---|---|
| Broad comparison | Concise account of similarities, differences and relevance to JD | `PortalExtractiveService.compare/_table` builds six category rows without starting QA; generic caution text | Identify JD criteria, build balanced evidence cells, explain supported commonality/tradeoff |
| Focused comparison | Answer the question across selected people | Independent per-document QA, then source/span assembly | Decompose once into an atomic question per owner; compute permitted comparison relations |
| Generation | Qwen helps communicate | Primary controller uses extractive service; historical `_model_claims` only accepts exact source quotes/evidence IDs | New versioned composition contract; merely enabling the old provider is insufficient |
| Follow-up | Retain comparison set when plural intent is clear | `_resolve` can clarify on “their qualifications” after two candidates unless comparison wording is explicit | Persist discourse focus, set, criteria and clarification slot; plural refers to active comparison set |
| Citation challenge | Reopen the earlier supporting passage | `portal_qa.answer` builds another retrieval question from `last_question` | Persist claim-to-evidence map; reopen that claim's original authorized source |
| Presentation | Direct answer, brief explanation, compact comparison when useful | Broad disclaimer/table and focused source extracts; chat tables can show raw owner IDs | Friendly candidate labels, conclusion/qualification first, table only for multiple criteria, sources expandable |
| Reliability | Clearly distinguish what is stated, inferred, unknown | Exact offsets and surrounding context are validated | Retain those controls; add explicit semantic support and relation checks before synthesis |

Code references: `src/candidate_screener/app/portal_extractive.py`, `portal_qa.py`,
`portal_controller.py`, `portal_ui.py`; model code under `llm/extractive_qa.py`
and `llm/qa_train.py`. These are static findings, not a new measured quality study.

## 2. Why model refinement alone is insufficient

1. Resolve the participants and criterion before model inference. A correct span
   about the wrong person is still a wrong answer.
2. Retrieve relevant, qualifier-bearing evidence for every participant. Current
   `_relevant` keeps at most two passages and broad cells use the first; evidence
   coverage must be measured before blaming model capacity.
3. Extract focused facts where useful. An exact span verifies its location, not
   whether it is an affirmative qualification. “Python” inside “No commercial
   Python experience” cannot become positive evidence.
4. Align facts to the same JD criterion and express a supported relationship.
   Two separately extracted degree names do not themselves answer a tradeoff.
5. Compose a short response and retain the comparison state. This is where Qwen
   can add conversational value; RoBERTa's start/end head cannot generate that
   dialogue, regardless of additional span fine-tuning.

The application performs these visible processing stages. No hidden chain of
thought is requested, exposed or persisted; retain concise conclusions and evidence.

## 3. Conversation style

- Greet once on entry/resume of a review: “You're reviewing Data Engineer. I can
  explain the requirements or compare the selected candidates.” Use the actual JD
  title. Do not repeat the greeting on each Streamlit rerun.
- Answer the specific question first, generally in two to four sentences. Use a
  small table for multiple criteria; a single criterion usually needs prose.
- Name candidates by the snapshot's display labels, not internal owner IDs.
- Say “The CV states…” for source claims. Distinguish a CV claim from verification.
  Use a specific limitation at the relevant point rather than appending the entire
  generic disclaimer to every answer.
- Preserve plural follow-ups when there is one active comparison set. Clarify
  singular “they” after a pair if the intended person is unclear. Ask one focused
  question, preserve the unanswered task, and resume after the reply.
- Do not end every answer with a question. Offer a next step only when useful.
- For “better”, “stronger” or “more qualified”, use a named criterion and scope:
  stronger documented evidence, never an invented overall hiring verdict.
- Unsupported opinion, aptitude, personality and protected-trait inferences are
  outside the evidence comparison; redirect to job-relevant source information.

## 4. Happy flow A: compare, narrow, switch criterion, inspect source

All examples below use invented documents. Fixture JD requires Python and
production deployment; a degree is preferred. Candidate 1 explicitly states
three years of Python and production deployments. Candidate 2 explicitly states
five years of Python, says “No production deployment experience”, and lists an
MSc as “in progress”. Candidate 1 lists a completed BSc. These statements support
the example; no inference from unrelated total tenure is needed.

| HR turn | Intended response and state transition |
|---|---|
| “Compare candidates 1 and 2 for this role.” | “Both CVs state Python experience. Candidate 1 has documented production deployment experience; candidate 2 explicitly states no production deployment experience [citations]. Candidate 2 reports more Python tenure, so the evidence shows a tradeoff rather than one candidate leading on every requirement.” Compact Python/deployment table; save pair + criteria. |
| “Who has more Python experience?” | “Candidate 2's CV states five years of Python, compared with three for candidate 1 [citations]. These are CV claims, not independently verified experience.” Keep pair, focus Python tenure; numeric relation is application-validated. |
| “What about their qualifications?” | “Candidate 1 lists a completed BSc; candidate 2 lists an MSc in progress [citations]. The MSc is not evidence of a completed master's degree.” Same pair, replace criterion with qualification status; no redundant candidate clarification. |
| “Where does it say the MSc is unfinished?” | Reopen candidate 2's stored claim/source excerpt including “in progress”, after authorization/source checks. Do not issue a fresh general search and silently substitute evidence. |
| “Now compare candidate 1 with candidate 3.” | Explicit participant replacement wins over prior pair. Retain a criterion only if wording/context makes that intention clear; otherwise use JD comparison defaults. Retrieve candidate 3, discard candidate 2 from active set. |

“More years” can be stated only where both sources explicitly tie comparable
durations to the same skill. Overlapping jobs, rounded dates and total tenure do
not establish additive skill-specific duration.

## 5. Happy flow B: clarification with task preservation

HR: “Compare their cloud experience.” With no selected candidates, ask “Which
candidates should I compare?” Persist cloud experience as the pending criterion.

HR: “1 and 3.” Recognize a participant answer to that specific pending slot,
not a new topic or a duration. Resume cloud comparison without asking for the
criterion again. If candidate 3 has no retrieved cloud evidence, say “I found
cloud evidence for candidate 1; I could not establish it for candidate 3 from the
retrieved CV passages.” Do not declare candidate 3 unqualified or rank them lower.

HR: “Does the second one have AWS certification?” Resolve “second one” to the
second member of the active ordered pair (candidate 3), distinguishing this
phrase from explicit “candidate 2”. Preserve expiry/issuer/status if stated.

## 6. Happy flow C: ranking explanation without invented causality

HR: “Why is candidate 1 first?” State the backend rank, then compare their
documented evidence to named JD requirements. If feature attribution is absent,
say that this describes supporting evidence rather than the scoring model's
causal explanation. “Who would you shortlist?” reports the saved shortlist and
offers criterion-based evidence discussion; it does not create a new model rank.

HR: “Prepare a comparison report.” Offer the existing report control. Chat can
explain what the report contains; existing UI confirmation executes the action.

## 7. Recovery flows

| Condition | Expected behavior |
|---|---|
| “They” after a pair, singular intent unclear | One clarification naming the pair; keep the pending criterion/task |
| New JD, changed ranking or revoked source | Clear active references; label historical turns with original snapshot; reauthorize previews |
| Generator missing, slow or invalid | Render the same comparison facts using concise application templates; label evidence mode accurately |
| One source unavailable or search truncated | Preserve the participant with an unavailable/partial cell; no apparent two-sided conclusion |
| Conflicting claims within one CV | Cite both, say source is inconsistent; do not choose the flattering statement |
| Source injection or forged citation | Treat source text as data; reject foreign IDs/actions, keep authorized evidence fallback |
| Long request comparing ten people | Show a compact complete matrix and offer a narrower discussion; never silently drop eight |

## 8. Definition of meaningful conversation

A completed comparison answers the requested relation against a criterion,
accounts for every requested participant (including unknowns), preserves source
qualifiers, and supports the next turn without making HR restate the context.
Citation count or valid JSON alone does not establish this outcome. Test whole
dialogues and human usefulness alongside extraction correctness.
