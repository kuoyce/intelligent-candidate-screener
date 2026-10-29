# Conversational QA: V3 Architecture and Methodology

This section describes the implemented v3 architecture and the documented local Qwen3.5-4B deployment. V3 remains experimental: local activation does not establish that its independent quality acceptance gates have passed. V1 remains the configured default unless explicitly overridden.

## 1. Final Architecture and Design Approach

The v3 Conversational QA system combines **source-based extraction, controlled comparison logic and LLM-generated explanations**. It helps recruiters discuss candidate evidence naturally while preserving factual accuracy, qualification context and source ownership.

V3 reuses the extractive foundation developed in v1 and the conversation-state and comparison components developed in v2. These components prepare evidence before generation and provide controlled answers when generation fails.

**Fallback is not a sequential switch from a v3 model to a v2 model and then a v1 model.** V3 invokes the shared extractive and comparison services with the v2 Qwen composer disabled. When v3 generation is unavailable or rejected, application code renders an answer from checked evidence.

The architecture has three complementary layers:

- **Evidence layer:** RoBERTa-SQuAD2, GLiNER-medium and source retrieval identify relevant information in authorized JD/CV documents.
- **Conversation and comparison layer:** Application logic resolves references, maintains dialogue state and constructs checked candidate comparisons.
- **Generation layer:** Frozen Qwen3.5-4B produces conversational explanations, subject to validation and deterministic fallback.

This is an **LLM-assisted, application-orchestrated CUI**. The LLM does not independently execute business operations or change rankings, scores or Fit decisions.

Sources: [service composition](../src/candidate_screener/app/portal_workspace.py), [v3 implementation](../src/candidate_screener/app/conversation_v3.py).

## 2. System Architecture and Execution Methodology

### Core components

| Component | Responsibility |
|---|---|
| Streamlit interface | Accept questions; display answers, comparison tables and supporting sources |
| Controller and dialogue state | Authorize access, resolve candidate references and preserve the active participants and criterion |
| Source retrieval | Retrieve relevant redacted passages from authorized JD/CV documents |
| GLiNER-medium | Identify entities that organize evidence retrieval |
| RoBERTa-SQuAD2 | Extract answer spans, including explicit no-answer outcomes |
| Evidence matrix and comparison rules | Represent supported facts, qualifications, gaps and candidate differences |
| Qwen3.5-4B | Generate concise explanations from the supplied evidence |
| Output validator and fallback renderer | Accept supported wording or construct a controlled alternative |
| SQLite workspace store | Persist operational review records and conversation state |

### Processing a recruiter question

1. **Resolve the request and its scope.** The application identifies the selected job, candidates and question type. Follow-ups reuse the active conversation frame. Ambiguous references trigger clarification. Greetings and supported out-of-scope cases are handled directly without model inference.
2. **Retrieve and extract relevant evidence.** The system retrieves authorized JD/CV passages. GLiNER entities assist evidence organization, while RoBERTa identifies relevant answer spans. Extracted spans retain surrounding context so that phrases such as “degree in progress” or “no commercial experience” are not reduced to misleading isolated terms.
3. **Construct a checked evidence representation.** Application logic verifies source ownership, offsets and source versions, then constructs evidence cells and comparison relationships. Statuses distinguish documented mentions, qualified evidence, explicit negatives and unresolved information. A skill mention is not treated as proof of proficiency.
4. **Build the generation context.** The current question, selected participants, job criteria, evidence matrix, supporting passages and bounded history are assembled within the model’s token budget.
5. **Generate and validate the answer.** Qwen produces an explanation. The application checks its format and supported factual content before displaying the accepted conclusion. Unsupported or unavailable generation invokes controlled rendering.
6. **Present and preserve provenance.** Explicit comparisons retain application-built tables. Supporting sources are available on request, while model identity, prompt identity and fallback reasons remain in diagnostics. Persisted answers retain their scope and provenance.

The Qwen3.5 deployment uses a pinned quantized GGUF model through local llama.cpp and a bounded service facade. Applicant processing remains local. Ranking is supplied separately through saved ranking records or `inference.Screener`; the conversational model explains available evidence without recomputing ranking decisions.

Sources: [extractive foundation](../plan/2026-10-24-grounded-qwen-comparisons/05-extractive-implementation.md), [Qwen3.5 runtime](../plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md), [SQLite store](../src/candidate_screener/app/portal_store.py).

## 3. QA Model and Fallback Mechanisms

The fallback design preserves useful evidence-based responses when natural-language generation cannot be accepted.

| Condition | System behaviour |
|---|---|
| Valid, supported Qwen response | Display the accepted conversational conclusion |
| Unsupported or invalid comparison response | Rebuild source-owned facts and use the controlled comparison renderer |
| Unsupported single-candidate response | Produce a checked summary or explicit uncertainty response |
| Model unavailable or timed out | Return an appropriate controlled answer or availability message; retain usable evidence controls |
| No relevant evidence | State that the information cannot be established from the reviewed material |
| Ambiguous candidate reference | Ask a focused clarification |
| Changed or unauthorized source | Reject stale factual context and revalidate access |
| Supported unrelated or sensitive-characteristic request | Return the configured out-of-scope response without generation |

### Reuse of v1/v2 functionality

- The **v1 extractive service** supplies answer-bearing evidence and comparison data.
- The **v2 conversation service**, instantiated without its generative composer, supplies conversation handling and controlled comparison rendering.
- **V3 adds prose generation and validation** above this foundation.
- Failure therefore falls back to **application-authored, source-grounded responses**, rather than another unrestricted generative attempt.

Sources: [fallback implementation](../src/candidate_screener/app/conversation_v3.py), [fallback text construction](../src/candidate_screener/app/conversation_context.py).

### Factual and runtime safeguards

| Category | Safeguard | Behaviour or constraint |
|---|---|---|
| Factual | Source ownership and fidelity | Preserve candidate ownership, numbers, units, negation and qualification status |
| Factual | Experience interpretation | Do not infer skill-specific experience from total employment duration |
| Factual | Missing evidence | Do not interpret missing evidence as proof that a candidate lacks a qualification |
| Factual | Decision boundaries | Reject invented rankings, scores, credential verification or hiring commitments |
| Factual | Balanced comparison | Preserve material counterevidence and qualification caveats |
| Factual | Conservative output validation | Accept supported factual patterns; reject unsupported wording and use a controlled fallback |
| Runtime | Shared generation deadline | Limit the selected Qwen3.5 path to a maximum 60-second shared turn budget |
| Runtime | Extraction deadline | Keep extraction capped at 30 seconds |
| Runtime | Worker isolation | Enforce deadlines and cleanup through an isolated generation worker |
| Runtime | Bounded recovery | Do not use an unbounded retry or prose-repair loop |
| Runtime | Cancellation semantics | A client timeout does not imply cancellation of computation in the separately managed model server |

The validator uses conservative registered factual patterns; it is not a general semantic-entailment model. Consequently, some faithful paraphrases may still fall back.

Source: [runtime limits and validation limitations](../plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md).

## 4. Prompt Engineering and Context Management

The prompt strategy combines **role instructions, evidence-grounded context, few-shot examples and constrained answer formatting**.

- **Role:** The model acts as an HR Data Analyst explaining candidate evidence.
- **Dynamic context:** Each request supplies job priorities, selected passages, an evidence matrix and the current question.
- **Few-shot examples:** Worked examples demonstrate technical comparisons and the treatment of ongoing degrees and expired certifications.
- **Answer format:** The current prompt requests a short evidence-check summary under `[Reasoning Steps]`, followed by `[Conclusion]`. The application displays the conclusion. The requested checks are bounded explanations, not detailed chain-of-thought.
- **Style:** Answers should be concise, preserve uncertainty and avoid source dumps, internal evidence IDs, code and JSON.
- **Trust boundary:** Document text, user assertions and history are data rather than instructions that can override system policy.

Source: [current packaged prompt](../src/candidate_screener/llm/hr_system_prompt_v3.txt).

### Memory and token budgeting

- The context contract permits up to **four recent validated exchanges**, subject to the actual token budget.
- The runtime caps the context at **4,096 tokens**, with up to **256 output tokens**.
- History supports conversational continuity; current authorized sources remain the factual authority.
- Old exchanges are removed as whole turns rather than truncated in ways that could remove qualifications.
- Ranked candidate review supports bounded saved memory; the initially implemented standalone JD-question surface does not provide equivalent persisted multi-turn memory.

There is no documented ReAct loop or autonomous tool-selection framework in this architecture. Routing, retrieval and business-action boundaries remain application-controlled.

Sources: [context contract](../plan/2026-10-25-hiring-desk-v3/01-prompt-and-context-contract.md), [generation adapter](../src/candidate_screener/llm/conversation_prose.py), [implemented memory limitations](../plan/2026-10-25-hiring-desk-v3/03-implementation-record.md).

## 5. Evaluation Methodology and Current Limitations

Evaluation distinguishes **successful evidence-based answers** from **successful LLM generation**. A correct fallback demonstrates resilience but does not demonstrate that Qwen answered correctly.

The documented methodology assesses:

- **Grounding:** Whether factual statements preserve source ownership, qualifications, negation and numerical details.
- **Dialogue correctness:** Candidate-reference resolution, criterion changes, follow-ups and clarification handling.
- **Generation coverage:** Accepted model-authored answers versus controlled fallback responses.
- **Task completion:** Whether a complete recruiter dialogue achieves its intended review task.
- **Usability:** Directness, naturalness, usefulness and unnecessary source repetition.
- **Performance:** End-to-end latency, timeout frequency and cold/warm execution.
- **Boundary enforcement:** Stale-source rejection, access isolation, prompt-injection resistance and prevention of unauthorized actions.

Testing combines synthetic regression fixtures, injected model clients, UI walkthroughs and separate real-model experiments. Independent human review and held-out scenario-family evaluation are specified as quality acceptance activities. Proposed final comparisons report denominators, confidence intervals and paired differences over independent scenario families, rather than treating each turn as an independent sample.

Source: [evaluation plan](../plan/2026-10-25-hiring-desk-v3/02-implementation-and-evaluation.md).

**Current evidence:** The Qwen3.5 60-second development walkthrough accepted **one of four factual turns before final refinements**. Subsequent prompt and grammar changes passed focused regressions, but the records do not establish post-refinement held-out generation accuracy. V3 was locally activated as an experiment; broader quality acceptance remains pending.

Source: [trial results](../plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md).

### Missing Information / Required Blanks

- Final held-out answer acceptance and fallback rates.
- Human-reviewed factual support, naturalness and task-completion results.
- Representative p50/p95 latency and concurrent-user performance.
- Post-refinement live-model evaluation.
- Confirmation of the configuration used for the final project demonstration.

These gaps are evaluation limitations, rather than achieved results or evidence that proposed targets have been met.
