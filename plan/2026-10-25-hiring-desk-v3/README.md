# Hiring Desk v3: inference-only conversational answers

25 October 2026. **Implementation approved by the owner: “approve and execute”.**
The approved scope is an opt-in inference-only implementation; promotion remains
subject to the quality gates below. No model training is authorized.

## Requirements and interpretation

Primary requirement: [ownership/06_inference-only_prompts](../../docs/ownership/06_inference-only_prompts).
Conversation process: [01-conversation-design.md](../../src/llm-plan/2026-10-19-hr-conversation-design/01-conversation-design.md).
Happy/negative flows: [ownership/05](../../docs/ownership/05_happyFlow-negativeCases_chatbot.md).

The product goal is an empathetic HR discussion: RoBERTa/GLiNER prepare relevant
source context, then frozen Qwen answers the current question naturally using
that context and bounded conversational memory. Normal replies contain no JSON,
code, chunk labels, raw-source dumps or repeated “according to the evidence”
prefaces. Greetings/thanks/clarifications are handled naturally.

“Inference-only” means no training or weight updates, no extraction job assigned
to the small generator, and no external-world facts added to applicant answers.
It permits synthesis of supported facts; it does not authorize inferred skills,
invented tenure, altered ranks or hiring commitments. The code still resolves
references, retrieves evidence, handles authorization and validates results.

## What must change from v2

| Current implementation | V3 change |
|---|---|
| `conversation_qwen.SYSTEM` requires JSON relation IDs | Separate plain-prose system prompt and generator; retain v2 adapter for rollback |
| Application renderer writes every final factual clause | Qwen writes the normal accepted answer; app guards, persists and displays it |
| Qwen mainly runs for balanced 2–3-person comparisons | Support grounded single-candidate, JD, comparison and explanation turns |
| Controller saves up to12turns but passes only `qa_context` | Pass bounded, validated, current-scope history plus typed frame explicitly |
| Full excerpts and source IDs appear in explanations | Keep provenance internally; show sources when requested or opened |
| Chat requires an active ranking snapshot | Add authorized JD-search context independent of a candidate-ranking snapshot |

The ownership note attributes rigidity to the small checkpoint's workload.
That is a hypothesis to test: v2's hard ID-only contract already prevents natural
generation. Do not attribute all failures to model size without the controlled
prompt/orchestration experiment.

## Reading order

1. [Prompt and context contract](01-prompt-and-context-contract.md): concrete proposed
   system prompt, model inputs, memory, answer validation and fallback behavior.
2. [Implementation and evaluation](02-implementation-and-evaluation.md): file ownership,
   delivery order, happy flows, negative cases, promotion gates and open questions.
3. [Implementation record](03-implementation-record.md): delivered code, regression
   evidence, real-model development findings and experimental rollout decision.

## Approval and rollout boundary

Record the v3 scope amendment as D66 and commit the finalized plan. The approved
default is a scoped v3 branch, `feat/hiring-desk-v3-inference-prompt`. Preserve uncommitted v2,
ONNX-ranking, report and owner-authored files. Do not mix unrelated fixes into v3.

Implement behind `HR_PORTAL_CHAT_VERSION=v3`, preserving v1/v2 behavior and red/black
styling. Use the existing local Qwen2.5-0.5B-Instruct, RoBERTa-base-SQuAD2 and
GLiNER-medium. No new model download, training, hosted applicant processing or
automatic service migration. Prompt/runtime identity must be visible in diagnostics,
not as technical clutter in the HR dialogue.

ONNX/SBERT candidate-ranking recovery remains a separate unfinished workstream.
Chat design and synthetic model tests do not depend on completing it. Newly
created JDs still need a ready ranking before candidate questions are possible;
v3's JD-only/search turns can work without inventing candidate results.

## Review status

Review recommendation: retain this architecture, but start with a bounded real-Qwen
vertical slice before full integration. Establish that natural paraphrases can pass
support checks at a useful acceptance rate; a longer system prompt alone cannot
establish this. Explicit comparison requests retain the application-built table.

The approved implementation has undergone independent code review and synthetic
development walkthroughs against the existing local Qwen. See the implementation
record for measured failures and validation. Quality promotion remains pending;
there has been no model training, dataset collection or active-service migration.
