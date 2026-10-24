# Hiring Desk v2: evidence-grounded HR conversation

24 October 2026. **Application implementation approved by the owner;
conditional training/resource experiments retain their separate gates.**

## Outcome and recommendation

Restore the conversational outcome in the [20–21 October amendment](../../src/llm-plan/2026-10-19-hr-conversation-design/06-hr-conversational-qa-amendment.md):
HR can compare applicants, change the comparison criterion, challenge a statement,
and continue naturally without repeatedly specifying the same people.

Use a hybrid: application-owned dialogue state and criterion-by-candidate evidence,
RoBERTa-SQuAD2 for focused source questions, and Qwen for bounded conversational
composition. Start by fixing orchestration and prompts with frozen weights.
Fine-tune Qwen only if conversational failures remain with adequate evidence;
fine-tune RoBERTa only if span/no-answer failures remain with the right passage.
Training either model cannot repair missing retrieval or lost conversation state.

This is an approved v2 scope, not a claim that v1's extractive implementation
is defective against its approved contract. The owner's implementation approval
amends D61 (entity/app) for bounded v2 composition. D61 (research) and the
research winner remain independent.

## Reading order

1. [Review and conversation specification](01-review-and-conversation.md)
2. [Architecture and model refinement](02-architecture-and-model-refinement.md)
3. [Delivery, evaluation and open questions](03-delivery-and-evaluation.md)

## Scope

English typed chat in the authenticated Hiring Desk, initially two or three
candidates per conversational comparison. Existing tables/reports continue to
support up to ten. Broad ten-candidate requests offer a compact matrix and ask
which subset/criterion to discuss; never silently omit participants.

Preserve v1's access controls, source ownership, qualification context, no-answer
states, saved reviews and existing operational confirmations. Compare evidence
against explicit job criteria; no generated overall Fit calls, score changes,
new ranks or unsupported explanations of the ranking model. A reviewer can ask
who has stronger *documented evidence for a stated criterion* without changing
the saved ranking. Missing evidence never means an applicant lacks a skill.

No implementation, live inference, downloads, model training, new applicant data
collection or benchmark scoring was performed to create this plan. Current local
manifests were read; historical timings are not fresh measurements.

## Review status

Read the amendment, current extractive scope/record, prior Qwen comparison repair,
portal resolver/controller/UI, span worker and existing QA trainer. A sub-agent
independently reviewed code-level gaps. Primary model/training documentation is
linked in the architecture document. Draft-only changes need link/static review;
the application test suite was not rerun for this documentation task.

## Approval boundary

Plan approval would authorize v2 application implementation and bounded inference
experiments using available local weights. The optional training phases require
their dataset, runtime, measured pilot cost and resource configuration to be
reviewed before launch. No startup training or automatic large model download.
Open resource questions and proposed defaults are recorded in document 03.
