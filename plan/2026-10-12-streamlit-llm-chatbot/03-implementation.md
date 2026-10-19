# Chatbot implementation and conversation-design entry point

Refactored 19 October 2026 at the user request. The original prototype record is
preserved verbatim; it must not be confused with newly requested portal features.

The new planning subfolder is
[src/llm-plan/2026-10-19-hr-conversation-design](../../src/llm-plan/2026-10-19-hr-conversation-design/README.md).

- [Conversation design process, sample dialogues and Mermaid flow](../../src/llm-plan/2026-10-19-hr-conversation-design/01-conversation-design.md)
- [Refactored implementation plan](../../src/llm-plan/2026-10-19-hr-conversation-design/02-implementation-plan.md)
- [Requirement coverage, acceptance and open questions](../../src/llm-plan/2026-10-19-hr-conversation-design/03-acceptance-and-open-questions.md)
- [Preserved original implementation record](../../src/llm-plan/2026-10-19-hr-conversation-design/04-prototype-history.md)

The new HR portal design specifies authentication, alerts, ranking-database top10,
read-only CV previews, a top8 package with all10 comparison, saved sessions and
confirmed internal department email. The20October refinement adds typed HR
questions and Qwen-based evidence-grounded conversation alongside dropdown actions;
see the [conversational QA amendment](../../src/llm-plan/2026-10-19-hr-conversation-design/06-hr-conversational-qa-amendment.md).
It excludes candidate outreach, appointment scheduling and conversation uploads.

These are proposed integrations, not delivered capabilities of the old local
prototype. Existing backend/RAG work remains reusable; no outbound action or
model run is performed by this documentation refactor.
