# HR portal chatbot: conversation design and delivery plan

19–21 October 2026. Status: guided local workflow and conversational QA implemented;
chatbox completion implemented and reviewed. Corporate identity/events/mail
remain unavailable.
See the [implementation record](05-implementation.md) for actual capabilities,
input provenance and limits rather than treating planned integrations as delivered.

## Read in order

1. [Conversation design process and flows](01-conversation-design.md): audience,
   persona, task inventory, dialogue states, sample turns and Mermaid happy/error paths.
2. [Implementation plan](02-implementation-plan.md): services, data contracts,
   dependencies and delivery gates.
3. [Acceptance criteria and open questions](03-acceptance-and-open-questions.md):
   requirement traceability, unresolved integration choices and evaluation.
4. [Original prototype implementation record](04-prototype-history.md): preserved
   historical record from the October chatbot prototype, not a statement that the
   portal capabilities requested here exist.
5. [Guided portal implementation record](05-implementation.md): current modules,
   verification, supplied extraction/ranking inputs and unresolved dependencies.
6. [HR conversational QA amendment](06-hr-conversational-qa-amendment.md): current
   proposed scope for typed HR questions, Qwen answers, memory, citations and delivery.

## Product boundary

Audience: authenticated HR employees on the hiring team. The assistant helps them
review an existing JD and its ranked applicants, inspect evidence, prepare a
shortlist/report and hand the reviewed package to the hiring department.

The revised design combines **typed HR conversation and dropdown navigation**.
The20October amendment supersedes the earlier dropdown-only restriction. It adds
no image, audio, video or file uploads. CV/JD content comes from authorized sources;
passwords and tokens remain outside conversation. The implemented portal includes
both typed questions and dropdowns. The
[21 October completion plan](../../../plan/2026-10-21-chatbox-completion/README.md)
addresses context sharing, retry recovery and verification.

Default review set: database-ranked top10 CVs for one authorized JD. Default
shortlist: ranks1–8 from that same ranking snapshot. The department package contains
those eight profiles/CVs plus a report comparing all ten. Smaller pools and missing
files are explicit incomplete states, never padded or represented as a complete
eight-person package.

Backend services own permissions, ranking, evidence identities and delivery. The
LLM supplies bounded explanations from validated evidence. It cannot choose email
recipients, modify scores, grant access or execute arbitrary actions.

## Relationship to existing work

The original [October implementation entry point](../../../plan/2026-10-12-streamlit-llm-chatbot/03-implementation.md)
now links here. The earlier [RAG implementation plan](../../../plan/2026-10-18-hr-conversational-rag/README.md)
remains the technical history for retrieval and QA experiments. For the portal
experience, the amendment integrates scoped free-text QA with the authenticated
portal rather than restoring the legacy upload UI. Existing model/research
restrictions and benchmark boundaries remain.
The production identity provider, ranking database, portal event feed, storage,
mail gateway and retention policy are unresolved dependencies, listed explicitly.

No model run is started, restarted or declared complete by this documentation
refactor. No email or notification has been sent.
