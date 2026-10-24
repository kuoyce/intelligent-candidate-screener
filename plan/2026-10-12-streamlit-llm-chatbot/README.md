# Streamlit LLM screening chatbot

19 October documentation refactor: the new HR portal conversation design and
implementation plan live in
[src/llm-plan/2026-10-19-hr-conversation-design](../../src/llm-plan/2026-10-19-hr-conversation-design/README.md).
The original implementation record is preserved there. The scope below describes
the historical local prototype; authentication, durable sessions, alerts and
department email are proposed new integrations, not completed features.

Date: 12 October 2026. Approved: 13 October 2026. Status: **wrapper implemented;
local Qwen/UI verified; real R2 scoring deferred until trained weights are restored**.

## Requested outcome

Extend the existing Streamlit application into a conversational wrapper around the
candidate-screening backend. A question-answering LLM, such as Qwen, explains
backend rankings, Fit / No Fit decisions and source evidence. The requested backend
includes cross-encoder re-ranking. The LLM must not invent scores or replace the
classification model.

Read [design specifications](01-design-spec.md), then the
[implementation sequence and acceptance criteria](02-implementation-plan.md).
Implementation and verification are recorded in [03](03-implementation.md).

## Verified starting point

- `src/candidate_screener/app/streamlit_app.py` already supplies a session-only UI,
  applicant ingestion and recruiter chat. `app/intents.py` answers using rules and
  templates. Extend these paths rather than create an unrelated second application.
- `pipeline.py` currently defaults to draft TF-IDF `text_v2`, accepts an injected
  ranker and returns evidence, entities and provenance. It deliberately sets
  `fit_call=None`; its `confidence` is a pool percentile, not a probability.
- `matching/base.py` defines `System`; `rerank/base.py` supplies `Reranker` and
  `SystemReranker`. `matching/cross_encoder.py` has A1/weak binary classifiers
  with validation-selected thresholds. Their `fit()` can train: application startup
  must instead load a frozen, approved artifact.
- `rerank/models.py` contains R1 pointwise and R2 pairwise cross-encoders. These
  ordering objectives do not establish a validated binary classification threshold.
- [P8 findings](../2026-10-06-extraction-and-rerank/17-p8-stage2.md) selected
  **SBERT weak ordering**, because no new reranker earned adoption. Therefore
  this repository does not establish the requested cross-encoder as its final
  adopted classification model. The findings also record a deferred cold check.

This plan accommodates an explicitly approved cross-encoder integration or
experimental configuration. It does not change the recorded research winner.

## Persistent open questions

| ID | Decision needed | Proposed direction / dependency |
|---|---|---|
| Q1 (model resolved; weights missing) | Which cross-encoder should the app use? | User directed to `rerank/models.py`: recorded selected R2 `r2-60412b3f1f1e`. Weights are absent; restore them. Application use remains distinct from research adoption. |
| Q2 (resolved: unavailable) | Does R2 supply validated Fit / No Fit? | R2 is pairwise ordering with no binary threshold. Show ranking and unavailable classification. No calibration/training is authorized. |
| Q3 (resolved for MVP) | Where does Qwen run? | Workshop-directed local CPU inference in the existing isolated transformer runtime; loopback only, no remote applicant transfer. |
| Q4 (prototype verified) | Which Qwen configuration and latency? | `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision/digest, float32 CPU, four threads, no quantization. Three synthetic smoke questions answered without fallback; actual local UI screening/free-form QA verified. Production SLA unspecified. |
| Q5 (resolved by approval) | Is MVP recruiter-only, or should the LLM also converse with applicants? | Recruiter QA over existing reviewed submissions. Applicant LLM intake remains an extension. |
| Q6 (resolved for MVP) | Which stage-1 retriever, shortlist size and maximum pool size should the app expose? | Explicit TF-IDF illustration retrieval, configurable shortlist (default 50), max pool 500; no research adoption. Frozen train-region retriever selection remains separate. |
| Q7 (resolved by approval) | Is session-only use sufficient, or are authentication, persistence and multi-user hosting required? | Session-only MVP. Hosting/persistence require a further design. |

The user approved implementation of this plan on 13 October 2026. Approval accepts
the proposed recruiter-only, session-only MVP and configurable top-50 retrieval
scope (Q5–Q7), not remote data transfer. Subsequent user clarification directed
Qwen use to the local workshop and backend use to `rerank/models.py` (Q1–Q4).
Qwen and local UI validation passed. Restoring selected R2 weights and its real
inference validation remain open. Missing R2 is explicitly excluded from the
working illustration prototype rather than reported as validated.
Implement independent contracts and configuration now; missing model configuration
has an explicit unavailable state. No training, research adoption or deployment is
authorized by this wrapper implementation. Create `feat/streamlit-llm-chatbot` and
commit the approved plan before code changes, as required by AGENTS.md.
