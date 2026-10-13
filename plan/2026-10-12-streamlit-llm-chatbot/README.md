# Streamlit LLM screening chatbot

Date: 12 October 2026. Approved: 13 October 2026. Status: **implementation in progress**.

## Requested outcome

Extend the existing Streamlit application into a conversational wrapper around the
candidate-screening backend. A question-answering LLM, such as Qwen, explains
backend rankings, Fit / No Fit decisions and source evidence. The requested backend
includes cross-encoder re-ranking. The LLM must not invent scores or replace the
classification model.

Read [design specifications](01-design-spec.md), then the
[implementation sequence and acceptance criteria](02-implementation-plan.md).

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
| Q1 | Which exact checkpoint is the intended final cross-encoder, and is its use an experimental app option or an adoption decision? | Supply artifact/config digest and its selection record. Do not silently choose R2 or rename it final. Blocks backend release. |
| Q2 | Does that artifact provide a validated Fit / No Fit threshold? | Use its recorded validation threshold; otherwise show ranking with classification unavailable. A new threshold needs a separate approved train/dev protocol. Blocks classification. |
| Q3 | Where will Qwen run, with what RAM/GPU and permitted data boundary? | Propose a local model service with configurable endpoint; remote deployment requires an explicit data handling decision. Blocks real-provider configuration. |
| Q4 | Which Qwen model/revision, quantization and answer latency budget? | Benchmark one compatible instruction-following model on the chosen host before selection; pin runtime and artifact. No checkpoint size is assumed. |
| Q5 | Is MVP recruiter-only, or should the LLM also converse with applicants? | Propose recruiter QA over existing reviewed submissions. Applicant LLM intake is a separate extension. Blocks conversation scope approval. |
| Q6 | Which stage-1 retriever, shortlist size and maximum pool size should the app expose? | Propose explicit retriever configuration and top-50 default subject to approval; document candidates outside the reranked set. |
| Q7 | Is session-only use sufficient, or are authentication, persistence and multi-user hosting required? | Propose session-only MVP, matching current app. Hosting and persistence need a further design. |

The user approved implementation of this plan on 13 October 2026. Approval accepts
the proposed recruiter-only, session-only MVP and configurable top-50 retrieval
scope (Q5–Q7), not an unspecified model artifact or remote data transfer. Q1–Q4
remain open for actual model configuration and measured provider validation.
Implement independent contracts and configuration now; missing model configuration
has an explicit unavailable state. No training, research adoption or deployment is
authorized by this wrapper implementation. Create `feat/streamlit-llm-chatbot` and
commit the approved plan before code changes, as required by AGENTS.md.
