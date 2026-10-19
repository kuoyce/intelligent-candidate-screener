# Implementation record — 13 October 2026

Status: local UI prototype implemented; Qwen live smoke passed. Real R2 inference
is deferred because its trained weights are absent. The whole plan is not marked
complete while that dependent feature and baseline verification failures remain.

## Delivered

- `app/screening_service.py`: immutable snapshots, independent retrieval/CE
  scores, bounded shortlist/tail, load-only verified checkpoints, explicit
  classification availability, source evidence, completion and timing provenance.
  Classification thresholds apply to full-precision raw logits before display.
  Existing `pipeline.py` and model/training code remain unchanged.
- `app/chat_schema.py`, `chat_context.py`, `chat_prompts.py`, `chat_service.py`:
  bounded intent interpretation and evidence selection, canonical backend facts,
  exact redacted citations, stable candidate IDs, snapshot-bound follow-ups,
  reference validation, deterministic fallback and clarification on ambiguity.
- `llm/base.py`, `qwen.py`: provider abstraction, structured text redaction,
  tokenizer budgets, bounded requests/output, loopback default boundary and
  optional service tokenizer. No annotation, training or unrestricted tools.
- `llm/qwen_assets.py`, `local_server.py`: the workshop's pinned Qwen2.5-0.5B
  provisioning and local CPU generation in the existing isolated transformer
  runtime. A loopback-only service supplies completion and exact token-count
  endpoints; generated tokens exclude the input prompt.
- `app/rerank_assets.py`: frozen manifest export from the recorded selected
  R1/R2 training-validation configuration, without grid selection or training.
  Default R2 is `r2-60412b3f1f1e`; its trained files are missing locally.
- `app/rerank_client.py`, `llm/local_reranker.py`: load-only local CE transport
  for the isolated Intel Mac runtime, finite-score/digest checks, bounded UTF-8
  batches and raw-logit inference. No substitute trained checkpoint is claimed.
- `app/session.py`, `streamlit_app.py`: settings/autodiscovery, snapshot/chat
  invalidation, explicit backend selection, QA/source rendering, model failure
  states, reset and bounded UI history. Existing reviewed interview ingestion
  is retained. Shared resources contain model weights, not applicant answers.
- `app/chat_smoke.py`: synthetic and configured live smoke reports with
  timings, budgets, client RSS and quote audits; no real applicant data.

## Approval and implementation choices

Branch: `feat/streamlit-llm-chatbot`. Approved plan commit: `0273882`.
Backend, QA and local runtime tasks were delegated; integration and cross-path
review were performed by the main agent. An independent review identified and
resolved specialized-answer omission, ambiguous plural follow-ups and fabricated
zero percentiles. UI integration exposed JSON redaction corrupting long decimal
scores and URL delimiters; redaction now visits textual JSON values instead.

The user clarified the backend using `src/candidate_screener/rerank/models.py`
and the LLM using the supplied workshop notebook. Its Qwen example uses
`pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")` with the selected
device. The wrapper uses the equivalent native chat template/generation path in
a local model service to preserve the repository's isolated neural runtime.
Qwen3 thinking parameters are not applied to Qwen2.5.

The research record selects R2 among the tested cross-encoders on validation,
but the registered overall stage-2 selection remains SBERT weak. R2 is pairwise
ordering, not a validated binary classifier. Application configuration reflects
that fact; there is no new threshold, adoption decision or final-set experiment.
The export digest authenticates restored local bytes only: the historical model
record lacks an original checkpoint digest to attest the restoration against.

## Model assets

Qwen model: `Qwen/Qwen2.5-0.5B-Instruct`.
Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
Checkpoint digest: `55fde3a06ebbfc5bc142d403f5b622ca3dd2ee958b2817665c4125ffcd85aca5`.
Local configuration: CPU, float32, no quantization, four threads,
4096-token context, at most 256 generated tokens, greedy decoding.
Only public model assets were fetched; no applicant text was transmitted.

Provisioning log: `logs/qwen-assets.log`; 361.72 seconds wall time and
520,548,352-byte peak RSS. This exceeded the initial estimate; subsequent similar
downloads should use the repository's long-job background workflow. Assets and
full inference responses remain git-ignored. No corpus source or annotation data
was added or modified.

## Verification

- QA: 68 synthetic tests; 34 authored question fixtures spanning intents,
  unavailable services, injection requests and invalid references.
- Local server/provider/assets/UI: 71 focused tests passed, including real
  loopback HTTP with synthetic inference doubles.
- Integration before local-service extension: 111 passed, 1 Torch-dependent skip.
- Regression proof: quote/ownership validation, specialized answers, ambiguous
  follow-ups, structured redaction, threshold precision, recorded selection,
  prompt slicing, provider redaction/budgets and session invalidation all have
  observed failing original or mutant runs. Logs include
  `chat-service-mutant.log`, `chat-service-review-before.log`,
  `chat-service-structured-before.log`, `screening-service-mutant.log`,
  `rerank-assets-mutant.log`, `local-qwen-server-mutant.log` and
  `chatbot-mutation-check.log` under `logs/`.
- Affected warm/cold runner initially selected no research records; logs:
  `logs/chatbot-check-selection.log`, `logs/chatbot-cold-checks.log`.
  No original training code or committed evaluation result changed. This does
  not retire the historical stage-2 cold-check debt.
- Synthetic smoke: four requests, four expected template fallbacks, six audited
  source quotes and six supported quotes. This is neither model accuracy nor
  live Qwen evidence. Report: `logs/chatbot-smoke.json`.

## Final verification and limits

Full suite (`logs/pytest.log`): **937 passed, 9 skipped, 2 failed**, 206.88 seconds.
Both failures are in `tests/test_baseline_golden.py`: the frozen vocabulary digest
and baseline figures do not reproduce on the local data/environment. The audit
`logs/baseline-change-audit.json` verifies that all 11 baseline dependency files
are unchanged from the approved-plan commit. No frozen record was overwritten.
The subsequent focused chatbot suite passed **182 tests**, with one base-environment
Torch skip (`logs/pytest-chatbot-final.log`). Later changes were checked narrowly.
Final affected warm/cold runner selected zero research records
(`logs/chatbot-check-selection-final.log`, `logs/chatbot-cold-checks-final.log`).

One provisioned model/runtime was measured. Initial synthetic live trials exposed
valid fenced JSON and schema copying; strict validation rejected these responses.
Single JSON fences are now supported without relaxing field/citation checks.
Versioned examples clarify the required schema; actual chat-format examples
support request interpretation by the small workshop model. All iterations were
on invented documents, not screening benchmarks or new labels.

Final live trial (`logs/chatbot-live-v3.json`): **3/3 answered, zero fallbacks,
9/9 source quotes supported**, including one free-form shortlist question. Request
times were 34.00, 17.96 and 19.97 seconds on CPU. This is a smoke trial, not a
quality estimate over arbitrary recruiter questions or a production latency SLA.
The local service verifies the expected model revision/digest/token budget through
`/v1/models` and records served Torch 2.2.2 / Transformers 4.48.3. Generation uses
greedy decoding and repetition penalty 1.0. The final smoke command fingerprints
system prompts and chat-format examples together.

Local services: Qwen at `http://127.0.0.1:8000`; Streamlit prototype at
`http://127.0.0.1:8501`. Streamlit's health endpoint returned `ok`. Live UI
verification with the invented demo and real Qwen is logged separately in
`logs/chatbot-local-ui-verification.log`.

The actual Streamlit UI trial passed: rendered the app, screened two invented
applicants, completed five-label entity extraction, and answered “Who is the
strongest applicant here?” through the real local Qwen service with three source
citations and no fallback. The displayed ranker is explicitly
`Illustration TF-IDF (submission pool)`. Both local services were left running
for the user's requested prototype; no public deployment was performed.

Real R2 scoring and any binary Fit / No Fit release remain unavailable until
the recorded trained cache is restored and verified. R2 itself has no validated
binary classification threshold; this wrapper does not invent one. Existing
baseline failures and historical stage-2 cold-check debt remain explicitly open.
