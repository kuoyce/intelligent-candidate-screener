# Design specifications

## 1. Responsibilities and data flow

```text
JD + reviewed candidate text
  → input validation / shared text_v2 processing
  → configured stage-1 retrieval
  → frozen cross-encoder scores on the shortlist
  → separately validated binary classifier, if available
  → evidence extraction + immutable screening snapshot
  → recruiter question → bounded evidence context → Qwen adapter
  → structured-answer validation → redacted Streamlit response + citations
```

Classification may reuse the cross-encoder only when its score semantics and
threshold were validated for binary Fit / No Fit. Otherwise the selected classifier
is a distinct frozen component or classification remains unavailable. Applying a
sigmoid to a pairwise score does not create a fit probability.

Qwen is the conversational interface. Ranking and classification run when the user
screens or changes inputs, not again for every chat message. Qwen cannot train,
dispatch annotation, change thresholds, alter candidate text or execute arbitrary
tools. No benchmark labels enter its context. D42 collection remains closed;
conversational inference is not a new annotator series or training-label source.

## 2. Backend contracts

Introduce a session-owned `ScreeningSnapshot` with snapshot ID; digest of scoring
inputs and configuration; JD text/title references; candidate results; retrieval,
reranking, classification and extraction provenance; completion status; and timings.
Model manifests record checkpoint revision/content digest, tokenizer, maximum
sequence length, truncation policy, text variant, threshold and its selection record.
Secrets and document contents are excluded from operational logs.

Each candidate carries an opaque ID, retrieval rank/score, optional reranked
rank/score, final display position, `rerank_status`, optional Fit / No Fit,
`classification_status`, optional validated probability, evidence and source lineage.
Use named score fields rather than blending incomparable retrieval and CE scores.
Retain legacy fields only with explicit compatibility semantics.

The reranked shortlist occupies its own ordered section. The retrieval-only tail
retains retrieval order and is labelled not reranked; it has no CE rank or fabricated
CE score. A top-k answer states whether it describes the reranked shortlist or the
entire pool. Classification coverage is explicit per candidate.

Validate finite scores, one score per input, ID alignment and uniqueness. Break
equal scores by opaque candidate ID. Apply classification thresholds at full score
precision before rounding. Model loading validates artifacts and fails clearly;
it cannot call training-capable `fit()` as a fallback. Missing weights must not
silently substitute TF-IDF while the UI reports cross-encoder results.

Use only CV/JD text and the permitted JD title (D53/D54). Preserve extraction
completion states and reviewed translation provenance. Evidence comparisons show
observations alongside scores; they do not claim causal attribution for a black-box
cross-encoder. Unknown experience and absent evidence remain unknown, not negatives.

## 3. Proposed package boundaries

| Location | Responsibility |
|---|---|
| `app/chat_service.py` | Session-independent QA orchestration and fallback |
| `app/chat_context.py` | Snapshot selection, redaction and token-budgeted evidence |
| `app/chat_schema.py` | Typed requests, answers, citations and validation |
| `app/chat_prompts.py` | Versioned grounding instructions and examples |
| `llm/base.py`, `llm/qwen.py` | Provider interface and configured Qwen service adapter |
| `app/screening_service.py` | Frozen model loading, retrieval/rerank orchestration, snapshots |
| `pipeline.py` | Reuse evidence generation and integrate explicit backend results |
| `app/intents.py` | Existing deterministic intents and degraded-mode answers |
| `app/streamlit_app.py` | Thin input/rendering transport and session state |

Confirm final file boundaries in implementation review; reusable logic must remain
independent of Streamlit and be tested with synthetic inputs.

## 4. Grounded QA contract

Support shortlist, candidate explanation, comparison, missing evidence, skill,
stated experience, entity and source questions. Resolve rank references against the
current snapshot; store stable IDs for follow-ups. Unknown or ambiguous references
produce clarification. Changing JD, candidate text or scoring configuration creates
a new snapshot and invalidates old evidence references and chat context.

Context contains only relevant redacted JD requirements, backend facts and selected
evidence spans. Reuse deterministic intents where possible; use bounded structured
LLM request interpretation for other questions, validating allowed operations and
arguments before selecting context. Do not send the entire applicant pool or rely
on conversational memory as a source of candidate facts. Budget prompt, evidence,
history and output tokens using the selected runtime's tokenizer. Preserve original
document offsets internally; citations refer to redacted display spans via an
explicit mapping because redaction can change offsets.

Return a structured answer with snapshot ID, answer blocks, candidate references,
evidence IDs and status (`answered`, `clarification`, `insufficient_evidence`,
`provider_unavailable`). Numeric facts, ranks and classifications are rendered
from validated backend references, not LLM-written values. Direct quotations are
rendered from validated source spans. Validate citation existence, candidate
ownership, snapshot identity and exact quotes. Citations alone do not establish
that narrative claims are supported: test factual consistency separately and
constrain the MVP to evidence summaries. Fall back to templates on invalid output.

Treat CVs, user questions and prior generated prose as untrusted data. The system
instruction forbids obeying instructions embedded in documents. The adapter has
no shell, filesystem or unrestricted tools. Redact input before model calls and
output before display with `annotation.redact`; never expose filenames as identity,
raw provider errors, secrets or unsafe HTML. Missing evidence is phrased as
“no evidence found in the supplied material.”

## 5. LLM serving and Streamlit UX

Use a provider interface with configurable endpoint/model, authentication from
environment or Streamlit secrets, bounded timeouts/output tokens, cancellation
where supported and at most one retry for transient errors. Structured-output
support is verified on the selected runtime; unsupported or malformed responses
use the deterministic fallback. Validate complete output before rendering factual
answers; a spinner is sufficient for MVP.

Qwen's official documentation describes local runtimes and OpenAI-compatible model
servers. This supports the proposed external service boundary, but does not select
a host or a checkpoint. See [official Qwen deployment guidance](https://qwenlm.github.io/blog/qwen3/)
and [Qwen repository](https://github.com/QwenLM/Qwen3).
Verify model-specific chat templates, non-thinking configuration and generation
settings when Q3/Q4 are resolved; do not copy settings across model families.

UI: keep the existing submission/review flow; provide JD/title inputs, a screening
action, labelled ranking/fit/evidence panels, recruiter chat and reset controls.
Display ranking scope, model identity and classification availability. Provide
clear loading, invalid-input, unavailable-model and stale-session states. QA
failure leaves the screening results available with template answers.

Cache immutable model resources by artifact/config digest. Keep submissions,
evidence, question history and snapshots in session state; no global document or
answer cache. Shared inference resources require concurrency-safe access. Reset
clears session documents and history; do not promise immediate zeroization of
runtime memory. Persistence, media changes and public hosting are outside the
proposed MVP until Q5/Q7 are answered.
