# Chatbot history: the earlier prototypes

**Historical.** This page keeps the record of the recruiter chatbot before the Hiring Desk:
the free-form/upload prototype (`legacy_streamlit_app.py`), the 19 October conversation-design
plan, the HR conversational RAG upgrade, the workshop setup with the local Qwen2.5-0.5B
service, the R2 cross-encoder backend and its manifest schema, and the Qwen configuration
table. None of it is the current way to use the app: current usage is in
[`hiring-desk.md`](hiring-desk.md). Moved from README.md on 1 Nov 2026.

## The legacy prototype

The earlier free-form/upload prototype is preserved separately:

```bash
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
```

The historical workshop setup below describes that prototype. Current portal
status is in the [extractive implementation record](../plan/2026-10-24-grounded-qwen-comparisons/06-implementation-record.md).

## Plans that led to the Hiring Desk

The [19 October conversation-design plan](../src/llm-plan/2026-10-19-hr-conversation-design/README.md)
defines the authenticated HR portal workflow, notifications,
stored sessions, top10 review and confirmed top8 department handoff (the current app hands off ten redacted CVs;
historical eight-CV packages remain readable). It preserves
the earlier implementation record and distinguishes planned integrations from
the current local prototype.

## The HR conversational RAG upgrade

The earlier [HR conversational RAG upgrade](../plan/2026-10-18-hr-conversational-rag/README.md)
introduced question-specific CV/JD passage retrieval, stable candidate follow-ups,
explicit filter previews and cited source mentions. The Hiring Desk v1 path ([`hiring-desk.md`](hiring-desk.md)) supersedes
its generator selection;
synthetic development measurements have not established production QA accuracy.
The extractive baseline is provisioned explicitly, never downloaded at UI startup:

```bash
PYTHONPATH=src uv run --project runtime/transformer --no-sync python -m candidate_screener.llm.qa_assets
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
```

Question examples: “Who best suits this role?”, “Compare 1 and 2”, “What
certifications does applicant 1 hold?”, and “What about applicant 2?”. Source
mentions retain context; a mention does not establish an active qualification,
and missing evidence does not prove a qualification is absent. The QA feasibility
corpus is authored synthetic data with a frozen family-disjoint manifest under
`output/chatbot-rag/`; it does not use the human GLiNER labels or frozen Fit labels.

## The workshop setup: the approved chatbot plan and local Qwen

The [approved chatbot plan](../plan/2026-10-12-streamlit-llm-chatbot/README.md) adds
session-only recruiter QA, immutable screening snapshots, frozen cross-encoder
loading and an optional Qwen service. Existing written CV and reviewed French
interview ingestion remains available.

The user-supplied workshop selects `Qwen/Qwen2.5-0.5B-Instruct` for local
generation. Provision it once and start the loopback service in a separate terminal:

```bash
uv run python -m candidate_screener.llm.qwen_assets
uv sync --project runtime/transformer
PYTHONPATH=src uv run --project runtime/transformer --no-sync python -m candidate_screener.llm.local_server --manifest data/processed/chatbot/qwen/manifest.json
```

For the historical prototype, start Streamlit as below. The provisioned manifest supplies local Qwen
settings automatically. The isolated service uses CPU, four threads and greedy
decoding; the app counts tokens through `/v1/tokenize`, so its environment needs
no Torch for Qwen. Loading is offline; only provisioning downloads public files.
`SCREENER_QWEN_DISABLED=true` forces templates.

## The R2 backend and frozen cross-encoder manifests

For the requested backend in `rerank/models.py`, prepare the recorded selected R2:

```bash
uv run python -m candidate_screener.app.rerank_assets
```

Streamlit and the local service discover the exported manifest automatically;
restart the service after preparation. The selected directory
`data/processed/rerank/r2-60412b3f1f1e` is git-ignored and may be absent from a
checkout; if so, restore its complete trained checkpoint and tokenizer before preparation. No training is
started. R2 has no validated binary threshold, so classification remains
unavailable. App use does not change the research selection of SBERT weak ordering.

```bash
uv sync --group app
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
uv run python -m candidate_screener.app.chat_smoke
```

With no model configuration, select **Illustration TF-IDF** and **Screen
demonstration** to use invented applicants and grounded template answers. The
illustration fits unsupervised IDF on submitted text; it is not an adopted benchmark
model. Ask `top 5`, `explain 1`, `compare 1 and 2`, `entities 1`, or `source 1`.
With Qwen, other phrasing is mapped to supported screening operations and the model
selects evidence. Backend facts, scores and quotes are rendered from validated
references. Generated narrative is excluded from the MVP. Invalid responses use
templates.

Provision an approved local Transformers sequence-classification checkpoint and
its tokenizer in the same directory. Native loading requires Torch/Transformers
in the app environment (`uv sync --group app --group ml` on a supported host).
On Intel macOS, the local service also supplies frozen CE inference through
`/v1/rerank`; it discovers the verified exported manifest at startup.
`SCREENER_CE_SERVICE_URL` selects this loopback transport, so the app needs no
main-environment Torch for R2. Set it to an empty string for native loading, or
pass `--reranker-manifest` to the service for another frozen manifest. App startup
performs no weights download or training.

Set `SCREENER_CE_MANIFEST` to a local JSON file with this schema:

```json
{
  "name": "Approved cross-encoder",
  "checkpoint": "checkpoint-directory",
  "tokenizer": "checkpoint-directory",
  "sha256": "replace-with-checkpoint_digest-result",
  "text_variant": "text_v2",
  "score_semantics": "ordering_logit",
  "selection_record": "path-or-id-of-approved-selection-record",
  "max_length": 384,
  "truncation_policy": "longest_first",
  "batch_size": 8
}
```

Paths resolve relative to the manifest. Compute the digest using
`candidate_screener.app.screening_service.checkpoint_digest(Path(...))`; it covers
relative filenames and bytes, including tokenizer files. Symlinked artifacts are
rejected. Length and truncation must match the approved artifact; this example is
not a model selection. The default shortlist is 50, the pool limit is 500, and
scoring documents are limited to 200,000 characters. The retrieval-only tail is
labelled **not reranked**. Retrieval scores and CE logits are separate fields.

Fit / No Fit requires `score_semantics: "binary_logit"`, a finite
`binary_threshold` **in raw logit units**, and `threshold_validation_record`.
Never copy a probability threshold into a logit manifest. For a model whose
documented output is `sigmoid(logit)`, an interior probability threshold `p` maps
to `log(p / (1-p))`; record that equivalence and the original validation lineage.
Ordering checkpoints, including R2, acquire no binary threshold through this
wrapper. Manifest declarations and file integrity are checked; statistical quality
of a supplied threshold is not independently revalidated. Without a validated
threshold, classification is **unavailable**. Percentiles are not probabilities.

## Configuring Qwen

Configure Qwen through environment variables, or matching string keys under
`[chatbot]` in `.streamlit/secrets.toml` (never commit secrets):

| Setting | Meaning |
|---|---|
| `SCREENER_QWEN_BASE_URL` | OpenAI-compatible URL, e.g. `http://127.0.0.1:8000/v1` |
| `SCREENER_QWEN_MODEL` / `SCREENER_QWEN_REVISION` | Exact served model and provisioned revision |
| `SCREENER_QWEN_SHA256` | Expected service digest; local provisioning supplies it automatically |
| `SCREENER_QWEN_TOKENIZER` | Matching tokenizer directory, or `service` for local `/tokenize` |
| `SCREENER_QWEN_RUNTIME` / `SCREENER_QWEN_QUANTIZATION` | Runtime name/version and quantization, or `none` |
| `SCREENER_QWEN_API_KEY` | Optional credential |
| `SCREENER_QWEN_CONTEXT_TOKENS` / `SCREENER_QWEN_OUTPUT_TOKENS` | Defaults 4096 / 512; match deployed model |
| `SCREENER_QWEN_TIMEOUT` / `SCREENER_QWEN_TEMPERATURE` | Defaults 30 seconds / 0.2; validate on selected model |
| `SCREENER_QWEN_JSON_MODE` | Default `true`; disable for runtimes lacking JSON response format |
| `SCREENER_QWEN_ENABLE_THINKING` | Optional `true`/`false` for compatible models/templates only |
| `SCREENER_QWEN_ALLOW_REMOTE` | Default `false`; explicit data-boundary decision and HTTPS required |
| `SCREENER_QWEN_DISABLED` | `true` forces templates and suppresses model autodiscovery |

Qwen supports OpenAI-compatible serving; chat templates and thinking options vary.
See the [official Qwen2.5 model and generation example](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct#quickstart)
before provisioning. The client counts tokens locally, reserves output space,
limits response bytes and retries transient errors at most once. HTTP redirects
and inherited proxies are disabled. Model revision/runtime declarations are
configuration provenance for external services. The provisioned local service's
revision, digest and token limits are checked against `/v1/models`, and its actual
runtime versions are recorded.

```bash
uv run python -m candidate_screener.app.chat_smoke --live --freeform
```

Live smoke uses invented documents and two QA requests (plus a free-form question
with `--freeform`), reports load/answer time,
token counts, client peak RSS and fallbacks, and exits unsuccessfully on missing
configuration or generated-answer fallback. Measure service memory separately.
The report separates Qwen QA from CE availability. A Qwen-only trial uses explicit
TF-IDF illustration retrieval if no CE manifest is configured; it never proves
R2 inference. Measure checkpoint/service compatibility separately.

Applicant material and chat context stay in session memory; shared resources hold
frozen models only. Document, JD/title or configuration changes invalidate results
and history. Reset clears session material. Redaction masks contact identifiers
but does not anonymize names. No persistence, public hosting, new annotation or
model adoption is included.
