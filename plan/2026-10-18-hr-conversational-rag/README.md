# HR portal assistant: hybrid factual QA and conversational RAG

18 October 2026. Status: **proposed for user review; implementation not started**.
Replaces the scope proposal in ../2026-10-17-chatbot-qa-improvement/README.md;
preserves that record as history. This is a new conversational product design,
building on reusable screening/provider services from the October chatbot MVP.

## 1. Outcome and scope

Help an HR user investigate which applicants best match an active job, compare
named applicants and ask factual follow-ups without uploading documents again.
Answers must use the active screening snapshot and verifiable CV/JD passages.
The first delivery is a local Streamlit demonstration with session isolation;
production authentication, durable storage and HRIS integration are a later scope.
Keep document uploads and recruiter-facing interactions in the existing app.

Examples of required behavior:

| Question | Response contract |
|---|---|
| Who best suits this Senior ML Engineer role? | Show the backend's top three within the active pool/filter scope; explain supported strengths and evidence gaps with citations. State unavailable reranking explicitly. |
| What certifications does candidate A hold? | Return all supported certification mentions found in retrieved CV passages, with quotes; do not infer certification from skill names. |
| Why is A ranked higher than B? | State backend ranks/scores, compare evidence by JD requirement, and distinguish observed evidence from a causal explanation of the ranker. |
| What about C? | Preserve active JD; resolve C by stable candidate ID and compare against the active shortlist, or clarify ambiguity. Explicitly named C is retrieved even outside the default top three. |
| Only consider candidates with evidence of AWS certification | Apply a visible recruiter-confirmed filter; label the result evidence-based and disclose incomplete coverage. Do not equate missing mentions with proof of absence. |
| How many years of PyTorch experience? | Report an explicit source statement, otherwise say it is not established. Do not invent tool-specific tenure from total employment years. |

The assistant explains screening results; it does not issue hiring decisions,
create scores, change ranks or invent Fit classifications. The repository's
research winner and frozen evaluation protocol remain unchanged.

## 2. Architecture and model responsibilities

Flow: Streamlit turn -> typed dialogue state -> query router -> bounded backend
operations -> factual QA or comparative synthesis -> claim/citation validation ->
rendered answer and source cards -> updated session state.

**Two retrieval paths are necessary.** Candidate retrieval takes a JD and the
permitted applicant pool, then the configured role-matching reranker scores that
shortlist. Evidence retrieval takes a question/requirement and searches passages
within the requested candidates and active JD. A good JD/CV relevance scorer is
not automatically a good question/passage relevance scorer. Start evidence search
with lexical retrieval and benchmark dense retrieval separately. A passage
reranker, if needed, has a separate task/model identity and evaluation.

Repository inspection: rerank/models.py defines R1 pointwise and R2 pairwise
encoder-only cross-encoders, plus R3 bi-encoder, using all-MiniLM-L6-v2. They are
not encoder-decoder generators. Preserve a typed ranker interface so a future
approved architecture can be integrated without changing answer contracts.
Recheck R2 artifact availability and digests: the previous MVP recorded missing
weights. Never train at app startup or silently substitute a model. A loaded
illustration ranker is visibly identified. R2 ordering scores are not probabilities
and do not establish a Fit threshold.

**Factual route.** Consult validated entity spans where appropriate, then retrieve
source passages and apply an extractive QA baseline for open factual questions.
Proposed external baseline: deepset/roberta-base-squad2, with explicit no-answer
handling. GLiNER's four labels do not cover every HR fact: certifications are not
automatically a fifth gold category, and extraction labels are not QA supervision.
Single-span QA is insufficient for exhaustive lists: retrieve multiple sections,
collect supported spans, deduplicate and disclose coverage; abstain if completeness
is not established. Keep negation, expired/ongoing qualifications and dates in
their surrounding context. All output spans must map exactly to source text.

**Synthesis route.** Establish the existing Qwen2.5-0.5B-Instruct as a controlled
baseline and benchmark Qwen/Qwen2.5-1.5B-Instruct as one external pretrained
generator candidate. These are proposed evaluation candidates, not promised
quality improvements. Pin revision, license record, tokenizer, runtime and artifact
hashes before use. Measure local memory/latency before selecting a model; do not
load all models concurrently. Default answer context is top three backend-ranked
candidates plus question-relevant JD requirements; explicit comparisons can
override this set. Candidate ranking and evidence ranking remain separate.

Generator input roles: system rules; structured question and bounded history;
untrusted quoted evidence payload with explicit IDs. A template is not the only
user-facing answer: allow explanatory sentences tied to atomic supported claims,
plus a structured requirement comparison table. The renderer retains backend
facts. Model-authored factual claims require validated support; unsupported claims
are removed or answered with an evidence-unavailable message. Citation existence
alone is insufficient to establish semantic support. MVP prose uses verified
claim types/relations; unrestricted paraphrases stay experimental until independently
evaluated. No free-text answer token is streamed as verified before validation.

## 3. Evidence and source contracts

Each passage carries snapshot_id, document_id, candidate_id (or JD owner),
document_digest, passage_id, original start/end, redacted display start/end,
source section, and page only when a parser actually preserved it. Parquet/plain
text has no assumed page number. Each entity carries its model/human provenance.

Each answer contains request_id, snapshot_id, status, resolved candidate IDs,
structured backend facts, claims with evidence IDs, answer text and citations.
Validate current snapshot, evidence ownership, exact source slices, claim type and
supported relation before rendering. Scores/ranks are copied from the snapshot.
Evidence gaps use statuses such as supported, contradicted, not_found or unknown;
do not encode unknown as a numerical zero skill gap. Distinguish required/preferred
JD requirements. An evidence matrix aids explanation; it is not a new scoring model.

Use sentence/section-aware chunks with actual tokenizer budgets and overlap for
boundary coverage. Keep complete source offset mappings and deduplicate repeated
spans. Never silently truncate long CVs. Pack per-candidate coverage before adding
extra passages, reserve output tokens, and expose incomplete retrieval/parser
coverage. Explicitly test facts late in long documents and cross-sentence facts.
Redaction applies before display/model input; preserve mapping to original source
privately. Do not stem source text or reuse display offsets on original documents.

## 4. Dialogue state and Streamlit experience

Use st.chat_input/st.chat_message and st.session_state holding a typed DialogueState:
session_id, active_jd_id/digest, snapshot_id, pool_digest, backend/model identity,
shortlist IDs, explicitly selected comparison IDs, recruiter filters with provenance,
last resolved candidates/requirement, bounded turn history and referenced evidence.
History is not evidence and cannot introduce candidate facts into the next answer.

Changing JD, document bytes, pool, filters or backend creates a new snapshot and
invalidates stale references/indexes/answers. Retain previous messages as labelled
historical results, not current facts. Pronouns resolve only when unambiguous;
clarify otherwise. Do not use pool-relative rank as permanent candidate identity.
Changing a constraint has a visible preview and explicit apply action before
reranking; ordinary factual questions do not mutate the active pool.

Layout: sidebar for active role/documents/model status and filters; main chat with
shortlist and side-by-side comparison cards; expandable cited excerpts with source
highlighting; suggested follow-ups; progress/status during retrieval/model calls.
Display ranking scope, unsupported claims withheld, and unavailable services in
plain language. Private session indexes stay session-owned; cache immutable model
weights with a concurrency guard, never applicant data in a global shared cache.
Streamlit reruns must not duplicate jobs, messages or model loads.

## 5. Training proposal and data boundaries

First compare pretrained baselines with identical retrieval/context. Fine-tuning
cannot repair omitted passages or wrong candidate resolution. Train only the
component responsible for measured development errors.

Prepare an initial **authored synthetic** corpus of 240 question turns across
48 document-family scenarios: 24 families/120 turns train, 12/60 dev, 12/60 test.
Keep every scenario's candidates, JD, paraphrases and dialogue turns in one split.
Cover answerable/unanswerable facts, lists, ranking, comparisons, filters, follow-ups,
ambiguity, contradiction and long-document evidence; all primary query families
occur across splits. This small corpus is a feasibility experiment, not proof of
real recruiter accuracy. Freeze manifest, answer references and test before tuning.
No LLM annotation dispatch or new real human collection is authorized by this plan.

Factual labels: question, context/document identity, answer spans/alternatives,
is_impossible, supporting passage IDs and coverage. Use SQuAD2-style single-answer
training only for compatible examples; list questions need a separate aggregation
target/evaluation. Preserve original character boundaries via fast-tokenizer
offset mappings, overflow windows, context-only masks and no-answer targets; never
round labels or mark an answer-containing overflow window falsely negative.

Conversation targets: resolved slots, route, approved backend operation, selected
evidence IDs and supported claim records. Start with intent/evidence-selection
adaptation; consider generator adapters only after their need is measured and
resources support them. Do not train on generated Fit labels, backend explanation
guesses, the ten human GLiNER examples, A2 eval/final judgements or independent
entity dev/test. Existing D51 permits only reranker training on its scoped LLM
batch; it is not QA training data authorization.

Prespecified training experiment: one extractor configuration first, seed 0,
learning rate 2e-5, three epochs, effective batch eight via accumulation if needed,
AdamW; select epoch and no-answer threshold on dev only. Pin model-specific context
and stride after a measured budget check, before the run. Save training/source/code
digests and timing. Compare unchanged pretrained weights versus adapted weights.
Generator adaptation requires a documented resource/config amendment before any
run; no speculative grid. One configuration is timed before further jobs, training
over five minutes follows the repository's tracked-background/end-turn rule.

## 6. Evaluation and acceptance

Freeze B0 existing chatbot, B1 improved retrieval + unchanged Qwen baseline,
B2 extractive QA + grounded synthesis hybrid, B3 adapted extractor (conditional).
Compare any larger generator on the same frozen retrieval configuration to isolate
its contribution. Tune on dev, select one configuration, then evaluate held-out test
once. Keep QA metrics separate from existing ranking benchmarks/research adoption.

Report factual exact match/token F1, list precision/recall, no-answer precision/
recall; evidence Recall@k and citation support/ownership; route/slot accuracy and
conversation-task completion; synthesis claim support, requirement coverage and
contradiction rate; p50/p95 end-to-end latency and peak memory. Report counts and
95% CIs, paired over document families (ranking comparisons over JDs); if A2 figures
are ever referenced, retain generic/targeted/total and their prescribed protocol.
Use structured reference comparison plus manual inspection of synthetic outputs,
not a judge model as sole truth. No new human evaluation batch without scope approval.

Mandatory deterministic tests: zero incorrect citation owners, stale snapshots,
modified backend facts or cross-session disclosures; all malicious-document
instructions remain inert; unavailable models abstain safely; explicit named
candidate works outside top three. All invariant tests must pass. Predeclare a
provisional product target of >=90% supported atomic claims and >=80% scripted
task completion on dev, with sample counts/CIs; these are targets, not current
results. Select only a configuration that meets hard invariants and improves dev
task completion over B0; otherwise report no validated improvement. Report held-out
results honestly even when targets fail. Latency SLA remains open until hardware
measurement; no silent model substitution to meet a number.

## 7. Delivery order and review

1. On approval create branch and commit this finalized plan, preserving dirty work.
   Freeze contracts, synthetic split manifest, baseline and acceptance definitions.
2. Implement source indexing/question-aware evidence retrieval and source mappings.
3. Implement typed dialogue manager, routes and bounded backend tool interface.
4. Integrate pinned extractive/generative baselines and validation/rendering.
5. Build the redesigned Streamlit flow and synthetic browser scenarios.
6. Run ablations, timed pretrained baselines and conditional extractor adaptation;
   review final configuration before held-out reporting. No production deployment.

Delegate independent evaluation, evidence retrieval/QA, and dialogue/UI work with
file ownership; integrate sequentially at shared contracts. Reuse existing services
where correct but do not preserve the old canned interaction design as a product
requirement. Code lives in package app/ and llm/; tests use synthetic fixtures and
fake providers, optional native smoke is separate. Show regression tests fail on
the prior behavior, run focused then full pytest once with logs, affected checks via
runner and cold checks where D55 requires. Record unrelated baseline failures
without changing frozen numbers. Review source grounding, race conditions,
memory and data boundaries. Update README, decisions and implementation records.

## 8. Open questions and explicit defaults

- Hardware/latency: start measurement on the existing local CPU runtime; model
  size choice remains conditional on timings. No cloud GPU or paid API assumed.
- R2 availability: verify actual checkpoint before integration. Missing weights
  remain an explicit unavailable state; restoration is separate from retraining.
- Real QA gold: none confirmed. Synthetic feasibility first; any future human QA
  batch needs its own document-disjoint scope and decision, separate from D56 spans.
- Portal deployment: local/session-only prototype proposed; auth, persistence,
  retention and multi-user hosting are not included in this implementation scope.
- User examples: primary scenarios above derive from this request; additional
  unsatisfactory real chat examples can inform dev before test is frozen.

## Primary references checked 18 October 2026

- Streamlit conversational apps/session state:
  https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
- Retrieve and rerank responsibilities:
  https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Transformers extractive QA/tokenizer offsets:
  https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/question_answering.md
- Extractive baseline: https://huggingface.co/deepset/roberta-base-squad2
- Generator candidate: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Generation settings:
  https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial.md
  Use do_sample=False for a reproducible baseline. Temperature/top_p only govern
  sampling when enabled; neither constitutes a grounding check.
