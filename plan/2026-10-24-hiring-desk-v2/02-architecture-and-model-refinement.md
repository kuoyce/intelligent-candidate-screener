# Architecture and model refinement

## 1. Model roles and recommendation

| Option | Benefit | Limit | Decision |
|---|---|---|---|
| Refine RoBERTa-SQuAD2 alone | Better domain spans, abstention and focused factual answers | Start/end predictions cannot compose a cross-document conversation | Keep as factual reader; insufficient alone for v2 outcome |
| Qwen with raw CVs | Flexible prose | Weak ownership/coverage control, large prompts, unsupported comparisons | Do not use as the primary design |
| Qwen over validated comparison evidence + RoBERTa reader | Conversation with explicit factual boundaries; independent model improvement | Needs state, relation contracts and evaluated rendering | Recommended |
| Application templates over the same evidence | Reliable direct comparisons without a generator | Less flexible phrasing and broad language understanding | Required baseline and operational fallback |

The [deepset model card](https://huggingface.co/deepset/roberta-base-squad2)
describes extractive QA trained with answerable and unanswerable SQuAD2 questions.
The [Transformers QA guide](https://huggingface.co/docs/transformers/tasks/question_answering)
uses source offsets and start/end training targets. Additional domain training
improves that task; it does not change its output into conversational generation.
Qwen's [0.5B instruction model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
describes a causal language model with structured-output instruction capabilities.
Those are reasons to evaluate it for composition, not evidence that our small
checkpoint already meets HR comparison quality requirements.

## 2. Existing runtime inventory, observed 24 October

| Component | Observed local manifest | What is unverified |
|---|---|---|
| Qwen | Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, unquantized, 4,096 context / 256 output token configuration | Current service availability, generation reliability and realistic end-to-end latency |
| RoBERTa | deepset/roberta-base-squad2, revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`; isolated runtime manifest records Torch 2.2.2 / Transformers 4.48.3 | Current installed-version parity, domain quality, threshold calibration |
| GLiNER | Existing isolated medium worker/cache remains the enrichment boundary | No claim that its confidence establishes a qualification |

Manifest existence is not a running service or proof of verified asset bytes.
Historical v1 records report a 0.5B CPU/four-thread service; re-inventory hardware,
available RAM, asset digests, runtime packages and endpoints before experiments.
The `qwen-rag` manifest at the inspected path is absent; this does not establish
whether a larger checkpoint exists elsewhere. Do not assume the 1.5B constant
in `qwen_assets.py` means that model is provisioned.

## 3. Turn pipeline

1. Reauthorize session, active JD and snapshot. Resolve explicit references using
   backend candidate IDs; validate any model-proposed resolution before use.
2. Update a typed dialogue frame: intent, ordered candidate set, criteria,
   focal candidate, pending slot/task, last answer and claim references. Explicit
   current references override history. A new JD/snapshot invalidates active scope.
   Support paraphrases such as “how do these two stack up?” without requiring
   the word “compare”. A schema-constrained Qwen intent suggestion can assist
   rule-based resolution, but the backend binds references and asks when ambiguous;
   generated IDs never grant scope. Reuse the frame for straightforward follow-ups.
3. Decompose a comparison into the same atomic criterion question for each CV
   and, where applicable, the JD. Resolve pronouns before QA, for example
   “their qualifications?” -> “What qualification and completion status does
   this candidate's CV state?” for each selected owner.
   Criteria are explicit JD phrases or reviewer-specified topics, not only the
   six v1 entity categories. Preserve terms such as “production deployment” and
   “AWS certification”; category labels are retrieval hints, not the whole question.
4. Retrieve complete sentence/section contexts with qualifiers. Reserve budget
   per owner and criterion, scan beyond document prefixes, retain conflicting
   passages, and expose truncation. Compare the existing lexical retriever with
   a section/alias-aware variant before considering new embeddings.
5. Use GLiNER as an index aid; invoke RoBERTa only for bounded factual questions.
   Preserve the full source context around each returned span. Unsupported
   extraction is unknown even when an entity was found.
6. Build comparison cells and permitted relations in application code. Retain
   negation, dates, units, completion and certification context. Uncertain polarity
   remains `unclear`; do not infer an affirmative qualification from a span.
7. Produce a concise deterministic answer, then optionally ask Qwen to choose
   emphasis/order and phrase supported relationships through the v2 contract.
8. Validate participant coverage, claims, relations and source versions. Recheck
   authorization/snapshot after inference, persist atomically with the existing
   turn ID/concurrency guards, and render only the validated representation.

## 4. Proposed contracts

Keep versioned JSON-compatible records in a proposed `app/conversation_schema.py`.

**DialogueFrame v2**: snapshot/JD identity, ordered `candidate_ids`, `intent`,
`criteria`, `focal_candidate_id`, `pending_task` with missing slot and original
question, `last_answer_id`, `last_claim_ids`, display-label mapping revision.
Store this independently of transcript text. Bounded validated previous turns
help phrasing; past prose is never used as CV evidence. Ordinals such as “second
one” resolve against the ordered active set, not a new saved rank.

**EvidenceCell**: owner/document/snapshot/source digest, criterion, original
question, normalized atomic question, extracted spans, complete redacted context,
evidence IDs/offsets, retrieval coverage, QA status, and `support_status`:
`explicit_positive`, `explicit_negative`, `qualified`, `conflicting`, `unclear`,
`not_found_in_retrieved_evidence`, `partial_source`, or `unavailable`.
These are source-claim states, not independently verified qualifications.
Positive/negative status is assigned only when a supported rule/evaluated semantic
step establishes the relationship; otherwise use `unclear` and show the context.
Phase 2 must implement this assignment explicitly: identify subject, qualification,
polarity and qualifier scope in the source context, with fixtures for unrelated
negation and conflicting sentences. Initial rules cover registered patterns only;
unhandled wording stays unclear rather than claiming general semantic understanding.

**ComparisonRelation**: criterion, participants, cell/claim IDs, relation type,
and derivation. Initial allowlist: `both_documented`, `explicit_contrast`,
`duration_greater`, `completion_status_differs`, `evidence_incomplete`,
`source_conflict`. Each has a dedicated application validator. Numeric ordering
requires comparable units, explicit skill linkage and nonconflicting source
claims; no adding overlapping employment or inferring tenure from proximity.
“Evidence incomplete” cannot become “candidate weaker”.

**AnswerPlan**: answer kind, requested/covered participants, selected cell and
relation IDs, ordered summary points, optional table rows, caveat codes, optional
clarification/action suggestion IDs, model/runtime/prompt provenance. Backend
rank facts have their own immutable slots. No arbitrary HTML or executable tools.

**PersistedAnswer v2**: validated visible text/blocks, claim-to-evidence map,
comparison/frame version, source/model/prompt digests, fallback reason and mode.
Keep legacy answers readable. Legacy context migrates by retaining known fields
and marking unknown discourse focus; never fabricate a previous comparison.

## 5. Safe conversational composition with Qwen

The existing `_model_claims` contract requires exact source text and returns only
selected excerpts. Create a new adapter/contract rather than relaxing that guard
for all historical calls. Preserve the old pathway for compatibility.

First v2 mode: Qwen returns an AnswerPlan containing supported cell/relation IDs,
emphasis and approved connective/caveat choices. The application realizes factual
clauses from validated slots and a small set of varied, reviewed sentence forms.
This enables direct comparative answers and continuity without pretending that
citation membership validates arbitrary prose. Templates must explain a relation,
not merely concatenate quotes as v1 does.

Example: the relation `duration_greater(candidate2, candidate1, Python, 5y, 3y)`
can realize “Candidate 2 reports more Python tenure: five years versus three.”
A `completion_status_differs` relation can contrast completed with in-progress
degrees without asserting an ordinal degree advantage or overall suitability.

Second, conditional experiment: permit short paraphrases, each attached to
specific cells/relations, only if the frozen evaluation shows useful gains.
Validate names, numbers/units, negation, modality, dates and owner alignment;
use an independently evaluated entailment check as additional evidence if useful.
It is not proof. Where support cannot be checked reliably, render the controlled
clause or full context. Never rely on Qwen judging its own answer as the sole guard.
Do not ship unrestricted free prose because JSON and citations pass validation.

Prompt separates role rules, question/frame, evidence and output schema. Treat
all CV text/history as untrusted content. Evidence includes full qualifiers,
not only naked spans. Use short request-local IDs mapped to canonical records.
At least one cell/status per participant must fit before composition; if not,
reduce criteria or offer a complete matrix instead of dropping people.

Start with the existing 0.5B and exact chat-template token counting. Proposed
experimental budget: 4,096 total tokens, 512 reserved output tokens, bounded
history, temperature 0 for repeatability, one schema repair within the same
end-to-end deadline. Existing 256 output tokens may truncate richer JSON; measure
before adopting the larger reservation. This is a proposed config, not a claimed
model limit. Omit hidden reasoning. Show no unvalidated factual token stream;
display progress and release validated blocks when complete.

## 6. Qwen refinement ladder

Q0: v1 extractive/table control. Q1: improved dialogue/evidence + application
comparison prose. Q2: Q1 + frozen 0.5B AnswerPlan composition. These isolate the
value of architecture from generation. Evaluate the conditional paraphrase mode
as a separate arm so extra fallbacks do not masquerade as model success.

If Q2 is safe but fails usefulness/style criteria, compare one [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
configuration, subject to asset/resource approval. Match prompt, evidence,
decoding and precision/backend where feasible; otherwise report a configuration
comparison, not a model-size causal claim. Record warm/cold latency and total
process-tree memory under realistic concurrent app/model use.

**Conditional LoRA/SFT:** use failure-labelled, reviewed synthetic conversations
to teach criterion switching, qualified contrasts, clarification, unsupported
requests and the AnswerPlan schema. Train only assistant target tokens with the
actual tokenizer/chat template; never teach hidden reasoning or use model ranks
as training targets. Include no-answer/partial cases and injection negatives.
Keep source facts in the input, not as knowledge to memorize.

Proposed single pilot: LoRA rank 8, alpha 16, dropout 0.05, learning rate 1e-4,
max sequence length 2,048, microbatch 1, accumulation 8, seed 0, at most 3 epochs;
select checkpoint by dev task completion with safety constraints. Exact target
module names and precision follow checkpoint inspection. Dev-selected settings
and any change to this recipe must be recorded before held-out evaluation.
Do not truncate away an owner's evidence to meet the sequence limit; reduce
the example scope explicitly. No parameter grid before one timed pilot.

LoRA is a resource strategy, not a guarantee of factuality. [PEFT documentation](https://huggingface.co/docs/peft/developer_guides/quantization)
describes adapters over quantized weights, including QLoRA. Do not assume the
documented GPU/bitsandbytes path works in this Mac/CPU environment. Freeze a
compatible backend/device/precision and pilot memory/time first; otherwise defer
training. Keep adapter and base digests separate; compare base/adapted models
with identical evidence, prompt and decoding. No automatic promotion.

## 7. RoBERTa refinement ladder

R0: pinned pretrained model and current threshold. R1: improved atomic questions,
retrieval and qualifier contexts with the same weights. R2: R1 plus dev-calibrated
no-answer threshold. R3: conditional span adaptation. Evaluate with Qwen disabled
so generation cannot hide reader errors; also report oracle-passage and actual
retrieval performance separately.

Build authored HR-domain span tasks for degree/completion, licenses/expiry,
technical skills, deployment, explicit skill duration, JD requirements, long CVs
and unanswerable queries. “Who is better?” is not a span training target; decompose
it into per-owner factual questions and compare in the application.

Use multiple acceptable exact spans where appropriate, full qualifier-bearing
contexts, and unanswerable negatives where the asked fact is not established.
Keep affirmative/negated pairs, near-miss skills, and total-versus-skill tenure
contrasts. Do not label an unobserved answer as missing because retrieval omitted
it; that is retrieval failure, not an unanswerable document.

Reuse audited parts of `qa_train.py`: train-family filtering, exact character
alignment and overflow coverage checks. Its current `training_examples` excludes
comparison tasks and mostly permits one answer; it needs an explicit versioned
atomic-question dataset adapter. Never feed unresolved follow-ups to a standalone
reader or silently round gold boundaries. Preserve SQuAD2 null targets and
document-level dev calibration; null-score margins are not calibrated probabilities.

Proposed initial adaptation preserves the existing 2e-5 learning rate, 384-token
length, stride 128, accumulation 8, seed 0 and at most 3 epochs. Audit boundary
and negative-window semantics before reuse; select a checkpoint/threshold on dev
only. Retain base weights, a separate adaptation manifest and a rollback switch.
Train only when oracle-passage failures justify it; improved conversational
phrasing is not a reason to retrain RoBERTa.
The current trainer saves the final epoch without dev selection. Implement an
explicit dev-evaluation/checkpoint-selection loop and separate threshold calibration
before claiming R3 follows this policy; the dataset adapter alone is insufficient.

## 8. Operational boundaries

Keep separate runtime processes and serialized resource budgets for Qwen,
RoBERTa and GLiNER. Prefer cached facts keyed by principal/workspace/source and
model/config digest; cache hits still reauthorize. Preserve queued/running/
partial/failed/unavailable states. Qwen failure must leave the complete comparison
matrix usable; adapted-reader failure can fall back to the pinned reader and then
labelled source contexts. No new private data in logs or committed model bytes.
