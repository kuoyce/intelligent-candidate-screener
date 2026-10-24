# Interim implementation and development evaluation

18 October 2026. Status: implementation in progress; no held-out test scoring.

## Frozen authored synthetic evaluation

Package code `app/rag_evaluation.py` authors 48 independent document families and
240 linked question turns. The frozen split is 24 families/120 turns train,
12/60 development and 12/60 test. Each family contains its own JD, four applicant
CVs, backend ranking fixtures, five turns and exact original-source oracle spans.
All eight primary task types occur in every split; filter previews are included.
Credentials include active, negated, expired and incomplete-training statements;
explicit tool tenure is distinguished from total employment. Long-document facts
occur after 65 work-log lines. An untrusted instruction attempts to change ranks.
These documents and labels are authored synthetic examples, never real applicant
data or model-generated gold. Repeated templated question structures limit the
external validity of this small feasibility corpus.

Frozen specification: `output/chatbot-rag/evaluation-spec.json`.
Corpus SHA-256: `92bfe2a63ca6fc6a397b3edc9897d39a494c409918d9f6d1a303866393930ee9`.
The specification records family/scenario/oracle digests, split membership,
metric definitions, denominators, acceptance targets and the held-out test gate.
Exact source boundaries and ownership are validated before scoring. Retrieval
Recall@5 measures whether an actual retrieved same-owner/document passage wholly
contains each oracle span, rather than demanding identical passage boundaries.
Citation ownership separately requires the current snapshot and exact quote slice.
Typed claim support requires matching source text, relation and supporting citation;
a citation ID alone does not prove a claim. Undefined metrics remain null.

## Frozen development measurements

Both runs below use the same 12 families/60 development turns. Each family is one
JD scenario. All CIs are 95% percentile intervals from 2,000 complete-family
bootstrap samples, seed 0. Paired differences use the same sampled families.
The backend order is an authored fixture, not an evaluation of a matching model.

| Measure and denominator | B0 legacy model-free | B1 lexical RAG model-free |
|---|---|---|
| Task completion, 60 turns | 7/60, 11.7% [6.7%,16.7%] | 14/60, 23.3% [16.7%,30.0%] |
| Route accuracy, 60 turns | 16/60, 26.7% [13.3%,40.0%] | 59/60, 98.3% [95.0%,100%] |
| Candidate slot accuracy, 60 turns | 16/60, 26.7% [13.3%,40.0%] | 59/60, 98.3% [95.0%,100%] |
| Evidence Recall@5, 77 oracle spans | 0/77, 0% [0%,0%] | 76/77, 98.7% [95.8%,100%] |
| Citation ownership | 18/18, 100% [100%,100%] | 123/123, 100% [100%,100%] |
| Citation relevance to authored oracle | 0/18, 0% [0%,0%] | 53/123, 43.1% [35.3%,52.1%] |
| Exact factual answers, 15 turns | 0/15, 0% [0%,0%] | 0/15, 0% [0%,0%] |

Paired task-completion delta: **+11.7 percentage points [3.3,20.0]**, n=12 paired
families/60 turns. Records: `B0-dev.json`, `B1-dev.json` and `B1-vs-B0-dev.json`
under `output/chatbot-rag/`. B1 is frozen; subsequent QA/router changes receive a
new configuration ID/record instead of replacing these measurements.

B1 development failure categories overlap: 45 turns lacked exact factual/list
answers, one failed routing/candidate resolution, one lacked complete reference
evidence and one failed no-answer behavior. Its factual service returned source
excerpts without an exact answer interface. The evaluator did not mine reference
answers from those excerpts or grant it oracle correctness. Source retrieval
improved, while the provisional 80% task-completion target remains unmet. There
are no emitted typed synthesis claims in these model-free runs, so the 90%
atomic-claim support target cannot be assessed; it is not treated as a pass.
Model-free answer-call latencies and clustered p50/p95 intervals are in each
record; snapshot construction/index preparation occurs outside those timed calls.
These timings do not characterize external model latency or peak memory.

## Verification and remaining work

Nine focused synthetic evaluation tests initially passed. Named invariants cover
family leakage, late facts, explicit fourth-candidate follow-ups, source offsets,
wrong citation owners, stale snapshots, fabricated claim text, incorrect relation,
undefined metrics, family-clustered uncertainty, paired denominators and test gate.
Full-suite and integrated QA verification remain pending with the implementation.

Next: expose actual validated extractive answers through the service, benchmark
that separately as B2 on development, and measure provisioned external baselines
before conditional adaptation. No B2 result or accuracy improvement is assumed.
No held-out test has been scored. Final configuration selection, hard invariants,
resource measurements, application review and required checks remain outstanding.

## Metadata-only provenance amendment (18 October 2026)

The initial family metadata incorrectly called these fixtures human-authored.
They are **agent-authored deterministic synthetic fixtures**, with no annotator
model dispatch and no human labelling claim. This literal and corpus version were
corrected explicitly before further development measurements. Original specification
`evaluation-spec-v1.json` and all B0/B1 measurements remain unchanged. Their v1
provenance metadata is superseded; their experimental measurements remain valid.

Current v2 specification SHA-256:
`b43ab125d19b82e836560ca5a444a54db41e8559d9aad5a1374809dea95a9231`.
The amendment `output/chatbot-rag/metadata-amendment-v2.json` records both hashes
and verified unchanged document/question/oracle/split content hash
`f09e4ad40e2ec50a5e6bce559e98c56f4cb37d73922b815c60a1c8b23aa7c901`.
No question, source text, answer boundary, relation, backend fact or split changed.
New B2 measurements use v2 metadata. Cross-version paired comparisons require
explicitly passing this amendment; they otherwise refuse different corpus hashes.

## B2 native extractive baseline, development only

B2 ran all 60 development turns across the same 12 JD/document families with one
native extractive-QA client worker. It uses current v2 authorship metadata; the
B1 comparison explicitly supplies the recorded metadata-only bridge. No held-out
test was scored and no B0/B1 measurement was replaced. Results are recorded in
`output/chatbot-rag/B2-dev.json`; paired comparisons are in `B2-vs-B1-dev.json`.
All intervals below use the same 2,000-resample 95% family-bootstrap procedure.

| Measure and denominator | B2 native QA |
|---|---|
| Task completion, 60 turns | 15/60,25.0% [18.3%,31.7%] |
| Route accuracy, 60 turns | 60/60,100% [100%,100%] |
| Candidate slot accuracy, 60 turns | 59/60,98.3% [95.0%,100%] |
| Evidence Recall@5, 77 oracle spans | 70/77,90.9% [81.3%,97.6%] |
| Citation ownership, 115 citations | 115/115,100% [100%,100%] |
| Citation relevance to authored oracle | 47/115,40.9% [31.5%,50.9%] |
| Exact factual answers, 15 turns | 0/15,0% [0%,0%] |
| Factual token F1, 15 turns | 0.360 [0.314,0.400] |
| List recall, 28 reference items | 0/28,0% [0%,0%] |
| No-answer recall, one impossible turn | 1/1,100% [100%,100%] |
| No-answer precision, 16 abstentions | 1/16,6.25% [0%,18.2%] |

B2−B1 paired completion delta is +1.7 percentage points [0.0,5.0], n=12 families/
60 turns. Evidence Recall@5 changed by −7.8 points [−15.2,−2.3], n=77 reference
spans clustered in 12 families. Token F1 changed by +0.360 [0.314,0.400], n=15
factual turns across those families. Neither the task-completion target nor a
validated synthesis-claim support target is met. No typed synthesis claims were
emitted, so claim support remains undefined.

The frozen strict tenure target includes its contextual phrase (“N years using
TOOL”). Minimal answers such as a number or “N years” would receive EM0 while
retaining token overlap. Exact-match failure must therefore be inspected alongside
source-grounding and boundary errors; it is not alone proof that every answer is
factually useless. References are not expanded after observing development outputs.

Static diagnosis of the seven evidence misses, without modifying oracles:

- Six misses are explicit certification follow-ups: HR24:t3,HR26:t3,HR28:t3,
  HR28:t4,HR32:t3,HR34:t3. The current inheritance path replaces any “what about”
  question with the previous requirement, including questions that explicitly ask
  a new certification-currentness fact. This can retrieve prior tool-tenure or
  list context instead of the requested expired credential. Inheritance should
  apply to underspecified follow-ups, preserving explicit factual constraints.
- HR28:t2 is the filter preview. Prefix stripping recognizes “candidates” but not
  “applicants”; the trailing “preview the evidence filter” action remains in the
  searched requirement phrase. Ordered-phrase evidence checks then correctly
  refuse a phrase absent from the CV. Parse command syntax separately from the
  requested qualification, preserving ordered-phrase and negation safeguards.

These are development implementation findings, not grounds to alter evaluation
labels. Parent implementation owns fixes and any subsequent separately identified
configuration. Conditional extractor adaptation remains an exploratory next step;
it cannot substitute for fixing omitted passages or command interpretation.

## Integration and training pilot

Approved plan branch `feat/hr-conversational-rag-2026-10-18`; plan commit
45ac274. The Streamlit application now uses a session-owned RagService and source
index, with stable candidate IDs, explicit filter preview/apply, source passages,
requirement evidence table and typed follow-up state. Full source documents stay
private to the screening snapshot. Changing inputs closes the QA worker and
invalidates private indexes/history. Factual spans are validated against the exact
redacted passage and owner before display. Original and display coordinates remain
separate. The legacy entities/source commands remain available.

Resolved the development findings above: explicit new factual follow-ups override
old question slots; only underspecified “what about ID?” inherits a prior question.
Filter command parsing supports applicants/candidates and separates the preview
action. Date/rank confusion, short-ID/article confusion and unrelated qualification
matching each have fail-before regression evidence. Backend facts are returned in
a structured record independently of conversational presentation.

Pretrained assets: deepset/roberta-base-squad2 revision
adc3b06f79f797d1c575d5479d6f5efe54a9e3b4; immutable per-file inventory under ignored
data/processed/chatbot/qa/extractive/. Worker is offline, CPU four threads, actual
fast-tokenizer overflow contexts with no-answer handling. It does not train or
download at startup. Source mentions are not asserted as current qualifications.

The fixed adaptation pilot passed: 45 compatible questions across 24 train-only
families, 137 windows; eight microbatches/one AdamW step took33.370seconds,
forecasting1801.984seconds (about30minutes) for three epochs. Source example SHA
a2367079034d511045130d79a40990698368bbd8272c8544bacd81ad2092b376.
List/rank/compare/filter/ambiguous tasks are explicitly excluded from single-span
training, with counts in logs/hr-qa-training-pilot.log. Exact source boundaries are
required; there is no rounding or dropping of incompatible answer spans.

Validation to date:55 focused tests passed before final list integration. The
first full suite had1167passed/10skipped and3failures: two known unrelated frozen
baseline failures plus a legacy entity-display regression. The entity command was
restored and its focused test passes. Subsequent changes require final focused and
broad validation before completion. Scoped runner cold selection found no existing
registered research record imports this isolated app/QA implementation; native
pilot/benchmark logs are its direct cold evidence. Unrelated dirty output/docs
would select all research checks, so this selection uses the scoped Python import
closure explicitly (logs/hr-rag-cold-selection.log).

Local review server: http://127.0.0.1:8502, health endpoint returnedok. This is a
prototype; quality targets remain unmet. Training result, adapted-versus-pretrained
evaluation with identical postprocessing, generator-size benchmark, final model
selection, any eligible held-out evaluation, and final review/checks remain open.

## Pre-training integration review

Qualification lists now use exact sentence/section contexts with source-local
offset remapping, deduplication and a shared32-context cap. Hitting the cap is
explicitly incomplete; no list is presented as exhaustive. The final focused
integration run passed69tests (logs/hr-rag-pretraining-focused.log).

The distinct B2-routing-list-v2 development record preserves previous runs:
15/60task completion (25%,95%CI18.3–31.7%),76/77evidence recall
(98.7%,95%CI95.8–100%),0/15strict factual EM, mean factual token F1.360
(95%CI.314–.400), and0/28list recall. Route/slot parsing fixes repaired the prior
retrieval regressions. QA quality targets remain unmet. This is an interim
diagnostic, not the final frozen pretrained-versus-adapted comparison; final
comparison must use the same recorded code, model artifacts and postprocessing.

The current evidence index is lexical, held only in the Streamlit session.
No dense passage embeddings or persistent vector database have been introduced.
Dense retrieval and the larger generator remain comparison work, not validated
improvements. Qwen asset tooling can provision the proposed1.5B baseline into a
separate directory and rejects overwriting a different existing model/revision.
That model has not been downloaded/benchmarked in this phase.

The application defaults to the pretrained extractive checkpoint. An experimental
checkpoint may be selected explicitly with SCREENER_QA_MANIFEST after validation;
training does not activate it automatically. The original reranker weights were
still absent under data/processed/rerank at inspection; only its batch parquet
was present. Local prototype uses the explicitly labelled illustration unless a
verified reranker is configured. No Fit threshold is inferred from ordering logits.

Final pre-training suite:1183passed,10skipped, two known unrelated frozen baseline
failures (logs/hr-rag-final-full-pytest.log),199.88seconds. The entity-display
regression is fixed. Focused source/model resource review:24passed,1optional skip
(logs/hr-rag-resource-review.log). There is no claim the full suite is green.

The approved fixed three-epoch adaptation is being launched once with its own
log/exit marker/Darwin resource measurement at logs/hr-qa-training.log, targeting
the new ignored directory data/processed/chatbot/qa/adapted/v1. Expected training
time is about30minutes from the pilot. It must finish and pass saved-model reload
and development comparison before any selection or activation. The Streamlit
prototype continues using the pretrained baseline. Plan remains in progress;
there is no adapted result or held-out score yet.
