# Grounded-answer design

## 1. Question and answer contracts

Support JD summaries, factual CV questions, lists of mentioned skills or
qualifications, comparisons against explicit JD requirements, supported strengths
and uncertainties, explanation of saved backend ordering, and bounded follow-ups.
Resolve scope to immutable JD/candidate/snapshot identities before retrieving.
Retain fast deterministic handling for common intents; ambiguous candidates or
criteria trigger one clear clarification. Avoid extra LLM calls merely to route
every question. Chat does not execute operational actions.

A successful comparison includes:

1. A direct, concise answer to the question.
2. A requirement-by-candidate matrix showing relevant evidence and uncertainty.
3. A short explanation of supported similarities/differences with citations.
4. Expandable original supporting excerpts, without forcing readers to assemble
   the comparison themselves.

Broad comparisons use the JD's explicit criteria and distinguish required from
preferred when the source does. Focused questions restrict the table to the
requested topic. Avoid generic 'good candidate' prose. Do not produce new ranks,
scores, hiring decisions or Fit calls. For 'why ranked higher', quote the saved
ordering and describe evidence differences; do not invent the ranker's causal
reasoning. Questions outside available evidence receive a useful limitation or
clarification rather than an unrelated extract.

## 2. Data flow

Authorized JD/CV source -> cached entity enrichment and passage index -> typed
requirement/evidence matrix -> one bounded Qwen synthesis request -> factual
validation -> answer/table/citations. The table can render before prose completes.

Keep the immutable ranking snapshot separate from a versioned QA evidence bundle.
Adding extraction must not silently change CSV ranks, snapshot scores, candidate
eligibility or the teammate's model. Conversation results record the QA engine
and evidence-bundle identities. Old messages remain historical, and generated
prose never becomes evidence for subsequent answers.

## 3. GLiNER medium enrichment

Reuse the existing offline `MediumRuntimeProxy` and pinned
`gliner-community/gliner_medium-v2.5` loader/runtime. Adapt it for application
work rather than importing Torch/GLiNER into the Streamlit environment.

Initial labels retain the configured inventory:

| Extraction prompt | Stored label | QA use |
|---|---|---|
| Technology/software tool/programming language | KNOWLEDGE | Locate technology evidence |
| Professional skill/competency | SKILL | Locate professional competencies |
| Job title/occupation | TITLE | Locate explicit role mentions |
| Academic degree/qualification | DEGREE | Locate education statements |
| Years of experience | EXPERIENCE_YEARS | Locate duration statements |

The existing adapter does not configure certifications, employers, employment
dates or skill-duration relations. Preserve text retrieval for these questions.
Adding labels later requires a versioned prompt/schema and separate validation;
do not silently turn technology mentions into certifications. No extraction-count
or confidence figure establishes entity accuracy without appropriate gold.

Extract from the exact authorized redacted text used by QA. Validate every span
as `text[start:end] == surface`, with document owner, digest, model revision,
label/prompt set, threshold, tokenizer/chunking versions and processing coverage.
Do not reuse offsets from original text on a redacted preview. Existing CSV
aggregate strings remain unverified mentions; they do not acquire span provenance
simply by passing through a new renderer. Existing campaign spans need matching
document/coordinate identity before reuse.

Entities are retrieval anchors, not standalone truth. Retrieve their surrounding
sentence/section, retaining negation, ongoing education, expired certificates,
requirements vs preferences and any temporal context. An extracted degree alone
cannot establish graduation; a skill mention does not establish proficiency.
Separate extraction confidence from assertion/evidence status. The legacy
`context_status='unqualified'` means no recognized qualifier, not an unqualified
candidate. An isolated 'five years' and 'Python' do not establish five years of
Python experience. Keep that link unknown unless source wording supports it.

## 4. Cache and worker lifecycle

Extract once per distinct source/configuration, not per question, JD/CV pair or
Streamlit rerun. Key the cache by authorization boundary, document/display digest,
text/redaction version, model/tokenizer digests, label prompts, threshold,
chunking and adapter versions. Validate authorization independently of cache hits.
Keep content in private git-ignored application storage, with bounded retention.

One managed medium worker uses a serialized queue/lock, explicit request IDs,
bounded startup/job waits and cleanup. The current stdin/stdout proxy is not a
ready concurrent service: protect request/response ownership and recover worker
failure/timeouts without assigning a late response to another document. Its
chunk deadline does not forcibly interrupt a single model call; enforce job
lifecycle limits at the worker boundary.

Prepare evidence asynchronously for existing CSV documents and new live-source
documents when their authorized source becomes available. Prioritize the selected
JD/CVs rather than launching a corpus-wide campaign. Show queued/running/complete/
partial/failed/unavailable states. Unavailable or incomplete extraction uses the
text path and never means zero qualifications. Serialize expensive extraction
and generation when resource profiling shows contention. Missing assets are an
unavailable state; do not download or train at startup.

## 5. Passage retrieval and requirement matrix

Reuse the tested source ownership, offset mapping and candidate-balanced
retrieval contracts in `rag_evidence.EvidenceIndex` where suitable. Extend the
primary portal through a service interface rather than reviving the legacy UI.
Start with lexical passage retrieval plus conservative curated aliases; add
GLiNER anchor boosts without excluding raw-text matches. Dense retrieval or a
separate passage reranker is conditional on measured remaining retrieval errors.

Use sentence/section windows with token-based budgets and overlap. Preserve
qualifying neighboring statements and explicit source coordinates. Replace the
two-short-excerpt limit with coverage per requested candidate and JD criterion.
Deduplicate overlapping evidence and reserve output tokens. Exhaustive lists
must disclose incomplete processing/retrieval rather than imply full coverage.

Each matrix cell carries a requirement ID, candidate ID, source-supported claim,
evidence IDs, qualifiers, processing coverage and status. Distinguish evidence
found, explicit contradictory/negative wording, not found in reviewed text and
unknown/incomplete. Missing extraction does not establish a skill gap. Do not
calculate a new fit score from the matrix. Normalize explicit numbers/units only
when source wording supports the relation; avoid inferred career arithmetic.

## 6. Grounded synthesis and validation

Qwen receives the question, bounded dialogue references, relevant JD criteria,
validated matrix cells and cited contextual passages. Ask for a small structured
answer plan with supported claim types and evidence IDs, plus concise explanatory
wording. Permit useful paraphrasing and comparison; do not retain the global
'return only IDs/no paraphrase' restriction.

Validate owner/snapshot/bundle identity, exact quoted spans, numerical facts,
requirement links, assertion polarity and qualification state. Rankings/scores
are copied from the backend. Every factual or comparative assertion needs
supporting cells/passages; prune or regenerate unsupported claims with a bounded
retry policy. Use a readable deterministic matrix/summary fallback when validation
fails, not an empty answer or an unexplained dump of quotes.

Citation existence alone does not prove semantic support. Structural checks must
be accompanied by typed relation/polarity checks and independent review of the
frozen evaluation outputs. Do not promise automatic proof of arbitrary generated
prose. Keep unrestricted reasoning experimental until it meets the agreed rubric.
Do not stream unvalidated factual tokens as verified evidence.

## 7. Generator/runtime pilot

Retain current Qwen2.5-0.5B-Instruct as a baseline. Test one
Qwen2.5-1.5B-Instruct candidate using a pinned GGUF/llama.cpp configuration suitable
for the current x86_64 CPU. Model-size improvement is a hypothesis. Quantization
may reduce memory/cost but needs quality and latency validation on this machine.
Do not presume that a larger model, larger context or more output tokens is free.

The existing local server hard-codes 4096 context/256 output tokens; client-only
changes are insufficient. Agree bounded synthesis budgets (initial candidate
512 output tokens), then update server caps, manifest/provenance, tokenizer
counting and client validation together. Increase only when measured coverage
needs it. Do not use the model card's maximum context as a default prompt budget.

Provide a backend adapter for exact token counts, model/template/artifact
identity, inference and health. llama.cpp's endpoints/metadata are not assumed
identical to the current custom `/tokenize` and attestation schema. Preserve
loopback-only deployment and existing secret/redaction boundaries. Load/test
one generator configuration at a time; avoid multiple full model copies per user.
