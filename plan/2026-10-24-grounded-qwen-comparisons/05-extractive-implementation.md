# Current v1 scope: GLiNER medium plus RoBERTa-SQuAD2

Authorized by the user's 'continue with the revised flow' instruction. This
supersedes generator synthesis, Qwen LoRA and larger-generator experiments in
earlier records. Implement on the existing branch; do not create a new branch.

## Design

1. Reuse the pinned isolated `runtime/gliner-medium` model to extract exact
   entity spans from authorized redacted JD/CV text. Cache once per source and
   model/label/chunking configuration. Retain coverage/qualifier provenance and
   explicit queued/running/complete/partial/failed/unavailable states. Extraction
   runs in a serialized managed background worker; UI reruns do not reload it.
2. Reuse `deepset/roberta-base-squad2`, the existing `ExtractiveQA`/`QaClient` and
   installed transformer runtime. Search only the active requested documents and
   relevant contextual passages; validate predicted text/start/end/owner before
   mapping to portal citations. Preserve no-answer decisions and surrounding
   negation, ongoing degrees, expired credentials and duration context. Do not
   invent answers from entities or other candidates' text.
3. Share deterministic requirement/category comparison between Compare candidates
   and comparison chat. Show the JD criterion and each candidate's supported
   answer/mention, context and citations. Missing evidence means not found or
   unavailable, never proved absence. No generated cross-candidate explanation.
4. Keep single-candidate fact questions, JD questions and bounded follow-ups in
   chat. Multi-candidate questions are extracted independently per document and
   assembled by the application. Qwen is not invoked by this primary path; its
   assets and historical turns remain intact and accurately labelled.
5. Preserve frozen CSV ranks/scores, the teammate's live-ranking adapter,
   ownership/revocation, 40-day auth, saved reviews, alerts and mail capture.

## Boundaries and fallback

GLiNER's configured labels remain KNOWLEDGE, SKILL, TITLE, DEGREE and
EXPERIENCE_YEARS. Certifications and other facts retain text passage retrieval.
Confidence is not proof; span slicing only verifies location, not interpretation.
Entity contexts cannot prove proficiency or skill-specific duration. Keep the
original-source/display-coordinate distinction explicit.

If a local model is missing/failed, explain availability and offer clearly
labelled relevant source excerpts, not a model answer or numerical zero.
Sentence/section evidence retains original offsets and qualifiers. Cached data
is private to the authorized owner/workspace; cache hits never grant access.
No downloads, training, new annotation or final-set tuning at startup.

## Implementation/verification

Use scoped sub-agents for the medium cache/worker adapter and extractive comparison
service while the parent integrates controller/UI/state and documentation.
Show new regression tests failing against previous behavior; use synthetic
documents and injected model clients for the normal test suite. Keep base tests
independent of data, Torch and Streamlit where appropriate.

Run focused tests, one full pytest suite and relevant checks through the runner.
Respect D55 if modifying model code, committing such code or completing a plan;
no cold model training or research scoring is implied by this inference wiring.
Do not mark the whole historical plan complete without its required checks.

Run a bounded real-model smoke on invented CV/JD text with exact-source citation
checks, time/RSS and explicit no-answer cases. Time a small workload first; jobs
estimated above five minutes run once in the background and end the turn. Cache
proof must show repeat questions require no medium inference. Test actual UI
comparison and fact chat, stale job switching, unauthorized cache access, partial
coverage, and unchanged CSV results. Retain logs rather than rerun to view output.

The linked Hugging Face guide's start/end span formulation applies to RoBERTa
here; no Qwen causal supervised fine-tuning remains authorized in this scope.
