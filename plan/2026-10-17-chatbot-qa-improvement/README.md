# Improve recruiter question answering

17 October 2026. Status: proposed for review. User requested training a QA model
or planning improvements to the Streamlit-backed chatbot. This document takes the
planning option; no training, downloads, new collection or deployment started.

## Existing design and measured gaps

The application uses local Qwen2.5-0.5B-Instruct, bounded intent interpretation,
immutable screening snapshots, validated evidence references, canonical rendering
and deterministic fallback. Prior validation used 34 authored question fixtures
and a small live smoke; that is not a broad recruiter QA quality estimate.
See the 12 October chatbot plan and its implementation record.

Code inspection: chat_context.build_context selects at most ten candidates and the
first twenty evidence items per candidate. messages_with_budget rejects oversized
contexts. The model cannot recover evidence excluded by this selection. Existing
follow-up handling resolves some references using prior candidate IDs; ambiguous
comparisons and missing evidence need explicit evaluation. The selected R2 weights
were missing in the previous record; recheck availability before claiming real
reranking or explaining its scores. Keep missing classification explicit.

## Proposed outcome and contracts

Answer recruiter questions about supplied CVs and the JD with relevant, clickable
source evidence, accurate candidate references and clear unavailable states.
Retain backend ownership of ranks, scores and Fit calls. Missing evidence means
not found in the supplied material, rather than proof a candidate lacks a skill.
Keep execution local and session-only. Redact documents before model exposure and
display. GLiNER suggestions remain predictions; human-confirmed spans are distinct.
Neither the ten entity-labelled documents nor their span labels constitute QA
instruction/answer training data.

## Implementation sequence and design

1. Freeze an evaluation specification before changes. Inventory existing fixtures;
   create synthetic document families covering ranking, comparison, requirements,
   skills, years, degrees, negation, follow-ups, unknown facts, long documents,
   ambiguous IDs, stale snapshots and instructions embedded in CVs. Partition by
   document family into development and held-out tests. Freeze held-out questions
   and expected reference sets before tuning. No real human annotation batch or
   LLM annotation dispatch is opened by this plan.
2. Record the current baseline: intent/argument correctness, candidate resolution,
   evidence recall, exact citation ownership/offset validity, correct abstention,
   fallback rate, end-to-end success and p50/p95 latency. Report denominators and
   uncertainty resampled over document families, with paired comparisons. Keep
   generated synthetic fixture performance separate from recruiter usability claims.
3. Add question-aware local evidence retrieval. Index sentence/paragraph spans of
   redacted session documents with stable document IDs, source coordinates and
   snapshot IDs. Start with existing lexical tooling; rank spans by question and
   JD requirement, reserve coverage for each requested candidate, deduplicate and
   pack using the provider's actual tokenizer budget. Replace positional first-20
   selection with relevance selection. If no existing document access contract
   supplies a required passage, extend the snapshot evidence interface explicitly.
   Never fabricate a quote or attach redacted offsets to original text.
4. Strengthen structured requests. Resolve candidate references against the current
   snapshot, support two-candidate comparisons and follow-up requirements, ask a
   concise clarification for unresolved ambiguity. Retrieve additional passages
   through bounded application calls. Keep the validated fact/reference output
   schema and canonical renderer; model-authored prose cannot introduce facts.
5. Improve Streamlit review: expandable source excerpts, visible candidate/JD
   ownership, previous-question suggestions, explicit local-model availability and
   distinguishable retrieval/rerank/classification statuses. Cache passage indexes
   per immutable snapshot, invalidate on source/model changes, prevent cross-session
   cache reuse. Loading a model or UI must never trigger training.
6. Evaluate on development fixtures after each substantive change. Compare the
   current implementation, retrieval improvement, and prompt/intent improvement
   separately. Freeze the chosen configuration before one held-out evaluation.
   Require all citation/ownership/stale-snapshot/privacy invariants to pass and no
   held-out end-to-end regression; report latency tradeoffs, no invented SLA.
7. Optional fine-tuning gate: only if development failures remain primarily intent
   or structured-reference errors after retrieval fixes, prepare a separate proposal
   for supervised adaptation of the existing local Qwen model. Targets are validated
   intent JSON and evidence-ID selections, never generated ranks/Fit labels. Use
   authored synthetic training examples with document-family disjoint validation;
   avoid final-set files, human entity gold and model-generated labels. Establish
   class coverage and a learning curve before choosing training-set size. Time one
   configuration first; freeze hyperparameters and compare to the same untrained
   baseline. Pin weights/tokenizer/runtime/digests; save adapters separately. Do not
   train concurrently with active entity inference or start an unbounded grid.

## Work packages and verification

After approval create a dedicated branch and commit this plan, preserving existing
dirty work. Delegate evaluation, retrieval/context and UI/state tasks with explicit
file ownership. Reusable modules live in app/ or llm/; Streamlit stays a transport.
Tests use synthetic data and fake providers; optional local-model smoke runs offline.
Demonstrate each regression fails before its fix. Run focused tests, then the full
suite once with logs, and registered affected checks through the runner (cold when
model/training code changes). Preserve the known unrelated baseline failures.
Review source citations, token budgeting, concurrent session isolation, malicious
document content, missing model behavior and stale references before review.
Update the chatbot implementation record, README and decisions as needed; report
actual measurements and unresolved limitations without changing research adoption.

## Open questions

- Q1: Primary failure cases from the user are not yet supplied. Start with the
  existing recruiter-only intents and long-document evidence coverage; add concrete
  examples when supplied without changing the frozen held-out suite.
- Q2: Target hardware and latency SLA are unspecified. Measure on the existing local
  CPU environment; do not select/download a larger model without a resource plan.
- Q3: Real R2 artifact availability must be checked at implementation time.
- Q4: Training is conditional and requires its concrete data/resource proposal;
  approval of retrieval/UI work does not silently authorize a new annotation batch.
