# GLiNER text cleanup with original-source evidence

Status: proposed; awaiting user approval before implementation.
Date: 14 October 2026.

## Requirement and findings

Improve entity extraction from CV resume_text and JD job_description_text by
cleaning text while preserving complete words. Repository inspection found no
stemming in extraction/transformer.py, extraction/fit_run.py or the data-layer
source paths searched. The current adapter submits original field substrings.
Removing an existing stemmer is therefore not an established fix. Already
stemmed upstream source text cannot be reliably reconstructed by guessing words.

The approved full-fit campaign is running and its identity includes code hashes.
Finish that frozen baseline before changing shared inference code. Preserve all
unrelated working-tree changes and existing source parquet files and checkpoints.

## Scope and frozen experiments

P0: current original-text extraction, retained as baseline.
P1: conservative text cleanup, with unchanged checkpoint, five label prompts,
threshold 0.5, overlap, CPU/thread settings and document budget.
Do not fine-tune, change prompts/thresholds, stem, lemmatise, remove stopwords,
lowercase, expand abbreviations or infer missing words in this experiment.
Section-aware chunking is a separate later experiment to avoid confounding P1.

## Design specification

Add reusable extraction/preprocessing.py. Return prepared text, versioned policy,
edit counts and an ordered mapping from prepared characters to original source
intervals. Identity preprocessing remains the default for existing callers.

P1 changes only confirmed formatting artifacts: CRLF/CR to LF; nonbreaking spaces
to ordinary spaces; runs of horizontal spaces/tabs to one space; remove a leading
BOM. Preserve line/paragraph boundaries, case, digits, punctuation and ordinary
hyphens. Preserve C++, C#, .NET, Node.js, CI/CD, version numbers, degree names,
negation and experience expressions. Do not strip HTML, repeated headers,
zero-width characters or join line-broken words automatically: these can alter
content and require a separately specified policy after artifact inspection.

Map every prediction from prepared offsets to its original source interval.
Validate bounds and mapping order. Store surface = original[start:end], never
the cleaned prediction string; record prepared surface separately only in local
audit data. Resolve context using original text. Keep original field strings and
original text_version; add preprocessing provenance without violating entity
schema. For spans containing collapsed whitespace, the original surface can
differ from the prepared surface and must still pass exact-source validation.

Chunk prepared text under existing actual tokenizer/prompt limits. Translate
submitted and quarantined coverage intervals back to original coordinates.
Track deliberately removed characters separately from unprocessed characters;
never claim source submission for removed text. Coverage completeness must require
all retained source content processed and every omission explained by the policy.
Reject malformed mappings; do not silently mark mapping errors as complete.

Integrate through the local adapter and isolated runtime proxy/worker consistently.
Thread an explicit policy through campaign configuration. Include policy, mapping
version and implementation digests in campaign identity; P0/P1 must have distinct
folders and cannot resume each other's checkpoints. Keep CV/JD separation and
deduplicate documents by original side/ID, not prepared text.

## Evaluation and decision criteria

First inventory formatting artifacts without displaying real document bodies.
Use existing compatible entity gold only after auditing labels, offsets, licensing,
splits and contamination exclusions. Predeclare eligible categories and exact-span
scoring in an implementation record before running comparisons. Split by original
document before chunking. No A2 final labels, eval-region tuning or new annotation.
Partial gold must not turn unannotated categories into negative examples.

For eligible gold, report per-label precision/recall/F1, document count and paired
document-bootstrap 95% CIs for P1 minus P0; report CV/JD separately where supported.
These are extraction metrics, not JD-query ranking metrics. Any matching evaluation
would require a separate approved protocol with query-level CIs and both strata.

Fit-pair reports measure mention distributions, zero-document rates, processing
states, coverage, runtime and peak RSS on identical original populations. They
cannot establish entity accuracy; A1-conditioned figures carry the A13 limitation.
Keep missing, failed, incomplete and completed-zero documents distinct.

P1 is experimental by default. All source-offset/coverage invariants must pass.
Promote only if compatible gold exists and the paired 95% CI for the predeclared
micro-F1 delta is entirely above zero; publish all category deltas and resource
costs. Without suitable gold, report operational changes and leave P0 the default.

## Implementation and verification sequence

1. Obtain plan approval, create a dedicated branch and commit this plan alone.
2. After the current campaign exits, delegate preprocessing/mapping implementation,
   integration/reporting and independent review with separate edit ownership.
3. Test synthetic invariants: identity policy; whole words and technical punctuation;
   CRLF/NBSP/tab cleanup; exact original surfaces after whitespace contraction;
   first/last/empty spans; overlap deduplication; coverage around removals;
   invalid mappings; proxy parity; resume rejection across policy identities.
   Show each guarded regression failing with its relevant guard disabled.
4. Time one deterministic P1 pilot and estimate the full campaign before launch.
   Jobs over five minutes run once with saved logs, exit marker and Darwin time -l
   peak RSS; end the turn while running. No simultaneous heavy jobs.
5. Add a thin reporting notebook after notebook 18; reports come from package code.
   Redact inside the notebook; cap text output at five rows/2,000 characters.
6. Review independently; run focused tests during iteration, full uv run pytest
   once with a saved log, then affected checks through the runner. Run required
   cold checks before marking complete. Register any added --check entry point.
7. Update README and implementation record. If corpus-content claims change,
   update data catalog and relevant cards in the same commit with superseded
   figures retained. Do not overwrite the frozen campaign or previous records.

## Persistent open questions

Q1: Approval of this conservative P1 policy and experiment scope is pending.
Q2: Does the user's reference to stemming concern upstream parquet content or
the desired prevention of future stemming? No local stemmer was found in the
inspected inference path. Upstream damage detection is an audit, not word recovery.
Q3: Which existing entity-gold populations support compatible categories and CV/JD
coverage? Resolve by a read-only inventory before claiming accuracy improvement.
Q4: Header/HTML/dehyphenation cleanup is deferred until actual artifacts and safe
rules are established; it is not silently included in P1.
