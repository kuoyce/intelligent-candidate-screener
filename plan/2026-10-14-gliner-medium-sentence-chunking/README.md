# GLiNER medium v2.5 with sentence-aware long-document extraction

Date: 14 October 2026. Status: user approved; implementation and validation underway.

## Requirements and authorization

Create a separate extractor using pretrained GLiNER medium v2.5. Segment long CVs
and JDs into sentences, then group sentences within the actual model token budget.
Approval authorizes implementation, delegated development/review, dependency and
model downloads, branch creation, plan commit, tests/checks, notebook execution,
and one complete campaign on the existing train/test fit parquet population.
Do not request further discretionary approvals for these approved actions.
Environment-enforced sandbox approval prompts can still be required and cannot
be overridden by this plan. No publication, merge or unrelated changes are included.

The verified published checkpoint is gliner-community/gliner_medium-v2.5, rather
than the requested unverified urchade/gliner_medium-v2.5. Approval explicitly
accepts this corrected identifier. This creates a new pretrained inference
configuration; no supervised fine-tuning or new trained weights are included.

## Current state and scope

Existing small-v2.1 extraction already uses overlapping tokenizer-budgeted word
windows, not arbitrary character splits. Preserve that baseline and its records.
Do not mutate identity-hashed shared inference files while its campaign runs;
wait for its terminal state before integration. Preserve unrelated local work.

The earlier conservative cleanup proposal remains separately unapproved. This
experiment reads original text unchanged: no stemming, lemmatisation, stopword
removal, lowercasing, whitespace rewriting or guessed word reconstruction.
CV input = resume_text/resume_id; JD input = job_description_text/jd_id.
Process each unique original (side, ID) once and attach entities to every pair.
Use the existing five label prompts, threshold 0.5, CPU and four threads.
Retain all documents, including failed, incomplete and completed-zero states.

## Model and runtime design

Add an explicit medium-v2.5 model profile and separate local model directory.
Resolve the repository to an immutable revision at acquisition; record license,
SHA-256 per required artifact, tokenizer/backbone revisions and runtime versions.
Download only the selected weight representation and necessary assets, not every
precision variant. Inspect configuration and checkpoint loading requirements first.
Verify that the checkpoint's supplied tokenizer is used; its model card specifies
load_tokenizer=True. Offline inference must succeed with complete local assets.

Test compatibility with the existing isolated pinned GLiNER runtime. If upgrades
are necessary, create a separately locked medium runtime rather than upgrading
the small baseline in place. Use uv exclusively and keep base tests independent
of optional neural dependencies. Pin spaCy in the medium runtime and use
spacy.blank("en") plus sentencizer; no statistical language model download.
Retain runtime diagnostics instead of hiding worker initialization errors.

## Sentence and chunk design specification

Add reusable extraction/sentence_chunking.py with a versioned strategy.
Use spaCy sentence spans and their original character offsets. Add deterministic
paragraph and bullet-list boundaries where punctuation-free resume content needs
them; preserve source whitespace and headings. Keep sentence detection separate
from the GLiNER word splitter and encoder tokenizer used for budget checks.
Regression fixtures cover initials, abbreviations, decimals, C++, C#, .NET,
Node.js, URLs, degree names and dates. Do not promise perfect linguistic boundaries.

Calculate the input ceiling from verified model/encoder/tokenizer limits and GLiNER
word limit; use a conservative 512-token cap until configuration and compatibility
tests establish a supported value. Reserve 16 tokens of encoder headroom, counting
the real prepared input including all entity prompts and special tokens. Reject
nonpositive effective budgets. Never count spaCy words as encoder tokens and
never rely on silent truncation to make an oversized chunk fit.

Greedily group consecutive complete sentences while the exact prepared chunk
fits both encoder and GLiNER word limits. Build chunks as original contiguous
substrings, not joined sentence strings. Carry one previous complete sentence
into the next chunk when it fits with new content; otherwise use up to 32 GLiNER
words of boundary context, reduced until it fits. Every iteration advances new
source content and cannot loop indefinitely on repeated overlap.

A single oversized sentence falls back to existing token-budgeted word-boundary
windows with up to 32-word overlap. Never slice inside a GLiNER word solely to fit.
Quarantine an individually unrepresentable word, identify its source interval,
and mark coverage incomplete. Record fallback and quarantine counts explicitly.
Sentence segmentation and overlap reduce boundary risk; entities longer than
the available context or incorrect sentence boundaries can still be split.

Validate every planned chunk using the actual inference preparation path before
submission. Convert chunk-local offsets directly to original document offsets;
enforce surface == source[start:end]. Compute context on original text. Deduplicate
only identical (field, start, end, label) predictions by maximum raw confidence;
retain other overlaps. Never infer entity accuracy from larger mention counts.

Coverage records expose planned/submitted source intervals, union coverage,
sentence counts, token budgets, fallback windows, quarantined spans, invalid
predictions and deadline interruptions. A complete state requires original source
coverage and no unexplained gaps. Keep the existing 120-second soft document budget
initially; an active model call is not preempted. Report incomplete documents
without silently increasing the budget or hiding them from denominators.

## Integration and artifacts

Thread model profile and chunk strategy through campaign CLI, local adapter and
proxy/worker protocol. Preserve old defaults and keep small/medium model choice
explicit; application default/research winner is unchanged by this experiment.
Campaign identity includes model, tokenizer, runtime, sentence/chunk configuration,
strategy code hashes, label prompts, threshold, source hashes and document budget.
No checkpoint sharing between profiles or strategies. Create profile-specific
latest pointers and explicit campaign-ID loading, with actionable messages when
a campaign has not finished. Never overwrite the old latest pointer accidentally.

Deliver a new thin notebook after existing notebooks 17/18 and aggregate records
under output/extraction/medium-sentence-comparison/. Source text, IDs, evidence and
weights remain under ignored data. Redact examples inside the notebook; five-row
and 2,000-character text caps apply. Show train/test and CV/JD separately.

Compare small-v2.1 word-window baseline with medium-v2.5 sentence-window results:
counts, category coverage, score distributions, completed-zero/incomplete/failed
rates, chunk/fallback counts, timings and peak RSS. This changes model and strategy
together, so it cannot isolate the benefit of sentence chunking. A controlled
ablation can be scoped separately; do not launch another full grid automatically.
Fit labels cannot provide entity precision/recall/F1; A1 label-conditioned summaries
carry A13. No final-set scoring, tuning, training or new annotation is authorized.

## Workflow and validation

1. On approval create a dedicated branch and commit the approved plan only.
2. Delegate model/runtime acquisition, sentence chunking, integration/reporting and
   independent review with separate edit ownership and staged dependencies.
3. Use synthetic tests for sentence packing, prompt-aware limits, whole-word
   fallback, overlap progress, exact offsets, complete source coverage, empty and
   whitespace inputs, multi-field CV/JD isolation, runtime parity, malformed model
   outputs, manifest mismatch and resume invalidation. Show guarded regression
   tests failing against the relevant old/disabled behavior.
4. Verify downloaded assets and offline medium loading; smoke-test short and long
   synthetic CV/JD inputs. Time a deterministic document-length pilot before the
   full population, reporting estimated time and peak RSS. If over an hour, state
   that estimate before launch; approval already authorizes this single campaign.
5. Launch only one heavy campaign at a time with saved log, exit marker and Darwin
   /usr/bin/time -l peak RSS. Use a tracked process; jobs over five minutes end the
   turn while running, with no foreground polling or duplicate launches.
6. Execute the reporting notebook, review evidence/PII/provenance, run full
   uv run pytest once with saved log and read it, run affected checks via the runner
   and required cold checks before marking complete. Register new --check commands.
7. Update README/data reconstruction instructions, model manifest documentation,
   implementation record and relevant catalog/cards for changed corpus claims.
   Preserve superseded figures. Report measured outcomes and any compatibility
   deviations. Do not claim improved accuracy without compatible entity gold.

## Acceptance criteria

Medium checkpoint loads offline from verified pinned assets. Sentence-aware chunks
fit actual prepared-input and word limits; oversized sentences use a tested fallback.
All returned evidence maps exactly to unchanged original CV/JD text. Coverage and
processing states remain auditable. Existing small extractor tests/defaults remain
valid. Campaign, notebook, review, suite and required checks finish with saved logs.
A complete report may contain explicitly disclosed incomplete/failed documents;
no blanket all-documents-success claim is permitted unless supported by counts.

## Persistent open questions

M1 resolved: user approved corrected gliner-community model identifier and full scope.
M2 implementation investigation: actual checkpoint context limits, tokenizer/backbone
assets and compatibility with GLiNER 0.2.21. Resolve from pinned artifacts and smoke
tests; document runtime changes rather than silently substituting another model.
M3 implementation investigation: measured CPU time/RAM and 120-second completion
rates. Publish pilot estimates and resulting states; do not tune on final labels.
M4 resolved scope: pretrained inference only, unchanged text, full existing fit
population; preprocessing cleanup/fine-tuning/application-default change excluded.

## Primary references

- Model card and tokenizer usage: https://huggingface.co/gliner-community/gliner_medium-v2.5
- Urchade's v2.5 collection: https://huggingface.co/collections/urchade/gliner-v25-66743e64ab975c859119d1eb
- spaCy rule-based sentence segmentation: https://spacy.io/api/sentencizer/
