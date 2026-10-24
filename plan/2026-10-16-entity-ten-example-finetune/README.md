# Learn from ten human-labelled documents and assist training annotation

Date:16October2026. Status: user approved; implementation underway. Real training
blocked by40unalignable human span boundaries; no labels dropped or changed.

## Verified starting point and scope

Campaign f5fadd3fda6068818192a3a8, annotator1: ten fully reviewed training documents,
4CVs/6JDs,130spans:80KNOWLEDGE,30SKILL,11DEGREE,9TITLE. No empty reviewed docs.
Inventory log logs/entity-ten-review-inventory.log. Do not require two extra
documents just to attain the earlier suggested5/5 balance. This is single-annotator
training data, not independently adjudicated entity test gold.

Approval covers a separate GLiNER fine-tuning experiment, timed CPU training,
suggestion generation for the remaining training documents, local UI integration,
tests and records. Preserve the frozen medium/small checkpoints and human reviews.
No new corpus, label categories, annotation batch or LLM dispatch is introduced.
Dev/test and annotator2's independent overlap stay blind to all suggestions.

## Freeze and validate the training snapshot

Export the latest completely reviewed annotator1 revisions for exactly these ten
documents into a new immutable ignored-data snapshot. Freeze campaign/guide/source
hashes, annotation revision IDs, categories and export digests. Fail if any selected
document is not entity train, if labels change, or source/spans fail validation.
Future human corrections are a new snapshot and model run, not a silent resume.

Align character gold to the pinned medium model's GLiNER words_splitter. Convert
to inclusive word-span training indices expected by GLiNER0.2.21. Print aggregate
unalignable-span counts only; do not round or silently discard labels. If any
cannot be represented, report that blocker with redacted bounded examples before
training; preserve the originals until humans resolve the guideline/alignment.

Sentence-aware training windows use original text and actual prepared prompt token
budgets. Never cut a labelled span across windows. Keep every source region and
negative window from exhaustively reviewed documents, but count each supervised
positive span once; mask partially visible/duplicate positive targets rather than
turning them into negatives. Reject impossible gold spans beyond model span width
or input budget explicitly. All chunks remain in their document's train split.

## First training experiment

Base: pinned gliner-community/gliner_medium-v2.5 revision
88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da and supplied tokenizer, four selected
descriptive label prompts only. No experience label supervision or Fit labels.
Keep preprocessing unchanged and no stemming. CPU four threads, float32, seed0.

Use the installed GLiNER0.2.21 collator and model forward/loss through a small
versioned PyTorch loop. The installed release lacks the latest train_model API;
do not assume examples for newer releases work. Verify collator/loss/save/reload
with synthetic data before consuming human labels; keep neural code in an isolated
worker and base tests synthetic/optional. No silent runtime dependency upgrades.

First run adapts the non-backbone entity representations with backbone frozen,
batch1, gradient accumulation4, AdamW learning rate1e-4, weight decay0.01,
max gradient norm1.0,3fixed epochs, no hyperparameter grid and no early stopping.
Record named trainable/frozen parameters, window count and effective steps.
Smoke-time one representative optimization step before the run and estimate total
runtime/RAM. If >1hour, state estimate before launch. If >5minutes, one tracked
background job writes logs/exit marker and Darwin time-l peak RSS; end the turn.

No development gold is currently available for selection. This fixed configuration
is an exploratory annotation-assistance pilot, not a validated improvement. Training
loss and scores on the ten training documents cannot establish accuracy. If the
frozen-head experiment fails or needs different hyperparameters, record findings
before proposing another run; do not silently sweep configurations.

Store a separate model directory with base provenance, training snapshot/runtime/
code/config hashes, trainable weights and reload checks. The loader must verify
the fine-tuned artifact manifest separately from medium_assets' fixed base hash
policy. Never disguise changed weights as the original pretrained checkpoint.

## Suggestion generation and UI behavior

Freeze the trained checkpoint before generating suggestions. Run only on the70
remaining entity-train documents, using existing sentence-aware extraction and
threshold0.5 with the four prompts; no dev/test inference is part of this pilot.
Skip any document already fully reviewed when publishing/loading suggestions.
Source/campaign/model/snapshot identity and complete/incomplete states are explicit.

Persist suggestions under ignored data separately from every annotator's reviews.
Map source offsets back to the frozen redacted display; reject masked/unalignable
suggestions rather than expose identifiers. Handle overlapping predictions by a
deterministic declared display policy (descending score, then boundaries/label),
retain all raw suggestions locally and report excluded overlaps. Do not pretend
the model's scores are calibrated confidence or present suggestions as gold.

Annotator1 can explicitly load a clearly labelled suggestion set on unreviewed
training documents, then add missing entities, delete errors, relabel and adjust
boundaries. Whole-document human confirmation remains mandatory. Record assistance
provenance on human-reviewed revisions so assisted data can never be mistaken for
blind gold. Starting/loading a page never auto-accepts or saves model predictions.
Existing saved drafts/reviews must not be overwritten. Suggestions unavailable
on dev/test and for annotator2's independent overlap, enforced server-side.

Keep existing document order/progress/assignments and all human labels intact.
UI startup loads verified artifacts only; never trains or downloads weights.
Allow ordinary blank annotation when a model or suggestion record is unavailable.

## Validation and reporting

Create dedicated branch and commit approved plan before implementation. Delegate
snapshot/alignment/training and suggestions/UI work with independent review.
Test document-disjoint training, exact tokenizer alignment, no split positives,
save/reload identity, backbone freeze and changed-snapshot resume refusal.
Test no dev/test or second-annotator suggestion exposure, no model-to-gold auto-save,
redaction/UTF16 mapping, draft/review preservation and assistance provenance.
Demonstrate relevant guard-disabled regressions failing. Browser-test synthetic
assisted editing and blinded views. Run full pytest once with logs and the runner's
relevant cold checks before completion; do not retrain unrelated records because
the runner broadly selects UI HTML. Preserve known frozen TF-IDF baseline failures.

Update README, A1 card, data catalog, implementation record and D56's provenance
rules together with new snapshot/model records. Summaries contain counts/digests,
never source text. No entity precision/recall/F1 claim until independent labelled
evaluation data exist. Human data remain local with backup instructions.

## Persistent open questions

T1 resolved: user explicitly approved this exploratory run and suggestion UI.
T2 unresolved blocker:90/130spans align;40do not (22extra boundary whitespace,
18inside tokenizer words). No width/budget failures. Original labels are unchanged;
humans must review proposed boundaries, including possible fused-source-word cases.
T3 implementation check: verify frozen-backbone parameter selection and loss API
in the installed0.2.21 runtime; record measured CPU cost before the full job.
T4 resolved default: development/test and annotator2 remain blind; no further
manual labels are required before this ten-document experiment can be attempted.
