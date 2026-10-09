# Visible entity records and transformer extraction exploration

6 October 2026. Status: approved by the user; implemented. See the implementation record.

## User request and current state

The user wants to see extracted entity fields and explore deep learning/transformer
models in place of the rule-based approach. Extend the approved local extraction
delivery and the original September extraction plan. Keep the rule outputs as a
comparison baseline, with transformer inference becoming the new experiment.
The existing notebook 11 has user modifications; preserve them when adding a view.

Current types are KNOWLEDGE, SKILL, TITLE and DEGREE. Records store entity_id,
doc_id, source, doc_type, field, text_version, start, end, label, surface,
canonical_id, method, extractor_version, confidence and context_status.
Local Djinni/A1 data remains available; DataTurks/SkillSpan and ESCO remain absent.
There is no NER gold or installed PyTorch/Transformers/GLiNER project dependency.

## Model research and decision

Primary sources consulted 6 October:

- [GLiNER repository](https://github.com/urchade/GLiNER): custom-label zero-shot NER
  using encoder models; local inference is supported.
- [GLiNER small v2.1 model card](https://huggingface.co/urchade/gliner_small-v2.1):
  English, 166M parameters, Apache-2.0. Proposed initial checkpoint, pinned to its
  resolved revision at download; selection favors a bounded exploratory run,
  not a claim that this checkpoint is best for resumes.
- [Hugging Face token classification](https://huggingface.co/docs/transformers/main/tasks/token_classification):
  fine-tuning requires token-label/subword alignment; source offsets and windows
  need separate attention for long fields.
- [SkillSpan research](https://aclanthology.org/2022.naacl-main.366/): domain-adapted
  models outperform non-adapted models in its JD experiments; this does not prove
  performance on resumes or this project's data.
- [Generic BERT NER card](https://huggingface.co/dslim/bert-base-NER): only PER,
  LOC, ORG and MISC labels; it is not a ready-made skill/title/degree extractor.

Propose GLiNER zero-shot inference now, without a new training campaign. Research
BERT/DistilBERT/DeBERTa token classification as a supervised follow-up, gated by
restored gold sources and an agreed training budget. Do not create fabricated gold
from rule predictions or treat rule agreement as accuracy. No current evidence
establishes that the transformer will outperform the rules.

## Delivery design

### Visible fields

Add a field dictionary and bounded real entity preview to notebook 11, preserving
current user edits. Show surface, entity label, source field, doc_type and context
status for up to five rows. Run redaction and assert_clean in the notebook's display
path; never show entire resumes. Current rule surfaces are bounded seed phrases.
Display coordinates separately from the redacted surface: offsets refer to the
original source text, never the modified display string. Update the earlier
synthetic-only wording to reflect this newly requested, redacted entity view.

For transformer outputs, a predicted type cannot guarantee that a span contains no
name. Default saved previews therefore use synthetic text; provide a local entity
inspection helper that redacts contacts and suppresses unreviewed spans for saved
outputs. Real predicted records remain in the local parquet for inspection. Do not
claim the regex redactor removes every name. No complete real source field is shown.

### Transformer adapter and notebook

Create `src/candidate_screener/extraction/transformer.py` and
`notebooks/13-transformer-entity-extraction.ipynb`. The notebook calls package code;
it does not define a model or training loop. Install compatible, resolved versions
through `uv add` (prefer optional `extraction-transformers` dependencies) and update
the lockfile. Cache model/tokenizer weights locally, never in git. Download only
the proposed checkpoint and required tokenizer/config artifacts. No resume/JD text
is sent to a remote inference service.

Start with custom descriptions mapped explicitly onto KNOWLEDGE (tools/languages),
SKILL (competencies), TITLE and DEGREE. Do not introduce dates, certificates,
employers or new label schemes silently. Keep model outputs and rule outputs in
separate run directories. Model-derived spans must preserve immutable field-relative
half-open offsets and satisfy original_text[start:end] == surface. Unknown canonical
IDs are null: matching an entity does not automatically map it to ESCO.

Generalize entity validation to distinguish rule confidence (null) from model scores
(finite bounded raw scores, explicitly uncalibrated). Scores do not establish
proficiency, fit or probability of correctness. Context may continue to use a
separately versioned heuristic; report the hybrid nature explicitly. A neural NER
model alone does not solve negation, aspirations or JD required/preferred status.

Chunk long fields at tokenizer boundaries with overlap under the actual model
limits. Record covered source intervals, preserve original offsets across chunks,
merge duplicate overlapping-window predictions deterministically, and report
truncation/quarantine rather than silently discarding source tails.

Proposed default: smoke test five CVs/five JDs, then 100 CVs/100 JDs from the same
frozen A2 training region at seed 0. Use CPU first; use MPS only when verified
compatible. Record wall time, device and memory observations. Cap exploratory
inference at 30 minutes and write an explicit partial/completion status if exceeded;
do not imply partial runs cover the whole requested sample. Scaling to 1,000 each
is configurable after runtime is measured, not the default for the initial trial.

### Comparison and evaluation gates

Report per-source/type entity counts, per-label document coverage, confidence score
distribution, runtime, covered/truncated fields and prediction differences from
rules on the identical document subset. Differences mean disagreement, not superiority.
Synthetic probes check aliases, offsets and long-window reconstruction but cannot
establish corpus accuracy. Preserve overlap policies and model label descriptions.

Record exact checkpoint revision, model/tokenizer/config artifact hashes, resolved
dependency versions, labels, threshold, chunk/overlap parameters, context policy,
source/partition digests, sample IDs and completion status. A default threshold
of 0.5 is a declared exploration setting, not a tuned or calibrated cut-off.
Persist real spans under data/processed/extraction/; commit aggregates/config only.
Do not demand bitwise score equality across devices; validate spans/provenance and
state the reproducibility tolerance where applicable.

Gold precision/recall/F1 remains blocked. A later supervised notebook can restore
B2 for independent skill/knowledge layers and B1 for resume-native repaired gold,
retain frozen splits/grouped folds, align subword labels, and evaluate sources
separately. B1 individual-skill segmentation remains silver, not individual gold.
No BERT training or extra human annotation is authorized by this proposed phase.

## Workflow and acceptance

After approval, create a new branch and commit the finalized plan. Delegate adapter,
notebook/view and independent review work under AGENTS.md. Preserve existing dirty
files and previous extraction artifacts. Add synthetic tests with mocked neural
outputs for offsets, window merging, confidence validation and missing models;
demonstrate guarded failures before fixes. Tests must run without weights/network.
Run a real smoke test and bounded sample inference, execute notebooks, validate
saved outputs (five records/2,000 characters), run uv run pytest and existing
derived/baseline checks, and document existing failures separately. Update README,
data reconstruction guide, A2 card/catalog and an implementation record together.

Acceptance: visible current entity fields; actual locally run transformer predictions;
exact source offsets and explicit coverage/completion; checkpoint provenance;
rule comparison on the same IDs; no gold-accuracy or ranking-benefit claim.
Missing model/dependency/network blockers must be reported, never replaced silently
with rule predictions labelled as transformer output.

## Open questions and proposed choices for approval

| ID | Question | Proposed resolution |
|---|---|---|
| TE1 | Immediate transformer approach? | GLiNER small v2.1 zero-shot for the same four types |
| TE2 | Dependencies and weights? | Install through uv and download pinned model artifacts for local inference |
| TE3 | Initial compute scope? | Smoke 5+5, then 100+100 train documents; 30-minute inference cap |
| TE4 | Show real extracted mentions? | Add redacted five-row current-rule preview; keep unreviewed neural surfaces local |
| TE5 | Supervised training and restored gold? | Research/design only here; defer execution until data and compute approval |

Approval resolves TE1–TE4 with these choices. TE5 and original EE2–EE5 remain
later-stage gates. Planning and model research do not alter current extraction code.
