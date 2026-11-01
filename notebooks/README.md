# Notebooks

This page covers the notebooks in this folder: a table of every notebook with one line each,
then the narrative for the entity-extraction series (08-16 of that series, 27-29), the GLiNER
campaigns, the medium-v2.5 experiment, the entity annotation UI, the ten-document pilot, the
tokenizer amendment and the French interview prototype. Notebooks report models; they do not
define them (AGENTS.md, Notebooks). The commands they rely on are in
[`docs/development.md`](../docs/development.md). Moved from README.md on 1 Nov 2026.

Launch them from the installed project environment, after `uv sync`:

```bash
uv run jupyter lab notebooks/
```

## All notebooks

Two series share the numbers 08-11: the reset's research findings (`08-final-annotation-set`,
`09-indomain-ladder`, `10-extraction`, `11-end-to-end`) and the entity-extraction series
(`08-entity-extraction-readiness` onwards). The one-line descriptions come from each
notebook's first cell.

| Notebook | What it covers |
|---|---|
| [`01-acquisition-overview`](01-acquisition-overview.ipynb) | What is on disk, how big it is, under what licence, and whether it matches the manifest |
| [`02-fit-benchmark-eda`](02-fit-benchmark-eda.ipynb) | A1, the `cnamuangtoun` fit benchmark |
| [`03-djinni-in-domain`](03-djinni-in-domain.ipynb) | A2 Djinni, the in-domain corpus of IT JDs and CVs |
| [`04-extraction-sources`](04-extraction-sources.ipynb) | B1 DataTurks and B2 SkillSpan, the extraction supervision |
| [`05-raw-formats`](05-raw-formats.ipynb) | C1/C2/C3 raw formats and the ingestion reality check |
| [`06-skill-vocabulary`](06-skill-vocabulary.ipynb) | D1 ESCO and D2 `data_jobs`, the skill vocabulary |
| [`07-baseline-tfidf-keyword`](07-baseline-tfidf-keyword.ipynb) | TF-IDF and BM25, the Stage 1 classical baseline on the leak-free A1 split |
| [`08-entity-extraction-readiness`](08-entity-extraction-readiness.ipynb) | What is available locally for entity extraction (no real resumes displayed) |
| [`08-final-annotation-set`](08-final-annotation-set.ipynb) | The final annotation set: one Fit / No Fit label per in-domain pair (D41, D47) |
| [`09-indomain-ladder`](09-indomain-ladder.ipynb) | The matching ladder scored against the final set |
| [`09-resume-evidence-and-themes`](09-resume-evidence-and-themes.ipynb) | Proposed evidence records and activity-backed themes, on synthetic evidence |
| [`10-entity-extraction-inputs`](10-entity-extraction-inputs.ipynb) | Validates inputs and samples 1,000 training CVs and JDs (seed 0) |
| [`10-extraction`](10-extraction.ipynb) | Skill extraction: rules vs a learned tagger |
| [`11-end-to-end`](11-end-to-end.ipynb) | Retrieval at pilot scale, the secondary benchmark, latency |
| [`11-rule-based-entity-extraction`](11-rule-based-entity-extraction.ipynb) | The versioned offline rules extractor on real A2 training documents |
| [`12-entity-extraction-evaluation`](12-entity-extraction-evaluation.ipynb) | Operational integrity, synthetic demonstrations and the status of held-out accuracy |
| [`13-transformer-entity-extraction`](13-transformer-entity-extraction.ipynb) | Local GLiNER small v2.1 zero-shot extraction on five entity types |
| [`14-entity-extraction-eda`](14-entity-extraction-eda.ipynb) | GLiNER and rules predictions on the same 100 CVs and 100 JDs |
| [`15-entity-distribution-and-matching`](15-entity-distribution-and-matching.ipynb) | Frozen entity distributions and matching in the 100 CV / 100 JD campaign |
| [`16-french-interview-translation`](16-french-interview-translation.ipynb) | French interview video to English screening evidence (prototype) |
| [`17-guide-new-model-testing`](17-guide-new-model-testing.ipynb) | Guide: testing a new model against the ladder |
| [`18-a1-learned-weights`](18-a1-learned-weights.ipynb) | Can weights learned from A1 beat the hand-set X2 score on A2? |
| [`19-learning-curves`](19-learning-curves.ipynb) | Do more LLM labels let a richer model beat X2? |
| [`20-error-analysis`](20-error-analysis.ipynb) | Where the score goes wrong on A2 (exploratory, diagnosis only) |
| [`21-error-findings-and-manual-review`](21-error-findings-and-manual-review.ipynb) | What is known about the scoring mistakes, and why some need a manual read |
| [`22-score-defect-fixes`](22-score-defect-fixes.ipynb) | Fixing the three defects the manual review found |
| [`23-ladder-on-clean-labels`](23-ladder-on-clean-labels.ipynb) | The ladder on the pairs where the humans agree |
| [`24-classifier-then-rerank`](24-classifier-then-rerank.ipynb) | A learned stage 1 and a reranker trained on `chat-1` (D61) |
| [`25-threshold-and-features`](25-threshold-and-features.ipynb) | Step 4: a lower stage-1 threshold, and feature arms |
| [`26-chat3-holdout-and-c2`](26-chat3-holdout-and-c2.ipynb) | A held-out test of stage 2 (`chat-3`), and C2 validated on it |
| [`27-entity-extraction-source-validation`](27-entity-extraction-source-validation.ipynb) | Saved extraction evidence validated against its sources |
| [`28-fit-gliner-comparison`](28-fit-gliner-comparison.ipynb) | Five-label GLiNER extraction across every fit pair |
| [`29-gliner-medium-sentence-comparison`](29-gliner-medium-sentence-comparison.ipynb) | GLiNER medium v2.5 with sentence-aware extraction |
| [`ownership/walkthrough`](ownership/walkthrough.ipynb) | Teaching notebook for a new owner: the research system one step at a time |

## Entity extraction

For preliminary entity-extraction work, start with
[`08-entity-extraction-readiness.ipynb`](08-entity-extraction-readiness.ipynb)
and its [execution plan](../plan/2026-09-13-entity-extraction/README.md).
The notebook checks what is available in this checkout, reports metadata without displaying
real resumes, and demonstrates the existing keyword helper on synthetic text. Missing NER
sources are reported explicitly; the historical acquisition record does not imply that every
source is installed locally. The plan covers skills, titles, qualifications and experience
evidence, with supervised extraction and theme construction as subsequent work.

Continue with [`09-resume-evidence-and-themes.ipynb`](09-resume-evidence-and-themes.ipynb)
and its [execution plan](../plan/2026-09-19-resume-themes/README.md) to explore the proposed
evidence records, activity-backed themes and requirement comparisons. Its worked examples
are manually specified synthetic evidence, not extractor predictions or accuracy results.

For executable extraction on local Djinni data, use these notebooks in reading order:

- [`10-entity-extraction-inputs.ipynb`](10-entity-extraction-inputs.ipynb): validate inputs and sample only the committed training region.
- [`11-rule-based-entity-extraction.ipynb`](11-rule-based-entity-extraction.ipynb): extract skills/tools, titles and qualifications with exact field-relative evidence spans.
- [`12-entity-extraction-evaluation.ipynb`](12-entity-extraction-evaluation.ipynb): check offsets, reproducibility and synthetic regressions, and report the status of labelled evaluation.

The [approved implementation plan](../plan/2026-10-05-entity-extraction-notebooks/README.md)
uses 1,000 training CVs and 1,000 training JDs by default. Its versioned seed vocabulary
and context rules provide a limited offline baseline; document coverage is descriptive,
not extraction accuracy. DataTurks/SkillSpan gold evaluation remains blocked when those
sources are absent. Real document text and entity surfaces stay in git-ignored local
artifacts; saved notebook outputs contain aggregates and synthetic examples only.


The same extraction can run without Jupyter: see [Extraction commands](../docs/development.md#extraction-commands).

Notebook 11 now includes a dictionary of the 15 stored entity fields and a five-row
redacted preview of actual rule-extracted mentions. Continue with
[`13-transformer-entity-extraction.ipynb`](13-transformer-entity-extraction.ipynb)
for locally run GLiNER predictions and a comparison with rules on the same training
documents. GLiNER extracts `KNOWLEDGE`, `SKILL`, `TITLE`, `DEGREE` and
`EXPERIENCE_YEARS`; the rules comparator supports the original four labels only.
Experience spans preserve stated durations, ranges and minimums in CV claims or
JD requirements. They do not infer tenure from dates or determine job fit.
The saved five-label run has its own aggregate record at
`output/extraction/transformer-experience-years-metrics.json`; the original
four-label record and historical notebook 14 results are retained.
After combining the extraction branches, the seed-rule inspection record is
preserved as `output/extraction/rule-inspection-metrics.json`.
`output/extraction/extraction-metrics.json` describes the separately evaluated
skill rules/NER campaign; the two records have distinct schemas and populations.
Neural scores are uncalibrated; context flags still use a separate heuristic, and
unlabelled coverage/disagreement does not establish accuracy or superiority.


The transformer stack runs in its own locked uv environment; its commands are in
[Extraction commands](../docs/development.md#extraction-commands).

For charts and real examples from the generated outputs, open
[`14-entity-extraction-eda.ipynb`](14-entity-extraction-eda.ipynb).
It compares GLiNER and rules on the same 100 CVs and 100 JDs, explores mention
frequencies, per-document counts, scores, field/context patterns and disagreements,
and exports figures to `output/extraction/eda/`. Model scores and threshold-retention
curves describe predictions, not extraction accuracy. This notebook uses existing
artifacts and regenerates only the matching rule predictions; it performs no
additional neural inference or training.

For validation against the original CV/JD fields, open
[`27-entity-extraction-source-validation.ipynb`](27-entity-extraction-source-validation.ipynb).
It audits historical run manifests, source hashes, exact entity slices and field
coverage, and displays bounded source context beside saved labels. Source matches
establish traceability; label accuracy and missing-entity recall require compatible
gold annotations. The [validation plan](../plan/2026-10-14-entity-extraction-validation/README.md)
records this distinction and the local gold-data availability.

The approved full-fit GLiNER campaign covers every pair in
`data/processed/fit/train.parquet` and `test.parquet`, extracting each unique CV
and JD once with five fixed entity categories. Its reporting notebook is
[`28-fit-gliner-comparison.ipynb`](28-fit-gliner-comparison.ipynb).
The [campaign record](../plan/2026-10-14-fit-pair-gliner-comparison/01-implementation.md)
tracks execution and validation status. Fit-conditioned distributions are
exploratory; A1 labels do not reproduce (A13), and they are not entity-span gold.

The separate [medium-v2.5 experiment](../plan/2026-10-14-gliner-medium-sentence-chunking/README.md)
uses `gliner-community/gliner_medium-v2.5` with sentence-aware, prompt-budgeted
chunks and overlapping word windows for oversized sentences. Original text and
entity offsets are preserved. It has its own runtime and campaign checkpoints;
the small-model default remains unchanged. Notebook
[`29-gliner-medium-sentence-comparison.ipynb`](29-gliner-medium-sentence-comparison.ipynb)
compares operational results after inference completes. Model and chunk strategy
change together; mention counts do not establish extraction accuracy.


## Entity annotation and the ten-document pilot

The separate [human entity annotation UI](../plan/2026-10-16-entity-gold-annotation/README.md)
supports selecting text spans, four category shortcuts, editing, undo and explicit
whole-document review. Try synthetic documents locally:

```bash
uv run python -m candidate_screener.entity_annotation.ui --demo --annotator reviewer
```

Open the printed `http://127.0.0.1:8766/` URL. Demo annotations are synthetic and
cannot measure the model. Real source/categories/batch/annotators must be confirmed
before creating a campaign under D56. See the
[boundary guide](../plan/2026-10-16-entity-gold-annotation/annotation-guide.md) and
[evaluation/training protocol](../plan/2026-10-16-entity-gold-annotation/training-protocol.md).
The frozen Fit annotation UI and labels are unchanged.

The approved [ten-document GLiNER adaptation pilot](../plan/2026-10-16-entity-ten-example-finetune/README.md)
uses saved human training spans. It validates exact tokenizer boundaries before
training and preserves original reviews. Model suggestions, once available, are
explicitly loaded only on untouched primary-annotator training documents; exports
retain assistance provenance. Independent annotation and dev/test stay blank.
The approved [character-boundary tokenizer amendment](../plan/2026-10-17-character-boundary-tokenization/README.md)
preserves every saved human boundary: training-derived lexical cuts and explicit
whitespace units align all 130 spans in the latest ten-document snapshot. The frozen
policy must accompany training, checkpoint reload and inference. This alignment
check establishes representability; it does not establish extraction accuracy.

## The French interview prototype

The [French interview prototype](16-french-interview-translation.ipynb)
demonstrates audio decoding, French speech transcription and Marian French-to-English
text translation. Full transcripts and media remain local; the notebook reports
aggregate results and separately reviewed safe excerpts. This is an ingestion
experiment, not an adopted ranking model or a three-class fit classifier. The
[implementation plan](../plan/2026-10-11-multimodal-entity-pipeline/README.md) describes
text-only pipeline changes, five-label GLiNER evidence, applicant transcript review
and the proposed chat upload flow. Live pipeline/UI integration awaits that plan's
approval.
