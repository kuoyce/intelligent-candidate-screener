# Frozen entity evaluation and subsequent training protocol

The original UI implementation did not train a model. The user subsequently
confirmed120A1documents, four categories, two annotators and explicitly approved
the [ten-document training pilot](../2026-10-16-entity-ten-example-finetune/README.md).
The original human snapshot is10train documents (4CV/6JD),130spans. Actual training
remains blocked by40span/token-boundary mismatches; labels are never rounded/dropped.

Primary-annotator training documents may receive explicitly requested model
suggestions only after a verified checkpoint exists. Record annotation_mode and
assistance provenance on every resulting review/export, even if the human deletes
all suggested spans. Dev/test and the independent second annotator remain blind.
Assisted annotation is training material, not blind accuracy-evaluation gold.

Freeze unique-document splits and duplicate groups before annotation or model
selection. All chunks and repeated pair appearances of a document stay in its
assigned entity split. For A1, original test documents cannot enter entity train
or dev; normalized-text duplicates across source splits are excluded. For A2 only
the original document-level train region is eligible. Calibration and previously
reviewed examples cannot become blind entity test examples.

Exports contain one latest exhaustively reviewed revision per annotator/document,
original-source spans, source digest, guideline and explicit split. Empty reviewed
gold is distinguishable from untouched text. Export paths remain under ignored
data; never commit original source text or entity gold. Existing Fit judgements
and final-set labels remain untouched. Independent annotators and adjudication
are separate datasets; single-annotator data must be reported as such.

First freeze checkpoint identity, selected categories, inference strategy and
prediction record. Score only exhaustively reviewed gold. Primary matching is
permitted only when every evaluated document has explicit complete/complete-zero
prediction coverage and a matching source digest; missing or interrupted inference
is never silently counted as zero predictions. Gold source hashes and duplicate
spans are validated independently of annotation export. Primary matching is
exact start, end and category; remove identical duplicate predictions only,
retain distinct overlapping predictions, ignore unselected categories. Report
micro precision/recall/F1, per category and CV/JD, document and gold-span counts,
with 95% bootstrap intervals resampling documents. Undefined denominators remain
undefined; bootstrap excludes undefined draws and they must not be called zero.
Partial-overlap scoring is not implemented and must not be substituted silently.

Before training, review the pinned runtime's supported training API. Align gold
to its exact word tokenizer, explicitly reject unalignable spans and ensure
chunks do not cut a gold mention. Preserve original offsets; repeated overlap
must not multiply evaluation counts. Train a separate checkpoint only on human
train spans, preserve the baseline, tune duration and hyperparameters only on
dev. Time one configuration before proposing a grid and report resource cost.
Freeze the selected model before comparing once on entity test; use paired
document bootstrap deltas on identical documents. Test-driven iteration makes
that test development and requires a new design. Fit labels cannot supervise
entity spans or establish extraction accuracy. No LLM/pseudo-gold collection or
training startup occurs in the annotation UI.
