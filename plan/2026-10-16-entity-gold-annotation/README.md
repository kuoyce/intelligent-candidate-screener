# Small human entity-gold batch, local span UI and fine-tuning protocol

Date: 16 October 2026. Status: implementation and real batch approved by user;
real A1 campaign created with four categories,120documents and two annotators.
UI/service/export implementation and validation are recorded in
[01-implementation.md](01-implementation.md).

## Objective and decision boundary

Collect human entity spans to measure the frozen GLiNER medium model and later
fine-tune a separate checkpoint. Pair Fit labels cannot serve as entity gold.
Annotation is by unique document, never repeated CV/JD pair rows.

D42 currently closes collection and annotation/ is frozen under D42. Approval must
explicitly authorize a new scoped decision in docs/decisions.md allowing this
human-only entity campaign, with a fixed population and cap. It must preserve all
existing Fit judgements, the A2 final set, annotation instruments and closed LLM
dispatch. Implement the new service under entity_annotation/ rather than editing
the frozen annotation app. Reuse annotation.redact as a read-only dependency.

## Proposed scope for approval

Four categories await confirmation. Recommended initial set: KNOWLEDGE, SKILL,
TITLE, DEGREE. EXPERIENCE_YEARS remains unevaluated/excluded if not selected.
Population awaits confirmation: both CV/JD from the A1 fit corpus, or A2 train.
Do not silently infer what the user means by A0/A1 or choose another corpus.

Proposed cap:120 unique documents,60CVs/60JDs. First8documents (4/side) calibrate
guidelines and measure labelling time; retain them only in training and reannotate
if the guideline changes. Proposed split80train,20dev,20test, balanced by side.
All splits are a new entity task, not a redefinition of existing fit splits.
For A1 prefer training/development documents from original train and test from
original test after cross-split document/text-overlap exclusion. For A2 select only
the document-level train region, subdivided for this entity task. Report achieved
inventory and pause population-dependent creation if insufficient eligible data.
Freeze group membership before labels or model selection; hash normalized text
for duplicate detection only, without changing annotation text. Exclude previously
reviewed examples from the blind test set where their identities are known.

## Annotation guide and local UI

Replicate the familiar local browser workflow with a new standard-library HTTP
server plus HTML/JS on127.0.0.1, separate port8766; thin transport and tested service.
Do not add a third-party hosted annotation service or transmit documents remotely.

One document at a time, with optional JD/CV pair navigation for convenience only.
Select text, press1–4 or a coloured label button, then add/delete/relabel/adjust a
span. Show highlighted spans, a label list, undo, explicit save, next/back and
progress. Support empty-but-reviewed documents, skipped/uncertain documents and
in-progress saves separately. Never equate untouched text with reviewed negatives.
Finish requires reviewing the whole document; model proposals alone are not gold.

Write a category and boundary guide before collecting final labels: distinguish
technologies from competencies and study disciplines from occupations; define
qualification boundaries, coordination, repeated mentions and nested spans.
Propose maximal informative spans with no redundant nested same-label spans;
ambiguous cross-category overlaps require an explicit guide rule/adjudication.
Dates, negation and educational durations are context, not automatic role evidence.

Gold evaluation mode starts blank with model predictions/scores/Fit labels hidden.
Annotators must add missed entities as well as reject false predictions. Training
documents can optionally expose suggestions after an independent first pass;
record suggestions and their model revision separately from accepted human spans.
Keep dev/test blind throughout collection. Suggestions do not become human labels
merely because a document was opened or saved.

Every displayed text passes annotation.redact. Freeze displayed text, original
source digest, redaction version and display-to-original boundary mapping. Reject
annotations intersecting masked identifiers or unrepresentable boundaries.
Convert browser UTF-16 selection offsets to Python code-point offsets explicitly;
test emoji/non-BMP characters and redaction length changes. Preserve original
evidence offsets without serving hidden original PII to the browser.

## Storage and quality control

Ignored data holds immutable document manifest, source/display text references,
annotator IDs, span start/end/label/surface, reviewed state, guideline version and
append-only revisions. Validate exact slices, bounds, category values and document
identity server-side. Atomic writes, optimistic revision checks and resumable
sessions prevent stale browser tabs from overwriting work. Export JSONL/Parquet
gold with explicit split and completeness; keep per-person labels and adjudication
separate. Do not write entity gold into judgements.csv.

Propose independent double annotation of20% across CV/JD and labels, followed by
human adjudication. If only one annotator is available, report single-annotator
gold and use a later blinded recheck; do not invent inter-annotator agreement.
Measure agreement without a target. Pilot time determines the actual workload
estimate;120documents is a proposal, not a claim that a small batch is sufficient.

## Evaluation and subsequent training

Before fitting, evaluate the frozen medium checkpoint against blind entity test
gold for selected categories only. Primary metric strict exact-boundary-and-label
micro precision/recall/F1; report per label, CV/JD, document counts, gold-span counts
and95%document-bootstrap CIs. Partial span scores are secondary and explicitly
defined. Ignore predictions outside the selected ontology rather than pretending
the other category is exhaustively annotated. Undefined metrics remain undefined.

Fine-tune a new medium checkpoint on human train spans; map original spans through
the exact pinned GLiNER word tokenizer and reject unalignable spans explicitly.
Chunks from a document stay in its assigned split; avoid cutting labelled spans
and prevent overlap from duplicating metric counts. Save the baseline untouched.
Tune training duration/hyperparameters only on dev; compare the final frozen model
once on entity test with paired document-bootstrap deltas. Repeated test-driven
iteration converts that test into development and needs a new evaluation design.

The first implementation delivers the UI, validated exports and scoring/training
protocol, not an immediately trained model: human labels must first be collected
and the compatible training API/config reviewed against the pinned runtime.
Time one training configuration before any grid and publish the cost estimate.
No training startup in the UI; no pseudo-gold or LLM annotation dispatch.

## Implementation workflow and validation

Resolve high-impact questions and obtain plan approval. Create a branch, commit
approved plan and scoped D42 decision, then delegate service/export and UI work
with independent review. Preserve unrelated working changes. Test offset mapping,
source hashes, review state, no-label documents, held-out blinding, split leakage,
stale writes, invalid labels and export/token alignment using synthetic data.
Show regression guards failing when disabled. Browser-test selection and editing
with synthetic text. Run full pytest once with logs and relevant runner checks,
including cold where required. Document launch/export commands and annotation
guide; no new collection starts before scoped decision and frozen manifest exist.

## Persistent open questions

G1 resolved: KNOWLEDGE, SKILL, TITLE and DEGREE confirmed.
G2 resolved: A1 fit documents confirmed.
G3 resolved:120documents,60CVs/60JDs;80train/20dev/20test confirmed.
G4 resolved: two annotators confirmed. IDs annotator1 (all120documents) and
annotator2 (balanced24-document independent overlap, as proposed by the plan).
G5 resolved: user approved proceeding with the isolated UI/protocol and scoped
human-only entity collection decision. Existing Fit collection stays closed.

Frozen real campaign: f5fadd3fda6068818192a3a8, stored under ignored
data/processed/entity-annotation/campaigns/. Human collection is open only for
this manifest and scope; no model output is prefilled.
