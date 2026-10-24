# Inspect medium GLiNER entity correctness against original CV/JD evidence

Date: 15 October 2026. Status: proposed; awaiting user approval.

## Scope and known facts

Inspect the completed medium campaign 3d836f87df64512b057b20ec, not whichever
campaign is latest later. It contains 41,027 spans across949documents and reports
all complete. Compare separately with small campaign901a26c9e785920223b07c9b.
The implementation preserved whole words and added sentence-aware chunks with
whole-word fallback; it did not stem text. Model and chunking changed together.
Existing notebook19 compares operational distributions, not semantic accuracy.

## Deliverables and design

1. Reuse medium_run.load_campaign to verify manifest identity, unchanged source
   digests and every saved checkpoint. Build a campaign audit adapter in reusable
   extraction code using validation.audit_entities with saved per-field coverage.
   Report counts, field coverage, invalid/quarantined predictions and fallback
   diagnostics by CV/JD. Do not infer label correctness from exact source slices.
2. Add a thin notebook20 for five labels (KNOWLEDGE, SKILL, TITLE, DEGREE,
   EXPERIENCE_YEARS) separately on CV/JD. Display deterministic representative
   spans beside bounded original redacted context, labels, offsets and raw scores.
   Show one boundary/fallback inspection section, repeated/overlapping-label spans,
   low-score and suspicious-title/degree queues. Heuristic flags identify cases
   to inspect, not adjudicated errors. No real full documents or new annotation
   labels are saved. Each text display <=5rows and <=2,000characters.
3. Compare small/medium evidence on exactly matching original documents and source
   hashes: shared spans, medium-only and small-only. Disagreement is not truth.
   Keep total requested, completed-zero, pending, failed and incomplete states
   explicit. Counts and raw confidence changes cannot establish accuracy gains.
4. Reinventory existing compatible entity gold and frozen evaluation records.
   If no compatible local gold/predictions exist, mark precision/recall/F1
   unavailable. Existing Fit labels and other tagger scores cannot supply these.
   Downloading datasets, new inference on gold, ontology remapping and new gold
   annotation are outside this inspection; any later accuracy benchmark needs a
   separately approved protocol identifying compatible labels, source splits,
   offset conventions, strict/partial matching and document-bootstrap CIs before
   scoring. Never tune on the A2 final set or reopen collection implicitly (D42).
5. Save aggregate audit records and an implementation report; real source snippets
   remain inside ignored data or redacted bounded notebook output. Update README
   links. Report what is verified and what remains unknown separately.

## Workflow and acceptance

On approval create a branch and commit the approved plan alone. Delegate package
audit/inspection helpers and notebook work, with independent review. Preserve
unrelated changes and all model checkpoints; do not retrain, rerun extraction,
change prompts/thresholds or repair predictions silently.

Use synthetic tests for wrong offsets, missing coverage, mixed campaign identity,
changed source text, dropped zero documents, and redaction/bounded previews.
Demonstrate guard-disabled regression failures. Execute notebook20; run focused
tests during iteration and the full suite once with saved logs. Use runner checks
for affected records, including the required cold tier before marking complete.
Do not overwrite frozen TF-IDF baseline records to hide the already documented
two baseline reproduction failures. No claim of improved accuracy without gold.

## Persistent open questions

I1 pending: user approval of this saved-evidence inspection scope.
I2 pending inventory: whether compatible local DataTurks/SkillSpan assets now exist;
they were absent in the previous validation. Their presence alone does not make
their annotations compatible with all five categories.
I3 unresolved accuracy limitation: the949fit documents lack verified entity-span
gold in the existing campaign; visual examples cannot estimate missing-entity
recall or population precision. No new annotation collection is authorized here.
