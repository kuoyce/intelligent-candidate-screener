# Entity annotation UI implementation record

16 October2026. Status: UI/service/exports/scoring implemented and synthetically
validated. Real batch configuration subsequently confirmed and UI launched below.
No human gold or trained checkpoint is claimed by the launch verification.

## Approval and scope

The user approved proceeding. Branch feat/entity-gold-annotation-2026-10-16;
approved plan commit98a2571; scoped D56 decision commit1a6143b. Frozen annotation/
code, Fit judgements, A2 final labels and LLM dispatch are unchanged. AGENTS.md
records D56's isolated exception. All earlier unrelated work remains in place.
Text/campaign, UI/browser and export/evaluation were delegated; independent review
identified and verified corrections to campaign provenance and asynchronous saves.

## Implemented behavior

New entity_annotation/ package contains a standard-library localhost server, span
editor, redaction mapping, immutable grouped document manifests, registered
annotator assignments, hash-chained review journal and gold exports. Explicit
whole-document confirmation distinguishes reviewed-empty from untouched/draft/
skipped/uncertain. UTF-16 browser boundaries map to exact original source slices;
masked spans, surrogate halves, overlaps, stale revisions and source changes fail
validation. Raw source, mapping and another annotator's work never reach the page.

All documents start without model suggestions or Fit labels. Pilot documents are
shown from training first, balanced across CV/JD. The real creation CLI requires
explicit source/four categories/counts/annotator choices, caps at120documents and
freezes source hashes and document-disjoint splits. A1 cross-original-split overlap
and known reviewed test examples are excluded; A2 uses only verified train-region
inputs and excludes CV Position. Second-annotator assignment uses approximately20%
stratified by side/split (actual fraction depends on rounded small-cell counts).

Exports are per annotator and revision; only completely reviewed documents become
gold. Atomic immutable JSONL/Parquet exports include source/revision digests. Strict
evaluation requires complete prediction-document coverage and source hashes so
missing inference cannot masquerade as zero entities. Scores include category/CV/JD
support and document-bootstrap intervals. No real accuracy figures are computed.
Fine-tuning is a documented subsequent protocol and requires human labels first.

## Verification and fixes

Full suite ran once:1,091passed,10skipped,2failed in168.59s.
Log: logs/entity-annotation-full-pytest.log. The failures are the already recorded
baseline golden reproduction/environment checks: local TF-IDF vocabulary/model
values differ from the frozen record. No baseline code/record was changed to hide
these failures. Optional browser checks skip in the base suite and passed separately
using ephemeral Playwright and installed headless Chrome.

Synthetic HTTP/browser suite:6passed, logs/entity-annotation-ui-browser-tests.log.
Actual-service browser smoke also passed, logs/entity-annotation-actual-service-browser.log:
eight temporary synthetic documents, selection after emoji, reviewed save, exact
original-source span, reopen/persistence, redaction and demo banner. Temporary
browser labels were removed with the synthetic test directory. No real corpus used.

Focused mapping/campaign/service/UI tests passed; export/evaluation tests14passed.
Guard-disabled regressions demonstrably fail in logs:
entity-annotation-mapping-regression.log (masked selection),
entity-annotation-revision-regression.log (stale write),
entity-annotation-ui-race-regression.log (save/navigation),
entity-gold-provenance-regression.log (foreign campaign),
entity-gold-export-regression.log (whole-document review).

Review fixes include complete source/guide/campaign provenance requirements,
explicit inference completion metadata, schema/leakage checks even for rehashed
manifests, journal replay revision validation, and save/load request race guards.
Owned tracked documentation passes git diff --check. An unrelated existing
entity_distributions.py trailing-whitespace issue was left untouched.

## Check runner limitation

The first scoped runner invocation included ui.html. Its generic rule treating
every non-Python package file as potentially read by every record selected all
checks. This was unintended: stop the scheduler and its retrieval_pilot/shortlist
children rather than run unrelated cold jobs alongside the suite. Its partial logs
are preserved under logs/entity-annotation-cold-checks/; failures include missing
optional ML dependencies, ESCO and pools. It is not a completed successful run.

The isolated Python import-closure cold invocation selects zero registered model
checks, logs/entity-annotation-scoped-cold.log and scoped-cold/summary.tsv. Existing
record modules do not import the new annotation package; HTML transport is covered
by HTTP and actual-service Chrome tests. No model/training implementation or final
score changes in this task justify retraining unrelated records.

## Launch and outstanding human choices

Synthetic demo launched on127.0.0.1:8766 with reviewer ID and tracked process;
log logs/entity-annotation-demo.log. UI/docs list explicit creation/export commands.
No real manifest was created. G1–G4 remain pending in the plan: selected four
categories, corpus, batch counts and one/two annotators. Answering those configures
the real campaign under the already approved D56 scope; it is not a new request
for implementation permission. Preserve local manifest/review/export backups once
collection begins because human decisions cannot be regenerated from raw sources.

## Real batch confirmed and launched

The user confirmed all four proposed categories,120A1documents (60CV/60JD) and
two annotators. Created immutable campaign f5fadd3fda6068818192a3a8 with explicit
KNOWLEDGE/SKILL/TITLE/DEGREE and80train/20dev/20test. Annotator IDs are annotator1
and annotator2. The first labels all120documents; second has the proposed balanced
24-document independent overlap. All real documents start blank, without model
predictions, scores, existing Fit labels or synthetic review records.

Stopped only the identified demo server processes; demo manifest and review journal
remain untouched. Real server is now on127.0.0.1:8766. Live session API verification
confirmed is_demo=false, same campaign identity, expected categories and these
assignments: annotator1 train40CV/40JD,dev10/10,test10/10; annotator2 train8/8,
dev2/2,test2/2. Both had zero reviewed/draft/uncertain/skipped at initial verification.
Logs: logs/entity-annotation-real-create.log, entity-annotation-real-ui.log and
entity-annotation-real-ui-verify.log. No further implementation/test-code changes
were needed; no rerun of inference, training or the full test suite was required.
Updated A1 card and data catalog alongside this new local population record.
