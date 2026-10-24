# Ten-example adaptation: implementation and alignment blocker

Historical blocker record. The approved17October2026
[tokenizer amendment](../2026-10-17-character-boundary-tokenization/01-implementation.md)
preserves this snapshot and its90/40result, while a new latest-reviewed snapshot
with frozen training-derived units represents130/130spans exactly. That resolves
tokenizer compatibility; native timing/training and full delivery remain pending.

17October2026. Status: implemented/tested training guards and read-only boundary
inspection; real training and suggestion generation blocked by human/token alignment.
Plan approved and committed before implementation: b30ae4b, branch
feat/entity-ten-example-finetune-2026-10-16. Work delegated across snapshot/alignment,
native trainer and suggestion storage/UI, with independent review. The root finished
UI integration after a delegated tool call remained approval-pending and was aborted.

## Verified human snapshot

Immutable local snapshot:
data/processed/entity-annotation/finetune-snapshots/f5fadd3fda6068818192a3a8-ten-reviewed-v1.
Ten completely reviewed annotator1 entity-train documents,4CV/6JD;130spans:
80KNOWLEDGE,30SKILL,11DEGREE,9TITLE. No empty reviewed documents. JSONL SHA256
b43ae0af620080d028913cf6ff53bc912af8cc794d9e33e154a4403c8a0b84b8.
This is single-annotator, unassisted training annotation, not held-out accuracy gold.
Source revisions/digests and guide version are frozen. Existing reviews are unchanged.

## Alignment result and implication

The actual pinned GLiNER medium tokenizer aligns90spans exactly.40cannot map to
its word boundaries:22whitespace-only edge mismatches,18inside-word cuts.
There are no width/budget violations (max_width12; effective368encoder tokens).
All training records are withheld when any span fails: the90representable spans
were not used as a silently reduced dataset. No real pilot or human-data training
has run. Logs: entity-ten-alignment.log, entity-ten-alignment-review.log and
entity-ten-boundary-review.log under logs/.

Some cuts are incomplete selections (e.g. a missing final letter); others are
correct entities inside fused source tokens, such as Excel within HighlightsExcel.
Do not expand correct gold to an unwanted heading merely to fit tokenization.
Whitespace/selection errors need human correction; valid inside-token entities
may instead require a separately specified tokenizer/preprocessing amendment.

Nonoverlapping gold-safe training windows preserve all representable source and
supervise each positive exactly once. This implements the plan's no-duplicate/no-
partial-positive requirement without overlap masking. There were27candidate windows,
but none were passed to training because of the global alignment blocker.

## Native training verification (synthetic only)

Implemented a versioned native GLiNER0.2.21 collator/PyTorch loop, avoiding unsupported
newer train_model APIs. Backbone token_rep_layer is frozen/eval;24representation
parameter tensors train and198backbone tensors remain frozen. Configuration uses
the approved3epochs/batch1/accumulation4/AdamW1e-4/weight-decay.01/norm1/seed0/CPU4,
with explicit checkpoint loss alpha.8,gamma0,smoothing0,sum reduction and label masking.
Head-only checkpoints verify immutable base/snapshot/runtime/code/artifact identities.

First synthetic smoke exposed GLiNER's shared-class-list training-label bug
(KeyError0). Passing per-example [PROMPTS] corrected it. The final synthetic smoke
passed one optimizer step, changed representations, verified an unchanged backbone,
saved/reloaded every state tensor exactly, and took1.03s for the optimization loop.
This is not an estimate of real-document training time and no model trained on
human labels exists. Synthetic checkpoint ID4f384ef7d3fa0fcc124dcc82 is marked
snapshot synthetic-only and was never published to the UI. Logs:
entity-finetune-synthetic-smoke.log (failed compatibility probe),
entity-finetune-synthetic-smoke-final.log and entity-finetune-smoke-summary.log.

## Safe inspection and assistance integration

The live local editor includes /boundary-review?annotator=annotator1:40escaped,
redacted, bounded cases with visible whitespace, tentative token coverage and
links to exact assigned documents. Stale source/revision cases withhold proposals.
No automatic or bulk correction exists. Annotator2 cannot see the report. Correct
labels should remain unchanged even where the current tokenizer cannot represent
them. Existing editor reselect/save creates ordinary new human revisions.

Suggestion artifacts are separate from reviews, with campaign/model/snapshot/source
identity, raw predictions and deterministic display-overlap filtering. No real set
is published because training is blocked. Once available, suggestions can explicitly
load only on untouched primary-annotator training docs. Save preserves assistance
provenance even after deleting all suggestions or omitting it from a later request.
Dev/test, independent annotator, saved drafts and prior reviews cannot be replaced.
Exports preserve annotation_mode; blind accuracy scoring rejects assisted rows.

Storage publication currently trusts the authorized generator to verify its model/
snapshot IDs and can leave a partial unpublished manifest after interrupted writes.
Before real generation, add the verified-model orchestration/atomic-manifest
publication checks; never publish the synthetic checkpoint. Real generation remains
pending with real training, not advertised as a completed model delivery.

## Verification

Focused data/trainer/provenance/boundary tests:36passed, logs/entity-finetune-focused.log.
HTTP/browser/service/boundary subset after editor integration:18passed,
logs/entity-assistance-browser-final.log. Live read-only Chrome check passed:
40report links, correct document deep-link, independent annotator denied and the
human journal SHA256 unchanged across the browser session. Log:
logs/entity-boundary-live-browser.log. No real labels were saved by tests.

Full suite once:1,113passed,10skipped,2failed in165.86s,
logs/entity-ten-finetune-full-pytest.log. Both failures are the already documented
frozen TF-IDF baseline reproduction/environment checks, not suppressed or rewritten.
The optional browser test passed separately with ephemeral Playwright/headless Chrome.
Cold import-closure runner selected zero existing registered checks for the isolated
package, logs/entity-ten-finetune-cold.log. No final-set scoring occurred.
The plan is not complete: real training is explicitly blocked.

## Next required input

Review the40boundary cases, correcting only genuine annotation errors. Do not
force valid entities into incorrect whole-token labels. Then freeze a new snapshot
revision, rerun exact alignment and resolve any remaining tokenizer incompatibility
before the already approved timed real pilot. Existing snapshot/reviews and pretrained
models are retained. No new permission to repeat the approved experiment is needed,
but changing its tokenizer/preprocessing policy requires an explicit design amendment.
