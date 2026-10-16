# Preserve human character boundaries with a frozen training-derived tokenizer

17October2026. User explicitly requested resuming adaptation of training
tokenization to preserve their character boundaries. This approves the scope
amendment to the already approved ten-example pilot; no further discretionary
implementation approval is required. Real labels/source text remain unchanged.

## Design

Keep the pretrained encoder and original lexical splitter as the base. From only
the ten human entity-train documents, learn a global vocabulary mapping each
affected lexical token surface to the union of internal character cut positions
needed by human start/end offsets. Apply these frozen word-surface rules globally
and identically in training and inference; never use per-document gold at inference.
Unseen lexical tokens retain the original splitter. No claim is made that every
unseen internal boundary is representable. This is a trained tokenizer component,
not an oracle and not manual label correction. Its vocabulary may contain private
training words and stays under ignored data with model weights.

Represent every whitespace code point as a separate source-offset unit with a
stable encoder-safe surrogate word (space/tab/newline/carriagereturn/whitespace).
Surrogates are model inputs only: text, labels, surfaces and source offsets are not
rewritten. Thus leading/trailing/partial whitespace selections remain exact,
including the22previously incompatible spans. Lexical cuts handle the18inside-word
selections, including correct Excel within HighlightsExcel and user-selected
partial-word endings, without guessing intended labels or expanding their text.

Build a complete contiguous unit partition of each source and verify every unit's
original offsets. Keep original word text for unsplit nonwhitespace units. Every
gold boundary must map exactly; zero whitespace trimming, dropping, snapping or
rounding is allowed. Persist source/snapshot digest, cut rules, surrogate policy,
version and canonical SHA256. The tokenizer's runtime interface accepts text only.

Derive the required maximum span width in the new units from training gold only,
freeze it in tokenizer/model provenance and restore it at model load. The pinned
markerV0 span head's width affects reshape/enumeration, not learned tensor shapes;
verify unchanged state keys/shapes and synthetic forward/reload before training.
Do not alter base artifacts. Check real encoder budgets including all four label
prompts and special tokens; existing sentence/gold-safe windows must cover each
source and supervise each of the130human spans exactly once. If a true encoder
budget limit remains, report it concretely instead of silently dropping labels.

## Integration and execution

Create branch/commit this amendment before implementation; delegate tokenizer/data,
training/checkpoint adaptation and inference/UI integration with independent review.
Freeze a new immutable latest-ten-reviewed training snapshot, preserving the prior
snapshot and journal. If the number is no longer ten, report the inventory rather
than silently widening the training population.

Update training/checkpoint manifest with tokenizer policy/digest and span width.
Load and verify that exact tokenizer for every synthetic smoke, real pilot, saved
checkpoint reload and suggestion inference. Keep fixed training configuration from
the prior approved pilot. Time one representative real optimizer group before full
training. Communicate an estimate above1hour before launch; jobs above5minutes run
once tracked/background with log, exit marker and Darwin time-l RSS, then end turn.

After a verified real checkpoint exists, generate suggestions only for remaining
unreviewed primary-annotator training documents, using the four prompts and
threshold.5. Verify model/snapshot/campaign/source identities before atomic publish;
never publish a synthetic checkpoint. Incomplete inference remains explicit and
cannot be presented as completed-zero. Recheck latest human review state before
publishing/loading. Preserve assisted provenance, saved drafts/reviews, original
labels and all baseline weights. Dev/test and annotator2 stay blind. No model
training or remote downloads at UI startup.

## Validation and records

Synthetic tests: leading/trailing/repeated whitespace, fused headings, internal-word
cuts, Unicode/non-BMP, rule unions across documents, unseen-word behavior, changed
snapshot rejection and no inference access to gold. Show the previous tokenization
fails the exact-boundary regression, then new policy passes. Confirm all actual
human labels align exactly and are supervised once. Validate markerV0 expanded
width forward/backward/save/reload, unchanged encoder weights and tokenizer parity.
Test suggestion publication provenance, no test/independent exposure, no automatic
gold saves and atomic set publication. Run focused tests while iterating, full suite
once and relevant runner cold checks. Retain known unrelated baseline failures.

Update previous pilot blocker record, A1 card, data catalog, README and decision log
in the same eventual change set. Store only aggregate counts/digests in committed
records. No accuracy improvement claim from ten training documents or training loss.

## Open questions

C1 resolved: user authorizes tokenizer adaptation while preserving their labels.
C2 implementation check: represent every saved boundary and actual prompt/token
budget under the frozen tokenizer; no implicit fallback to gold-free base rules.
C3 implementation check: measure expanded-span compute/RAM and full job estimate.
C4 fixed limitation: unseen-token internal cuts may remain unrepresentable; measure
on later independent gold rather than treating this pilot as proven accuracy.
