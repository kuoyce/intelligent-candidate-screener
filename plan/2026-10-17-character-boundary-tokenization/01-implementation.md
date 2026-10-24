# Character-boundary tokenizer: implementation and actual alignment

17 October 2026. Status: tokenizer implemented and actual alignment verified;
native forward/reload, measured optimization cost, training, suggestion publication
and broad validation remain in progress. The plan is not complete.

## Frozen sources and historical blocker

The original v1 snapshot is retained unchanged. Its JSONL SHA-256 is
b43ae0af620080d028913cf6ff53bc912af8cc794d9e33e154a4403c8a0b84b8.
The original pinned lexical tokenizer represented 90 of 130 spans; 40 were
unrepresentable: 22 whitespace-only edge mismatches and 18 inside-word cuts.
Those were compatibility counts, not extraction precision/recall/F1. No subset
was silently used for training. See the original
[implementation record](../2026-10-16-entity-ten-example-finetune/01-implementation.md).

The new immutable latest-reviewed snapshot is under ignored data:
`data/processed/entity-annotation/finetune-snapshots/f5fadd3fda6068818192a3a8-ten-reviewed-v2/`.
It contains exactly ten primary-annotator entity-train documents (4 CVs, 6 JDs)
and 130 spans. Latest human revisions differ from v1; the export records those
revisions rather than overwriting the old snapshot. No source text, saved span or
review journal was modified by snapshot/tokenizer preparation. No entity dev/test
or independent annotator labels entered derivation.

V2 JSONL SHA-256:
6971d013d19ed0303974b10b4a3c88a822c284f886f23a894da4b599dfc6f920.

## Implemented frozen policy

`entity_annotation/character_tokenization.py` retains GLiNER's pinned base regex
and learns the union of required internal offsets for each exact lexical surface
from the ten training documents. There are 17 affected lexical surfaces. The
same text-only callable applies the frozen cuts globally at training and inference;
it has no per-document label input. Unseen lexical surfaces retain base splitting.
The private vocabulary stays under ignored data with model artifacts.

Every whitespace code point receives one source-offset unit and an encoder-safe
surrogate: space, tab, newline, carriagereturn or whitespace. Source text and gold
surfaces remain unchanged. Unit intervals partition the full original source
contiguously, including leading/trailing/repeated whitespace and non-BMP text.

The manifest freezes policy/version, lexical cuts, surrogate rules, training
source/revision/span digests, exact serialized snapshot SHA-256 and a canonical
snapshot fingerprint. Canonical manifest identity is verified on construction;
changed snapshots are rejected. Maximum gold width is derived only from training
labels: 25 new units. No gold offsets are trimmed, rounded or expanded.

Policy file: `tokenizer-policy.json` within snapshotv2. Policy SHA-256:
f349feb1e9da90d31c328c3ebaee27369800edf7bf2abce4e1a7cc0eb6114ec3.

## Actual pinned-runtime alignment

Loaded the verified local GLiNER medium checkpoint offline, derived the policy
using its actual base splitter, installed the frozen tokenizer and width geometry,
and prepared training examples using all four real descriptive label prompts.
The serialized snapshot digest matched the immutable export exactly.

Result: 10 documents, 130 gold spans, 130 supervised exactly once, zero
unalignable spans, zero width/budget violations and zero unrepresentable units.
There are 48 nonoverlapping gold-safe training windows. Effective encoder budget
is 368 prepared tokens including prompts/special tokens; GLiNER unit ceiling is
384. All positive spans remain within a single window; negative source regions
are retained. The former 90/40 result is historical; this is representability on
training data, not a claim of 100% extraction accuracy.

Local `alignment.json` stores native training records and original source-window
provenance. Aggregate log: `logs/entity-ten-character-alignment.log`. This step
performed no optimizer updates, training or suggestion publication.

## Focused validation

Twelve tokenizer/data tests passed, log
`logs/entity-character-tokenization-tests.log`: exact original offsets, leading/
trailing/repeated/Unicode whitespace, fused/internal-word cuts, global rule unions,
unseen surfaces, snapshot changes, manifest tampering, nonoverlap and negative
regions. A mutation replacing the new tokenizer with the old lexical policy
makes the exact-human-boundary regression fail, retained in
`logs/entity-character-tokenization-regression.log`.

## Remaining work

Verify expanded markerV0 width forward/backward, unchanged pretrained tensor
shapes and backbone weights, checkpoint save/reload and tokenizer parity. Time a
representative real optimizer group before launching the approved fixed pilot;
state estimates over one hour and run jobs over five minutes once in the background
with logs, exit marker and Darwin peak RSS. Complete guarded suggestion generation,
UI/provenance checks, full suite and relevant cold runner checks before reporting
completion. No independent entity accuracy result is available.
