# Torch-free local D63 inference

25 October2026. Approved by the owner's “okay proceed” after the ONNX proposal.
Continue on `feat/hiring-desk-v2-conversation`; preserve unrelated working changes.

## Contract

Keep frozen L1, F12, role-family/TF-IDF/concept parts, text preprocessing, embedding
dimension, cosine computation and pool scaling. Replace only SBERT execution for
the opt-in local backend. Export existing trained weights; do not train or tune.
Research/default Torch paths and scores remain unchanged. D64 is explicitly
unavailable for the ONNX backend; never silently substitute D63 for a D64 request.

## Sequence and ownership

1. Parent inspects saved checkpoint and installs CPU ONNX/tokenizer dependencies
   in a separate dependency group, without installing the incompatible `ml` group.
2. Parent exports the frozen transformer using the already-provisioned isolated
   Torch2.2.2/Transformers4.48.3 runtime if compatible. If not, produce a reviewed
   Colab export path rather than retraining or modifying source weights.
3. Runtime agent builds CPU ONNX + standalone tokenizer inference. Preserve mean
   pooling over attention-masked tokens, chunk-level L2 normalization, 800-character
   whitespace-normalized chunks, mean across chunks, then document L2 normalization.
   Use a separate ONNX cache namespace until verification; never overwrite original
   SBERT caches with an unverified backend.
4. Integration agent wires `Screener.load(embedding_backend='onnx')`, configured
   portal adapter and D64 disabled control. Default remains `torch` for research.
5. Parent verifies unseen invented texts (short, long, Unicode, blank), reference
   embeddings and cosine similarities against Torch before publishing validation.
   Register atol1e-5/rtol1e-4 for embeddings before results; no loosening after looking.
   Record source/export digests and both runtime versions. Failure blocks promotion.
6. Independent review, regression proofs, full tests and central affected-check
   selection. Request approval before long/cold research checks as required.
7. Test D63 ranking an invented JD through the actual adapter and a temporary local
   portal database with an authorized fixed pool. Validate ready snapshot, top10,
   source ownership and retry after simulated missing assets. No real JD is submitted
   or modified without its explicit existing UI request. Report runtime readiness.

## Artifacts and boundaries

Ignored export: `data/processed/inference/sbert-onnx/`, including model, tokenizer,
reference fixture and source/runtime manifest. Canonical source checkpoint remains
`data/processed/matching/sbert-weak-3255167debe9/`. Model bytes stay ignored.
No eval labels, new Fit judgements, new training data or final-set score enters
this conversion. Existing embedding-cache comparisons are numerical regression
checks rather than a new research benchmark or tuning set.

Proposed dependency group `inference` contains CPU ONNX Runtime and standalone
tokenizers. All installation uses uv. Main application continues without importing
Torch/sentence-transformers in its ONNX ranking path; other isolated QA/chat
workers retain their independent existing runtimes.

## Acceptance and open questions

- Exact configured tokenizer special tokens, truncation/padding and maximum length
  must be established from checkpoint files, not assumed from a model card.
- No ONNX inference activation before parity passes; missing/incompatible assets
  remain explicit unavailable states.
- A same-model numerical export can have float noise; compare embeddings/cosines
  and synthetic ordering without claiming bitwise equality or unchanged ties.
- The Intel Mac CPU wheel must be verified; do not change unrelated global package
  constraints merely to make the incompatible Torch ML group install.
- No blocking scope question: export locally if supported, otherwise Colab remains
  the approved alternative. New training and D64 conversion are outside this task.

Implementation and measurements will be appended below. Plan not complete yet.
