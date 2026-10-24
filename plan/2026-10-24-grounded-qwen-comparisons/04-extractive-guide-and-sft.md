# v1 amendment: GLiNER comparison evidence and Qwen fine-tuning

Superseded by the user's subsequent extractive-only direction. No Qwen training
was started. See `05-extractive-implementation.md` for current authorization.

Authorized by the user on 24 October 2026. Keep the current branch. This amendment
supersedes the earlier exclusion of fine-tuning; it does not reopen annotation,
Fit labels, A2 final-set tuning or model training at application startup.

## Hugging Face extractive-QA guide mapping

The [linked guide](https://huggingface.co/docs/transformers/main/tasks/question_answering)
trains a start/end span predictor with `AutoModelForQuestionAnswering`. The
repository's separate RoBERTa QA adaptation already uses that objective, exact
offset mappings, context windows and no-answer targets. Hiring Desk's Qwen path
is a causal language model used for generation/selection, not that span head.

Apply the guide's source/answer alignment and held-out evaluation principles to
the evidence layer. For Qwen itself, use causal supervised instruction/response
training with loss on assistant tokens, preserving its chat template. LoRA is a
small-parameter adaptation, not a replacement of source grounding. Do not copy
the guide's random example split or upgrade the pinned runtime to Transformers
main; use document-family-disjoint splits and compatible pinned dependencies.

## Initial fine-tuning configuration

- Base: existing pinned Qwen2.5-0.5B-Instruct; preserve original weights/manifest.
- Method: CPU float32 LoRA, rank 8, alpha 16, q_proj/v_proj, dropout 0.05;
  seed 0, assistant-only loss, context 512 tokens, microbatch 1, accumulation 4,
  learning rate 0.0002, initially two epochs, four threads. Freeze this before
  seeing validation outcomes; no hyperparameter grid.
- Data: deterministically authored synthetic JD/CV evidence/claim dialogue
  families, separate train/dev/test identities and digests; no model-generated
  Fit labels, real applicant PII, A2-eval documents or final judgements.
- Targets: source-supported comparison explanations, uncertainty, qualifiers,
  candidate ownership and refusal to invent ranks/qualifications. No target
  answer can cite an absent or wrong-owner claim.
- Save adapters separately in ignored application model data with exact base,
  tokenizer, training data/config/code and runtime provenance. Retain a base
  model rollback. A trained candidate is not automatically the research winner.

Time one short pilot before full training, record steps/time/RSS and extrapolate.
Any long training job starts once in the background with exit marker/peak RSS;
end the turn and wait for completion, as required by repository instructions.
If estimated work exceeds an hour, communicate before starting it.

Evaluate base and adapted models on identical frozen dev/held-out cases and
grounding invariants before activation. Synthetic results establish integration
and controlled behavior, not broad real-world answer quality. If adaptation does
not improve supported answers, retain it as experimental and keep the validated
base/fallback rather than misrepresent success. The larger quantized generator
pilot remains conditional after the current-model adaptation/resource results.

## Implementation order

1. Add cached medium extraction and the criterion-by-candidate evidence matrix
   to Compare candidates, preserving existing CSV/live rankings and guards.
2. Connect grounded comparison answers to the same matrix and source references.
3. Implement and validate the isolated LoRA data/training/evaluation/export path.
4. Run the timing pilot; launch the bounded training job only after independent
   application work and checks are ready. Review/activate on completion.

References: [PEFT LoRA v0.14](https://huggingface.co/docs/peft/v0.14.0/package_reference/lora),
[Transformers causal LM recipe](https://huggingface.co/docs/transformers/v4.48.3/tasks/language_modeling).
