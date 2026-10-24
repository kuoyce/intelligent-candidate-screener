# Proposed choices, open questions and sources

## Defaults for approval

- Balanced local answer quality and response time. An optional preference question
  has been presented; no response has yet changed this default.
- Grounded synthesis and a requirement matrix are the first changes.
- GLiNER uses existing pinned medium weights and five configured labels; it is
  cached enrichment with a measured adoption gate, not a per-turn dependency.
- Start with one quantized Qwen2.5-1.5B-Instruct candidate after the retrieval/
  matrix path works. Quantization/backend choice is a measured pilot, not an
  assertion that this model will meet local quality/latency targets.
- Keep broader passage text alongside entities; no skill-gap deductions from
  missing extraction. No new certification label or relation model by default.
- No fine-tuning, new entity/fit gold, cloud inference or ranking changes.

## Open questions

| Question | Proposed resolution | Status |
|---|---|---|
| May chat move beyond exact quote selection? | Permit supported paraphrasing and comparisons with a typed evidence ledger/validation; decision log amendment after approval | Approval required |
| Quality vs response-time priority? | Balanced; matrix-first UI and short synthesis, with declared latency targets | Optional preference pending |
| Is GLiNER worth its cost? | Compare G1/G2 with the same generator/cases; adopt only on meaningful benefit | Measurement after approval |
| Which generator/runtime meets the targets? | Current 0.5B baseline plus one 1.5B quantized CPU pilot | Measurement after approval |
| Who reviews realism/usefulness on the frozen acceptance sample? | User/team reviewer(s); freeze rubric/sample first; no messages sent by the agent | Confirm during evaluation setup |
| Future fine-tuning? | Separate proposal only if audited retrieval and stronger prompted inference still fail | Out of current scope |

## Local evidence

- `src/candidate_screener/app/portal_qa.py`: explicit selector-only system prompt,
  two excerpts per source, 400-character cutoff, exact-quote validation.
- `src/candidate_screener/llm/local_server.py` and local Qwen manifest: CPU float32,
  four threads, 4096 context and 256 output caps; single serialized HTTP inference.
- `extraction/medium_assets.py`: pinned medium identity/revision/digests.
- `extraction/medium_runtime.py`, `medium_worker.py`, `medium_inference.py`:
  offline isolated process, exact field-relative spans and coverage diagnostics.
- `extraction/transformer.py`, `schema.py`, `rules.py`: five-label mapping and
  bounded negation/learning flags, not a full credential/skill-duration resolver.
- `app/rag_evidence.py`, `chat_schema.py`: reusable passage/ownership/citation
  contracts from prior approved RAG work. These are not fully wired into the
  primary portal's current quote-selector path.
- `plan/2026-10-14-gliner-medium-sentence-chunking/01-implementation.md`:
  historical four-document pilot shows nontrivial first-use/extraction costs.
  Its recorded timings have no CI and are exploratory operational context, not a
  new benchmark or promise. Do not replay the whole campaign to plan a chat cache.

## External primary references checked 24 October 2026

- [Pinned model family: gliner-community/gliner_medium-v2.5](https://huggingface.co/gliner-community/gliner_medium-v2.5).
- [GLiNER paper, NAACL 2024](https://aclanthology.org/2024.naacl-long.300/): entity
  extraction architecture/task; it does not establish this chatbot's quality.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and
  [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct):
  published instruction-model candidates, not local quality guarantees.
- [Official Qwen llama.cpp guide](https://qwen.readthedocs.io/en/v2.5/run_locally/llama.cpp.html):
  x86 CPU and quantized GGUF inference options. Custom tokenizer/attestation
  adapter integration still needs implementation and local measurement.
