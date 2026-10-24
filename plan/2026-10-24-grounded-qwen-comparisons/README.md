# v1 extractive QA and GLiNER comparisons

Status: authorized revised extractive implementation on 24 October 2026. Stay on
`feat/hiring-desk-v1-2026-10-23`; do not create branches. The latest user direction
replaces Qwen generation/fine-tuning with the existing RoBERTa-SQuAD2 model.
`05-extractive-implementation.md` is the governing specification; earlier
generative/pilot/training proposals remain historical and are superseded.

## Recommendation

Implement: authorized source -> cached GLiNER medium entities/context ->
RoBERTa-SQuAD2 extractive answers -> deterministic cited comparison table.
The primary chat and Compare candidates action share this evidence path. Preserve
CSV/live rankings, authentication and saved reviews. No Qwen fine-tuning,
generative comparison, larger-generator pilot or training is in current scope.

### Historical generative recommendation (superseded)

Replace the current excerpt-selector answer contract with grounded explanations
and a JD-requirement comparison matrix. Add the existing GLiNER medium runtime
as cached evidence enrichment, then evaluate whether it improves answers beyond
the improved retrieval/matrix alone. Benchmark one larger quantized Qwen model
only after the answer/evidence contract is working. The user's amendment adds
a bounded LoRA supervised fine-tuning stage for the existing Qwen checkpoint.

GLiNER can help locate and organize mentions; it neither upgrades Qwen's weights
nor supplies the comparison reasoning/answer interface. Its effect on this
chatbot must be measured. Running it on every message would add avoidable work.

## Historical diagnosis for the superseded generative proposal

| Constraint verified in this checkout | Consequence | Proposed change |
|---|---|---|
| `portal_qa.SYSTEM` requires evidence IDs and forbids paraphrasing | Even a better model cannot provide a normal explanatory answer under this contract | Permit bounded, supported synthesis with typed claims and citations |
| Regex intent/reference routing and lexical evidence selection | Natural paraphrases and multi-part questions remain brittle | Typed question plan, better passage retrieval and explicit clarification |
| Two excerpts per document; statements over 400 characters are skipped | Lists, long CV sections and multi-requirement comparisons lack coverage | Sentence/section windows and a requirement-by-candidate evidence budget |
| Primary portal mostly uses unverified aggregate table mentions | Useful extraction capabilities are outside the chat evidence path | Optional exact-span medium enrichment with source context/provenance |
| Local Qwen manifest is 0.5B, CPU float32, four threads, 4096 input/context and 256 output-token limits | Small capacity and short answer budget constrain synthesis; runtime cost needs separate profiling | Controlled model/runtime pilot, coordinated limits and bounded prompts |
| Passing two synthetic selections checked plumbing, not realistic answer usefulness | Previous integration success did not establish conversational quality | Frozen comparison scenarios, independent review and separate quality/latency metrics |

Source: `app/portal_qa.py`, `llm/local_server.py`, the local Qwen manifest and the
existing GLiNER medium implementation. These are observations, not newly measured
accuracy or latency claims.

## Historical delivery proposal (superseded)

1. Freeze answer behavior, evaluation cases and stage timings.
2. Build structured comparison answers with improved passage retrieval, using
   existing sources and Qwen first.
3. Integrate cached GLiNER medium spans and measure its incremental benefit.
4. Pilot Qwen2.5-1.5B-Instruct with an x86 CPU quantized runtime; retain the
   current 0.5B path for comparison/fallback. Promote only on measured benefit.
5. Review output quality, state isolation and cost; document the chosen setup.

The expected answer has a direct overview, a comparison table and an explanation
of supported differences/uncertainties, followed by expandable source excerpts.
Its facts remain tied to the active JD and selected CVs. Backend ranks/scores/Fit
decisions and the research winner remain outside the generator's authority.

## Historical approval scope (superseded by the current extractive scope)

The user authorized the new grounded-answer contract, cache/index/matrix,
pinned medium integration and Qwen fine-tuning in v1 on the current branch.
See `04-extractive-guide-and-sft.md` for the frozen initial training scope.
Cloud inference, new Fit labels, new entity-gold collection and ranking changes
remain excluded. Model
provisioning is an explicit preparation step, never app-startup training.

Preserve existing CSV reviews, the teammate's live-ranking adapter boundary,
authentication, saved reviews, alerts and local email flow. Commit the finalized
plan on the current branch before implementation. The user's explicit
same-branch instruction supersedes prior branch-creation text in this plan.

## Documents

- `01-design-spec.md`: evidence, comparison and generation contracts.
- `02-implementation-and-validation.md`: phases, controlled comparisons and gates.
- `03-decisions-and-sources.md`: proposed defaults, open questions and references.
- `04-extractive-guide-and-sft.md`: approved fine-tuning amendment and guide mapping.
- `05-extractive-implementation.md`: latest authorized implementation contract.
- `06-implementation-record.md`: delivered inference flow, review and retained validation evidence.
