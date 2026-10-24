# Phased implementation and validation

## Phase 0 — freeze acceptance and establish baseline

After plan approval, create a branch, commit the plan and record the new grounded
answer contract in the decision log. This explicitly supersedes the current
excerpt-only chatbot behavior, while retaining rank/score/Fit boundaries.

Freeze 48 authored synthetic engineering questions across 24 independent JD/CV
families (12 development families and 12 held-out families, two questions each).
Cover broad/focused comparisons, factual/list questions, JD summaries, follow-ups,
late evidence, negation/ongoing/expired qualifications, skill-specific tenure,
missing/incomplete sources, conflicting statements and source injection. These
are authored tests, not human gold or evidence of general real-world accuracy.

Prepare a small separate redacted A2-train acceptance sample only after fixing
the case manifest/rubric. Keep its document families separate from development.
Use no final Fit labels/judgements or A2-eval documents for prompt/model tuning.
Independent user/team review rates answer usefulness/support; no Fit or entity
span collection is reopened, and no annotation LLM dispatch is used.

Instrument resolution, cache/load/extraction, passage search, matrix building,
prompt tokens, generation, validation and total time. Separate first-use costs
from warm turns; record peak process RSS and queue time. Review existing real
records before running any new timing job. Time one representative configuration
before deciding the remaining experiment budget.

## Phase 1 — answer/matrix and retrieval, without GLiNER

Implement the typed question plan, candidate/requirement evidence matrix and
grounded answer interface. Reuse existing sources and passage-index contracts.
Add per-criterion evidence coverage and readable explanations. Keep CSV and live
source adapters independent. Do not silently reinterpret unverified CSV fields
as source proof. Preserve bounded history and job-switch isolation.

Acceptance: the user receives an actual comparison and direct answer, with
qualifiers/uncertainties preserved, not merely a pair of quoted passages. A
specific skill/degree/years question gets relevant evidence or a useful unknown.
The system cannot mutate ranks/scores or claim a credential is verified.

## Phase 2 — optional GLiNER medium enrichment

Implement a managed worker/cache adapter around the pinned medium runtime,
source validation and entity-anchored passage retrieval. Protect concurrent
requests, hard worker failure, partial coverage and cleanup. Precompute/cache the
selected sources; source-only comparison remains usable while extraction waits.

Acceptance: repeated questions on unchanged documents invoke no new extraction;
changed text/model/prompts invalidate the enrichment. All span ownership/offset
invariants pass. A failed/incomplete run cannot produce a false 'no skill' result.
Certification and other unconfigured categories retain their text evidence path.

## Phase 3 — generator/runtime experiment

Profile the improved answer path on 0.5B before provisioning a larger model.
Then pin/license/document one 1.5B quantized candidate, update runtime capability
validation and run a small smoke test before the frozen cases. Prefer matching
quantization/backend when comparing model sizes; report confounded comparisons
as configurations rather than attributing all change to model size.

Measure quality, generation speed and memory. No promotion based solely on a
synthetic happy-path answer or model-card claims. If the larger model is too slow,
retain a fast validated matrix and a shorter/lighter synthesis configuration.
No fine-tuning is included. A future training proposal requires a separate
approved data/split/resource plan after retrieval/format limitations are resolved.

## Controlled comparisons

| Configuration | Purpose |
|---|---|
| G0: current quote-selector + current 0.5B | Preserve the actual baseline |
| G1: improved passages/matrix/synthesis + same 0.5B | Measure the answer-contract/retrieval improvement |
| G2: G1 + cached medium entity anchors | Isolate the incremental GLiNER contribution |
| G3: G2 + candidate generator/runtime | Assess the capacity/runtime tradeoff |

G1 vs G2 uses the same generator, prompt budgets, documents and questions. G3
model-size comparisons require a matched runtime control where feasible. Run
experiments sequentially after timing one configuration; no unbounded model grid.
Do not rerun failed generations merely to choose a pleasing answer. Retain the
first-run outcomes and label any changed-configuration retest.

## Promotion gates and metrics

Hard gates: all owner/snapshot/citation/number/negation/credential/incomplete-
coverage invariants pass; no invented ranks, Fit calls, tool actions or credentials;
all requested comparison candidates are represented or explicitly unavailable.

Use a frozen rubric for relevance, answer completeness, useful comparison,
supported claims, qualifier preservation and honest uncertainty. Review both
matrix and prose. Record model acceptance vs fallback separately so template
success is not reported as generator success. GLiNER is promoted only if its
paired answer-quality/coverage benefit justifies its memory and first-use cost
without increasing unsupported claims. Report inconclusive results honestly.

Proposed balanced UX targets: a cached evidence matrix within one second and a
short validated answer within ten seconds at p95 on the tested local workload.
These are acceptance targets, not measured promises. Cold extraction/model load
is reported separately with visible progress, never hidden inside a warm metric.
If they are infeasible, adjust the model/budget or obtain agreement on the
tradeoff rather than claiming the target was met.

Report n and uncertainty with quality/latency summaries, paired by JD/query
family. Cluster bootstrap over independent JD families, not candidate pairs or
individual claims. For real A2 samples, show generic/targeted/total together. No
ranking effectiveness result or entity F1 is claimed from these QA scenarios.
Do not imply complete source processing means complete/correct extraction.

## Review, testing and long jobs

Use scoped sub-agents for evidence/cache, comparison/synthesis and runtime/eval
after agreeing interfaces. An independent reviewer checks qualifier preservation,
claim support, migrations, state isolation and model-failure behavior.

Show regression tests failing against the guarded behavior, iterate narrowly,
then run the full `uv run pytest` and affected records through the checks runner.
The current checkout has two previously observed baseline golden failures;
record them separately rather than alter frozen model records to hide them.
Follow D55 for any model/training commit, final scoring or plan completion. Cold
requirements must be scoped/scheduled explicitly; this plan does not waive them.

All jobs estimated over five minutes start once in the background with a retained
log, exit marker and peak RSS, then end the turn for completion notification.
Estimate the full workload from the first configuration and disclose totals over
an hour before starting. Do not launch heavy model jobs alongside each other.

Update README, decisions, operational data documentation and implementation
records. Keep applicant text, entity caches and generated evidence private and
git-ignored. Preserve the existing account expiry, CSV reviews, live-ranking
interface, alerts and mail procedure.
