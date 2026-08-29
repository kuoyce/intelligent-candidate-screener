# Phase 5 — Unified Judging Wave

**Date:** 29 August 2026
**Status:** **Revised 29 August 2026 before approval — D25–D27 below narrow it.** The
judging wave is deferred; what remains on the critical path is one annotation session and
two code tasks, both now implemented (`bfafc9c`, `0b1d81e`).
**Predecessors:** [`plan/2026-08-23-derived-artefacts/`](../2026-08-23-derived-artefacts/) (task 3.4, Q14) ·
[`plan/2026-08-29-pool-precision-bias/`](../2026-08-29-pool-precision-bias/README.md) (Q18)
**Scope:** One annotation instrument serving both objectives — in-domain pair rating and
pooled top-k judging — across both corpora. No modelling.

## Contents

| File | Purpose |
|---|---|
| [`01-design-spec.md`](01-design-spec.md) | The judgement layer, the overlay, the dual-reporting policy, acceptance checks |
| [`02-work-plan.md`](02-work-plan.md) | Sequenced tasks, what runs now vs. after Stage 4 |

## Why this phase exists

Two planned pieces of work turned out to be the same physical act — a person reading a JD and
a CV and answering a 3-way question:

| | Origin | Why we ask | Which pairs are selected |
|---|---|---|---|
| **Objective A** | Task 3.4 | Build the in-domain evaluation set | A representative, banded sample |
| **Objective B** | Q18 step 4 · Q14 wave 2 | Judge what a system actually surfaced | The union of every system's top-k |

**D17 already made the label identical** — A1's 3-class scheme doubles as nDCG's graded
relevance. So the two objectives differ only in *selection*, and one instrument can serve both
if it records why each pair was selected. This plan is that instrument.

It supersedes the wave-2 half of [`02-work-plan.md` §Q14](../2026-08-23-derived-artefacts/02-work-plan.md#q14--pool-depth-and-whether-to-collect-rankings)
and step 4 of [the Q18 analysis](../2026-08-29-pool-precision-bias/README.md#proposed-path);
both now point here rather than each carrying half a design. Wave 1 of task 3.4b is
**unchanged** and is not owned by this plan.

## Decisions

| # | Decision | Closes | Consequence |
|---|---|---|---|
| **D20** | Our judgements are a **separate committed layer**, never merged into `pools.csv`. Every A1 figure is reported twice: *A1-only* and *A1+ours* | **Q28** | Confirmed by a mechanism the question did not mention: `pools.csv` is byte-reproducible from seed 0 and `verify --derived` asserts it. Human labels are not a function of a seed, so they **could not** have lived there without destroying that check. See [`01-design-spec.md` §2](01-design-spec.md#2-the-judgement-layer) |
| **D21** | Wave 2 covers all **64 graded-scoreable** queries, not the 31 strict | **Q29** | The 31 strict-scoreable queries are a subset of the 64 (a query with a `Good Fit` necessarily has one under `graded`), so this subsumes rather than replaces. Queue roughly doubles: 320 top-5 slots per system, not 155 |
| **D22** | **One wave, run once after the last stage exists.** No incremental per-stage judging | **Q30** | No stage is scored against an answer key an earlier stage helped write. **Cost, stated for the record:** every Stage 1–3 precision figure is provisional until the wave lands, and will be superseded upward. §"What D22 costs" below |
| **D25** | *New, and the one that changes this plan's shape.* Manual annotation is capped at **one session, ~250 judgements, ~2.5 team-days**: task 3.4b's 200 in-domain pairs plus ~50 blind A1 recheck pairs. The judging wave leaves the critical path | **The budget question** | Supersedes the *scope* of D22 and D23 — their reasoning stands on record, their schedule does not. See §"Why the wave was cut" |
| **D26** | *New.* **Q18 closes as a standing limitation.** The attainable precision ceiling is recorded in code and reported beside every Precision@k, raw and normalised both; Recall@10 is the headline retrieval metric | **Q18, Q27**; dissolves **Q26** | One of the three closes the Q18 analysis itself listed. Implemented: `metrics.attainable_ceiling`, `metrics.ceilings`, emitted into `pools-yield.json` |
| **D27** | *New.* The retrieval scoring pass over the 3.3 pools runs **now** | **Q24** | Implemented as `evaluation.retrieval`. It cost no annotator time and had been deferred twice. §"What Q24 actually returned" |
| ~~**D23**~~ | *Superseded in scope by D25, 29 Aug 2026.* The wave covers **all 100 test queries**, including the 36 that are currently unscoreable. Widening *n* is a first-class objective of the wave, not a byproduct | **Q31** | Supersedes D21's 64 within the same wave — D21's reasoning (graded, not strict) still governs the *relevance definition*; D23 governs *query coverage*. Upper bound rises to **500 top-5 slots per system** from 320. §"What D23 buys" below |

## What D22 costs, and what it changes about Q26

Two consequences the decision carries that were not in the question:

**1. Every interim precision figure is provisional.** Stages 1–3 will be reported, discussed and
possibly acted on before the wave runs. Each must carry the ceiling (task 5.2) and an explicit
"provisional, pending the judging wave" line, and each will be **superseded in place** when the
wave lands — the repo's *Superseded {date}* convention, not a silent edit. There is no version of
D22 where the interim numbers are final.

**2. Q26 no longer means what it meant.** Q26 asked for a pre-registered contamination threshold
*X* — the decision rule for **whether to run a judging wave**. D20–D22 commit to running it. The
gate Q26 was guarding is gone.

Q26 does not disappear, it changes job: *X* is now the threshold above which the pre-wave A1
figures are declared **unusable rather than merely provisional** — i.e. whether Stage 1–3 interim
results get retracted or just revised. That still has to be fixed before the wave's output is
seen, for the reason originally given. **Q26 is re-scoped, not closed** — see below.

## Open questions

| # | Question | Status | Blocks |
|---|---|---|---|
| ~~Q24~~ | Score an existing system over the 3.3 pools | **Closed 29 Aug 2026 by D27** — run, recorded in `output/baselines/retrieval-metrics.json` | — |
| ~~Q26~~ | What contamination rate makes the pre-wave figures retractable? | **Dissolved 29 Aug 2026 by D26.** Q26 gated *whether to run a wave*; D22 removed that gate and D25 removes the wave. Under a standing limitation there are no "pre-wave" figures to retract — the figures are final and carry the limitation | — |
| ~~Q27~~ | Precision@k raw, ceiling-normalised, or both — and does the ceiling enter `pools-yield.json`? | **Closed 29 Aug 2026 by D26: both, and yes.** `retrieval-metrics.json` carries `value`, `ceiling` and `pct_of_attainable` on every precision row; `pools-yield.json` carries the ceilings and the unwinnable-slot arithmetic | — |

**No open questions remain on the evaluation side.** The only one still live anywhere in
Phase 3 is Q12 (the length gap), whose recommendation is written and which blocks nothing.

## Why the wave was cut *(D25)*

Not because Q18 was wrong. Because of what it costs and what it is measured against.

**1. Nothing the project is graded on is Precision@k.** The proposal's success measures (§7)
are Extraction F1, **Recall@10 > 0.80 / > 0.90**, **Macro-F1 > 0.85**, and ≥85% in-domain
agreement over 200+ pairs. Precision@k appears once in the entire proposal, in §4's table of
candidate techniques, and never as a target. The judged-supply ceiling is a precision artefact
by construction — it exists because some queries have fewer than *k* relevant documents, and
`pools.attainability` now measures Recall@10's ceiling at **0.983**, with 93.5% of queries able
to reach 0.90. Macro-F1 is computed on labelled pairs, where no distractor appears at all.

**2. The arithmetic against the budget.** The proposal allots ~30 team-days total (§8). The
plan as approved would have spent **6–8.5 team-days on annotation** — 20–28% of the project —
and D22 scheduled the largest piece after Stage 4 froze, landing it in weeks 8–11 alongside the
conversational MVP (4 days) and final testing (3 days). D25 spends **~2.5 team-days**, before
Stage 2 starts, and returns 3.5–6 team-days to the stages the research question is about.

**3. The pattern this plan was part of.** Commits `ae08d28` and `96cbe07` produced ~590 lines
of plan, four decisions and three files, and moved the open-question count from 1 to 3. They
produced no labelled pair and no annotation code, while the guide Phase 3 called *"the real
blocker"* stayed unwritten. Each analysis was individually right. The loop was not.

**What is kept.** D20 (separate layer) and D21 (graded definition) stand — D20's reasoning was
mechanical, not preferential. The judgement schema is written now, before any label exists, and
`annotation/queue.py` records `selection_reason` per pair, so a later wave appends to the same
layer rather than migrating it. The wave is deferred to an optional week-11 batch of 100–150
pairs, not cancelled, and D22/D23's reasoning is on record for whoever runs it.

## What Q24 actually returned *(D27)*

Run at last, and it does **not** support the comfortable reading.

Of TF-IDF's 500 top-5 slots over the N100 pools, **416 (83.2%) are unjudged distractors**.
The bias Q18 raised is *large*, not negligible, and the raw Precision@5 of 0.142 is a floor
rather than an estimate.

But the question actually in doubt was whether the bias **compresses or inverts the ranking**
between systems — Q18's argument was that the penalty grows with quality, because a better
system surfaces more relevant-but-unjudged documents. Measured, the unjudged share falls
monotonically as the system improves:

| | Recall@10 strict | P@5 strict | % of attainable | unjudged share of top-5 |
|---|---|---|---|---|
| TF-IDF | **0.298** [0.216, 0.391] | 0.142 | 18% | **83.2%** [78.6, 87.4] |
| BM25 | 0.196 [0.125, 0.279] | 0.103 | 13% | 86.6% [82.6, 90.2] |
| Random floor | 0.128 [0.061, 0.212] | 0.071 | 9% | 94.0% [91.8, 96.0] |

A better system puts **more** judged documents at the top, not fewer. The mechanism that would
have made the pools unfit for comparison is not present in the observed direction — which is
the evidential basis for D26 that the Q18 analysis, reasoning from argument alone, could not
have had.

**This is exposure, not contamination.** How many of the 416 are genuinely relevant still
requires a human, and D26 is the decision not to find out. TF-IDF's Recall@10 of 0.298 against
a 0.80 target is also worth stating plainly: the Stage 1 baseline is far from the proposal's
bar, which is a result, not a problem to tune away before reporting.

## What D23 bought, and what it risked *(retained; superseded in scope by D25)*

D21 sets the *relevance definition*; **D23 sets the query coverage to all 100.** The 36 additional
queries are currently excluded from every figure — they carry no judged relevant document at all,
so `metrics.score` drops them as undefined, not zero.

If a system surfaces a genuinely relevant resume for one of those queries and a human marks it
`Good` or `Potential`, **that query becomes scoreable**, and *n* rises above 64.

**Why this is worth the extra ~180 judgements.** Phase 3 recorded its own verdict twice over: *"at
n = 31 the interval is the finding"*, and the measured random-ranker Recall@10 CI spans a factor of
three. Nothing tightens a bootstrap CI like more queries — the precision correction the wave was
originally built for does not touch the interval width at all. D23 is the only part of this plan
that attacks the constraint Phase 3 named as binding.

**The risk, stated plainly.** These are the queries A1's own labellers found nothing relevant for.
The hit rate may be low, and a low hit rate means judgements spent for no gain in *n*. That is a
real possibility and it is accepted knowingly: **the ~180 judgements are spent before the return is
known**, and there is no cheaper probe — a query only becomes scoreable once a human has said so.

Two consequences that must not be lost:

1. ***n* becomes view-dependent.** A1-only keeps *n* = 64 (31 strict); A1+ours may be higher. A
   figure carrying *n* from one view and a value from the other is wrong, and `Figure` cannot catch
   it — it validates that *n* exists, not that it matches. Task 5.7 owns this.
2. **The newly-scoreable queries are a biased subset** — they became scoreable *because* a system
   surfaced something relevant. Reporting A1+ours over all 100 mixes queries whose relevant set was
   found independently with queries whose relevant set was found by the systems under test. Task
   5.7 reports the newly-scoreable queries as a **named stratum**, never silently pooled.

## Progress

| Task | Status |
|---|---|
| 5.1 judgement schema | **Done** 29 Aug 2026 — `judgements.csv` header committed; the overlay `layer.apply` was **dropped**, since the A1 recheck compares against labels that already exist and changes no relevance. Nothing consumes an overlay until a wave runs |
| 5.2 attainable ceiling | **Done** 29 Aug 2026 — `metrics.attainable_ceiling`, all four figures pinned, depth-invariance asserted, emitted into `pools-yield.json` |
| 5.3 annotation guide | **Done** 29 Aug 2026 — `docs/annotation-guide.md`, covering both corpora (A16) |
| 5.4 A1 recheck | **Queue built** 29 Aug 2026 — 50 pairs stratified 15/15/20 across A1's classes, blind in the session queue. Awaits the session |
| 5.5 queue builder | **Done** 29 Aug 2026 — `annotation/queue.py`. Sized at 250, not at the wave's 500-per-system union |
| 5.6 the session | **Not started** — ~2.5 team-days, gated on people, not on Stage 4 (D25 replaces D22's ordering) |
| 5.7 re-report | **Not started** — κ per corpus, the A13 result, and the limitation line |
| ~~wave 2~~ | **Deferred by D25** to an optional week-11 batch |

## Explicit assumptions

| # | Assumption | If wrong |
|---|---|---|
| A16 | The same annotators, guide and 3-class scheme are valid across both corpora — A1 US resumes at 5,134 median chars and Djinni EE/UA CVs at 1,525 | The guide gains a per-corpus section; the κ figures are reported per corpus rather than pooled. Task 5.4 is the early warning |
| A17 | Wave-2's top-k union is materially smaller than the sum of per-system top-k, because stages agree at the top | If they disagree more than expected the queue grows toward 4 × 500 under D23. Task 5.5 measures it with **zero** annotator time before anyone is scheduled |
| A18 | The A1 pool universe (193 resumes) is large enough that "judge the union of top-5" reaches a meaningfully different set than "judge everything" | At 193 candidates over 64 queries, judging *all* pairs is 12,352 judgements — infeasible. The top-k restriction is what makes the wave exist at all |
