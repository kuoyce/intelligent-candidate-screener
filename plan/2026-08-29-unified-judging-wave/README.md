# Phase 5 — Unified Judging Wave

**Date:** 29 August 2026
**Status:** Draft for approval
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
| **D23** | *New.* The wave covers **all 100 test queries**, including the 36 that are currently unscoreable. Widening *n* is a first-class objective of the wave, not a byproduct | **Q31** | Supersedes D21's 64 within the same wave — D21's reasoning (graded, not strict) still governs the *relevance definition*; D23 governs *query coverage*. Upper bound rises to **500 top-5 slots per system** from 320. §"What D23 buys" below |

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
| **Q24** | Score an existing system over the 3.3 pools | **Open**, from Phase 4 | 5.5 queue sizing. Needed for a budget, not for the wave |
| **Q26** | *Re-scoped by D22.* What contamination rate makes the pre-wave figures retractable rather than revisable? Fix before 5.6's output is seen | **Open** | 5.7 reporting policy |
| **Q27** | Precision@k raw, ceiling-normalised, or both — and does the ceiling enter `pools-yield.json`? | **Open** | 5.2. **Now harder:** D20 creates *two* ceilings, A1-only and A1+ours, and D23 lets *n* differ between them |

## What D23 buys, and what it risks

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
| 5.1 – 5.7 | Not started — plan awaiting approval |

## Explicit assumptions

| # | Assumption | If wrong |
|---|---|---|
| A16 | The same annotators, guide and 3-class scheme are valid across both corpora — A1 US resumes at 5,134 median chars and Djinni EE/UA CVs at 1,525 | The guide gains a per-corpus section; the κ figures are reported per corpus rather than pooled. Task 5.4 is the early warning |
| A17 | Wave-2's top-k union is materially smaller than the sum of per-system top-k, because stages agree at the top | If they disagree more than expected the queue grows toward 4 × 500 under D23. Task 5.5 measures it with **zero** annotator time before anyone is scheduled |
| A18 | The A1 pool universe (193 resumes) is large enough that "judge the union of top-5" reaches a meaningfully different set than "judge everything" | At 193 candidates over 64 queries, judging *all* pairs is 12,352 judgements — infeasible. The top-k restriction is what makes the wave exist at all |
