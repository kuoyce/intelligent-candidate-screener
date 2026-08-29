# Q18 — Pool precision bias: findings, and what still has to be decided

**Date:** 29 August 2026
**Status:** **Open, awaiting sign-off.** Analysis only — nothing here is implemented, and no
figure below is yet reproducible from the package (see *Provenance*).
**Origin:** Q18, raised in
[`plan/2026-08-23-derived-artefacts/05-implementation.md`](../2026-08-23-derived-artefacts/05-implementation.md)
§5 and left open there.
**Blocks:** any reported Precision@k on the task 3.3 pools, for every stage.
**Superseded in part, 29 Aug 2026:** step 4 below is now designed in
[`plan/2026-08-29-unified-judging-wave/`](../2026-08-29-unified-judging-wave/README.md), which merges it with
Q14's in-domain wave 2 into one instrument. D20–D22 there close Q28–Q30; **Q26 is re-scoped by
D22** — it no longer gates whether the wave runs, only how the pre-wave figures are treated. D23
widens the wave to all 100 test queries, so *n* is no longer fixed at 64/31.

## Why this document exists

Q18 was written to close itself: *"decide once the Stage 1 baseline exists and the size of the
bias can be seen."* The Stage 1 baseline now exists (commit `6de2d3c`, Phase 4). The closing
condition was tested against that and **does not hold** — not because the baseline is missing,
but because "the size of the bias" is three separate quantities, and one of them cannot be
measured by any amount of computation.

This document separates them, records the one that is measurable today, and states what
decision is actually being asked for.

## The decomposition

| | Sub-question | Evidence required | Status |
|---|---|---|---|
| **Q18a** | How much precision is lost to **judged supply** — can a perfect system even reach 1.0? | Arithmetic over the committed pool manifest. No model, no annotators | **Answered below** |
| **Q18b** | How many **unjudged distractors** does a real system rank into its top-k? | One scoring pass of an existing system over the 10,000 N100 pool pairs | **Blocked on Q24**, deferred out of Phase 4 by design |
| **Q18c** | Of the distractors a system surfaces, what fraction are **genuinely relevant**? | Human judgement. Nothing else | **Not computable — see the circularity below** |

Q18 as worded asks all three at once, which is why it has stayed open: Q18a was never
attempted, Q18b is one cheap step away, and Q18c is not a measurement problem at all.

## Q18a — answered: there is a second bias, and it is not the distractor assumption

The pools carry a precision ceiling that has nothing to do with unjudged distractors: **for many
queries there are not five judged-relevant resumes in existence to put in the top five.** A
perfect ranker scores below 1.0 by construction.

Measured on the committed `docs/data/manifests/pools.csv`:

| Metric | Scoreable *n* | **Max attainable mean** | Queries able to reach 1.0 |
|---|---|---|---|
| Precision@5, strict | 31 | **0.7806** | 20 / 31 |
| Precision@5, graded | 64 | **0.7188** | 30 / 64 |
| Precision@10, strict | 31 | 0.5677 | 6 / 31 |
| Precision@10, graded | 64 | 0.4734 | 7 / 64 |

Judged-relevant supply per scoreable query, N100:

| Definition | *n* | Median | Mean | Range | Queries with <5 | with <10 |
|---|---|---|---|---|---|---|
| strict | 31 | 6 | 5.87 | 1–12 | 11 | 25 |
| graded | 64 | 4 | 4.92 | 1–13 | 34 | 57 |

Put slot-wise: across the 31 strict-scoreable queries there are **155 top-5 slots, of which only
121 can be filled by a judged-relevant document.** 34 of them (22%) are unwinnable by any system.
Graded is worse — 90 of 320 (28%).

Three consequences:

1. **The ceiling is a property of the split, not of pool depth.** It is *identical* at N20, N100
   and Nfull (0.7806 / 0.7188 to four decimals at every variant), because adding distractors
   changes the denominator of precision but not the supply of judged-relevant documents.
   Choosing a different primary variant does not mitigate it.
2. **It changes how every Stage 1–4 precision figure reads.** Against a random-ranker floor of
   0.071 (strict) and a ceiling of 0.781, a system scoring 0.45 sits at **58% of attainable**,
   not 45%. Reporting the raw figure without the ceiling understates every system by a fifth,
   uniformly and silently.
3. **It retroactively justifies deviation W8 with a number.** Phase 3 retired Precision@10 for
   Precision@5 on density grounds. Precision@10's ceiling is 0.5677 strict — a metric whose best
   possible value is barely above half. The decision was right; the reason is stronger than the
   one recorded.

This bias is deterministic, system-independent and computable from an artefact already in git.
It should be recorded whatever is decided about Q18b and Q18c.

## Q18c — the circularity, stated plainly

**The judging wave is the only instrument that can measure the quantity that decides whether the
judging wave is warranted.**

To know whether the downward precision bias is acceptable, we need to know what fraction of
surfaced distractors are truly relevant. To know that, someone must judge them. There is no
scoring pass, no held-out trick and no amount of compute that substitutes.

Therefore Q18 **cannot be discharged by measurement alone**, and waiting for more evidence
before deciding is not a strategy — it is the status quo that has kept it open. It needs one of:

- a **decision rule fixed in advance** ("accept the bias if contamination is below X%"), then a
  bounded sample to evaluate against it; or
- an accepted **standing limitation**, with the ceiling reported alongside every figure and no
  judging wave; or
- an unconditional decision to run the wave.

## The argument that the current defence may not hold

`pools.py` states the position the pools were built on:

> The pools are a comparison instrument between systems, not an estimate of production precision.

That defence is sound **only if the bias is roughly constant across systems.** There is reason to
think it is not, and that it runs the wrong way:

> A better system surfaces more genuinely relevant resumes. In a pool that is 93–97% unjudged,
> more of those surfaced resumes are unjudged distractors, which score as false positives.
> **The penalty therefore grows with system quality.**

If that holds, the bias does not shift the scale uniformly — it **compresses the differences
between systems**, and in principle can invert a ranking. That is precisely the failure mode that
would make the pools unfit for the comparison they were built for.

**This is an argument, not a measurement.** It is exactly what Q18b would size, and it is the
strongest reason not to leave Q18 open indefinitely.

## Proposed path

Cheapest first. Steps 1–2 are mechanical and need no annotator time; step 3 is the decision.

| # | Step | Cost | Needs |
|---|---|---|---|
| 1 | **Record Q18a.** Add the attainable-ceiling computation to `evaluation/metrics.py` so it is reproducible, emit it into the pools manifest, and report it beside every Precision@k figure | Small | Sign-off on Q27 |
| 2 | **Run Q18b.** Score an existing system over the N100 pools and count how many top-5 slots are filled by unjudged distractors, per query | Small — one scoring pass, no judgement | Q24 unblocked |
| 3 | **Fix the decision rule *before* looking at step 3's output** | — | **Q26** |
| 4 | **Judge a bounded sample** — now designed as [Phase 5](../2026-08-29-unified-judging-wave/02-work-plan.md#56--wave-2-the-judging-run), widened to all 100 test queries by D21/D23: only the distractors that actually reached some system's top-5 | **≤155 pairs strict, ≤320 graded**, per system, deduplicated across systems — and only the subset that is not already judged | Annotator time, `docs/annotation-guide.md` (task 3.4a) |

Step 4's bound is the useful surprise. Judging the pools is infeasible; judging **only what a
system actually surfaced** is smaller than task 3.4's planned 200-pair in-domain wave, and it
reuses the same annotation guide. Q18's framing — "does a judging wave belong in the plan?" —
implied a cost that the top-k restriction does not carry.

## Decisions requested

| # | Question | Why it cannot be defaulted |
|---|---|---|
| **Q26** | What contamination rate makes the bias unacceptable? Fix X **before** step 2's output is seen | Choosing the threshold after seeing the number is how a null result gets talked into significance. It must be pre-registered |
| **Q27** | Is Precision@k reported raw, ceiling-normalised, or both — and does the ceiling enter `pools-yield.json` as a committed figure? | It changes every published precision number by ~22%, and the project's convention is that a figure travels with what bounds it *(cf. `Figure` refusing to exist without its n)* |
| **Q24** | Unblock the scoring pass in step 2? | Already open from Phase 4, deliberately deferred there. Q18b cannot proceed without it |

## Provenance

Every figure above was measured on 29 Aug 2026 from `docs/data/manifests/pools.csv` at commit
`6de2d3c`, using `evaluation.metrics.DEFINITIONS` for the relevance schemes and the same
"queries with no relevant document are excluded, not scored as zero" rule `metrics.score`
enforces — so *n* = 31 (strict) and 64 (graded) match the counts already reported elsewhere.

**These figures are not yet reproducible from the package.** They were computed ad hoc, which is
exactly what step 1 fixes; per the *Where code goes* rule the computation was not committed
under `plan/`. Until step 1 lands, treat them as measured-but-unpinned, and re-derive before
citing them in a report.

The random-ranker comparison points (Precision@5 strict 0.071 [0.039, 0.110] n=31; graded 0.044
[0.022, 0.066] n=64) come from `uv run python -m candidate_screener.evaluation.metrics --sanity`
and are already reproducible.

## Explicit assumptions

| # | Assumption | If wrong |
|---|---|---|
| A13 | The judged labels themselves are correct. This document treats the A1 judgements as ground truth and asks only about the *unjudged* remainder | If A1's labels are noisy, the ceiling is soft rather than hard and step 4 should sample judged documents too |
| A14 | "Contamination" is estimable from a top-k sample without re-judging the whole pool. The quantity of interest is the rate *among surfaced distractors*, not among all distractors | If a system's surfaced distractors are unrepresentative in a way that matters, the sample needs stratifying by rank |
| A15 | One system's exposure generalises well enough to set policy for Stages 1–4 | If Stage 3/4 systems surface distinctly different distractors, step 2 repeats per stage — cheap, since it is one scoring pass |
