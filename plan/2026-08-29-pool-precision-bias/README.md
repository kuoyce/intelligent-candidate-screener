# Q18 — Pool precision bias: findings, and what still has to be decided

**Date:** 29 August 2026
**Status:** **Closed 29 August 2026 by D26** — as a *standing limitation*, which is one of the
three closes this document itself listed. Every figure below is now reproducible from the
package, and Q18a/Q18b are both measured. See *Provenance* and *How this closed*.
**Origin:** Q18, raised in
[`plan/2026-08-23-derived-artefacts/05-implementation.md`](../2026-08-23-derived-artefacts/05-implementation.md)
§5 and left open there.
**Blocks:** any reported Precision@k on the task 3.3 pools, for every stage.
**Resolved 29 Aug 2026.** Step 4 below — the judging wave — was designed in
[`plan/2026-08-29-unified-judging-wave/`](../2026-08-29-unified-judging-wave/README.md) (D20–D23) and then
**deferred by D25** on cost against the proposal's success measures. Steps 1 and 2 were run
instead: the ceiling is in code (`metrics.attainable_ceiling`) and the exposure is measured
(`evaluation.retrieval`). **Q26 is dissolved, Q27 is closed, Q24 is closed.**

## How this closed

| Sub-question | Outcome |
|---|---|
| **Q18a** — how much precision is lost to judged supply? | **Answered and pinned.** `metrics.attainable_ceiling` reproduces all four figures and `tests/test_metrics.py` asserts depth-invariance. Emitted into `pools-yield.json` |
| **Q18b** — how many unjudged distractors reach the top-k? | **Measured.** 83.2% of TF-IDF's 500 top-5 slots, [78.6, 87.4]. `output/baselines/retrieval-metrics.json` |
| **Q18c** — what fraction of those are genuinely relevant? | **Deliberately not measured (D26).** It needs a human, and the wave that would supply one costs 4–6 team-days against a metric no proposal target names |

**The argument in "the current defence may not hold" resolves against itself.** It reasoned
that the penalty *grows* with system quality, compressing or inverting the ranking. Measured,
the unjudged share falls monotonically as quality rises — random 94.0%, BM25 86.6%, TF-IDF
83.2% — because a better system puts *more* judged documents at the top. The failure mode that
would have made the pools unfit for comparison is not present in the observed direction. That is
the evidence this document, reasoning from argument alone, could not have had.

**What is conceded.** The bias is *large* — a raw Precision@5 of 0.142 against an 83% unjudged
top-5 is a floor, not an estimate. D26 does not claim otherwise. It reports the floor with its
ceiling beside it, and headlines **Recall@10**, whose own ceiling is 0.983 and which is the
metric the proposal's success measures actually name.

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
| ~~4~~ | ~~**Judge a bounded sample.**~~ **Deferred by D25** to an optional week-11 batch. The design survives in [Phase 5](../2026-08-29-unified-judging-wave/02-work-plan.md#56--the-session-d25) for whoever runs it | — | — |

Step 4's bound is the useful surprise. Judging the pools is infeasible; judging **only what a
system actually surfaced** is smaller than task 3.4's planned 200-pair in-domain wave, and it
reuses the same annotation guide. Q18's framing — "does a judging wave belong in the plan?" —
implied a cost that the top-k restriction does not carry.

## Decisions requested

| # | Question | Why it cannot be defaulted |
|---|---|---|
| ~~Q26~~ | What contamination rate makes the bias unacceptable? | **Dissolved by D26.** Q26 was the gate on *whether to run a wave*. Under a standing limitation there is no wave and no "pre-wave" figure to retract — the figures are final and carry the limitation. The pre-registration argument was right and is retained above for whoever revives the wave |
| ~~Q27~~ | Raw, ceiling-normalised, or both — and does the ceiling enter `pools-yield.json`? | **Closed by D26: both, and yes.** Every precision row in `retrieval-metrics.json` carries `value`, `ceiling` and `pct_of_attainable`; `pools-yield.json` carries the ceilings and the unwinnable-slot arithmetic |
| ~~Q24~~ | Unblock the scoring pass in step 2? | **Closed by D27**, run 29 Aug 2026. It cost no annotator time and had been deferred twice — which is most of why Q18 stayed open |

## Provenance

Every figure above was measured on 29 Aug 2026 from `docs/data/manifests/pools.csv` at commit
`6de2d3c`, using `evaluation.metrics.DEFINITIONS` for the relevance schemes and the same
"queries with no relevant document are excluded, not scored as zero" rule `metrics.score`
enforces — so *n* = 31 (strict) and 64 (graded) match the counts already reported elsewhere.

**These figures are now reproducible from the package** *(step 1 landed 29 Aug 2026,
`bfafc9c`)*:

```bash
uv run python -m candidate_screener.data.pools --feasibility   # ceilings + slot arithmetic
uv run pytest tests/test_metrics.py                            # the four figures, pinned to 4 dp
```

`metrics.attainable_ceiling` uses the same exclude-zero-relevant rule `score` enforces, so its
*n* matches by construction rather than by coincidence.

The random-ranker comparison points (Precision@5 strict 0.071 [0.039, 0.110] n=31; graded 0.044
[0.022, 0.066] n=64) come from `uv run python -m candidate_screener.evaluation.metrics --sanity`
and are already reproducible.

## Explicit assumptions

| # | Assumption | If wrong |
|---|---|---|
| A13 | The judged labels themselves are correct. This document treats the A1 judgements as ground truth and asks only about the *unjudged* remainder | If A1's labels are noisy, the ceiling is soft rather than hard and step 4 should sample judged documents too. **Now testable:** 50 already-judged A1 pairs, stratified 15/15/20 across the three classes, are in the session queue blind (`annotation/queue.py`). This is the one assumption here that gets measured |
| A14 | "Contamination" is estimable from a top-k sample without re-judging the whole pool. The quantity of interest is the rate *among surfaced distractors*, not among all distractors | If a system's surfaced distractors are unrepresentative in a way that matters, the sample needs stratifying by rank |
| A15 | One system's exposure generalises well enough to set policy for Stages 1–4 | If Stage 3/4 systems surface distinctly different distractors, step 2 repeats per stage — cheap, since it is one scoring pass |
