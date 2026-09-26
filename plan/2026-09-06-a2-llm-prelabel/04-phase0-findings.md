# Phase 0 Findings — the judge on the in-domain corpus

**Date:** 6 Sep 2026 · **Run 5**, draw `a2_generic`, n=400, instrument `75e00ea3`
(post-D34), model `claude-haiku-4-5` · `docs/data/manifests/llm-a2-generic.csv`

The first measurement of this judge against `yc` on the corpus the project actually
targets. Every figure below is over the `generic` stratum alone and none may be pooled
with `targeted`, which is not yet dispatched *(D29)*.

## Headline

| | A2 `generic` (n=400) | A1 recheck (n=100) |
|---|---|---|
| kappa(yc, llm), 3-class | **0.380** [0.310, 0.447] | 0.352 |
| kappa(yc, llm), binary | **0.637** [0.553, 0.708] | 0.614 |
| raw agreement | 0.605 | — |

## Q36 is answered: agreement does not degrade on A2

This was the open question — the judge was calibrated against A1 pairs, and A2 documents
are a third the length (median 1,549 JD / 1,742 CV against A1's 5,134). It carries. Both
figures are marginally *higher* than the A1 recheck's and the confidence intervals overlap
heavily, so the honest statement is **no material difference**, now measured at four times
the n with an interval less than half as wide.

That is a real result and it removes a stated risk from the plan. It does not, on its own,
license anything.

## The finding that matters: one boundary, one direction

Per-class agreement, conditioned on `yc`'s label:

| `yc` said | n | judge agreed |
|---|---|---|
| `Good Fit` | 101 | **0.119** |
| `Potential Fit` | 138 | 0.797 |
| `No Fit` | 161 | 0.745 |

Confusion, rows `yc`, columns `llm`:

| | Good Fit | Potential Fit | No Fit |
|---|---|---|---|
| **Good Fit** | 12 | **85** | 4 |
| **Potential Fit** | 4 | 110 | 24 |
| **No Fit** | 0 | 41 | 120 |

**85 of `yc`'s 101 `Good Fit`s came back `Potential Fit`.** Only 4 came back `No Fit`. The
disagreement is not spread across the scheme and it is not bidirectional — it is almost
entirely one boundary, resolved downward almost every time.

### This is D34's residue, and it is the shape D34 predicted

D34 diagnosed the tie-break — *"choose the lower one"* reading to a model as *any doubt →
down, transitively* — and fixed it by barring the tie-break from reaching `No Fit` and
scoping it to the `Good`/`Potential` boundary. **That worked**: only 4 of 101 `Good Fit`s
fall all the way to `No Fit`, against the pre-D34 instrument's 26 of 30 on A1.

What D34 left in place is the tie-break *at the boundary it was scoped to*, and there it is
still resolving down almost unconditionally. The instrument now has one failure mode
instead of a transitive one, and it is legible.

### The binary collapse is a different proposition

0.637 against 0.380, and the reason is visible in the matrix: `Good → Potential` does not
cross the fit/no-fit line. The judge is substantially reliable on *is this candidate
plausible at all* and unreliable on *is this candidate strong*.

## What this licenses

**Not three-class substitution.** The gate's per-class clause is `> 0.40 on every class`
and it is **absolute, not relative** — 0.119 fails it, and no value of
kappa(yc, annotator2) rescues a figure that low. That conclusion is available now, before
the second annotator arrives, and it does not depend on Phase 2.

**Binary substitution stays open**, and is now the more interesting question. Phase 2's
relative gate applies to it unchanged.

## What this does *not* establish

**Not that the judge is wrong.** It establishes that `yc` and the judge disagree at one
boundary, one-directionally. Which of them is miscalibrated is exactly what one annotator
cannot determine — the same structure as D33's original problem, one boundary along. If
`yc` is generous with `Good Fit`, the judge is right and the in-domain labels need
recalibrating; if `yc` is right, the instrument needs a second calibration pass.

**The second annotator resolves this**, and their `Good Fit` rate against `yc`'s is the
single most informative number Phase 2 can produce. Task 2.1's draw should be checked to
carry enough `Good Fit` pairs to measure it — a draw that happens to be thin there answers
the wrong question.

## Interruption, and what it cost

The run hit the account's usage limit 193 pairs in. **202 of 395 spawns failed**, all
`exit 1` with **both streams empty** — which is what a quota looks like from `claude -p`,
so `QUOTA_MARKERS` matched nothing and they were labelled `exit1`.

Under the pre-F1 code every one of those would have been written to `llm-recheck-raw.jsonl`
as a `<judge exited 1: …>` string, absorbed into the resume set, and never retried. The
re-run would have reported `remaining: 0`, and `a2_report` would have computed a kappa over
**198 latency-ordered survivors**. That figure was computed, out of interest, before the
retry: **0.301 three-class, 0.554 binary** — plausible, wrong, and 0.079 below the real
number. Nothing anywhere would have said so.

Instead the pairs were quarantined, `judge()` raised, `a2_report` refused to score the
stratum, and the retry after the limit reset judged **exactly the 202**, 0 failures. Final
coverage 400/400.

F1–F3 were written two commits before they were needed. They are the reason this document
quotes 0.380 and not 0.301.

## Open questions

| # | Question | Status |
|---|---|---|
| **Q38** | Does the `Good`/`Potential` boundary need a D34-style calibration pass? | **New.** It is the whole of the residual disagreement. A guide edit is a new judge — new `agent_sha256`, and no figure pooled with run 5 |
| Q35 | kappa(yc, annotator2), and specifically on `Good Fit` | Open — second annotator incoming. Now the decisive measurement |
| Q36 | Does agreement differ A1 vs A2? | **Closed: no material difference**, 0.380 [0.310, 0.447] against 0.352 |
| Q37 | May `yc` and `llm` labels coexist in one evaluation layer? | Deferred to Phase 3 |
| Q34 | Option D's 639 pre-D34 rows | Untouched, deferred, no run number reserved |

## Next

Phase 1: dispatch `a2_targeted` as run 6 **before** its human session, which is what makes
it prospective. Its `generic` counterpart is now a retrospective baseline to compare it
against.
