# Q45 — closed: the graded instrument stays committed

**Date:** 10 Sep 2026
**Decision:** **D39**. Q45 closes as **keep `5d986c82`**.
**Decided by:** the user, 10 Sep 2026, on the reading in
[`07-graded-instrument-findings.md`](07-graded-instrument-findings.md) §9.

---

## The conflict

Move 3 returned **outcome B**, and outcome B's registered action said two things that
cannot both be done:

> The committed instrument stays `75e00ea3`, **and** the score column survives as the
> reusable part.

Only `5d986c82` asks for a score. Move 3 bundled all four changes — the `Good Fit`
positive condition, the rescoped tie-break, the score, and the removed `model:`
frontmatter — into **one instrument on purpose**, so that no finding could attribute
anything to one clause. That deliberate bundling is what makes the two clauses
incompatible: you cannot keep the score without keeping the guide edit that asks for it.

The registration did not anticipate this because it wrote outcome B for the case where the
edit fails *and* is separable from the score. It was not separable, by its own design.

## The decision

**Keep `5d986c82` as the committed instrument.**

The reasoning, in the order it was weighed:

1. **Outcome B says the edit did not fix the boundary. It does not say the edit made
   anything worse**, and no registered statistic says so either. Consensus binary held its
   floor on both strata, no gate row that read `pass` or `tie` reads `fail`, the
   `yc`-minus-`van` gap fell on `generic`, and the unregistered paired supplement found
   the gain **strictly nested** — the graded instrument got right everything the baseline
   got right, plus 3 and 4 more. There is no measured cost to keeping it.
2. **Reverting pays 600 spawns to return to a judge that measures the same and emits
   less.** The scores would describe an instrument the repository no longer commits, which
   is Q34's situation exactly — unusable without a re-judge.
3. **The score is the part of move 3 that has independent value**, and outcome B's own
   wording says so. It ranks a consensus `Good Fit` above a consensus `Potential Fit` at
   AUC **0.832** / **0.756** — the boundary the emitted label collapses — and held-out cut
   points reach three-class kappa 0.601 and 0.406. Discarding it to honour the literal
   text of a clause that was written without foreseeing this case would be following the
   registration against its own purpose.

**What this does not do.** It does not promote the graded instrument on the strength of its
figures — outcome B stands, unamended, and **Q38 stays closed**: the `Good Fit` boundary is
not reachable by editing the instrument. It does not commit a cut point. It does not make
**D36**; nothing here licenses LLM labels as ground truth.

## Consequences on file

- Every A2 figure is quoted under `5d986c82`. Runs 5-8 keep their figures under
  `by_instrument`, keyed by the superseded `75e00ea3` — **superseded, not withdrawn**.
- The **A1 recheck** has no rows under the committed instrument, so
  `llm-recheck-report.json` reads `status: no overlap` at section level and D33's and
  D34's figures live under `per_instrument` (`47fc2e48` 0.142 / 0.287; `75e00ea3`
  **0.085 / 0.352**). This is the state fix M3 exists to make legible: *never measured*,
  not *measured empty*.
- Re-judging the A1 recheck under `5d986c82` is **~100 spawns and is not scheduled**. It
  is not needed for anything: A1 is not the evaluation corpus going forward
  (`plan/2026-09-10-a2-reference-design/`), and the figures it would refresh are quoted
  against a label set that fails its own reproducibility check.

## What closes, and what this plan leaves behind

**The judge-calibration plan is complete.** Three moves, all landed:

| move | question | answer |
|---|---|---|
| 1 | is there a third reference? | **D37** — consensus, and it licenses measurement, not substitution |
| 2 | is it the model? | **outcome D** — stratum-dependent; `sonnet` found `yc`, not the truth |
| 3 | is it the instrument? | **outcome B** — the boundary is not reachable by editing it |

**Q38 closes**, and with the user's decision of 10 Sep 2026 that there will be **no third
annotator**, it closes permanently rather than pending budget. Model and guide have both
been tried; the reference was the remaining lever and it is not available.

Carried forward unresolved: **Q34** (option D's pre-D34 rows), **Q40** (binary
substitution not defensible on both strata), **Q41**, **Q42**, **Q43** (runs 1-6's model
column was asserted, not passed), **Q44** (whether a judge that tracks one annotator is
useful at all).

Raised by the decision to move to A2: **Q46**, what a reference label *is* on that corpus.
That is a design question, not a calibration one, and it has its own plan.
