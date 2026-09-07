# Judge calibration — what to do with the LLM when the humans are spent

**Status:** draft, awaiting approval
**Date:** 8 Sep 2026
**Depends on:** D29 (two strata, never pooled), D33 (the judge exists), D34 (the
calibrated guide), D35 (one draw per run, one file per draw), Phase 2 of
`plan/2026-09-06-a2-llm-prelabel/` (an annotator is an axis).
**Supersedes:** the A2 plan's Phase 3. **D36 stays unmade** — nothing here licenses LLM
labels as ground truth.

## The situation

The in-domain campaign is closed: 600 pairs, 1,034 labels, both strata fully
double-labelled by `yc` and `van`. There is no budget for a third annotator and no
budget for a re-labelling wave (D25 capped manual annotation at one session; Phase 2
spent the extension). **The only remaining lever is the judge** — its instrument, its
model, or what we measure it against.

Phase 2 left the judge refused on both clauses of its gate:

| | `generic` | `targeted` |
|---|---|---|
| human ceiling kappa(yc, van) | 0.250 [0.140, 0.365] n=126 | **0.214** [0.112, 0.309] n=200 |
| kappa(yc, llm) | 0.380 [0.310, 0.447] n=400 | 0.249 [0.150, 0.339] n=200 |
| kappa(van, llm) | 0.142 [0.050, 0.236] n=126 | 0.204 [0.115, 0.295] n=200 |
| `Good Fit` per-class, judge | 0.119 / 0.062 | 0.121 / 0.067 |
| `Good Fit` per-class, **yc vs van** | 0.338 | 0.300 |

The relative clause (0.75 x ceiling) stopped measuring — judge and ceiling are one
overlapping cluster (**Q39**). The absolute per-class clause (0.40 on every class)
refuses the judge — but the last row says the **two humans fail it too**. A gate no
human pair can pass is not measuring the judge.

## The answer this plan proposes

Three moves, in strict order, cheapest first. Each is independently useful and each
stops cleanly if the next is not approved.

**1. Change what the judge is measured against, before changing the judge.** The
repository has two annotators and has never used them together. Against the
**two-annotator consensus** — the pairs where `yc` and `van` gave the same label — the
judge looks different and, for the first time, *stable across strata*:

| reference | n | 3-class | binary | `Good Fit` |
|---|---|---|---|---|
| `generic`, consensus vs llm | 60 of 126 (47.6%) | **0.389** [0.233, 0.541] | **0.729** [0.545, 0.892] | 0.136 |
| `targeted`, consensus vs llm | 97 of 200 (48.5%) | **0.401** [0.259, 0.536] | 0.584 [0.421, 0.732] | 0.167 |

Measured 8 Sep 2026 from the committed labels; **zero spawns, zero human hours**. On
the consensus subset the residual error collapses to **one cell** — consensus
`Good Fit` returning llm `Potential Fit`, 17 of 22 on `generic` and 14 of 18 on
`targeted` — while `No Fit` agreement is 0.955 / 0.772. The "two boundaries in opposite
directions" reading of Phase 2 was substantially an artefact of measuring against `yc`
alone, whose `No Fit` rate rose to 0.580 on `targeted` while `van`'s stayed at 0.360.

This does **not** license substitution (§3 of the design spec): consensus gives a
reference on 48% of pairs, and substitution needs one on 100%.

**2. Run 7/8 — the model control, one variable.** The committed instrument
(`75e00ea3`) under `claude-sonnet-5` over the same 600 A2 pairs, holding the guide,
the pairs and the humans fixed. This closes **Q32**, open since run 1 was `sonnet` and
run 4 `haiku` with no control, and it must come **before** any guide edit — otherwise a
model change and a guide change land together and neither is attributable. ~600 spawns,
no human time.

**3. Run 9/10 — a graded instrument, only if 2 does not move it.** The residual is one
threshold, so a D34-style pass at the `Good`/`Potential` boundary is the obvious edit.
But a guide edit is a new `agent_sha256` and forces a full 600-pair re-judge that pools
with nothing that came before. Since that is paid once, buy the version that makes
future boundary moves free: have the judge emit a **graded score** beside the class, so
the threshold becomes a post-hoc parameter. Today the judge's `Potential Fit` bucket
holds consensus-`Good`, consensus-`Potential` and consensus-`No Fit` with nothing to
sort them by.

## What this plan explicitly rejects

**Ensembling repeated runs of one judge.** Self-consistency is 0.857-1.000, so the
variance an ensemble averages away is already near zero. A *cross-model* disagreement
set is a different thing and falls out of move 2 for free.

**Chasing kappa(yc, llm) upward.** Beyond ~0.25 on `targeted` that means agreeing with
`yc` more than `van` does — fitting one annotator, not calibrating a judge. `van` and
`yc` are equally valid readings of the guide and the plan may not prefer one.

## A latent bug move 2 would trigger, and its prerequisite fix

`a2_frames` builds `llm` as a per-pair **majority across every run sharing the committed
`agent_sha256`**. A `sonnet` run of the *same* guide has the same `agent_sha256`. So run
7 would be pooled with run 5 into one series, and `majority` returns `None` on a tie —
which is every pair where the two models disagree. Those pairs are **dropped**, and what
survives is exactly the subset two models agreed on: a non-random, easier sample, with a
smaller n that nobody ordered and a kappa that would jump for the wrong reason. No
exception, no wrong row count, a plausible figure — the D35 shape, one axis along.

**Fix M1 is a prerequisite to any dispatch in this plan**: model is an axis, and
`a2_frames` must key on `(agent_sha256, model)`.

## Documents

- [`01-design-spec.md`](01-design-spec.md) — the reference axis, the rewritten gate, what each move licenses
- [`02-work-plan.md`](02-work-plan.md) — tasks, order, acceptance checks, cost

## Open questions carried in

- **Q32** — `sonnet` (runs 1-3) against `haiku` (runs 4+) with no control. **Move 2 answers it.**
- **Q38** — whether the `Good`/`Potential` boundary needs a D34-style pass. Sharpened by the
  consensus figures: the residual is one cell. **Move 3 acts on it, gated on move 2.**
- **Q39** — the relative gate no longer measures at a 0.21 ceiling. **Design spec §4 replaces it.**
- **Q40** — binary substitution is not defensible on both strata. Re-examined under the
  consensus reference and the interchangeability gate.
- **Q42** — `yc`'s `No Fit` shift between strata is confounded with D34. Unresolved and
  **unresolvable without human time**; recorded as a standing limitation.
