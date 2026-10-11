# Move 3 — the graded instrument, read against `06-graded-registration.md`

**Date:** 10 Sep 2026
**Registers against:** [`06-graded-registration.md`](06-graded-registration.md), committed
before any run 9 or run 10 prompt existed.
**Runs:** 9 (`a2_generic`, 400) and 10 (`a2_targeted`, 200), instrument `5d986c82`,
`claude-haiku-4-5`, `response_scheme = label+score`.

**The outcome is B — no material change.** Read in the registered order below.

---

## 1. What ran

600 spawns, 600 parsed returns, 600 scores, zero unparsed rows. Both runs reached exact
coverage before anything was collected.

The dispatch cost five sittings on run 9 and four on run 10, every interruption an
account usage limit:

| run | sitting | attempted | answered | cumulative coverage |
|---|---|---|---|---|
| 9 | 1 | 400 | 68 | 68 / 400 |
| 9 | 2 | 332 | 0 | 68 / 400 |
| 9 | 3 | 332 | 97 | 165 / 400 |
| 9 | 4 | 235 | 128 | 293 / 400 |
| 9 | 5 | 107 | **107** | **400 / 400** |
| 10 | 1 | 200 | 36 | 36 / 200 |
| 10 | 2 | 164 | 47 | 83 / 200 |
| 10 | 3 | 117 | 38 | 121 / 200 |
| 10 | 4 | 79 | **79** | **200 / 200** |

A retry pass was budgeted by the registration and is not a deviation. **F1/F2 did the job
it was built for**: every failed spawn was an `exit 1` with both streams empty, was
quarantined rather than recorded as a return, and stayed owed. Sitting 2 of run 9 answered
*nothing* and the run still reported itself short — pre-F1 that sitting would have filed
332 pairs as judged and computed a kappa over the 68 that answered fastest.

**Operational note, not a code fault.** `/low-priority` on the parent session does not
reach the `claude -p` subprocesses a judging pass spawns: sitting 2 was run under it and
returned `exit 1: You've hit your session limit` on all 332.

## 2. The sanity checks

All eight pass.

| # | check | result |
|---|---|---|
| 1 | `--coverage` exact before `--collect` | 400/400 and 200/200 |
| 2 | `llm-judge-failed.jsonl` clean for runs 9/10 | 0 outstanding |
| 3 | every run 9/10 row carries the new `agent_sha256` | `5d986c82` only |
| 4 | every parsed row scored, 0-100; no run 1-8 row scored | 600 scored, range 3-87 / 5-85; 0 on runs 1-8 |
| 5 | `model` matches the dispatch record | `claude-haiku-4-5`, `verify --derived` |
| 6 | `self_consistency` refuses runs 5 vs 9 | listed under `not_self_consistency` |
| 7 | the A1 recheck figures do not move | intact under `per_instrument` — **see fix M3** |
| 8 | `prompt_sha256` differs from run 5's / run 6's | **0 of 400** and **0 of 200** identical |

Check 8 is the one that establishes what changed: the prompt now asks for a score, so a
matching hash would have meant the old prompt was re-sent. None matched.

Check 7 needed a code fix to be *readable*, not to be true — §6.

## 3. The registered statistics

Against the **consensus** reference (D37), per stratum. Coverage: **60 of 126 (47.6%)** on
`generic`, **97 of 200 (48.5%)** on `targeted`. Consensus selects the pairs two humans
found unambiguous, so **every figure here is an upper estimate**, and it licenses
measurement and not substitution. Baseline is `75e00ea3` under `haiku` (runs 5/6), and it
reproduces the registration's table exactly.

| | statistic | `generic` baseline → graded | `targeted` baseline → graded | bar |
|---|---|---|---|---|
| **G** | consensus `Good Fit` per-class | 0.136 → **0.273** (3/22 → 6/22) | 0.167 → **0.389** (3/18 → 7/18) | ≥ 0.40 |
| **C** | consensus `Good` → llm `Potential` | 0.773 → **0.591** (17/22 → 13/22) | 0.778 → **0.500** (14/18 → 9/18) | < 0.50 |
| **K** | consensus three-class kappa | 0.389 → **0.483** [0.324, 0.641] | 0.401 → **0.385** [0.239, 0.530] | — |
| **B** | consensus binary kappa | 0.729 → **0.729** [0.547, 0.883] | 0.584 → **0.549** [0.385, 0.710] | ≥ 0.540 / 0.422 |
| **I** | gate, three-class (van / yc) | fail / pass → **fail / pass** | tie / tie → **tie / tie** | lose no pass or tie |
| **I** | gate, binary (van / yc) | tie / pass → **tie / pass** | pass / pass → **tie / pass** | lose no pass or tie |
| **A** | kappa(yc,llm) − kappa(van,llm), shared | 0.313 → **0.268** | 0.045 → **0.058** | do-no-harm / ≤ 0.20 |
| | llm `Good Fit` issue rate | 4.0% → **11.0%** | 3.0% → **9.5%** | — |

## 4. The outcome, read in the registered order

**Outcome C — the improvement bought elsewhere. Does not hold.** G rises on both strata,
so the clause is live, and none of its three violations fired: B clears its floor on both
(0.729 ≥ 0.540, 0.549 ≥ 0.422); no gate row that read `pass` or `tie` reads `fail`; A is
0.058 ≤ 0.20 on `targeted` and **fell** on `generic` (0.313 → 0.268), which is the
do-no-harm form the registration declared for that stratum, and the finding says so as it
was told to. This is the outcome `sonnet` produced in move 2 and this instrument does not:
it did **not** buy `Good Fit` recall by getting harsher elsewhere.

**Outcome A — the boundary is fixed. Does not hold**, and not narrowly. G must clear 0.40
on both strata and reaches **0.273** and **0.389**; C must fall below 0.50 and reaches
**0.591** and **0.500** — the latter exactly half, which is not below it. Both clauses miss
on both strata.

**Outcome B — no material change. This is the reading.** The registered test is whether G's
interval overlaps the baseline's on both strata, and it does, comfortably:

| stratum | baseline G | graded G | overlap |
|---|---|---|---|
| `generic` | 0.136 [0.047, 0.333] | 0.273 [0.132, 0.482] | yes |
| `targeted` | 0.167 [0.058, 0.392] | 0.389 [0.203, 0.614] | yes |

(Wilson 95%, the deterministic interval for a proportion; the repo's `kappa_ci` bootstrap
is for kappa, not for a per-class rate.)

**Therefore Q38 closes as "the boundary is not reachable by editing the instrument."**
Model (move 2) and instrument (move 3) have both now been tried against this corpus, and
this is the end of the calibration road the plan opened. It is a real finding, not a null
one: the `Good Fit` boundary has resisted a model change and a guide change that each
moved the point estimate and neither moved it far enough to matter at this n.

### 4a. An unregistered supplement, labelled as one

The registered test compares two **independent** intervals, but the two instruments judged
the *same* consensus pairs with the *same* human labels — a paired design, which the
independent test wastes power on. Reported here because it is information the registered
test discards, and **it does not overturn the outcome**:

| stratum | n | both right | baseline only | graded only | neither | McNemar exact |
|---|---|---|---|---|---|---|
| `generic` | 22 | 3 | **0** | 3 | 16 | p = 0.250 |
| `targeted` | 18 | 3 | **0** | 4 | 11 | p = 0.125 |

The paired test agrees with the registered one — neither stratum reaches significance. What
it adds is that **the gain is strictly nested**: `baseline only = 0` on both strata, so the
graded instrument got right everything the baseline got right and then some. The
improvement is real in direction and consistent across strata; it is simply too small at
n=22 and n=18 to separate. That is worth recording because it is the difference between
"no effect" and "an effect this corpus cannot resolve", and **outcome B's wording is the
latter**.

## 5. The score, and what it is worth under outcome B

The score is the part of move 3 that survives the outcome, and it works.

**S1 — the score separates the boundary the label collapses.** No threshold, nothing
fitted, so nothing here is at risk of being fitted-and-quoted.

| stratum | consensus `Good` | consensus `Potential` | consensus `No` | AUC Good vs Potential | AUC Potential vs No |
|---|---|---|---|---|---|
| `generic` | median 65 (q1 50, q3 73) | 43.5 (35.5, 48) | 10 (8, 15) | **0.832** | 0.930 |
| `targeted` | 65 (62.75, 74.75) | 50 (40.5, 65) | 18 (12, 32) | **0.756** | 0.888 |

This is the object the 600 spawns bought. The emitted **label** agrees with consensus
`Good Fit` 27% and 39% of the time; the **score** ranks a consensus `Good Fit` above a
consensus `Potential Fit` 83% and 76% of the time. The information the boundary needs is
present in the judge's output and the three-class label is discarding it.

**S2 — cut points fitted on all consensus pairs.** Fitted, and says so; `fitted_on` names
every pair, and `assert_score_figure_names_its_fit` refuses to emit the block without it.
**Not comparable with the emitted label's kappa.**

| stratum | cut points | three-class kappa | per-class | note |
|---|---|---|---|---|
| `generic` | `No` < 19, `Good` ≥ 49 | 0.749 [0.600, 0.875] | 0.773 / 0.813 / 0.909 | healthy |
| `targeted` | `No` < 51, `Good` ≥ 55.5 | 0.547 [0.411, 0.677] | 0.889 / **0.091** / 0.982 | **degenerate** |

`targeted`'s full-data fit is worth naming as a failure of the objective, not of the score:
a `Potential Fit` band 4.5 points wide emits **2** of that class out of 97 and scores 0.091
on it. Maximising overall three-class agreement against a reference whose majority class is
`No Fit` (57 of 97) buys agreement by nearly abolishing the middle class. Any future
adoption of a cut point needs an objective that cannot do this — per-class-balanced, not
overall agreement.

**S3 — the one out-of-sample figure.** Cut points fitted on a seeded half of the consensus
pairs and quoted on the other half, the split fixed before any score existed.

| stratum | fitted on | quoted on | cut points | three-class kappa | binary kappa |
|---|---|---|---|---|---|
| `generic` | 30 | 30 | `No` < 21.5, `Good` ≥ 49 | **0.601** [0.365, 0.800] | 0.591 [0.308, 0.857] |
| `targeted` | 48 | 49 | `No` < 51, `Good` ≥ 72.5 | **0.406** [0.214, 0.606] | 0.622 [0.392, 0.830] |

Read against the emitted label's consensus kappa on the same pairs (0.483 `generic`, 0.385
`targeted`), a re-scored label is **better on `generic` and level on `targeted`** — and both
intervals are wide enough to contain the emitted label's figure. The registration required
this to be reported as such rather than replaced by the full-data fit that cannot miss, and
it is.

**No cut point is committed by this move**, per the registration. Adopting one is a separate
decision needing a reference on 100% of pairs, which consensus is not.

## 6. Fix M3 — a kappa over no pairs is not a figure

**Surfaced by check 7 and fixed in this commit.** `judge_frames` returns `llm` for the
**committed** instrument alone, so the commit that changed the guide left the A1 recheck
draw judged by no current instrument — and `report()` emitted `a1_vs_llm` and
`human_vs_llm` as `kappa: NaN, n: 0`. Regenerating the report turned D34's committed
kappa(A1, llm) = **0.085** and kappa(yc, llm) = **0.352** into NaN, and emptied
`disagreements` from 72 entries to 0.

Nothing was ever lost — both instruments' figures were intact under `per_instrument`
(`47fc2e48`: 0.142 / 0.287 over runs 1-3; `75e00ea3`: **0.085 / 0.352** over run 4) — but a
reader could not tell *never measured* from *measured empty*, which is the same
absent-vs-withdrawn ambiguity `a2_report` was given `by_instrument` for in task 3.1 and
which `report()` was not. `pairwise_over` now skips a pair with no common index, the
per-class blocks emit `{}` rather than NaN, and the report carries `status`
(`scored` / `no overlap`) and `note_no_overlap`. The leg with no `llm` dependency —
kappa(A1, yc) at n=100 — is untouched and stays quotable.

Regression test:
`test_a_recheck_report_with_no_committed_rows_says_so_rather_than_reporting_nan`, shown to
fail against the unguarded code with exactly the `NaN` / `n: 0` shape.

## 7. Deviations

- **D2 of this plan** (declared in the registration): move 3 got an advance-registration
  document, so the findings are `07-` rather than `06-`.
- **M3** (above): a fix to `report()` surfaced by registered check 7. It changes no
  figure — it changes whether an uncomputed figure is printed as NaN.
- No deviation from the registered statistics, bars, or outcome order. Every number in §3
  and §4 is the one the registration named, computed the way it said.

## 8. Open questions

- **Q38 — closes.** The `Good Fit` boundary is not reachable by editing the instrument.
  Model and guide have both been tried. What remains untried is the reference: both humans
  bracket the judge from far above on `Good Fit` and **do not clear the 0.40 bar against
  each other** on it (0.338/0.759 `generic`, 0.300/0.545 `targeted`), so the class is
  `unidentified_by_instrument` for the human pair too. A third annotator, not a third
  guide, is what would move this.
- **Q45 — raised here, and it needs a decision.** Outcome B's two clauses cannot both be
  honoured. It says *"the committed instrument stays `75e00ea3`"* **and** *"the score
  column survives as the reusable part"* — but the score exists only because `5d986c82`
  asks for it, and move 3 bundled all four changes into one instrument on purpose ("no
  finding may attribute anything to one clause of it"). Reverting the guide destroys the
  score and 600 spawns; keeping it overrides the registration's stated action. **The tree
  is left as it stands — `5d986c82` committed — because that is the reversible option and
  the one that preserves the data**, and because the graded instrument is not worse than
  the baseline on any registered statistic. This is flagged, not decided: it is the user's
  call, and §9 states what each choice costs.
- **Q40** — unchanged. Binary substitution is still not defensible on both strata.
- **Q34, Q41, Q42, Q43** — untouched by this move.

## 9. What the two readings of Q45 cost

**Keep `5d986c82` committed** (the current tree). Every A2 figure is quoted under it; runs
5-8 stay readable under `by_instrument`; the score is on file for all 600 A2 pairs and a
future boundary move is a re-scoring rather than 600 spawns. Cost: the committed judge is
one the registration's outcome-B clause said would not be committed, and the A1 recheck
report reads `no overlap` at section level until the recheck is re-judged under it
(~100 spawns).

**Revert to `75e00ea3`.** The registration's outcome-B action is honoured literally and the
A1 recheck report's section level comes back. Cost: the 600 scores describe an instrument
the repository no longer commits, which is Q34's situation exactly — they become unusable
without a re-judge, and S1's finding that the score separates the boundary the label
collapses is stranded with them.

**Recommendation: keep it.** Outcome B says the guide edit did not fix the boundary; it does
not say the edit made anything worse, and no registered statistic says so either. Reverting
pays 600 spawns to return to a judge that measures the same and emits less.
