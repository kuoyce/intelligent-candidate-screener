# Move 3 — the reading, registered before the dispatch exists

**Date:** 8 Sep 2026
**Committed:** before any run 9 or run 10 prompt exists. Everything move 3 concludes is
read against this document and nothing else.
**Registers:** work-plan tasks 3.3, 3.4, 3.5.
**Precedent:** [`04-run7-registration.md`](04-run7-registration.md). Move 2's four
outcomes and their cut points were fixed in advance and the result — outcome D — was one
the registration named but nobody expected. That is the whole value of doing this: a
result that can be read either way afterwards measures nothing.

**Deviation D2 of this plan.** The work plan named one move-3 document,
`06-graded-instrument-findings.md`, and no registration task (move 2 had one as task
2.1). Move 3 both edits the instrument *and* **fits a parameter** against the reference
it is then scored on, so it has strictly more freedom to be read after the fact than
move 2 did. This document takes number 06 and the findings become
`07-graded-instrument-findings.md`.

---

## 1. What the edit is

One instrument change, three parts, one new `agent_sha256`, one full re-judge. It is
already in the working tree at the time of writing and is committed with this document.

**(a) `Good Fit` gets a positive condition** (design spec §6a). D34 did this for `No Fit`
— *"a claim, not a residue"* — and it held: the transitive collapse into `No Fit` is
gone. The same sentence is now written for `Good Fit`: assign it when the CV evidences
the role's **core activity** in a substantive capacity and neither step 1 nor step 2
applies, and **then assign it** — a missing tool, industry, employer size or year is a
screening question the guide has already ruled on, not a reason to withhold.

**(b) The tie-break is scoped to the one question it is for.** D34 scoped it to the
`Good`/`Potential` boundary and barred it from reaching `No Fit`. It is now scoped
further, to *whether the core activity is evidenced at all*, and explicitly does not
apply when the activity is evidenced and something else is missing. That case is decided.

**(c) The judge emits a `score`, 0-100, beside the class** (design spec §6b). This is the
part worth paying 600 spawns for: after it, moving a boundary is a re-scoring of labels
on file rather than a new instrument and another 600 spawns. The guide describes the
scale as *how strongly the CV evidences the role's core activity, at a career stage the
role can use*, gives five band descriptions, and says in terms that **the bands are not
label boundaries and the judge should not try to make them line up** — a score and a
label that disagree are information, and a judge that reconciles them internally destroys
it.

**(d) The agent definition's `model: haiku` frontmatter is removed.** M2 made the
subprocess argv the authority on which model answers (`judge_argv` passes `--model`). A
`model:` line in the committed agent file is a second, unenforced assertion of the same
fact — exactly the shape M2 removed — and it disagreed with runs 7/8 on their face while
they ran `sonnet`. It changes `agent_sha256`, which move 3 changes anyway, so it is free
here and would not be free later. **It is confounded with (a)-(c) inside the new
instrument** and no finding may attribute anything to one clause rather than another.

**The model does not change.** Runs 9 and 10 are `claude-haiku-4-5`, the committed model
after move 2. Changing the guide and the model together would leave neither attributable
— the mistake move 2 exists to avoid, one move later.

## 2. The dispatch

| run | draw | pairs | model |
|---|---|---|---|
| **9** | `a2_generic` | 400 | `claude-haiku-4-5` |
| **10** | `a2_targeted` | 200 | `claude-haiku-4-5` |

Two run numbers because D35 gives a run one draw. `--coverage` must report **400/400** and
**200/200** before either `--collect`, and `llm-judge-failed.jsonl` must hold no run 9 or
10 row: run 5 lost 202 of 395 spawns to a usage limit and would have reported itself
complete pre-F1. **A retry pass is budgeted and is not a deviation.**

## 3. The sanity checks, all of which must pass before any figure is read

1. `--coverage` exact on both runs before `--collect`.
2. `llm-judge-failed.jsonl` clean for runs 9 and 10.
3. **Every run 9/10 row carries the new `agent_sha256`**, and it is what
   `docs/annotation-guide.md` renders (`verify --derived`, `agent_definition_is_current`).
4. **Every parsed run 9/10 row carries a score** in 0-100, and no run 1-8 row carries one
   (`verify --derived`, `score scheme` and `score range`).
5. `model = claude-haiku-4-5` on every row, matching the dispatch record
   (`assert_run_holds_one_model`).
6. **No figure pools runs 9/10 with runs 5-8.** `by_instrument` keys them apart; runs 5
   and 9 differ on `agent_sha256`, so `self_consistency` must refuse the pair and list it
   under `not_self_consistency`.
7. The A1 recheck report does not move — `judge_frames` reads `RECHECK_CSV` alone.
8. **`prompt_sha256` for a given pair must differ from run 5's / run 6's.** The prompt now
   asks for a score, so identical hashes would mean the old prompt was re-sent. This is
   the inverse of move 2's check 3, and load-bearing for the same reason: it establishes
   that one thing changed and which.

## 4. The registered statistics

All against the **consensus** reference (D37), per stratum, with coverage attached
(60 of 126, 47.6% on `generic`; 97 of 200, 48.5% on `targeted`) and the selection-bias
note: consensus selects the pairs two humans found unambiguous, so every figure below is
an **upper** estimate.

Baseline is the committed instrument `75e00ea3` under `haiku` — runs 5/6, as committed at
`fa5532b`. `sonnet` (runs 7/8) is shown because move 2 measured it, and is **not** a
target: it reached its `Good Fit` recall by getting harsher elsewhere.

| | statistic | `generic` haiku | `targeted` haiku | (`sonnet`, for reference) |
|---|---|---|---|---|
| **G** | consensus `Good Fit` per-class agreement | 0.136 (n=22) | 0.167 (n=18) | 0.364 / 0.389 |
| **C** | share of consensus `Good Fit` → llm `Potential Fit` | 0.773 (17/22) | 0.778 (14/18) | 0.273 / 0.611 |
| **K** | consensus three-class kappa | 0.389 [0.235, 0.545] | 0.401 [0.259, 0.539] | 0.415 / 0.522 |
| **B** | consensus binary kappa | 0.729 [0.540, 0.889] | 0.584 [0.422, 0.732] | 0.475 / 0.722 |
| **I** | gate verdicts (retain `van` / retain `yc`), three-class | fail / pass | tie / tie | see §5 |
| **A** | kappa(yc, llm) − kappa(van, llm) on the shared pairs | 0.455 − 0.142 = **0.313** | 0.249 − 0.204 = **0.045** | 0.188 / 0.312 |
| | llm `Good Fit` issue rate | 4.0% | 3.0% | 10.8% / 8.0% |

**The bars, fixed now:**

- **G ≥ 0.40** — the per-class bar, absolute, unchanged since Phase 0 and unchanged by
  move 2 missing it at 0.364 / 0.389. It is still 0.40.
- **C < 0.50** — the residual collapse is no longer the majority of the class.
- **B ≥ the baseline's lower CI bound** — 0.540 on `generic`, 0.422 on `targeted`. This is
  the **sonnet guard**: `Good → Potential` does not cross the fit/no-fit line but
  `Potential → No Fit` does, and move 2's unregistered finding was a judge that bought G
  by moving 9 of 16 consensus `Potential Fit` pairs into `No Fit`, halving B on `generic`.
- **A ≤ 0.20** — declared now, before the run. Beyond that the judge agrees with `yc`
  substantially more than `van` does, which is fitting one annotator rather than
  calibrating a judge; the plan's rejection list named it in advance and `sonnet` did it
  at 0.312 on `targeted`. `generic`'s baseline is **already 0.313** and therefore already
  over the bar — so on `generic` A is registered as a *do-no-harm* clause (it may not
  rise) rather than a bar to clear, and the finding must say which it applied.
- **I must not lose a row** — no gate row that reads `pass` or `tie` under the baseline
  may read `fail` under the graded instrument.

## 5. The outcomes, and what each licenses

Read in this order. The first that holds is the reading.

**Outcome C — the improvement is bought elsewhere.** G rises but *any* of: B falls below
its registered floor, a gate row that passed or tied now fails, or A rises above 0.20 on
`targeted` / above its baseline on `generic`. → **Report and do not commit the graded
instrument.** This is `sonnet`'s failure mode reached by a different route; the committed
instrument stays `75e00ea3` and the guide edit is reverted or held unmerged. Checked
first, because it is the one an eager reader would skip.

**Outcome A — the boundary is fixed.** G ≥ 0.40 **and** C < 0.50 on **both** strata, with
no outcome-C violation. → **Commit the graded instrument** as the judge, re-quote every
A2 figure under the new `agent_sha256`, and close **Q38**. It licenses the judge as a
second annotator in a future double-labelling wave (design spec §4) and still licenses
nothing about ground truth: **D36 stays unmade** whatever this move returns.

**Outcome B — no material change.** G's interval overlaps the baseline's on both strata.
→ **Q38 closes as "the boundary is not reachable by editing the instrument"**, which is a
real finding and the end of the calibration road this plan opened: model (move 2) and
instrument (move 3) have both been tried against the same corpus. The committed
instrument stays `75e00ea3`, and the score column survives as the reusable part.

**Outcome D — stratum-dependent.** The conditions of A hold on exactly one stratum. →
**D29 forbids picking the stratum that suits.** The committed instrument does not change,
and the finding records which stratum moved and by how much. This is what move 2 returned
and it is not an unlikely outcome here.

## 6. The score, and what a fitted cut point is not

The score is registered separately, because it is what makes future boundary moves free
and it is worth something even under outcomes B, C and D.

**S1 — does the score separate the classes the label does not?** Reported per stratum
against consensus: the score's distribution per consensus class, and its rank separation
of consensus `Good Fit` from consensus `Potential Fit` — the one cell the label collapses.
No bar, because the point is descriptive: a score that separates a boundary the label
does not is the object this move was buying.

**S2 — cut points fitted against consensus, per stratum.** Two thresholds maximising
three-class agreement with consensus. Every fitted figure carries:

- `fitted_on` — the pair ids it was fitted on, not a prose note (work-plan task 3.4, with
  `test_a_fitted_cut_point_names_the_pairs_it_was_fitted_on` as its regression test);
- the consensus coverage and the selection-bias note, like every consensus figure;
- an explicit statement that **a fitted cut point is not an out-of-sample result on the
  pairs it was fitted on**, and that its kappa may not be compared with the emitted
  label's as though the two were measured the same way.

**S3 — one genuinely held-out estimate.** The consensus pairs of each stratum are split in
half by a seeded draw; cut points are fitted on one half and quoted on the other. n is
about 30 on `generic` and 48 on `targeted`, so the interval will be wide and is reported
as such. This is the only number in move 3 that may be read as an out-of-sample estimate
of what a re-scored judge would do, and it is registered **before** it is computed
precisely so that a wide interval cannot afterwards be replaced by the full-data fit.

Cross-stratum transfer (fit on `generic`, quote on `targeted`) is **not** registered as
out-of-sample: the two strata are different populations by D29, so transfer between them
measures the populations as much as the cut points. It may be reported as a curiosity,
labelled as one.

**No cut point is committed by this move.** S2 and S3 report what re-scoring would do.
Adopting a re-scored label as the judge's output is a separate decision with its own
document, and it needs a reference on 100% of pairs, which consensus is not (design
spec §3).

## 7. What move 3 does not do

- **It does not make D36.** Unchanged from the design spec §7 and unchanged by any
  outcome above.
- **It does not touch the A1 recheck.** `judge_frames` reads `RECHECK_CSV` alone; the
  D33/D34 figures do not move, and runs 1-3 against 4+ remain incomparable (Q32, outcome
  D).
- **It does not re-open Q34.** Option D's 639 pre-D34 rows stay unanalysed, and run
  numbers 9 and 10 are taken by this move as the work plan says they are.
- **It does not re-judge under `sonnet`.** A graded instrument under a second model is two
  axes moving at once. If it is ever wanted it is a further dispatch with its own runs.
- **It does not resolve Q42 or Q41.** Both need human time this project does not have.
