# Move 2 — the reading, registered before the dispatch

**Date:** 8 Sep 2026, written and committed **before** any run 7 or run 8 prompt exists.
**Closes:** Q32 — `sonnet` (runs 1-3) against `haiku` (runs 4+) with no control, open
since 5 Sep 2026.
**Why in advance:** a result that can be read either way afterwards measures nothing.
The statistics, their cut points and the decision each outcome licenses are fixed here,
and `05-model-control-findings.md` may only report against this document.

---

## 1. The experiment is one variable

| held fixed | how it is held |
|---|---|
| the instrument | `agent_sha256 = 75e00ea3`, the committed judge, byte-identical. `assert_agent_definition_current` refuses a dispatch against anything else |
| the pairs | the same 600: all 400 of `a2_generic`, all 200 of `a2_targeted`, from `indomain_rows()` |
| the prompts | re-rendered by the same code from the same rows, so **every `prompt_sha256` must equal run 5's / run 6's for the same pair**. This is checked, not assumed (§5) |
| the humans | `yc`, `van` and `consensus` as they stand today; no annotation happens in this move |
| the strata | two runs, because D35 gives a run one draw |

| varied | |
|---|---|
| the model | `claude-haiku-4-5` (runs 5, 6) against `claude-sonnet-5` (runs 7, 8) |

**Run 7** = `a2_generic`, 400 pairs, `claude-sonnet-5`.
**Run 8** = `a2_targeted`, 200 pairs, `claude-sonnet-5`.

Both land in the existing draw files beside runs 5 and 6, separated by the `model`
column. This is safe only because M1 landed: before it, `a2_frames` took a per-pair
majority across every run sharing `agent_sha256`, and a second model would have been
folded into that majority — dropping every pair the two models disagreed on and keeping
exactly those they agreed on.

## 2. The incumbent — what `haiku` scores today

Everything below is from `output/annotation/llm-a2-report.json` as committed at
`ca35e5a`, and is the comparison set. No figure in this table may be recomputed after
run 7 lands.

| statistic | symbol | `generic` (run 5) | `targeted` (run 6) |
|---|---|---|---|
| consensus `Good Fit` per-class agreement | **G** | **0.136** | **0.167** |
| share of consensus `Good Fit` returning llm `Potential Fit` | **C** | **0.773** (17/22) | **0.778** (14/18) |
| consensus three-class kappa | **K** | **0.389** [0.235, 0.545] | **0.401** [0.259, 0.539] |
| consensus binary kappa | **B** | 0.729 [0.540, 0.889] | 0.584 [0.422, 0.732] |
| kappa(yc, llm), whole stratum | | 0.380 | 0.249 |
| kappa(van, llm), whole stratum | | 0.142 | 0.204 |
| llm `Good Fit` issue rate | | 4.0% | 3.0% |

The consensus reference covers 60 of 126 pairs on `generic` and 97 of 200 on `targeted`,
and every consensus figure here is an **upper** estimate for the reason design spec §1
gives. That does not weaken the comparison: both models are scored against the *same*
subset, so the bias is common to both and the **difference** is what this move reads.

## 3. The primary statistic, and the cut points

**Primary: G**, consensus `Good Fit` per-class agreement, per stratum. It is the cell the
whole plan turns on — the judge issues `Good Fit` at 4.0% / 3.0% against `yc`'s
25.2% / 16.5% and `van`'s 51.6% / 30.0%, and against consensus the residual error is
almost entirely one cell.

**Secondary: C**, the share of consensus `Good Fit` the judge returns as `Potential Fit`.
G and C are near-complements today but need not be: a judge that moved those pairs to
`No Fit` instead would leave G unchanged and lower C, which is a different — and worse —
outcome, so both are registered.

**Tertiary: K**, consensus three-class kappa with its bootstrap CI, used only for the
"materially worse" test and for the overlap test.

Bootstrap CIs are `kappa_ci`, 2000 resamples, `seed=0`, as everywhere else in this
repository.

## 4. The four outcomes, evaluated in this order

Exhaustive and mutually exclusive by construction. The first one whose condition holds
is the reading.

### Outcome C — `sonnet` is materially worse. **Report and stop.**
**Condition:** K(sonnet) falls below the *lower bound* of K(haiku) on **either** stratum
— that is, K < **0.235** on `generic` or K < **0.259** on `targeted`.
**Licenses:** the committed model stays `claude-haiku-4-5`. Runs 7/8 stay on file as a
recorded negative. Move 3 proceeds under `haiku`. Q32 closes as "the model matters and
`haiku` is the better one on this corpus", which also means runs 1-3 and runs 4+ on the
A1 recheck **may not** be compared and the D33/D34 figures acquire a model caveat.

### Outcome A — the failure was model capacity. **Commit `sonnet`; move 3 is not needed.**
**Condition:** on **both** strata, G ≥ **0.40** — *or* on **both** strata, C < **0.50**.
**Licenses:** commit `claude-sonnet-5` as the judge's model, re-quote every A2 figure
under it, and close Q38 without a guide edit. Move 3's part (b) — the graded score —
remains separately arguable on its own merits but is no longer required by the boundary.

### Outcome D — the effect is stratum-dependent. **Move 3 proceeds.**
**Condition:** outcome A's condition holds on exactly **one** stratum.
**Licenses:** nothing about substitution. **D29 forbids picking the stratum that suits**,
so a one-stratum improvement is not an improvement to the judge; it is a finding that the
model effect interacts with the population. Q32 closes as "stratum-dependent"; runs 1-3
against runs 4+ on the A1 recheck stay incomparable, since A1 is a third population.

### Outcome B — no material model effect. **Move 3 proceeds.**
**Condition:** none of the above — K(sonnet)'s CI overlaps K(haiku)'s on both strata and
G stays below 0.40 on at least one, with C at or above 0.50 on at least one.
**Licenses:** Q32 closes as "no material model effect on this corpus", which
retroactively licenses comparing runs 1-3 (`sonnet`) with runs 4+ (`haiku`) on the A1
recheck — the control D33 and D34 never had. The `Good Fit` failure is then the
instrument or the class definition, not the model, and **move 3 is the indicated next
step**. The committed model stays `haiku` on cost grounds, and the report says so.

**None of the four licenses substitution.** Whatever the model, the consensus reference
covers 48% of pairs, three-class per-class agreement is what the gate reads, and D36 stays
unmade. A model change cannot license what a reference on half the corpus cannot support.

## 5. Sanity checks that gate the reading

Every one of these must pass **before** `05-model-control-findings.md` quotes a figure.
A failure here means the experiment did not hold its one variable, and the run is
re-dispatched rather than interpreted.

1. **Coverage is exact.** `--coverage` reports 400/400 for run 7 and 200/200 for run 8
   before either `--collect`. Run 5 lost 202 of 395 spawns to a usage limit and, pre-F1,
   would have reported itself complete and scored 0.301/0.554 over 198 latency-ordered
   survivors instead of the true 0.380/0.637 over 400. Budget for a retry pass.
2. **`llm-judge-failed.jsonl` holds no run 7 or run 8 row** at collect time.
3. **The prompts are byte-identical to runs 5/6.** For every pair, run 7's
   `prompt_sha256` equals run 5's and run 8's equals run 6's. This is the *proof* that
   only the model varied; without it the comparison is between two dispatches, not two
   models.
4. **`agent_sha256` is `75e00ea3`** on every run 7/8 row.
5. **Every run 7/8 row carries `model = claude-sonnet-5`**, and the dispatch record for
   those runs names the same — enforced at judge time by
   `assert_run_holds_one_model` (M2) and re-checked in `verify --derived`.
6. **`self_consistency` reports nothing for run 5 vs run 7** (or 6 vs 8). Those are model
   comparisons; M1 routes them to `not_self_consistency`.
7. **The A1 recheck report does not move.** `judge_frames` reads `RECHECK_CSV` alone and
   no A2 row reaches it.

## 6. What this move does not do

- It does not touch the guide. `agent_sha256` is unchanged, which is the point.
- It does not re-open Q34, and reserves no run number for it.
- It does not resolve Q43 (runs 1-6's model provenance is an assertion). Runs 7/8 are the
  first with an enforced one.
- It does not make D36.
