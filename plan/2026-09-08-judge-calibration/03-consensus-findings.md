# Move 1 — the consensus reference, and the gate that replaces the old one

**Date:** 8 Sep 2026
**Cost:** 0 spawns, 0 human annotation hours. Everything here is computed from labels
already on file.
**Artefact:** `output/annotation/llm-a2-report.json`, rebuilt.
**Prerequisites landed in the same change:** M1 (model is an axis), M2 (the recorded
model is the model that ran).

---

## 1. What was measured

Nothing new was collected. The repository has held two annotators' labels on the
in-domain corpus since 7 Sep 2026 and had never used them together: every judge figure
was keyed by *one* annotator, and the pairs where the two agreed were not a series.
They are now.

`consensus` is the label **every** annotator present gave, over the pairs where two or
more looked. It is written for *n* annotators so a third does not change the API.

| stratum | multi-labelled | agreed | coverage | consensus `Good`/`Potential`/`No Fit` |
|---|---|---|---|---|
| `generic` | 126 | 60 | **47.6%** | 22 / 16 / 22 |
| `targeted` | 200 | 97 | **48.5%** | 18 / 22 / 57 |

Against it, the committed instrument (`75e00ea3`, `haiku`, runs 5 and 6):

| stratum | 3-class | binary | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|---|---|
| `generic`, n=60 | **0.389** [0.235, 0.545] | **0.729** [0.540, 0.889] | 0.136 | 0.688 | 0.955 |
| `targeted`, n=97 | **0.401** [0.259, 0.539] | 0.584 [0.422, 0.732] | 0.167 | 0.727 | 0.772 |

Both reproduce the figures the plan was approved on.

**The residual is one cell.** Consensus `Good Fit` returning llm `Potential Fit`:
**17 of 22** on `generic`, **14 of 18** on `targeted`. Nothing else in either confusion
exceeds 12, and consensus `No Fit` is held at 0.955 / 0.772.

**Phase 2's "two boundaries in opposite directions" was substantially an artefact of the
reference.** That reading came from measuring against `yc` alone, whose `No Fit` rate
rose to 0.580 on `targeted` while `van`'s stayed at 0.360. Against consensus the
`No Fit ↔ Potential Fit` boundary largely closes and one boundary remains.

## 2. What a consensus figure may and may not be read as

Three rules, all enforced in code rather than in prose. They are the content of
**proposed D37: the reference is a fifth axis** a figure may not be pooled across,
beside `agent_sha256`, `draw`, `stratum` and `annotator`.

**It is not an improvement on the single-annotator figure.** kappa(consensus, llm) =
0.401 beside kappa(yc, llm) = 0.249 on `targeted` is not the judge getting better; the
two are computed over different sets of pairs — 200 and 97 — and it is a different
question asked of a different half of the corpus. `a2_report` has no field holding both,
and `test_consensus_and_single_annotator_figures_are_not_the_same_field` asserts the
two `n` differ and are both reported.

**Coverage travels with it, the way D26's ceiling travels with every Precision@k.**
`assert_consensus_carries_coverage` refuses to emit a block without
`n_multi_labelled`, `n_agreed`, `coverage` and the selection note, so the report fails
to build rather than quoting 0.401 unqualified.

**It is an upper estimate, not an unbiased one.** The consensus subset is not a random
subsample: it selects the pairs both humans found unambiguous. A judge scores higher
there for the reason a student scores higher on the easy half of an exam.

**It therefore does not license substitution.** Measurement asks how good the judge is,
and a reference on 48% of pairs answers it. Substitution asks whether the judge may
produce the labels, and that needs a reference on 100% — the missing 52% is *exactly*
the population a substituting judge would label alone, with no reference, no appeal and
no way to detect being wrong, and it is the harder half by construction. **D36 stays
unmade.**

## 3. The gate is rewritten (Q39 closes, Q40 re-examined)

The old gate had two clauses and both assumed a ground truth this corpus does not have.
The relative clause divided by a human ceiling of 0.214 while the judge sat at
0.204–0.249 — one overlapping cluster, so "the gate passed" and "the gate is not
measuring" produced the same output. The absolute per-class clause refused the judge on
`Good Fit` at 0.119 while **the two humans fail the same bar against each other**
(0.338 / 0.300).

So the question is changed rather than the threshold. Each row **retains** one annotator
and asks whether the judge, standing in the other's seat, is as good a partner as the
person it replaced — every figure on the pairs **all three series share**.

| stratum | retained | replaces | kappa(human, human) | kappa(human, llm) | 3-class | binary (human / llm) | |
|---|---|---|---|---|---|---|---|
| `generic` n=126 | `yc` | `van` | 0.250 [0.134, 0.366] | **0.455** [0.330, 0.569] | **pass** | 0.419 / **0.690** | pass |
| `generic` n=126 | `van` | `yc` | 0.250 | **0.142** [0.051, 0.232] | **fail** | 0.419 / 0.431 | tie |
| `targeted` n=200 | `yc` | `van` | 0.214 [0.119, 0.312] | 0.249 [0.157, 0.346] | tie | 0.292 / 0.371 | pass |
| `targeted` n=200 | `van` | `yc` | 0.214 | 0.204 [0.112, 0.295] | tie | 0.292 / 0.394 | pass |

Read: on `generic` the judge is a **good replacement for `van`** and a **poor
replacement for `yc`**; on `targeted` it is indistinguishable from either. **Binary
passes or ties all four.**

**Nothing here is separated.** Every interval overlaps every other at n=126–200. The
verdict column is a declared indifference band (`INTERCHANGEABILITY_BAND = 0.05`), fixed
before the run rather than chosen after it, and `pass` means "a better partner than the
human it replaces, by more than the band" — never "better than a human".

**Note the restriction matters.** kappa(yc, llm) is 0.380 at n=400 and **0.455** on the
126 pairs `van` also labelled. Reading the first against a ceiling measured on the second
compares two populations, and that is what
`test_the_gate_compares_on_the_pairs_the_three_series_share` guards.

**What passing licenses.** The judge may serve as a **second annotator** in a future
double-labelling wave — a role whose specification is being about as unreliable as a
human. It does not license the judge as an answer key: no agreement figure between two
fallible raters is a claim about correctness.

### The per-class clause, rescoped — and a deviation from the design spec

The bar stays absolute at 0.40 and is now applied to the **human pair as well as the
judge**. Per-class agreement is conditioned on a reference, so a human pair yields two
numbers per class, and the class is marked `unidentified_by_instrument` if **either**
falls below the bar. Requiring both to fail would declare `Good Fit` identified on the
strength of the direction pointing from the rarer user of the class to the more frequent
one.

| class | `generic` human pair | `targeted` human pair | unidentified |
|---|---|---|---|
| `Good Fit` | 0.338 (van) / 0.759 (yc) | 0.300 / 0.545 | **yes** |
| `Potential Fit` | 0.432 / 0.364 | 0.324 / 0.431 | **yes** |
| `No Fit` | 0.917 / 0.415 | 0.792 / 0.491 | no |

**Deviation D1 from the design spec.** §4 named `Good Fit` alone as unidentified. Applied
in both orientations the clause also flags **`Potential Fit`, on both strata** — the two
annotators recover each other's `Potential Fit` at 0.324–0.432. So **two of the three
classes are not identified by the instrument, and only `No Fit` is held consistently**.
This strengthens move 3's premise rather than weakening it, and it widens what a guide
edit would have to address: a pass aimed at the `Good`/`Potential` boundary alone leaves
the `Potential` class as loose as it found it.

## 4. Prerequisites, and the two bugs they close

Both were latent, both are the D35 shape — no exception, no wrong row count, a plausible
figure — and both were shown to fail against the code they guard before being fixed.

**M1 — `a2_frames` would have blended two models.** `llm` was a per-pair majority across
every run sharing the committed `agent_sha256`, and a model change does not change that
hash. Move 2's `sonnet` run of the same guide would have been a two-element majority:
`majority` returns `None` on a tie and `.dropna()` deletes it, so **every pair the two
models disagreed on would have been silently dropped** and the survivors would be exactly
the pairs two models agreed on. Measured against the pre-fix code on a staged 10-pair
draw where the models disagree on half: the live series returns **n=5 of 10**, and all
five are the agreeing pairs. `llm` now keys on `(agent_sha256, model)`, other models are
returned under `per_model`, each stratum names its `llm_model`, and
`self_consistency` refuses a run pair that differs on either axis — runs 5 and 7 are a
model comparison, and reporting them under the field D33 used to bound how high
kappa(A1, llm) could have reached would be a between-judge figure wearing the name of a
test-retest one. They are listed under `not_self_consistency` with their provenance.

**M2 — the recorded model was an assertion, not a measurement.** `MODEL` was a module
constant written into the dispatch record while the subprocess was launched as
`claude -p --agent a1-judge` with **no `--model` flag**. `judge_argv` now passes it, and
`assert_run_holds_one_model` refuses a run whose prompts name two models and a `--judge`
asked for a model the dispatch did not name. `--model` is on `--dispatch` and `--judge`.

**Runs 1–6 are not re-stamped** — their model provenance rests on the constant having
been correct at the time, and re-stamping would replace one assertion with another made
later. **Q43** records the open question: whether that provenance is trustworthy enough
to keep quoting, or whether the D33/D34 figures must carry a caveat. Raised by M2, not
resolved by it.

## 5. What moved, and what did not

`output/annotation/llm-a2-report.json` gains `consensus`, `interchangeability`,
`llm_model`, `models_on_file`, `run_provenance`, `not_self_consistency` and
`note_never_pooled_references` / `note_never_pooled_models`. **Every figure that was in
it before is unchanged**: 0.380 / 0.142 on `generic`, 0.249 / 0.204 on `targeted`,
ceilings 0.250 / 0.214.

`judge_frames`, `llm-recheck-report.json` and the committed D33/D34 figures are
untouched — the A1 recheck is a different code path and a different file, and no A2 row
reaches it.

## 6. Acceptance

```
uv run pytest                                          226 passed
uv run python -m candidate_screener.data.verify --derived   all derived checks passed
uv run python -m candidate_screener.baselines.run --check   the baseline reproduces
```

Eight regression tests were added, each **shown to fail** against the code it guards:

| test | against the pre-fix code |
|---|---|
| `test_two_models_of_one_guide_are_never_one_series` | live series `assert 5 == 10` |
| `test_a_model_comparison_is_not_reported_as_self_consistency` | run5-vs-run7 scored as reliability |
| `test_a_run_is_judged_by_the_model_its_dispatch_names` | argv has no `--model` |
| `test_a_pair_two_annotators_read_differently_is_not_in_the_consensus` | `['j__c0','j__c1'] == ['j__c0']` |
| `test_a_consensus_figure_never_travels_without_its_coverage` | coverage absent, no refusal |
| `test_consensus_and_single_annotator_figures_are_not_the_same_field` | `assert 8 == 2` |
| `test_the_gate_compares_on_the_pairs_the_three_series_share` | `assert 8 == 4` |
| `test_a_class_the_humans_miss_is_unidentified_not_failed_by_the_judge` | `assert False is True` |

## 7. Open questions after this move

- **Q32** — the model control. Unchanged; move 2 answers it. M1 and M2 are now in place,
  which was the condition for dispatching it.
- **Q38** — the `Good`/`Potential` boundary. Sharpened: against consensus the residual
  is one cell (17/22, 14/18), and both humans fail the 0.40 bar on that class against
  each other. Widened by deviation D1: `Potential Fit` is unidentified too.
- **Q39 — closes.** The relative clause is replaced by the interchangeability gate.
- **Q40** — binary substitution. Under the interchangeability gate binary passes or ties
  all four rows, and against consensus it is 0.729 / 0.584. Still **not licensed**: the
  consensus reference covers 48% of pairs and cannot license substitution at all
  (§2), and the two strata's binary figures remain far apart. Carried.
- **Q42** — `yc`'s `No Fit` shift between strata is confounded with D34. Unresolvable
  without human time. Standing limitation, unchanged.
- **Q43 (new)** — runs 1–6 record a model that was never passed to the subprocess.
  Raised by M2, not resolved by it.
