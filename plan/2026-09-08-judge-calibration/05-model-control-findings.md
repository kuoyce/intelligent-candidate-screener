# Move 2 — the model control. Q32 closes, and the answer is **outcome D**

**Date:** 8 Sep 2026
**Registered in advance:** [`04-run7-registration.md`](04-run7-registration.md), committed
at `7c5262b` **before** any run 7 or 8 prompt existed. Everything below is read against
that document and nothing else.
**Cost:** 600 spawns, zero human annotation hours, zero retries.
**Runs:** 7 (`a2_generic`, 400, `claude-sonnet-5`), 8 (`a2_targeted`, 200,
`claude-sonnet-5`).

---

## 1. The experiment held its one variable

All seven registered sanity checks pass.

| # | check | result |
|---|---|---|
| 1 | coverage exact before collect | **400/400** and **200/200** |
| 2 | `llm-judge-failed.jsonl` clean for runs 7/8 | **0 rows**. No retry pass was needed |
| 3 | **every `prompt_sha256` equals run 5's / run 6's for the same pair** | **0 drift**, same pair set, 400 and 200 |
| 4 | `agent_sha256` = `75e00ea3` | on every row |
| 5 | `model = claude-sonnet-5` on every row, matching the dispatch record | enforced at judge time by `assert_run_holds_one_model`, re-checked by two new `verify --derived` clauses |
| 6 | run 5 vs 7 not reported as self-consistency | `self_consistency` is empty; `run5_vs_run7` and `run6_vs_run8` are under `not_self_consistency` |
| 7 | the A1 recheck report does not move | `judge_frames` reads `RECHECK_CSV` alone |

Check 3 is the load-bearing one: the prompts are **byte-identical**, so this is two
models and not two dispatches. Runs 7 and 8 are also the first runs in this repository
whose model was *passed to the subprocess* rather than asserted beside it (M2).

## 2. The registered statistics

Consensus reference, per stratum. `haiku` is runs 5/6 as committed at `ca35e5a`;
`sonnet` is runs 7/8.

| | | `generic` | | `targeted` | |
|---|---|---|---|---|---|
| | | haiku | **sonnet** | haiku | **sonnet** |
| **G** | consensus `Good Fit` per-class | 0.136 | **0.364** | 0.167 | **0.389** |
| **C** | share of consensus `Good Fit` → llm `Potential Fit` | 0.773 | **0.273** | 0.778 | **0.611** |
| **K** | consensus three-class kappa | 0.389 [0.235, 0.545] | **0.415** [0.255, 0.574] | 0.401 [0.259, 0.539] | **0.522** [0.382, 0.664] |
| **B** | consensus binary kappa | 0.729 | **0.475** | 0.584 | **0.722** |
| | llm `Good Fit` issue rate | 4.0% | **10.8%** | 3.0% | **8.0%** |

Consensus `Good Fit` confusion (n=22 `generic`, n=18 `targeted`):

| | Good | Potential | No |
|---|---|---|---|
| `generic` haiku | 3 | **17** | 2 |
| `generic` **sonnet** | **8** | 6 | 8 |
| `targeted` haiku | 3 | **14** | 1 |
| `targeted` **sonnet** | **7** | 11 | 0 |

## 3. The reading, in the registered order

**Outcome C — materially worse. Does not hold.** K(sonnet) = 0.415 and 0.522, against
the registered floors of 0.235 and 0.259. Sonnet is not worse on the tertiary statistic
on either stratum.

**Outcome A — model capacity. Does not hold.** Both limbs fail:
- G ≥ 0.40 on **both** strata: **0.364** and **0.389**. Both move a long way — +0.228
  and +0.222, roughly tripling and doubling — and both **miss the registered bar**. The
  bar was 0.40 before the run and stays 0.40 after it.
- C < 0.50 on **both** strata: 0.273 on `generic` but **0.611** on `targeted`.

**Outcome D — the effect is stratum-dependent. This is the reading.** Outcome A's
condition (G ≥ 0.40 *or* C < 0.50) holds on **`generic`** (C = 0.273) and not on
**`targeted`** (G 0.389, C 0.611) — exactly one stratum.

**Q32 closes as "stratum-dependent".** Three consequences follow from the registration:

1. **Move 3 proceeds.** The `Good`/`Potential` boundary is not fixed by changing the
   model.
2. **D29 forbids picking the stratum that suits.** `generic`'s C = 0.273 is not a licence;
   it is half of a result whose other half is 0.611.
3. **Runs 1-3 (`sonnet`) and runs 4+ (`haiku`) on the A1 recheck remain incomparable.**
   The control D33 and D34 never had is still missing for that corpus — A1 is a third
   population, and this move measured a model effect that varies *between* populations.
   The D33/D34 figures keep their model as a stated property, not a nuisance parameter.
4. **The committed model stays `claude-haiku-4-5`.** Outcome D licenses no model change,
   and `a2_frames` now defaults to the committed model rather than the newest run, so
   dispatching this control did not silently change what `vs_llm` names.

## 4. What sonnet actually did — and why it is not "better"

**It is a different judge, not a more accurate one.** The plan's own rejection list says
so in advance: *"Chasing kappa(yc, llm) upward. Beyond ~0.25 on `targeted` that means
agreeing with `yc` more than `van` does — fitting one annotator, not calibrating a
judge."* That is precisely what happened.

| stratum | reference | haiku | sonnet | |
|---|---|---|---|---|
| `targeted` | `yc` (n=200) | 0.249 | **0.453** | toward `yc` |
| `targeted` | `van` (n=200) | 0.204 | **0.141** | away from `van` |
| `generic` | `yc` (n=400) | 0.380 | 0.358 | slightly away |
| `generic` | `van` (n=126) | 0.142 | 0.191 | slightly toward |

On `targeted`, sonnet at 0.453 against `yc` sits **more than twice the human ceiling of
0.214** while falling to 0.141 against `van`. A judge that agrees with one annotator far
more than the two annotators agree with each other has not found the truth; it has found
`yc`. The interchangeability gate says the same thing in its own terms: sonnet **fails**
the `retain van` row on both strata (0.191 against a 0.250 ceiling, 0.141 against 0.214)
where haiku failed it once and tied once.

**And it bought its `Good Fit` recall by getting harsher elsewhere.** Sonnet issues
`No Fit` on 53.0% of `generic` against haiku's 37.0%, and consensus binary kappa on
`generic` **falls 0.729 → 0.475** — with kappa(yc, llm) binary falling 0.637 → 0.481 and
kappa(van, llm) binary 0.431 → 0.287. `Good ↔ Potential` does not cross the fit/no-fit
line but `Potential ↔ No Fit` does, and sonnet moved 9 of 16 consensus `Potential Fit`
pairs to `No Fit` on `generic`. **This was not a registered statistic and is reported as
an unregistered finding**: the registration made K the only "materially worse" test, and
K rose. It is recorded here so the next reading of Q40 has it.

**The per-class clause is unchanged by the model.** `Good Fit` and `Potential Fit` remain
`unidentified_by_instrument` under both models on both strata — that is a property of the
human pair, and no model change can move it.

## 5. What did not move

Every haiku figure. Runs 5 and 6 are untouched and reproduce exactly: 0.380 / 0.142 on
`generic`, 0.249 / 0.204 on `targeted`, consensus 0.389 / 0.401, ceilings 0.250 / 0.214.
`llm-recheck-report.json` and the committed D33/D34 figures do not move.

## 6. Artefacts

- `docs/data/manifests/llm-a2-generic.csv` — **800 rows** (400 haiku run 5, 400 sonnet run 7)
- `docs/data/manifests/llm-a2-targeted.csv` — **400 rows** (200 haiku run 6, 200 sonnet run 8)
- `output/annotation/llm-a2-report.json` — each stratum now carries `by_model`, a complete
  figure set per model (`vs_llm`, `consensus`, `interchangeability`, `label_distribution`).
  The section-level `vs_llm` / `consensus` / `interchangeability` are the **committed**
  model's block repeated inline, named by `llm_model`. No field holds a series built from
  two.
- `verify --derived` gains `model provenance` (a row's model matches its dispatch record)
  and `one model per run`.

## 7. Acceptance

```
uv run pytest                                               226 passed
uv run python -m candidate_screener.data.verify --derived   all derived checks passed
uv run python -m candidate_screener.baselines.run --check   the baseline reproduces
```

## 8. Open questions after this move

- **Q32 — closes**, as "the model effect is stratum-dependent". It does *not* license
  comparing runs 1-3 with runs 4+ on the A1 recheck.
- **Q38** — the `Good`/`Potential` boundary. **Move 3 is the indicated next step.** Its
  premise is strengthened: a model with visibly more headroom recovers consensus
  `Good Fit` at 0.364 / 0.389, still short of the bar, and pays for it in binary
  agreement on `generic`.
- **Q40** — binary substitution. **Weakened further.** It was already not defensible on
  both strata under haiku (0.637 / 0.371); sonnet inverts which stratum is the good one
  (0.481 / 0.591) without making both defensible. Carried.
- **Q43** — runs 1-6's model provenance is an assertion. Runs 7/8 are the first with an
  enforced one; the question is unchanged for the earlier runs.
- **Q44 (new)** — sonnet reaches kappa 0.453 against `yc` on `targeted` while falling to
  0.141 against `van`, against a human ceiling of 0.214. Is a judge that tracks one
  annotator more closely than the annotators track each other *useful* for anything, or
  is it only a measurement of `yc`? The interchangeability gate says it is a bad
  replacement for `van` specifically. Not actionable without a third annotator, and D25
  has no budget for one.
