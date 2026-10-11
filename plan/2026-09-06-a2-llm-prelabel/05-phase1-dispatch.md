# Phase 1 — `a2_targeted` pre-labelled, human leg pending

**Date:** 7 Sep 2026 · **Run 6**, draw `a2_targeted`, n=200, instrument `75e00ea3`
(post-D34), model `claude-haiku-4-5` · `docs/data/manifests/llm-a2-targeted.csv`

The 200 pairs of batch 2 are judged. **Their human session has not started**, which is the
point: the judge saw no label and the annotator will see no machine label, so when the
session finishes the comparison is genuinely prospective and held out. Run 5 (`generic`) is
retrospective by construction and is the baseline this will be read against.

Same instrument and same model as run 5, so the two strata are comparable — separately
quoted, never pooled *(D29)*.

## Status

| Stratum | Draw | Run | LLM | Human | Scored |
|---|---|---|---|---|---|
| `generic` | `a2_generic` | 5 | 400/400 | 400 (`yc`) | **0.380** / 0.637 binary |
| `targeted` | `a2_targeted` | 6 | 200/200 | **0** | pending — `no overlap` |

`a2_report` reports `targeted` as `no overlap` rather than inventing a figure. It will
score itself the moment human labels land; nothing further needs dispatching.

## What is already readable without the human leg

The judge's **marginal distribution**, which needs no reference labels:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| llm, `generic` (n=400) | 0.040 | 0.590 | 0.370 |
| llm, `targeted` (n=200) | 0.030 | 0.575 | 0.395 |
| **`yc`, `generic` (n=400)** | **0.252** | 0.345 | 0.402 |

**Two things follow.**

**The instrument is stable across the strata.** The judge's three proportions move by at
most 1.5 points between `generic` and `targeted`, on populations selected by different
rules. Whatever it is doing, it is doing consistently, so the prospective figure on
`targeted` is likely to land near `generic`'s 0.380 — which is a prediction this plan
makes *before* the human labels arrive, and therefore one worth recording and checking.

**It strengthens the Q38 hypothesis.** `No Fit` is nearly matched (0.370 against `yc`'s
0.402), but the judge issues `Good Fit` at **4.0% where `yc` issues it at 25.2%** — a
sixfold under-issuance. Phase 0 established the disagreement is one-directional at the
`Good`/`Potential` boundary; this says the judge does not merely disagree case by case,
it **almost never reaches `Good Fit` at all**.

A near-total absence of a class is the signature of a mis-set threshold rather than of
case-by-case judgement differences, which would scatter. That is an inference about shape,
not a proof: `yc` over-issuing `Good Fit` remains consistent with the same numbers, and
**one annotator still cannot separate the two**. It does make the threshold reading the
more economical explanation, and it sharpens what Q38 would fix — the boundary's position,
not the reasoning around it.

## What this does not change

Phase 0's refusal stands. Three-class substitution fails the gate's absolute per-class
clause at 0.119 on `Good Fit`, and nothing here revisits that.

## Interruption

The account's usage limit landed again, 100 pairs in: **13 of 100 spawns failed**, all
`exit 1` with both streams empty. Quarantined, `judge()` raised, and the retry after the
reset judged exactly the 13 plus the outstanding 100 — 113 attempted, 0 failed, coverage
200/200. Second time in two runs; the recovery cost was one command each time.

## Next

1. **Run the batch 2 human session.** Nothing blocks it — `build_queue` serves all 200,
   and the guard for that is `test_a_pair_the_judge_has_labelled_is_still_owed_by_the_human`.
2. Re-run `--a2-report`; `targeted` scores itself.
3. Check the prediction above: does `targeted` land near 0.380?
4. Phase 2 needs the second annotator, and their draw must carry enough `Good Fit` pairs
   to measure the boundary — a draw thin there answers the wrong question.
