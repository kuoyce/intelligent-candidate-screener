# Phase 3 — Implementation Record

**Branch:** `feat/derived-artefacts` · **Covers:** task 3.2 only. 3.1, 3.3, 3.4 and 3.5 are
not started.

## 1. What was built

```
src/candidate_screener/data/
  ids.py             content-addressed document identity (design spec §2)
  fit_split.py       task 3.2 — pooling, the doubly-disjoint split, the yield survey
  build.py           `build --all --seed 0`; a registry the later tasks join
  verify_derived.py  `verify --derived`; the acceptance checks, likewise a registry
docs/data/manifests/ fit-split.csv, fit-split-yield.json, fit-split-survey.json,
                     fit-pair-conflicts.csv, fit-c4-overlap.csv, README.md
data/processed/fit/  train.parquet, test.parquet   (git-ignored)
```

```bash
uv run python -m candidate_screener.data.build     --all --seed 0
uv run python -m candidate_screener.data.verify    --derived
uv run python -m candidate_screener.data.fit_split --survey   # the evidence, writes no artefact
```

All five acceptance checks pass: identity (994 IDs, 0 unresolved, no repeats), disjointness
(0 documents in more than one split, no cross-split resume or JD overlap), accounting
(3,338 discarded + 4,649 assigned = 7,987 usable), determinism (rebuild at seed 0 reproduces
the committed manifest), and the C4 contamination guard.

## 2. The split as built

**30% hold-out, no val fold, seed 0.**

| Split | Pairs | Resumes | JDs | JDs w/ Good Fit | Median Good Fit per JD |
|---|---|---|---|---|---|
| train | 3,990 | 446 | 245 | 90 | 11 |
| test | **659** | 158 | **100** | **31** | **6** |
| discarded | 3,338 | — | — | — | — |

Documents: 450 / 193 resumes and 246 / 105 JDs assigned train / test.

## 3. Findings — three of them change decisions already recorded

### 3.1 The predecessor's yield table does not reproduce, and 25% was the wrong fraction

`03-acquisition-action-plan.md` §3.2 recommended a 25% hold-out on the strength of a table
claiming **30 JDs with a Good Fit** — "slightly more than the shipped test split's 28". The
sampling mechanism behind that table was never recorded. It was not reproduced by any of five
natural procedures tried (fresh vs. carried RNG state, `pandas.sample`, `random.sample`,
sorted vs. unsorted uniques), and no attempt was made to search for a seed that would match:
that would be fitting the evidence to the conclusion.

Measured by the committed implementation, at seed 0 across 20 seeds `[min–max]`:

| Hold-out | Test pairs | Test JDs | **JDs w/ Good Fit** | Train pairs |
|---|---|---|---|---|
| 15% | 169 [118–227] | 49 [37–49] | 12 [7–20] | 5,841 [5,540–6,099] |
| 20% | 298 [238–374] | 67 [58–67] | 20 [11–27] | 5,159 [4,906–5,457] |
| 25% | 442 [418–620] | 82 [78–85] | **24 [15–32]** | 4,577 [4,073–4,801] |
| **30%** | **659 [610–838]** | **100 [94–102]** | **31 [23–40]** | **3,990 [3,541–4,149]** |
| 35% | 885 [861–1,117] | 116 [113–122] | 36 [27–48] | 3,527 [3,006–3,611] |

**25% yields 24, not 30.** The evaluation strength the plan was approved on costs a 30%
hold-out, and that costs 587 training pairs against 25% — 13%, for 29% more scoreable
queries. Approved by the team 23 Aug 2026.

### 3.2 A single draw was never evidence

`jds_with_good_fit` is *n* for every retrieval figure task 3.3 can report. At a 25% hold-out
it ranges **15 to 32** across seeds — a spread wider than the gap between adjacent hold-out
fractions. Choosing a fraction from one draw would have been choosing noise. `--survey` now
reports `seed0 [min–max]` for every cell, and the seed is fixed at 0 and committed so every
stage and every team member scores identical documents.

This does **not** license re-drawing until a favourable seed appears. Seed 0 was fixed before
the survey ran; the spread is reported so the reader knows the width of what they are reading.

### 3.3 A held-out val fold is not worth its price — D16's stated fallback is taken

D16 approved a three-way split, with the plan's own escape clause: *"if val cannot reach that,
fall back to cross-validation on train and say so."* Measured at seed 0:

| Design | Train pairs | Val pairs | Val JDs w/ Good Fit | Test unchanged? |
|---|---|---|---|---|
| 30% / no val | 3,990 | — | — | — |
| 30% / 10% val | 3,022 | 63 | 10 | yes |
| 30% / 15% val | 2,602 | 157 | 11 | yes |

A val fold costs ~968–1,388 training pairs and returns a fold of 10–11 scoreable queries —
too thin to tune against, and thinner than the test set it is supposed to protect. Because
splits are assigned `test` first, raising `val_frac` never moves a document out of test: the
cost falls entirely on train.

**Taken instead:** 5-fold doubly-disjoint cross-validation *inside* train (`fit_split.cv_folds`)
— same disjointness rule, 799 evaluation pairs in total across folds, 10–17 good-fit JDs and
2,466–2,655 fitting pairs per fold, and it spends nothing permanently. `val.parquet` is
deliberately absent rather than empty.

### 3.4 Pooling surfaced 7 duplicate pairs, 6 of them self-contradictory

All 7 lie inside the shipped **train** set, so this is a defect of the source, not of pooling.
6 carry two different labels for the same `(resume, jd)`. A pair the publisher labelled both
ways is not evidence in either direction, so all of its rows are dropped; the one
self-consistent duplicate keeps a single row. 8,000 → **7,987 usable**. Every case is logged
by ID in `fit-pair-conflicts.csv`. This confirms the "6 pairs carry conflicting labels" already
recorded on the A1 card, and locates them.

Also confirmed while pooling: the shipped split is fully **JD-disjoint** (280 + 71 = 351) and
leaks only on the resume axis — 476 test resumes also appear in train.

### 3.5 Open for task 3.3 — relevance density drops from 18 to 6, so Recall@10 may be back

The catalog, the A1 card and the README all state **"do not report Recall@10"**, on the
grounds that the median query has 18 relevant resumes and Recall@10 is therefore capped at
0.56. That figure is the *shipped test split's* density. In the doubly-disjoint split each JD
keeps only its held-out resumes, and the test-side median falls to **6 Good Fit per JD** —
under which Recall@10 is attainable for the median query.

This is **not** actioned here. It changes a metric decision (D9) recorded in three documents
and belongs to task 3.3, which should re-derive the attainability curve on the actual test
pool before anything is edited. Flagged so it is not discovered after the first results exist.

## 4. Deviations from the design spec

| # | Deviation | Why |
|---|---|---|
| W1 | Hold-out is **30%**, not the 25% the predecessor plan recommended | §3.1 above — measured, and approved by the team |
| W2 | **No val fold**; `val.parquet` is not written | §3.3 above — D16's stated fallback, taken on measured cost |
| W3 | `fit-split.csv` carries only `train`/`val`/`test`, never `discarded`, though the spec's schema lists it | `discarded` is a *pair*-level outcome — a pair whose two documents landed in different splits. A document is held out, never discarded. The count is in `fit-split-yield.json` |
| W4 | The "not comparable" note lives in `fit-split-yield.json`, `docs/data/manifests/README.md` and the A1 card — **not** as a `#` header inside `fit-split.csv` | A comment header makes the CSV misparse under a plain `read_csv`. A footgun in the authoritative split manifest is worse than a note one file away |
| W5 | Task 3.2 lives in `fit_split.py` with its own CLI, with `build.py` / `verify_derived.py` as thin registries | The survey is analysis and must be runnable without writing an artefact; the builders are a pipeline. Later tasks register in one line each |
| W6 | No manifest carries a `generated` timestamp, unlike `acquisition-manifest.json` | Byte-reproducibility is an acceptance check. With a timestamp every re-run diffs and the check tests nothing. The git history is the timestamp |
| W7 | `fit-c4-overlap.csv` written — not in the spec's file list | The spec asks `verify --derived` to guard against the rejected C4 derivatives. A pass/fail alone is not applyable; the IDs and their splits are |

## 5. Open questions

| # | Question | Status |
|---|---|---|
| Q14 | Pool depth in-domain, and whether to collect rankings | **Open** — recommendation in `02-work-plan.md`; belongs to 3.4 |
| Q16 | Where more labelled in-domain data comes from if a later stage runs short | **Deferred by design** (D18) |
| **Q17** | *New.* Does the test-side density of 6 Good Fit per JD reinstate Recall@10? | **Open** — §3.5 above; decide inside task 3.3, before results exist |

## 6. Next

Task **3.3** (retrieval pools) is unblocked and is the compute critical path to the first
reportable number. Task **3.4a** (the annotation guide) remains the longest human lead and can
run in parallel.
