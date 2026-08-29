# Phase 3 — Implementation Record

**Branch:** `feat/derived-artefacts` · **Covers:** tasks 3.2 and 3.3. 3.1, 3.4 and 3.5 are
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

## 3A. Task 3.3 — retrieval pools

### Built

```
src/candidate_screener/data/pools.py            the pools
src/candidate_screener/evaluation/metrics.py    Recall / Precision / nDCG, bootstrap CI, guards
docs/data/manifests/                            pools.csv, pools-yield.json
data/processed/pools/pools.parquet              (git-ignored)
```

```bash
uv run python -m candidate_screener.data.pools        --feasibility   # writes nothing
uv run python -m candidate_screener.evaluation.metrics --sanity       # random-ranker floor
```

**100 queries over a 193-resume candidate universe.** Variants are nested — N20 ⊂ N100 ⊂ Nfull
— so a sensitivity run differs from the primary run by depth alone. `N` is a floor, not a cap:
one query judged against 29 resumes keeps all 29 in its N20 pool rather than losing a
judgement to hit a round number.

| Variant | Pool size | Judged per pool (median) | Queries w/ relevant (strict / graded) |
|---|---|---|---|
| N20 | 20–29 | 4 | 31 / 64 |
| **N100** | 100 | 4 | 31 / 64 |
| Nfull | 193 | 4 | 31 / 64 |

### Q17 is answered, and it reverses D9's metric set

The ban on Recall@10 rested on a median of 18 relevant resumes per query. That is the
**shipped** split's density. The leak-free split holds resumes out, so each query keeps only
its held-out judgements. Measured on the built N100 pools:

| | Relevant/query (median) | Recall@10 reaches 0.90 | Recall@50 reaches 0.90 |
|---|---|---|---|
| strict (`Good`), *n*=31 | 6 | **93.5%** of queries | 100% |
| graded (`Good ∪ Potential`), *n*=64 | 4 | **93.8%** of queries | 100% |

**Recall@10 is reinstated as the primary recall metric. Recall@50 is retired** — with 6
relevant documents in a 100-deep pool every query's ceiling is 1.0, so it saturates and
separates nothing; over N20 it is not even defined. Precision@10 is capped at 0.6 for the
median strict query, so **Precision@5** is the better-behaved precision figure.

**Adopted: Recall@10, Precision@5, nDCG@10** — each with its *n* and a bootstrap CI, under
both relevance definitions. This is D9 operating as written ("decided from the data, not by
preference"); it is nonetheless a reversal of a rule stated in three documents, all now
updated with a superseded-notice rather than a silent edit.

### The reporting discipline is code, not convention

At *n* = 31 the interval is the finding, so three rules are enforced by the module:

- A `Figure` **cannot be constructed without its query count** — `n=0` raises.
- A `k` deeper than the shallowest pool raises, rather than silently collapsing to
  `k = pool depth`. `Recall@50` over N20 is refused with an explanatory error.
- Queries carrying no relevant document are **excluded, not scored as zero** — that would
  report the pool's label sparsity as a property of the system.

Bootstrap resamples **queries**, not pairs: resampling 659 correlated judgements as if they
were independent observations would produce an interval several times too narrow.

### The floor: a random ranker on N100

Not a result — the number every later system must beat, and the metric code's own self-test.

| Metric | strict (*n*=31) | graded (*n*=64) |
|---|---|---|
| Recall@10 | 0.128 [0.061, 0.212] | 0.074 [0.043, 0.110] |
| Precision@5 | 0.071 [0.039, 0.110] | 0.044 [0.022, 0.066] |
| nDCG@10 | 0.084 [0.047, 0.128] | 0.057 [0.034, 0.084] |

These match the analytic expectation for a random ranker (Recall@10 ≈ 10/100, Precision@10 ≈
6/100), which is the check that the metric code is doing what it claims. Note the interval
width — Recall@10's CI spans a factor of three. **At this sample size the CI is the result**,
and any Stage 1–4 comparison that ignores it is reading noise.

### Open, carried into modelling

**The distractor assumption is now doing most of the work.** A1 judges a median of 4 resumes
per test JD, so a 100-deep pool is ~96% assumed non-relevant, and precision is biased downward
by construction. This was always stated as a limitation; the leak-free split makes it larger,
not smaller. The pools are a comparison instrument between systems, not an estimate of
production precision, and the manifest header says so.

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
| W8 | Metrics are **Recall@10, Precision@5, nDCG@10**, not the spec's Recall@50 / Precision@10 / nDCG@10 | §3A above — measured on the built pools. Recall@50 saturates and Recall@10 no longer does |
| W9 | The third pool variant is **Nfull = 193**, not N477 | 477 was the shipped test split's resume count. The leak-free split holds out 193 |
| W10 | The metric guard refuses **any k deeper than the pool**, rather than special-casing Recall@10 | The general rule catches the specific one, and survives Q17 reversing which k is safe |
| W11 | Metrics live in a new `candidate_screener.evaluation` package, not under `data/` | Building an artefact and scoring one are different jobs with different lifetimes; the split keeps `data/` about data |

## 5. Open questions

| # | Question | Status |
|---|---|---|
| Q14 | Pool depth in-domain, and whether to collect rankings | **Open** — recommendation in `02-work-plan.md`; belongs to 3.4 |
| Q16 | Where more labelled in-domain data comes from if a later stage runs short | **Deferred by design** (D18) |
| ~~Q17~~ | Does the test-side density of 6 Good Fit per JD reinstate Recall@10? | **Closed** — yes. Recall@10 reinstated, Recall@50 retired, Precision@5 adopted. §3A above; catalog, A1 card and README updated |
| **Q18** | *New.* With ~96% of a 100-deep pool assumed non-relevant, is the downward precision bias acceptable for Stage 1–4 comparison, or does a judging wave over system top-k output belong in the plan (as Q14 proposes in-domain)? | **Open — analysed 29 Aug 2026 in [`plan/2026-08-29-pool-precision-bias/`](../2026-08-29-pool-precision-bias/README.md).** The stated closing condition ("once the Stage 1 baseline exists") is now met and does **not** close it: the question is three quantities, one of which (the contamination rate) is not computable at all. A second, previously unrecorded bias was found — a judged-supply ceiling capping Precision@5 at 0.781 strict / 0.719 graded regardless of pool depth. Decisions Q26/Q27 requested |

## 6. Next

The compute critical path to the first reportable number is **clear**: the split and the pools
exist, the metrics are implemented and guarded, and the random-ranker floor is measured. What
remains is a system to score.

Task **3.4a** (the annotation guide) is now the longest lead and is human-gated. Tasks **3.1**
(DataTurks repair) and **3.5** (vocabulary) are independent and can run in any order.
