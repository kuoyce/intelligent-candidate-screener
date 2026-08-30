# Phase 3 — Implementation Record

**Branch:** `feat/derived-artefacts` · **Covers:** tasks 3.2, 3.3, and a preliminary of 3.4
(the A2 train/eval partition and a data-science labelling shortlist, D19). 3.1, the 200-pair
in-domain set itself, and 3.5 are not started.

## 1. What was built

```
src/candidate_screener/data/
  ids.py             content-addressed document identity (design spec §2)
  fit_split.py       task 3.2 — pooling, the doubly-disjoint split, the yield survey
  a2_finetune.py     task 3.4 preliminary — A2 train/eval partition (D19) + data-science shortlist
  build.py           `build --all --seed 0`; a registry the later tasks join
  verify_derived.py  `verify --derived`; the acceptance checks, likewise a registry
docs/data/manifests/ fit-split.csv, fit-split-yield.json, fit-split-survey.json,
                     fit-pair-conflicts.csv, fit-c4-overlap.csv,
                     a2-partition.csv, a2-partition-report.json,
                     a2-datascience-shortlist.csv, a2-shortlist-report.json, README.md
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

## 3B. Task 3.4 preliminary — A2 train/eval partition and data-science shortlist

### Why this exists

Q16 asked what happens once a stage needs more *labelled* A2 data than the 200-pair in-domain
set, and was deferred by design. It stopped being deferred this session: labelled A2 pairs are
now wanted for supervised fine-tuning as well as evaluation, which conflicts with D18 ("A2 …
never contributes fit labels"). **D19** resolves this the way Q16's own text anticipated —
partition A2 at document level into a training-eligible region and an evaluation-eligible
region *before either is sampled* — reusing `fit_split.assign_documents` (the same
sort-permute-slice draw already applied to A1 under D8) rather than re-deriving it.

### Built

```
uv run python -m candidate_screener.data.a2_finetune --partition --seed 0
uv run python -m candidate_screener.data.a2_finetune --shortlist --seed 0
uv run python -m candidate_screener.data.verify --derived --task a2-partition a2-shortlist
```

**Partition** — every JD `id` and CV `id` (352,147 documents total) independently assigned to
`train` or `eval`, seed 0, `eval_fraction=0.25` (an assumption, not a specification — see D19):

| | eval | train |
|---|---|---|
| JD (141,897) | 35,474 | 106,423 |
| CV (210,250) | 52,562 | 157,688 |

**Shortlist** — within `region == train` only, filtered to `Primary Keyword ∈ {Data Science,
Data Engineer, Data Analyst}` (verified: no separate `AI Engineer`/`ML Engineer` keyword
exists — those titles fold into `Data Science`), banded by experience (`0-1`/`2-3`/`4-6`; JD
`Exp Years`'s categorical scale tops out at `5y`, so no JD populates a `7+` cell), 15 JDs × 5
CVs per (keyword, band) cell:

| Keyword | Pairs | JDs | CVs |
|---|---|---|---|
| Data Analyst | 225 | 45 | 204 |
| Data Engineer | 225 | 45 | 170 |
| Data Science | 225 | 45 | 208 |
| **Total** | **675** | **135** | **582** |

All acceptance checks pass: partition identity (0 unresolved, no repeats) and disjointness (0
documents in >1 region); shortlist eval-region isolation (0 stray JDs/CVs), keyword scope, and
pair-id uniqueness; both manifests reproduce byte-for-byte from their recorded seed.

### Deviation from the approved session plan

The plan additionally proposed adding a `MANIFESTS` constant to `config.py`. Implementation
found `fit_split.py` already defines `MANIFESTS = DOCS_DATA / "manifests"` locally and
`pools.py` imports it from there rather than from `config.py`. `a2_finetune.py` follows the
existing convention (`from candidate_screener.data.fit_split import MANIFESTS`) instead of
adding a second definition — `config.py` is unchanged.

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
| W8 | Metrics are **Recall@10, Precision@5, nDCG@10**, not the spec's Recall@50 / Precision@10 / nDCG@10 | §3A above — measured on the built pools. Recall@50 saturates and Recall@10 no longer does. **Reinforced 29 Aug 2026:** Precision@10's ceiling is 0.5677 strict, and Recall@10's is 0.983 — the choice was right and the reason is stronger than the one recorded |
| **X2** | *New, 29 Aug 2026.* Task 3.4b's builder was one-shot: growing `n_jds` from 40 to 60 kept the same 40 JDs but re-drew the candidates for 38 of them, changing **190 of 200 `pair_id`s** | The same "pure function of (corpus, seed), byte-identical across rebuilds" discipline `pools.csv` uses — correct for a derived artefact, wrong for an annotation campaign that runs over days and grows. Found by asking what happens when a session pauses or the scope widens, not by a failing check. Fixed by **D28**: frozen append-only batch specs, per-JD seeds keyed on `jd_id`, and `choose_jds` sorting before permuting so a publisher re-upload cannot silently reselect. Nothing was labelled yet, so no data was lost |
| **X1** | *New, 29 Aug 2026.* `pools.parquet` sourced resume text from `test.parquet`, leaving **1,755 of 10,000 N100 rows** (17.6%) with a null `resume_text` — every one a distractor | The candidate universe is all 193 test-assigned resumes but only 158 appear in a judged test pair; the other 35 had every pair discarded by the doubly-disjoint rule. Nothing caught it: the null count *was* computed, but assigned to `report` **after** the manifest had been written, and no consumer existed until `evaluation.retrieval` (Q24, deferred twice) tried to vectorise the pools. `pools.attach_text` now sources from `load_pooled()` and **asserts** rather than warns. `pools.csv` is byte-identical; only the parquet and the yield report changed |
| W9 | The third pool variant is **Nfull = 193**, not N477 | 477 was the shipped test split's resume count. The leak-free split holds out 193 |
| W10 | The metric guard refuses **any k deeper than the pool**, rather than special-casing Recall@10 | The general rule catches the specific one, and survives Q17 reversing which k is safe |
| W11 | Metrics live in a new `candidate_screener.evaluation` package, not under `data/` | Building an artefact and scoring one are different jobs with different lifetimes; the split keeps `data/` about data |

## 5. Open questions

| # | Question | Status |
|---|---|---|
| ~~Q14~~ | Pool depth in-domain, and whether to collect rankings | **Closed 29 Aug 2026 — option (a).** Wave 1 is built (`indomain-pairs.csv`, 200 pairs). In-domain metrics are **P@5 / nDCG@5**; P@10 is not reported and `metrics.score`'s depth guard refuses it. The top-1 shortlist pick is kept. **Wave 2 deferred by D25** — design retained in [`plan/2026-08-29-unified-judging-wave/`](../2026-08-29-unified-judging-wave/README.md) |
| ~~Q16~~ | Where more labelled in-domain data comes from if a later stage runs short | **Closed 29 Aug 2026 by D19** — partition A2 into document-disjoint `train`/`eval` regions before sampling either; labelled `train`-region pairs may be used for fine-tuning. §3B above |
| ~~Q17~~ | Does the test-side density of 6 Good Fit per JD reinstate Recall@10? | **Closed** — yes. Recall@10 reinstated, Recall@50 retired, Precision@5 adopted. §3A above; catalog, A1 card and README updated |
| ~~**Q18**~~ | With ~96% of a 100-deep pool assumed non-relevant, is the downward precision bias acceptable for Stage 1–4 comparison, or does a judging wave over system top-k output belong in the plan? | **Closed 29 Aug 2026 by D26, as a standing limitation** — one of the three closes the analysis itself listed. Analysed in [`plan/2026-08-29-pool-precision-bias/`](../2026-08-29-pool-precision-bias/README.md), remedied without the wave. **Q18a** (judged-supply ceiling: P@5 capped at 0.781 strict / 0.719 graded regardless of pool depth) is now `metrics.attainable_ceiling`, pinned to 4 dp with depth-invariance asserted, and emitted into `pools-yield.json`. **Q18b** is measured (D27): **83.2%** of TF-IDF's 500 top-5 slots are unjudged distractors, so the bias is *large*. **Q18c** is deliberately not measured. The compression argument resolves against itself — the unjudged share **falls** with system quality (random 94.0%, BM25 86.6%, TF-IDF 83.2%), so the pools rank systems in the right order. Recall@10, ceiling **0.983**, is the headline |

## 6. Next

The compute critical path to the first reportable number is **clear**: the split and the pools
exist, the metrics are implemented and guarded, and the random-ranker floor is measured. What
remains is a system to score.

Task **3.4a** (the annotation guide) is now the longest lead and is human-gated. Tasks **3.1**
(DataTurks repair) and **3.5** (vocabulary) are independent and can run in any order.
