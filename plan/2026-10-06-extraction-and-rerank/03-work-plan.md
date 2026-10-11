# 03 — Work plan

Estimates are team-days and are guesses. The LLM batch is calendar-bound (04 §4), so the
critical path is **P6**, not the effort total (~14.5 d). Every phase ends with workflow step 4
(`uv run pytest`, every `--check`) and a record `NN-*.md` in this directory. Every regression
test must be shown to fail against the code it guards (`AGENTS.md`, Testing 2).

```
P0 ─► P1 ─► P2 ─► P3 ─► P4 ─┐
 │     └──────────────────────┼─► P5 (registration) ─► P7 (stage 1, X rungs) ─┐
 └─► P6 (LLM batch, calendar-bound; starts after P0, runs while P1–P5 proceed) ─► P8 (stage 2) ─► P9
```

P6 starts as soon as P0 is approved, because it waits on session limits, not on the code.

## P0 — Decisions and housekeeping (0.5 d)

| # | Task | Acceptance |
|---|---|---|
| 0.1 | Resolve the rules review (README table): D42 scoped reopen, D43 amended, D47 amended for ordering figures. Log as **D50–D52** in `docs/decisions.md`, each linked to this plan | Each decision has a one-line entry and a plan link. A test greps that every `D\d+` cited resolves |
| 0.2 | `text_v2` recorded as a decision (**D53**) | Entry present |
| 0.3 | Branch from `main`, commit this directory alone | Plan commit contains nothing else |
| 0.4 | Update `AGENTS.md` for the amended rules (D42 scope, D43, D47 ordering), keeping it under 1,800 words (D45) | Word count; every changed rule links to its decision |

## P1 — `text_v2` and re-scoring the existing ladder (1 d)

| # | Task | Acceptance |
|---|---|---|
| 1.1 | `matching/corpus.py`: a `text_variant` argument; `v2` builds CV text without `Position` (`annotation/sample.py` is frozen, so the field list is mirrored, not imported-and-edited) | Named invariant test: a CV with a distinctive `Position` string never contains it under `v2` |
| 1.2 | **Residual-leak report:** share of A2 CVs and JDs whose remaining text still contains the removed `Position` (seniority stripped, lower-cased) | Number and CI in the record. No scrubbing |
| 1.3 | Retrain what read CV text: TF-IDF IDF from the A2 **train** region and **SBERT weak (3b)** and the weak-pair inputs, all on `v2` train text. (The existing models saw `Position` in their CV text, so scoring them on `v2` alone would be a train/test mismatch) | A `--check` for the `v2` record. Cache keys include `text_variant` |
| 1.4 | Re-score the existing text rungs on `v2` (random, TF-IDF, BM25, SBERT zero-shot, SBERT weak) | `exploratory`; both variants side by side. Metadata rungs are unchanged by `v2` |

## P2 — Data checks and the role-family taxonomy (1.5 d)

| # | Task | Acceptance |
|---|---|---|
| 2.1 | Profile title or role fields in ESCO, `data_jobs`, livecareer and resume-atlas (fill rate, distinct values, PII) | Marked **Verified** only from `profile` output; recorded in `docs/data/profile-metrics.json` and the data catalog |
| 2.2 | Profile A2-train `Position` and `Primary Keyword`: class sizes, how many normalised families fall under each keyword, seniority and language tokens | Table of T-A vs T-B (02 §3.2) with evidence |
| 2.3 | Title-in-text check: share of sampled JDs and `v2` CVs where a title span can be found | Number with *n*. Decides span extraction vs classifier for E2 |
| 2.4 | **Decision (05, Q-B1):** the taxonomy | Chosen by the user from the evidence, before any extractor is scored on A2 |

## P3 — E2: title → role family (2 d)

| # | Task | Acceptance |
|---|---|---|
| 3.1 | Alias table from ESCO alt labels, `data_jobs` titles and A2-train `Position` | Coverage of A2-train `Position` strings reported |
| 3.2 | Rules: title-span extractor for JD and `v2` CV text | Precision / recall on a hand-checked sample (size per Q-B2) |
| 3.3 | Classifier text → role family, trained on A2-train, **input with the title field removed**; validation split disjoint by JD and CV | Macro-F1 and accuracy vs normalised `Position` on A2-train validation; never tuned on the final set |
| 3.4 | Evaluation on the A2 final-set JDs and CVs (`v2` text): role-family accuracy vs the oracle | Reported with *n*; the oracle keyword is a measurement, never an input |

## P4 — E3 and E4: experience and seniority (1.5 d)

| # | Task | Acceptance |
|---|---|---|
| 4.1 | Experience-years rules for JD phrases and CV date ranges (overlapping ranges merged) | Band accuracy vs `Exp Years` / `Experience Years` on A2-train, then on the final-set documents |
| 4.2 | Seniority from title tokens plus years | Accuracy vs seniority tokens in `Position` |
| 4.3 | Named invariant tests: overlapping date ranges are not double-counted; "5+ years" parses as a minimum; ranges with an open end | Each fails against a naive implementation |

## P5 — Ordering evaluation and registration (1 d)

| # | Task | Acceptance |
|---|---|---|
| 5.1 | Promote the scratchpad check to `evaluation/ordering.py` (nDCG with tied-score mean gain, pairwise over differing grades, JD-level bootstrap, paired deltas, `llm_free` and fit-only subsets) | Named invariant tests: score ties give no tie-break luck; a JD with all-equal grades is excluded and counted; a constant scorer scores 0.5 pairwise |
| 5.2 | `evaluation/ordering.py --check` against a committed record `output/matching/ordering-metrics.json` | Reproduces |
| 5.3 | **`05-registration.md`, committed before any X rung or held-out reranker is scored:** rungs, comparisons, the S2 bars (text-only), the ordering bar (beats random), and which results are already `exploratory` (01 §4) | File committed first; the scoring tasks cite its commit |

## P6 — The LLM scoring batch (details in [04](04-llm-scoring-batch.md)) (1 d active + calendar)

| # | Task | Acceptance |
|---|---|---|
| 6.1 | Scoped reopen of dispatch (D50), with tests that every other dispatch still refuses | Tests |
| 6.2 | Build the batch: JDs and CVs from the A2 **train** region only, none from the final set or the eval region, 10 CVs per JD | Manifest of ids only, with a digest; a test that no id is in the final set or the eval region |
| 6.3 | Dispatch, bank answers as they arrive, retry the quarantined pairs | `--coverage` full before collecting |
| 6.4 | Collect into `llm-train-scores.csv`, keyed by (`agent_sha256`, model, run, draw); never into `judgements.csv` | Validation: every row parsed, scores in 0–100 |

## P7 — Stage 1: text-only retrieval and the X rungs (1.5 d)

| # | Task | Acceptance |
|---|---|---|
| 7.1 | `Retriever` interface (02 §4) and adapters for TF-IDF, SBERT (zero-shot, weak) and the X rungs | Interface test |
| 7.2 | Score X1–X4 on `v2` per 02 §3.4, with the oracle as a reference row | `held-out`; generic / targeted / total with *n* and CI; paired deltas for the registered comparisons; within-family AUC |
| 7.3 | Retrieval pilot, text-only: Recall@50 against S2 and Recall@10 reported | S2 verdict (pass / miss) with CI. Recall@10 with no bar |

## P8 — Stage 2: rerankers (3 d)

| # | Task | Acceptance |
|---|---|---|
| 8.1 | `Reranker` interface (02 §4) | Interface test |
| 8.2 | Cross-encoder trained on the batch's train split, in the two fixed forms (pointwise, pairwise). Hyperparameters and δ chosen on the batch's validation split | Record includes the selection grid and validation figures. No file reads the final set |
| 8.3 | Bi-encoder trained on the same data | Same record |
| 8.4 | Score every reranker, the LLM score as a ranker, and the baselines with `evaluation/ordering.py` on the final set | nDCG and pairwise, full and `llm_free` and fit-only, generic / targeted / total, paired deltas vs TF-IDF and SBERT weak |
| 8.5 | Stage 1 + stage 2 together: Recall of fits in the top 10 on 500-pools, plus the ordering metric on judged CVs; unlabelled shortlisted CVs counted, not scored | Counts reported |

## P9 — Pipeline, UI, documentation (1.5 d)

| # | Task | Acceptance |
|---|---|---|
| 9.1 | `pipeline.py` and the UI read extracted fields, never the metadata | A test that the pipeline never touches a `Primary Keyword` or `Position` column |
| 9.2 | Findings record `NN-findings.md`, README results, `docs/decisions.md`, the data catalog and cards for anything profiled | Every measured figure read from an `output/…` record |
| 9.3 | `uv run pytest`, every `--check`, `data.verify --derived` | All green, or the miss stated |
