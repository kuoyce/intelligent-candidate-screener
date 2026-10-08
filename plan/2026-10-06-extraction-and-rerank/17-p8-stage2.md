# 17 — P8: stage 2 rerankers (design and implementation record)

Branch `feat/stage2-rerank`. Registered as amendment **A3** ([06 §8](06-registration.md))
before any reranker was trained. This file is the build plan; results are appended below
once they exist.

## 1. Modules

| Path | Content |
|---|---|
| `rerank/base.py` | `Reranker` protocol (02 §4: `score(jd, cvs) -> list[float]`), and `SystemReranker`, which adapts a fitted `BaseSystem` |
| `rerank/batch.py` | Run-11 training frames on `text_v2`: `train` / `validation` (ids from `llm-train-scores.csv`, texts from the train-region sources), within-JD pairs with gap ≥ δ. Reads neither `a2-final-labels.csv` nor `judgements.csv` |
| `rerank/models.py` | R1 (CE pointwise), R2 (CE pairwise), R3 (bi-encoder, pointwise or pairwise), all from the base MiniLM. Grid, training, validation selection, stop-rule figures. Weights are cached under `data/processed/rerank/` and keyed by a config fingerprint. Each model is a `BaseSystem`, so the ladder tools and `rerank_top` take it unchanged |
| `evaluation/stage2.py` | Scores the selected rerankers, SBERT weak, TF-IDF, SBERT zero-shot, random and the LLM score on the final set with `ordering`. Runs O1–O5, the §5 selection and P8.5. Record: `output/matching/stage2-metrics.json`; `--check` (registered in `checks`, cold) |
| `tests/test_rerank.py`, `tests/test_stage2.py` | Synthetic-only invariant tests (§3) |

## 2. Tasks and acceptance (03 §P8)

| # | Task | Acceptance |
|---|---|---|
| 8.1 | `Reranker` interface | Interface test: `SystemReranker` returns one score per CV, in input order |
| 8.2 | R1, R2 on run-11 train, selected on validation | The record holds the full grid with each point's validation pairwise, plus the chosen point; no training module reads the final set (AST test) |
| 8.3 | R3, same data | Same |
| 8.4 | Final-set ordering: all systems; O1–O5 verdicts; §5 selection | Full, `llm_free`, `fit_only`; generic / targeted / total; *n* and CI; paired deltas vs TF-IDF and SBERT weak |
| 8.5 | X2 → stage 2 on the 500-pools | Recall@10 for both, paired Δ, unjudged-in-top-10 count |

## 3. Invariant tests (each shown to fail against a mutant)

1. Pairwise pairs never cross JDs, and every pair has a gap ≥ δ, with hi > lo.
2. Batch frames: the train and validation JD sets are disjoint, the CV sets are disjoint,
   and the CV text has no `Position`.
3. The validation selection metric is `ordering.pairwise` against the LLM score, and the
   first grid point wins a tie.
4. Stage-2 selection walks the cost order; a candidate that fails O5 is skipped; a
   justified dearer candidate displaces a cheaper one, and a non-justified one does not.
5. No module under `rerank/` other than `scoring_batch` names `a2-final-labels` or
   `judgements` (D46).

## 4. Cost

12 trainings on CPU (800 train pairs, up to ~3k within-JD pairs). This is measured on the
first run and recorded here. The cold `stage2 --check` retrains, and the warm tier reuses
cached weights.

## 5. Open

- Q-A1b (replicate) stays open. Without it, the score's run-to-run noise is unknown, and
  the record says so.

## 6. Results (9 Oct 2026; record `output/matching/stage2-metrics.json`)

**Run.** Code and tests by the implementation sub-agent, reviewed here. One training run of
the 12 grid points from scratch: 3 h 30 min wall, peak RSS 5.3 GB (so `stage2` is
`run_alone`). The pairwise CE (R2) was the slow part, about 1.5 h for its 4 points. The warm
`--check --warm` reproduces the record (4 min 12 s, 2.1 GB). **The cold `--check` has not
been run yet.** The user chose to commit on the warm result (deviation from D55, logged
here). Tests: 639 passed. The 5 invariant tests in §3 each fail against their mutant.

**Validation (run-11 batch, 20 JDs; pairwise vs the LLM score).**

| Rung | Grid (validation pairwise) | Selected |
|---|---|---|
| R1 CE pointwise | 2e-5/1 ep 0.558; 5e-5/3 ep 0.619 | 5e-5/3 ep |
| R2 CE pairwise | δ10 2e-5/1 0.633; δ20 2e-5/1 0.633; δ10 5e-5/3 0.645; δ20 5e-5/3 0.606 | δ10, 5e-5/3 ep |
| R3 bi-encoder | point 0.711, 0.717; pair δ10 0.693, δ20 0.694, δ10 0.727, δ20 0.734 | pairwise δ20, 5e-5/3 ep |

Selected cross-encoder: **R2** (0.645 > 0.619). Stop rule: **not triggered**. Every rung
beats random on validation (Δ vs 0.5: R1 +0.119, R2 +0.145, R3 +0.234, all CI lower > 0).

**Final set (pairwise, `yc`, `full`; generic / targeted / total, n = 56 informative JDs).**

| System | generic | targeted | total |
|---|---:|---:|---:|
| random | 0.500 | 0.500 | 0.500 |
| TF-IDF | 0.774 | 0.792 | 0.780 |
| SBERT zero-shot | 0.736 | 0.749 | 0.740 |
| SBERT weak | 0.828 | 0.767 | 0.809 |
| R1 | 0.651 | 0.556 | 0.620 |
| R2 | 0.649 | 0.547 | 0.616 |
| R3 | 0.839 | 0.749 | 0.810 |
| LLM score (exploratory) | 0.874 | 0.778 | 0.843 |

CIs, nDCG, `llm_free`, `fit_only` and the other references are in the record.

**Verdicts (A3.8; `mean` agrees on every one).** O1 R2 vs SBERT weak: **harmful**
(Δ −0.193 [−0.247, −0.139]). O2 R2 vs TF-IDF: **harmful**. O3 R3 vs SBERT weak: **not
resolved**. O4 R2 vs R3: **harmful**. O5: every reranker passes (R1 +0.120, R2 +0.116,
R3 +0.310).

**Selection (§5).** No reranker beats SBERT weak, so **stage 2 is SBERT weak ordering**, with
no new model. That is the registered finding.

**P8.5 (X2 below S2; decides nothing).** Recall@10 total: X2 0.300; X2 → SBERT weak top-50
re-order 0.258; X2 → R2 0.209. Re-ordering X2's shortlist with either stage-2 system
**lowers** Recall@10 of Fits. The unjudged CVs in the top 10 (472 / 499 / 516 of 600) are
counted, not scored.

**Reading.** The cross-encoders are about equally weak on the batch's validation split
(0.62–0.64) and on the final set (0.62), so this is not a transfer gap. A3.2 starts them from
a bi-encoder checkpoint with an untrained one-logit head. On 800 pairs that start learns
little. A cross-encoder pre-trained as one (e.g. an MS MARCO checkpoint) is not cached, and
switching to one now would be post-hoc. Supervision from the LLM batch does not improve on
SBERT weak's metadata pairs either (O3 not resolved, Δ ≈ 0). A second limitation is that the
judge saw `Position` and the inputs do not (A3.1).

## 7. Deviations

1. **Micro-batching** in the pairwise loop (8 pairs per forward pass, gradients accumulated
   to the registered batch of 16). 16 pairs = 32 CE sequences of 384 tokens ran the 15 GB
   machine out of memory on the first run. The step is unchanged, which a test checks.
2. **Cold check deferred** (above).
3. **Waiting on the run burned the session limit.** The sub-agent waited out the run with
   foreground polls, and that hit the limit. `AGENTS.md` "Running tests and checks" rules
   7–8 now forbid this.
