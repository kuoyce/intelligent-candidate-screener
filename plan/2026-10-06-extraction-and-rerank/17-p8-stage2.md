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
