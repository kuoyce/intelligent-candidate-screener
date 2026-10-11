# Metadata-free matching: entity extraction, then retrieve-and-rerank

**Status:** **approved 6 Oct 2026** (user). Implemented on `feat/extraction-and-rerank`,
branched from `main` (`928e4bd`). Q-A1b (the replicate) is still open and is asked before
the batch is dispatched.
**Follows:** [`plan/2026-09-26-top-down-reset/`](../2026-09-26-top-down-reset/README.md)
(executed 27 Sep 2026, D48 and D49).
**Files:** [01-context.md](01-context.md) · [02-design-spec.md](02-design-spec.md) ·
[03-work-plan.md](03-work-plan.md) · [04-llm-scoring-batch.md](04-llm-scoring-batch.md) ·
[05-open-questions.md](05-open-questions.md)

## Why this plan exists

D48 selected **metadata + TF-IDF** (AUC 0.849) where a `Primary Keyword` exists on both
sides, and the weakly fine-tuned **SBERT** (0.854) for text only. The first result depends
on a Djinni field that an open-world JD or CV does not have, and `Primary Keyword` is often
too coarse to mean a role (Python covers backend, frontend, data and more). The user's
direction (6 Oct 2026):

1. **No metadata at inference.** Replace the fields with entities extracted from the text,
   starting with the easiest to extract that brings the most value. The method stays rules
   first, but more targeted.
2. **Retrieve a shortlist (top 10), then rerank it**, with a trained reranker (a
   cross-encoder is one candidate). The stage-1 classifier and the stage-2 reranker are
   developed **concurrently** behind a fixed interface.
3. **Reranker supervision from the LLM's 0–100 score**, not from weak keyword signals.
   This reopens D42 for **one scoring batch** (see [04](04-llm-scoring-batch.md)).
4. **A2 is the final evaluation, with the `Position` text removed**, so no title field
   leaks into the text a system reads.

## What is decided (user, 6 Oct 2026)

| # | Decision | Where it lands |
|---|---|---|
| 1 | Primary Keyword is an **oracle for measurement only**, never read at inference | 02 §1 |
| 2 | Ordering is validated with **nDCG and pairwise agreement**, not "pick in top 3" | 02 §5 |
| 3 | Ordering is scored on **judged CVs only**; unlabelled retrieved CVs are not slotted in | 02 §5 |
| 4 | The ranking orders **only the CVs the first stage calls Fit** | 02 §4 |
| 5 | Ties are handled by counting **only pairs whose human grades differ** (pairwise) and mean-gain tie groups (nDCG) | 02 §5 |
| 6 | The LLM only has to **beat guessing**; no human-ceiling bar | 02 §5 |
| 7 | The read-only LLM-score check is run (done, exploratory, 01 §4) | 01 |
| 8 | Baselines for the ordering metric: **random, TF-IDF, SBERT** | 02 §5 |
| 9 | Stage-1 bars: the registered **S2 thresholds unchanged, measured text-only** (0.80 classical, 0.90 advanced), plus Recall@10 **reported with no bar** | 02 §4 |
| 10 | **Reopen D42 for one LLM scoring batch**, to train the reranker | 04 |
| 11 | Final evaluation on **A2 with `Position` removed** from the text | 02 §2 |
| 12 | Check existing data for job titles first; **train elsewhere, infer on A2** | 02 §3, 03 P2 |
| 13 | Batch size **B**: 100 JDs × 10 CVs = 1,000 calls; `generic` pair design only | 04 §2 |
| 14 | `van` does not label the remaining 274 pairs; `yc`-only pairs are a separate row | 02 §5 |

## The rules review (approved 6 Oct 2026, not yet logged)

The user approved all three changes below. They are logged as D50–D52 in P0. Each item is a conflict
between this plan and a live rule, and none of it is assumed:

| Rule | Conflict | Proposed change |
|---|---|---|
| **D42** | Dispatch refuses in code | Reopen for **one** batch, scoped by corpus, instrument and size; every other dispatch still refuses (04 §5) |
| **D43** | "No system may be trained on LLM labels" | Allow training on the 0–100 scores of **that batch only**, never on the final-set pairs; report every ordering figure beside the **`llm_free`** subset |
| **D46** | Nothing fitted or tuned on the final set | Unchanged. The batch is outside the final set, tuned on its own validation split |
| **D47** + `AGENTS.md` rule 2 | "No figure is computed on three classes" | Add **ordering** metrics on the graded human labels as a secondary figure. Classification stays Fit / No Fit |
| **D44** / rule 8 | Bars decided before looking | Register this plan's bars before any held-out reranker is scored (05-registration, task P5) |
| `AGENTS.md` rule 5 | Fitting code never reads `judgements.csv` | Unchanged. Training reads the new batch file; only evaluation reads human labels |

## Constraints and assumptions

- Python 3.12, `uv` only. Tests need no `data/`. No CI: step 4 is `uv run pytest` plus
  every `--check`.
- **Confirmed (6 Oct):** about **20 days** remain. The 26 Sep plan counted about 30
  days remaining, and 10 have passed. This plan is ~14 team-days of work plus a
  calendar-bound LLM batch (04 §4).
- The LLM instrument is unchanged: `5d986c82`, `claude-haiku-4-5` (the one series in the
  final set).
