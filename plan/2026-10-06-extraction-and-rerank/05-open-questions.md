# 05 — Open questions and assumptions (persistent list)

Updated 6 Oct 2026. Status: **open**, **answered** (with date), or **parked** (decided to
discuss later).

## Open

| ID | Question | Why it matters | Recommendation |
|---|---|---|---|
| Q-A1b | Include the ~100-call replicate (re-score 10% to measure the score's noise)? | Without it the score's run-to-run noise is unknown | Yes; ~7% more calls |
| Q-B2 | Hand-check sample size for the title extractor, and who labels it | Gives a ground truth where the keyword is too coarse | ~100 strings per side; labelled by the user or a teammate (no LLM, D42) |
| Q-D2 | The exact `llm_free` subset for ordering: reuse `final_set.LLM_FREE_TIERS` (pairs where no LLM label entered the final label) | The subset is the guard against the LLM scoring itself | Reuse unchanged |
| Q-D3 | `text_v2` also needs the training texts re-cut (P1.3): are retrained TF-IDF and SBERT weak accepted as the `v2` baselines? | Otherwise the baselines saw `Position` in training | Yes |

## Answered

| Date | Question | Answer |
|---|---|---|
| 6 Oct | Q-B1: role-family taxonomy and method | **T-A, split generic**: `Primary Keyword`, with `Data Science`, `Lead` and `Other` split into normalised-title families. Extracted by a **text → family classifier**, title spans as features and UI evidence (evidence: 11-p2-data-checks.md) |
| 6 Oct | Q-A3: How is dispatch unfrozen? | Answered in P6.1 ([15](15-p6-batch.md)): a new module `rerank/scoring_batch.py` imports the frozen instrument from `llm_recheck` and runs its own guarded loop; `annotation/` is not edited and `DISPATCH_CLOSED` stays `True` |
| 6 Oct | Q-A1: Batch size | **B**: 100 JDs × 10 CVs = 1,000 calls (80 train / 20 validation JDs). The ~100-call replicate was not addressed, so it stays an open sub-question (Q-A1b) |
| 6 Oct | Q-A2: Pair mix | Decided by the assistant at the user's request: copy the `generic` design only (8 same-keyword + 2 other, TF-IDF bands). Revisit after P2 |
| 6 Oct | Q-C1: Mean-human reference | `van` does **not** label the remaining 274. The 274 `yc`-only pairs are a separate, labelled row |
| 6 Oct | Q-C2: Rules review | Approved: D42 scoped reopen for one batch, D43 amended (train on that batch's scores only, report `llm_free`), D47 amended (ordering figures as a secondary metric) |
| 6 Oct | Q-D1: Time remaining | About 20 days, confirmed |
| 6 Oct | Which graded signal trains the reranker? | The LLM 0–100 score from one scoring batch, not weak keyword signals |
| 6 Oct | Ordering metric | nDCG and pairwise agreement, not "pick in top 3" |
| 6 Oct | Unlabelled retrieved CVs | Not scored; judged CVs only |
| 6 Oct | Ties | Accepted (pairwise over differing grades; nDCG with tie groups) |
| 6 Oct | Human ceiling for the LLM | None: it only has to beat guessing |
| 6 Oct | Read-only LLM-score check | Done: 01 §4 |
| 6 Oct | Baselines | Random, TF-IDF, SBERT |
| 6 Oct | S2 bars | Unchanged, measured text-only; Recall@10 reported, no bar |
| 6 Oct | Primary Keyword at inference | Never; oracle for measurement only |
| 6 Oct | Final evaluation text | A2 with `Position` removed (`text_v2`) |
| 6 Oct | Does `yc` cover what `van` labelled? | Yes, already: `yc` labelled all 600, `van` 326 |

## Assumptions (to confirm)

1. The LLM instrument and model are unchanged (`5d986c82`, haiku). If that model is no
   longer available, the series changes and so does D43's "one series".
2. The per-sitting yield (40–130 pairs) holds. It was measured once, on 8–10 Sep.
3. The A2 `train` region has enough English, non-empty JDs and CVs to build 100 pools of 10
   excluding the final set. Counts so far are document counts (106k JDs, 158k CVs), before
   the English filter; P6.2 measures it.
4. Rules-first extraction can reach a useful role-family accuracy. P3 tests it. If not, the
   classifier alone carries E2, and the plan says so.
