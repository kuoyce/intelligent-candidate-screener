# Stage 2: the LLM reranks X2's shortlist into a top 10

Status: **superseded 18 Oct 2026 by [classifier-then-rerank](../2026-10-18-classifier-then-rerank/README.md)
(D61).** `chat-1` is complete (60 JDs, 6,000 pairs) and is now the stage-2 target. Branch
`exp/stage2-llm-rerank`. Decision D59 (amended D43, now superseded): the LLM may be the
stage-2 reranker.

## Goal (owner's framing)

A recruiter has hundreds of applicants per job. **Stage 1** (X2: extracted rule + TF-IDF, cheap
and fast) narrows them to a shortlist of K. **Stage 2** (the LLM, expensive) reranks the
shortlist and hands over the **top 10**. The grey zone of the LLM's Fit / No Fit threshold does
not matter here: only its ordering of the shortlist does.

## Steps

1. **Stage-1 recall@K curve** (`evaluation/recall_curve.py`, record
   `output/matching/stage1-recall-curve.json`). X2's recall at K = 10, 20, 50, 100, 150, 200, 250
   of 500-CV pools: on dev-select (150 A2 train-region JDs; relevance = same oracle role family,
   a proxy) and on the final pools (60 JDs; relevance = judged final-set Fit; unjudged counted
   not relevant, an upper bound).
2. **Choose K** by the rule below, on dev-select only (D46).
3. **Stage-2 evaluation without new calls** (existing LLM scores, runs 9/10, on the 600 judged
   pairs): ordering within the human Fits (`fit_only`), and on `full` / `llm_free`, against X2 and
   the other rerankers. Already in the records; collected in one notebook.
4. **End to end (needs new LLM calls, D42; not decided).** The LLM scores every unjudged CV in
   X2's top K for each JD; the top 10 by LLM score is compared with X2's own top 10.

## Rule for K (registered 17 Oct 2026, before the curve was read)

- **K = the smallest K in the grid whose dev-select recall is at least 0.90** (the advanced
  retriever bar of S2). If none reaches it, K = 250 and the miss is stated.
- The final-pool recall at that K is reported, never used to choose.
- Owner to confirm the 0.90; if changed after the curve is read, the change is labelled
  `exploratory`.

## Cost of step 4

- **K calls per JD at most**, not the pool size: only the shortlist is reranked. The judged CVs
  in the shortlist already have an LLM score (runs 9/10), so the new calls are the **unjudged**
  CVs in X2's top K.
- Measured cost of a dispatch: runs 9 and 10 (600 pairs) took **nine** session windows, about
  40-130 pairs per window, because each call is a `claude -p` subprocess on the account's
  session limit (memory note, 8-10 Sep 2026).
- Evaluation caveat: the LLM's top 10 can contain unjudged CVs. They count as not relevant (pool
  distractors are random CVs, mostly of other roles), so recall@10 of judged Fits is a lower
  bound for the LLM's true quality; a proper precision@10 needs grades on those CVs.

## Open questions

1. The 0.90 recall bar for K (proposed above).
2. Step 4: reopen D42 for the shortlist calls? With how many JDs (all 60, or a sample) and which
   K? Calls ≈ (unjudged CVs in the top K) × JDs.
3. Reuse the runs 9/10 scores for the judged CVs, or score the whole shortlist in one new run?
   Reuse saves 600 calls but mixes runs within one ranking (AGENTS: never pool across runs).

## Result of step 1 (17 Oct 2026; record `output/matching/stage1-recall-curve.json`)

X2 recall@K of 500-CV pools (mean per JD, 95% CI over JDs):

| K | dev-select (150 JDs, proxy) | final pools (60 JDs) | generic | targeted |
|--:|---|---|--:|--:|
| 10 | 0.350 [0.31, 0.39] | 0.300 [0.24, 0.36] | 0.286 | 0.326 |
| 20 | 0.544 [0.50, 0.59] | 0.503 [0.42, 0.58] | 0.506 | 0.498 |
| 50 | 0.727 [0.69, 0.77] | 0.779 [0.71, 0.85] | 0.795 | 0.749 |
| 100 | 0.844 [0.81, 0.87] | 0.902 [0.85, 0.94] | 0.900 | 0.905 |
| 150 | 0.886 [0.86, 0.91] | 0.939 [0.89, 0.97] | 0.930 | 0.956 |
| **200** | **0.911** [0.89, 0.93] | **0.960** [0.93, 0.98] | 0.962 | 0.956 |
| 250 | 0.927 [0.91, 0.95] | 0.969 [0.95, 0.99] | 0.969 | 0.970 |

**By the registered rule, K = 200** (the smallest K with dev-select recall >= 0.90). Final-pool
recall there is 0.960. The dev proxy has about 32 relevant CVs per JD (same role family) against
about 6 Fits per final-set JD, so it is the harsher test; K = 100 already keeps 90% of the final
pools' Fits. Final-pool R@50 0.779 matches the stage-1 record.

## Owner decisions, 17 Oct 2026 (D60)

- **K = 100**, all **60 JDs** (6,000 pairs; X2's top 100 holds about 7 judged CVs per JD).
- Scored by hand in a claude.ai chat with **`claude-haiku-5-5`**: a new series `chat-1`, never
  pooled with runs 9-11. The judged CVs are scored in the same chats, so one ranking holds one
  series.
- **New listwise instrument**: the a1-judge guide verbatim, 20 CVs per message, JSON-array reply.
- **One chat per JD** (scale consistency across a JD's five batches); the shortlist is shuffled
  (seed 0) so batch drift is noise, not bias; batch drift is checked afterwards.
- Ties in the LLM's top 10 are broken by X2 rank; the evaluation reports how many ties there are
  and how many slots the tie-break decided.
- Grading the unjudged CVs in the top 10s is decided after the scores are in, once the count is
  known. The sample JD already suggests the LLM's top 10 is mostly unjudged CVs.

Tool: `rerank/chat_rerank.py` (plan, prompt, validation, saving; tested) and
`rerank/chat_rerank_app.py`:

    uv run --group app streamlit run src/candidate_screener/rerank/chat_rerank_app.py

Replies: `output/rerank/chat-rerank-replies.csv` (one row per CV, latest wins) and
`chat-rerank-replies.raw.jsonl` (the pasted text). JD 1 batches 1-2 are the owner's test pastes.
