# Stage 2: the LLM reranks X2's shortlist into a top 10

Status: **in progress (owner, 17 Oct 2026).** Branch `exp/stage2-llm-rerank`. Decision D59
(amends D43): the LLM may be the stage-2 reranker.

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
