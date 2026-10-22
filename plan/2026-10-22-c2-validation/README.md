# C2 validation: stage 2's top k → C2, chosen on chat-3; the reference's coverage gap

Status: **registered 22 Oct 2026, before any scoring.** Owner's task of 22 Oct 2026, with the owner's
answers of the same day (below). Branch `exp/chat3-holdout`. Follows
`plan/2026-10-21-chat3-holdout` (D62, D63).

## Why

The reported system (D63) is L1 on F12, trained on `chat-3`, ranking all 500 CVs of a pool. C2
(Qwen3-Reranker-0.6B, zero-shot, ~5 s per pair on CPU) is too slow for 500 CVs, so the idea is a
cascade: L1 narrows to its top k, then C2 helps order those k into the top 10. Two gaps:

1. **C2 was never run on chat-3's 48 JDs.** Every C2 figure so far is on the 60 test JDs, after
   seeing them, with one cut (k = 30) and one fusion weight. All of it is **exploratory**: steps
   5a-5d of plan 2026-10-18 and `output/rerank/chat3-vs-c2.json`, `chat3-l1-pool.json` (L1 top 30
   → C2 + L1: −0.002 [−0.026, +0.024] against `chat-1`).
2. **The reference covers X2's top 100** (`chat-1`) plus S1's extension (`chat-2`). A CV a cascade
   lifts into its top 10 from outside that set has no score and counts as a miss.

## Owner's decisions (22 Oct 2026)

1. The 60 final-set JDs may be reused for one more test, **exploratory**, and only if the chat-3
   gate (below) passes.
2. chat-4 covers the **pooled top 10s** of the options compared (IR pooling), not a fixed top 30.
   The agent reports if more is needed.
3. **No "C2 as an L1 feature" option.** C2 is heavy and runs on the top k only.
4. Ties go to the cheaper option: smaller k, then smaller C2 weight (explained to and accepted by
   the owner).

## Design

### Menu (fixed now)

Within L1's top k of a JD (ties by pair id), the order is `mm(L1) + w · mm(C2)`, each min-max
scaled within those k. CVs below the top k keep L1's order underneath.

| | k = 20 | k = 30 | k = 50 |
|---|---|---|---|
| w = 0.25 | ✓ | ✓ | ✓ |
| w = 0.5 | ✓ | ✓ | ✓ |
| w = 1.0 | ✓ | ✓ | ✓ |

Nine options, beside **L1 alone** (no C2).

### Selection on chat-3 only (5-fold CV by JD)

- L1 on F12 (D63's model) **out of fold** on chat-3: each JD is ranked by an L1 fitted on the other
  four folds (`chat_targets.folds`, as §4 of the chat-3 plan). On chat-3 the ranking is over X2's
  top 100, the CVs chat-3 scored. D63 found that equal to ranking all 500 (top 10 identical in
  59 / 60 test JDs).
- C2 scores each JD's out-of-fold top 50 (the largest k). Top 20 and top 30 are subsets: 48 × 50 =
  **2,400 pairs**, one background job.
- Each option's Recall@10 against chat-3's tie-aware top 10, mean over the 48 JDs (total), with
  generic / targeted, JD-bootstrapped CIs, nDCG@10, chat No Fits, and the paired Δ vs L1 alone.
- **Winner:** the highest total Recall@10 (3 dp); ties to the smaller k, then the smaller w.
- **Gate:** the winner's Δ vs L1 alone on chat-3 must be > 0. If not, C2 is retired here and the
  60 test JDs are not touched again.

### Test on the 60 final-set JDs (exploratory; only if the gate passes)

1. L1 fitted once on all of chat-3 (§5's model; it must reproduce §5's 0.484413), applied to all
   500 CVs of each test JD (D63). C2 on its top k (cached for X2's top 100; the rest scored).
2. **Before any metric:** the top 10s of the winner, L1 alone and X2's order are pooled. The pooled
   CVs without a `chat-1` or `chat-2` score form **chat-4** (ids only).
3. The owner scores chat-4 (below). Then, once:
   - Reference: `chat-1` + `chat-2` + `chat-4`, the latest score per pair (as step 4a). `chat-1`
     alone, with unscored CVs as misses, is reported beside.
   - **Bar:** the winner's Recall@10 Δ vs L1 alone (total) ≥ +0.02 with the CI above 0. Met:
     "C2 cascade supported (exploratory)". Not met: "C2 retired". No retuning after the test.
   - Generic, targeted and total, *n*, JD-bootstrapped CIs, paired deltas; nDCG@10 and No Fits
     beside.

### chat-4 (evaluation only, never training)

- Per JD with pooled unscored CVs: a **fresh chat**, the new CVs plus ≥ 5 already-scored anchors
  (a systematic sample over the JD's score order), padded to a multiple of 20, shuffled (seed 0).
  Same instrument (`chat_rerank.instrument_sha256()`) and model as `chat-1`. Every document passes
  `annotation.redact`. Run label `chat-4`, its own plan and reply files (as chat-3), so no record
  built on `chat-1` / `chat-2` changes.
- **Draft decision (not in `docs/decisions.md` until the owner approves):**

  > (Number assigned when logged.) D42 reopened once more, for evaluation only: **`chat-4`**, the owner's claude.ai scoring of
  > the CVs pooled from the top 10s of the C2-validation test's options (plan 2026-10-22) that lack a
  > `chat-1` or `chat-2` score, on the 60 final-set JDs. Fresh chat per JD, ≥ 5 anchors, the
  > `chat-1` instrument and model; no Claude subprocess. `chat-4` is never a training target and is
  > never used to choose anything.

### Code and records

| What | Where |
|---|---|
| Out-of-fold L1 lists and C2 on chat-3 (step B) | `evaluation/c2_validation.py --score` → C2 cache only |
| Selection and gate | `--select` → `output/rerank/c2-validation-select.json` |
| Test lists and chat-4 plan (before any metric) | `--pool` → `output/rerank/c2-validation-pool.json`, `data/interim/chat4-plan.csv` |
| Test (after chat-4) | `--test` → `output/rerank/c2-validation-test.json` |

Records hold ids, digests and metrics only. New code lives in its own module.

### Cost

- C2 on 2,400 chat-3 pairs at the measured 5.0-7.2 s per pair: **3.3-4.8 h**, one background job
  (`/usr/bin/time -v`, exit marker). The fusion grid costs seconds once C2 is cached, so the C2 job
  is the "one configuration" AGENTS rule 10 asks to time. It logs every 100 pairs.
- The test: C2 on the test top-k pairs outside the cache (~100-150, ~15 min), chat-4 (likely
  20-40 CVs plus anchors, ~1 batch per affected JD), then minutes.

## Assumptions

- On chat-3, X2's top 100 → L1 stands in for L1 over all 500 (D63's equivalence). It is the only
  ranking chat-3 can score fully.
- The C2 weight is relative to L1 after min-max scaling within the top k, as in step 5b.
