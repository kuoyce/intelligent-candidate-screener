# C2 validation: stage 2's top k → C2, chosen on chat-3; the reference's coverage gap

Status: **registered 22 Oct 2026, before any scoring; tested 22 Oct 2026 (bar not met); C2 then
adopted by the owner (D64, post hoc).** Owner's task of 22 Oct 2026, with the owner's
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

## Step B record (22 Oct 2026)

`c2_validation --score`, `logs/c2_validation_score.log`: C2 on 2,400 chat-3 pairs (48 JDs × the
out-of-fold L1 top 50), none cached. 13,657 s of scoring (5.7 s per pair), 3 h 50 min end to end,
peak RSS 5.1 GB.

## Selection record (22 Oct 2026): winner k = 50, w = 0.25; the gate passes

`c2_validation --select` → `output/rerank/c2-validation-select.json`, 2 min 51 s. Out of fold on
chat-3, 48 JDs, Recall@10 against chat-3's tie-aware top 10:

| Option | R@10 [CI] | generic / targeted | Δ vs L1 alone [CI] | nDCG@10 | Chat No Fits |
|---|---|---|---|---|---|
| L1 alone | 0.513 [0.450, 0.575] | 0.468 / 0.605 | – | 0.818 | 1.12 |
| top 20, w 0.25 | 0.526 | 0.474 / 0.630 | +0.013 [−0.000, +0.027] | 0.828 | 1.02 |
| top 20, w 0.5 | 0.518 | 0.471 / 0.611 | +0.004 [−0.015, +0.023] | 0.826 | 1.00 |
| top 20, w 1.0 | 0.508 | 0.451 / 0.621 | −0.005 [−0.034, +0.023] | 0.825 | 0.94 |
| top 30, w 0.25 | 0.531 | 0.482 / 0.630 | +0.018 [+0.003, +0.033] | 0.830 | 1.02 |
| top 30, w 0.5 | 0.505 | 0.456 / 0.605 | −0.008 [−0.029, +0.013] | 0.826 | 1.00 |
| top 30, w 1.0 | 0.504 | 0.446 / 0.621 | −0.009 [−0.040, +0.022] | 0.822 | 0.94 |
| **top 50, w 0.25** | **0.532** [0.468, 0.593] | 0.483 / 0.630 | **+0.019 [+0.003, +0.034]** | 0.830 | 1.02 |
| top 50, w 0.5 | 0.503 | 0.449 / 0.611 | −0.010 [−0.032, +0.013] | 0.825 | 0.96 |
| top 50, w 1.0 | 0.506 | 0.451 / 0.615 | −0.008 [−0.040, +0.025] | 0.821 | 0.96 |

- **Winner: top 50, w = 0.25** (0.532; top 30, w 0.25 is 0.531, so no tie at 3 dp). **Gate: passes**
  (+0.019 > 0). The test on the 60 JDs goes ahead (owner's decision 1).
- C2 helps only as a light second opinion. At w = 0.25 it adds +0.013 to +0.019 at every k; at w ≥ 0.5
  it does not help at any k. Every option cuts chat No Fits (1.12 → 0.94-1.02).
- Read with care: the winner is the best of nine, so its +0.019 is optimistic. It sits just under the
  test bar (+0.02); the test has to find at least the CV gain to meet it.

## Pooling record (22 Oct 2026): chat-4 is empty

`c2_validation --pool` → `output/rerank/c2-validation-pool.json`, `logs/c2_validation_pool.log`,
30 min (C2 on the 216 test top-50 pairs not yet cached, 6.8 s per pair). L1 fitted on all of chat-3
reproduced the chat-3 test's 0.484413 over the pools. No metric of the options was computed.

- The top 10s of the winner (L1 top 50 → L1 + 0.25 C2), L1 alone and X2's order, pooled over the 60
  JDs: **every CV already has a `chat-1` or `chat-2` score.** Unscored per option: 0 / 0 / 0.
- So **chat-4 has no pairs**. Nothing needs scoring, and the draft decision reopening D42 is not
  needed (withdrawn; never logged). Reference A of the test is `chat-1` + `chat-2`, complete for
  every top 10 compared.
- Gap 2 of the task, for this comparison: closed by `chat-2` already. CVs beyond the top 10 (e.g.
  L1's ranks 31-50) remain partly unscored, but no compared option puts one in its top 10.

## Test record (22 Oct 2026; exploratory): bar not met, C2 retired

`c2_validation --test` → `output/rerank/c2-validation-test.json`, `logs/c2_validation_test.log`,
6 min. Run once. 60 final-set JDs; reference A = `chat-1` + `chat-2` (complete for every compared
top 10); reference B = `chat-1` alone.

| Option (60 JDs) | R@10, ref A (gen / tar) | Δ vs L1 alone [CI] | nDCG@10 | Chat No Fits | R@10, ref B |
|---|---|---|---|---|---|
| X2's order | 0.417 (0.419 / 0.414) | −0.058 [−0.088, −0.029] | 0.775 | 1.73 | 0.419 |
| L1 alone (D63) | 0.475 (0.484 / 0.456) | – | 0.820 | 1.38 | 0.484 |
| **L1 top 50 → L1 + 0.25 C2** | 0.480 (0.487 / 0.467) | **+0.005 [−0.010, +0.021]** | 0.832 (Δ +0.012 [+0.005, +0.019]) | 1.28 | 0.490 (Δ +0.006 [−0.010, +0.022]) |

- **Bar (Δ ≥ +0.02 with the CI above 0): not met.** +0.005 is a quarter of the bar, CI across 0 (generic
  +0.003, targeted +0.011). As registered: **C2 is retired**. No retuning.
- What C2 does do, on both chat-3 and the test, is order the top 10 a little better: nDCG@10
  +0.012 (CI above 0), and 0.1 fewer chat No Fits. It does not bring more of the chat's top 10 in.
- Human check (exploratory, 422 judged CVs): no change. Per-JD AUC vs L1 −0.003 [−0.021, +0.013]
  (`llm_free` −0.003); pairwise on yc −0.006 [−0.018, +0.006].
- On chat-3 the same option was +0.019 (the best of nine, so optimistic); on the test it is +0.005.
  That is the same pattern as steps 5b-5d: C2 adds 0 to +0.02, within noise, at ~5-7 s per pair.

## Report to the owner (task of 22 Oct 2026)

- Plan: this file, registered at `523cfeb` before any scoring.
- First C2 batch (step B): 2,400 chat-3 pairs, 3 h 50 min, 5.7 s per pair, peak 5.1 GB.
- CV table: the selection record above (winner k = 50, w = 0.25, +0.019 [+0.003, +0.034]; gate passed).
- Test: +0.005 [−0.010, +0.021] vs L1 alone; bar not met; C2 retired.
- chat-4: 0 CVs on every JD (all pooled top-10 CVs were already scored); no batch, no decision entry.
- Owner decisions used: reuse of the 60 JDs (yes, gated), pooling for chat-4, no C2-as-feature option,
  the tie order (smaller k, then smaller w).

## Owner's decision after the test (22 Oct 2026): adopt C2 (D64)

Owner: "lets make the decision to use C2 (based on the fact that both recall and ndcg is better)."
Recorded as **D64**: the reported system is L1 (chat-3) over the pool → its top 50 reordered by
L1 + 0.25 × C2 (the option chosen on chat-3, unchanged).

- This **overrides the registered bar**, which the test did not meet; the decision was made after
  seeing the test, so it is post hoc. Every report gives the bar and the miss beside the result.
- The evidence as it stands: nDCG@10 +0.012 [+0.005, +0.019] (CI above 0); Recall@10 +0.005
  [−0.010, +0.021] (point estimate higher, CI across 0); chat No Fits 1.38 → 1.28; human check
  unchanged (AUC −0.003, pairwise −0.006, both CIs across 0).
- Nothing is retuned: k = 50 and w = 0.25 stay as chosen on chat-3, although top 30 (0.531 on
  chat-3, ~40% fewer C2 calls) was within 0.001.
- Cost in the system: ~50 C2 calls per JD, ~5 min of CPU per JD at 5-7 s per pair.
- Not done: the recruiter app (`pipeline.py`) still runs its own TF-IDF draft ranker; wiring the
  reported system into it is a separate task.
