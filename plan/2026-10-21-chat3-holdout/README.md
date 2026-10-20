# chat-3: train stage 2 on new train-region JDs; the final set becomes test-only

Status: **approved by the owner, 21 Oct 2026** (open questions 1-4 answered below). Follows
`plan/2026-10-18-classifier-then-rerank` (steps 1-5d). Decision D62 below. Branch `exp/chat3-holdout`, from
`exp/classifier-then-rerank`.

## Why (owner, 21 Oct 2026)

L2 is trained on `chat-1` and scored against it, on the same 60 JDs on which every stage-2 choice
was made (~40 configurations). Out of fold by JD, but not a held-out test. Training is cheap (L2
fits in seconds); the cost is scoring. So: score 48 **new** train-region JDs by chat ("chat-3"),
train stage 2 on them only, and test once on the 60 final-set JDs, which nothing is then fitted on.

What it fixes and what it does not:

- **Fixed:** the stage-2 model never sees a final-set JD or a `chat-1` score. S1 (D50) and X2
  (dev set) already never did, so the whole pipeline is fitted outside the final set.
- **Also fixed (owner, 21 Oct: "should we plan to re-do L2 with this new set?"):** the stage-2
  design (L2 and F9) was chosen on the 60 JDs. It is now re-chosen on chat-3 alone, by
  cross-validation, from a small menu fixed here (§4). Only the menu itself comes from steps 3-4b.
- **Not fixed:** the judge is the same (Claude), so stage 2 still learns its taste; the human
  labels stay the only independent reference.

## Decision D62 (in `docs/decisions.md`)

> Stage 2 is trained on chat-3 and tested on the final set (owner, 21 Oct 2026). (1) D42 reopened
> once more: **chat-3**, the owner's claude.ai scoring of X2's top 100 for 48 A2 **train-region**
> JDs (32 `generic`, 16 `targeted`), with `chat-1`'s listwise instrument and model; no Claude
> subprocess. (2) **D61 amended:** `chat-1` is no longer a training target or used to choose one. It is the stage-2 test
> reference only. (3) **D61 part (3) withdrawn:** D46 applies to stage 2 again; nothing is fitted
> or tuned on the final-set JDs, `chat-1` included. (4) The stage-2 design is chosen on chat-3 alone,
> by JD-fold cross-validation, from a menu fixed before chat-3 is scored (L1 or L2, on F12 or F9).

## Design

### 1. JDs (48, drawn once, seed 0)

From the dev documents (`dev_pools.dev_documents_from`: A2 train region, P3's validation part
minus its calibration draw, minus weak pairs), so the role classifier and SBERT weak behind the
features never trained on them. Eligible as in `dev_pools.draw_pools` (a JD title; ≥ 8 CVs of its
keyword). Excluded: the 400 dev-pool JDs (dev-train and dev-select keep their roles) and D50's JDs.
Supply, counted 21 Oct 2026: 8,980 eligible JDs outside the dev pools.

| Stratum | JDs | Drawn how |
|---|---|---|
| `generic` | 32 | Uniform at random from the eligible set, as the final set's `generic` is unstratified |
| `targeted` | 16 | From the final set's four targeted families, in its proportions (7 / 6 / 5 / 2 of 20): Business Analyst 6, Data Science 5, Data Engineer 4, Data Analyst 1 (largest remainder). Supply: 267 / 161 / 83 / 54 |

32 / 16 keeps the final set's 2 : 1 ratio, and 48 JDs × 100 = 4,800 pairs is what L2 trained on
per fold.

### 2. Pools and shortlist

- **Pools:** 500 dev CVs per JD, `dev_pools`' recipe: 8 of the JD's keyword, 2 of another, 490
  random. The final set's pools are its 10 labelled picks (80% keyword match in both strata) + 490
  random, so the mix is the same. One difference: the final set's picks were banded by experience,
  these are not.
- **Shortlist:** X2 (unchanged, digest-guarded) top 100 per JD.

### 3. Scoring (the owner)

`chat_rerank` as for `chat-1`: the shortlist shuffled (seed 0), short ids, batches of 20, one
conversation per JD, the same instrument (`chat_rerank.instrument_sha256()`) and model
(claude-haiku-5-5 in claude.ai), run label `chat-3`, replies in the same append-only file. Every
document passes `annotation.redact`. **240 batches** (`chat-1` was 300), through the existing app.

### 4. Stage 2: chosen and trained on chat-3 only

- **Menu (fixed now):** the four contenders of steps 3 and 4b, with their exact settings: L1
  (bounded ridge) or L2 (monotone HistGradientBoosting), on F12 or F9.
- **Choice:** 5-fold cross-validation by JD on the 48 chat-3 JDs (seed 0). Each arm's out-of-fold
  Recall@10 against chat-3's tie-aware top 10, total. The highest wins; a tie (to 3 dp) goes to the
  smaller model, F9 before F12 and L1 before L2. Reported with CIs, but nothing else is chosen.
- **Fit:** the winner, once, on all 4,800 chat-3 pairs. No other tuning.
- The code that chooses and fits reads only chat-3 replies, never `a2-final-labels.csv`,
  `judgements.csv` or `chat-1`.

### 5. Test: the 60 final-set JDs, all folds, scored once

| Arm | Stage 1 | Stage 2 | Trained on |
|---|---|---|---|
| X2's order | X2 | – | nothing |
| **Stage 2 (chat-3)** | X2 top 100 | §4's winner | chat-3 |
| S1's order | X2 top 100 | S1 probability | D50 |
| L2 (chat-1, out of fold) | X2 top 100 | L2 on F9 | `chat-1` (the old figure, for comparison; no rule) |

Reference: `chat-1` (it scored exactly X2's top 100). Metrics as step 3: tie-aware Recall@10,
nDCG@10, chat No Fits; generic / targeted / total; JD-resampled CIs; paired deltas.

**Rules, fixed before stage 2 (chat-3) is scored on the 60:**

1. **Headline:** stage 2 (chat-3) passes if its Recall@10 Δ vs X2's order (total) is ≥ +0.05 with the
   CI above 0 (S3's bar, owner 18 Oct 2026).
2. **Chat vs chat-free stage 2:** stage 2 (chat-3) beats S1's order if its Recall@10 Δ (total) is > 0
   with the CI above 0.
3. **Humans (secondary):** stage 2 (chat-3) vs X2's order, per-JD AUC on the final label (`full`), Δ > 0
   with the CI above 0; pairwise on yc's grades and `llm_free` reported beside. These figures are
   confirmatory for stage 2 (chat-3), whose scores nobody has seen; the same labels were seen for other
   systems.

Whatever the result, it is reported as the result: no retuning, no second test on these JDs.

### 6. Code and records

| What | Where |
|---|---|
| The 48-JD draw and their pools | `evaluation/chat3.py --build` → `output/rerank/chat3-pools.json` (ids and digests only) |
| The chat-3 plan for the app | `rerank/chat_rerank.py --build-chat3` (own run label and short ids) |
| Choice, training and test | `evaluation/chat3.py --select` → `output/rerank/chat3-select.json`; `--test` → `output/rerank/chat3-test.json` |
| Tests (inline synthetic) | the draw's quotas and exclusions, no final-set or eval-region id, the fit never reads the test |

### 7. Cost

- Agent: about half a day of code. Pools, X2 and features: ~10 min of compute.
- Owner: 240 batches.
- Choice and test: minutes.
- During the labelling (owner, 21 Oct): the D55 cold run the old plan owes, after the chat-3 build
  so the two do not compete for RAM. `rerank_chat` (~9 h, retrains L3), `feature_arms` (~13 min),
  `stage1_classifier` (~7 min), `stage1_threshold` (~3 min): about 9.4 h, one background run.

## Open questions (answered by the owner, 21 Oct 2026)

1. **C2 on chat-3 during the labelling: no.** The CPU runs the D55 cold run instead (§7). C2's
   scores on the 60 test JDs stay cached; no C2 arm in this test.
2. **Stage 1 for chat-3: X2's top 100.** `chat-1` covers exactly that list, so every test figure is
   complete and comparable with steps 3-5.
3. **Generic draw: any family**, including the four targeted ones, as the final set's was.
4. **Re-do L2 on chat-3: yes** (§4: re-choose the design on chat-3 by cross-validation, then fit).

## Assumptions

- The chat-3 judge behaves like `chat-1`'s. A drift in the scale between sessions (`chat-2` ran
  +3.4) matters little, since L2 ranks CVs within a JD.
- 48 JDs are enough, since L2 trained on 48 per fold before. The choice in §4 sees ~38 per fold.
