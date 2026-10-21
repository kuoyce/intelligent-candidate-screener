# chat-3: train stage 2 on new train-region JDs; the final set becomes test-only

Status: **approved by the owner, 21 Oct 2026; chat-3 collected, §4 chose L1 on F12, §5 tested on
22 Oct 2026: all three rules pass.** Owed: a `chat3 --check` and the remaining cold checks. Follows
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

## Build record (21 Oct 2026)

`evaluation/chat3.py --build` → `output/rerank/chat3-pools.json`, `data/interim/chat3-plan.csv`,
`logs/chat3_build.log`. 2 min 29 s, peak RSS 3.9 GB.

- 48 JDs (32 `generic`, 16 `targeted`: Business Analyst 6, Data Science 5, Data Engineer 4, Data
  Analyst 1) from 9,380 eligible dev JDs, 405 of them excluded (dev pools and D50).
- 24,000 pool pairs; X2's top 100 = 4,800 pairs in 240 batches. X2 reproduced the stage-1 record
  first.
- No overlap with the eval region, the final set, the weak pairs, D50's JDs or the dev-pool JDs
  (all 0, checked in code).
- The text builders for train-region documents give exactly what `chat-1` read: on the final set,
  60 / 60 JD texts and 600 / 600 CV texts are equal (checked 21 Oct 2026).
- chat-3 keeps its own plan and replies (`chat_rerank.CHAT3_*`); the app takes `-- --chat3`.
  Prompts are about the size of `chat-1`'s (~65,000 characters for a batch of 20).

## Collection record (chat-3 complete, 21 Oct 2026)

`output/rerank/chat3-replies.csv` (+ `.raw.jsonl`), saved 20 Oct 17:00 to 21 Oct 15:42 UTC by the
owner. Checked before use:

- 240 of 240 batches, 4,800 pairs, one latest score each; every row run `chat-3` and the build's
  instrument; no batch saved twice; no pair outside the plan; no chat-3 row in `chat-1`'s file.
- Scores are integers 0-100. Labels agree with scores: Good Fit 60-95 (mean 79.6), Potential Fit
  8-78 (44.6), No Fit 1-35 (8.9); none is a No Fit ≥ 60 or a Good Fit < 50.
- Close to `chat-1`: mean score 28.4 vs 30.3; No Fit / Potential / Good 57.5 / 30.1 / 12.4% vs
  57.2 / 29.0 / 13.8%.
- Per JD: score SD 10.2 to 23.6 (median), at least 12 distinct scores, Spearman with X2 0.12 to
  0.51 (median), never below 0. Folds (5, by JD, stratified): 11 / 10 / 9 / 9 / 9 JDs.
- Looked at by hand: the one batch running against X2 (ρ −0.33, a Data Science JD's X2 ranks
  81-100) scores every CV 2-32 with reasons naming a different role each, i.e. noise among
  near-zero scores; the JD with 99 Fits is a React Native posting whose pool is mostly JavaScript
  developers (96 Potential Fit, mean 44). Neither changes anything.

## §4 record (21 Oct 2026): stage 2 = L1 on F12

`chat3 --select` → `output/rerank/chat3-select.json`, `logs/chat3_select.log`, 2 min 46 s. Out of fold
on chat-3 (5 JD folds, 48 JDs), Recall@10 against chat-3's tie-aware top 10:

| Arm | R@10 total [CI] | generic / targeted | nDCG@10 |
|---|---|---|---|
| X2's order (context) | 0.472 [0.412, 0.530] | 0.414 / 0.588 | 0.794 |
| **L1 F12** | **0.513** [0.450, 0.575] | 0.468 / 0.605 | 0.818 |
| L1 F9 | 0.511 [0.450, 0.570] | 0.468 / 0.599 | 0.815 |
| L2 F12 | 0.507 [0.445, 0.565] | 0.477 / 0.568 | 0.809 |
| L2 F9 | 0.502 [0.441, 0.558] | 0.465 / 0.574 | 0.809 |

The rule picks L1 F12 (0.513; no tie at 3 dp). The four arms lie within 0.011 of each other, well
inside their CIs: on chat-3 the menu barely matters. Steps 3-4b had picked L2 F9 on `chat-1`; the
linear model is also the one whose weights can be read.

**Before §5:** the cold checks of what the test reads (D55, any final-set score): `stage1` (X2),
`role_family`, `extraction`, `dev_pools`, `final_set`, `ordering`, `indomain_v2`, ~35 min; today's
`rerank_chat`, `feature_arms`, `stage1_classifier` and `stage1_threshold` passed already. The rest
of the cold suite (`stage2` ~3.5 h, superseded records) is not read by the test and runs before the
write-up.

## §5 record (22 Oct 2026): all three rules pass

`chat3 --test` → `output/rerank/chat3-test.json`, `logs/chat3_test.log`, 7 min 12 s, peak RSS 5.6 GB.
Run once, after the cold checks of everything it reads passed (`logs/d55_cold_pretest.log`: `stage1`,
`role_family`, `extraction`, `dev_pools`, `final_set`, `ordering`, `indomain_v2`; with 21 Oct's
`rerank_chat`, `feature_arms`, `stage1_classifier`, `stage1_threshold`). Stage 2 = L1 on F12, fitted
once on the 4,800 chat-3 pairs. 60 final-set JDs, reference `chat-1`, nothing fitted on them.

| Pipeline (60 JDs) | Trained on | R@10 (gen / tar) | Δ vs X2 [CI] | Δ vs S1's order [CI] | nDCG@10 | Chat No Fits |
|---|---|---|---|---|---|---|
| X2's order | nothing | 0.419 (0.421 / 0.416) | – | −0.034 [−0.063, −0.005] | 0.776 | 1.70 |
| X2 top 100 → S1's order | D50 | 0.454 (0.464 / 0.432) | +0.034 [+0.005, +0.063] | – | 0.801 | 1.55 |
| **X2 top 100 → stage 2 (chat-3)** | **chat-3** | **0.484** (0.497 / 0.459) | **+0.065 [+0.036, +0.096]** | **+0.031 [+0.007, +0.056]** | 0.822 | 1.35 |
| X2 top 100 → L2 (`chat-1`, out of fold; no rule) | `chat-1` | 0.497 (0.503 / 0.485) | +0.078 [+0.046, +0.110] | +0.043 [+0.019, +0.068] | 0.826 | 1.30 |

1. **Headline: pass.** +0.065 Recall@10 over X2's order, CI above 0 and above the +0.05 bar
   (generic +0.076, targeted +0.044).
2. **Chat vs chat-free: pass.** +0.031 over S1's order (stage 2 trained on D50 labels), CI above 0.
3. **Humans: pass, narrowly.** Per-JD AUC on the final label vs X2, `all`: +0.068 [+0.002, +0.143],
   48 JDs (generic +0.073, targeted +0.061). Beside it, not part of the rule: `llm_free` +0.040
   [−0.046, +0.135] (36 JDs; targeted −0.106 [−0.213, −0.018], 11 JDs); pairwise on yc's grades
   +0.104 [+0.049, +0.165] (55 JDs), `llm_free` +0.109 [+0.044, +0.183] (52 JDs).

**Reading.**

- On JDs it never saw, a stage 2 trained on chat scores of other JDs beats X2's order by more than
  the bar and beats the chat-free order. Owner's question of 18 Oct ("is L2 just tuned to
  `chat-1`?"): mostly no.
- The old L2 (trained and chosen on these 60 JDs) is only +0.012 [−0.011, +0.036] above the clean
  stage 2. That is the optimism of training and choosing on the reference: about 0.01 Recall@10,
  not significant.
- What training on chat scores buys over the chat-free S1 order is +0.031 on untouched JDs (steps
  5c-5d estimated 0.03-0.04 on seen ones).
- The human evidence is weaker than the chat evidence: rule 3 passes on `all` by a CI edge of
  +0.002, and the `llm_free` AUC is not above 0 (pairwise on grades is, in both subsets). Report
  both.

**Still owed before this plan is complete:** a `--check` for `chat3` (rebuild the selection and the
test from the committed replies, diff the records) and its cold run (D55); the rest of the cold
suite (`stage2` ~3.5 h and the superseded records) before the write-up.

## Comparison with the C2 arms of steps 5b-5d (22 Oct 2026; exploratory)

Owner: "compare this against the C2 results". `chat3 --compare-c2` → `output/rerank/chat3-vs-c2.json`,
`logs/chat3_compare_c2.log`, 7 min 36 s. Same 60 JDs, `chat-1` reference, cached C2 scores only;
the refitted stage 2 reproduces §5's 0.484. Paired against stage 2 (chat-3). "C2 + stage 2 (chat-3)"
is new. C2's fold-0 scores were seen in the 5a pilot; folds 1-4 give the same picture.

| Pipeline (60 JDs) | Trained on the test JDs' chat scores? | R@10 | Δ vs stage 2 (chat-3) [CI] | nDCG@10 | Chat No Fits | Human AUC vs X2 |
|---|---|---|---|---|---|---|
| X2 top 100 → C2 alone (5d) | no | 0.379 | −0.106 [−0.144, −0.067] | 0.751 | 1.63 | +0.057 |
| X2's order | no | 0.419 | −0.065 [−0.096, −0.036] | 0.776 | 1.70 | – |
| X2 top 100 → S1's order | no | 0.454 | −0.031 [−0.056, −0.007] | 0.801 | 1.55 | +0.044 |
| X2 top 100 → C2 + S1 (5d) | no | 0.464 | −0.021 [−0.050, +0.009] | 0.813 | 1.30 | +0.094 |
| X2 top 100 → C2 + stage 2 (chat-3) (new) | no | 0.480 | −0.005 [−0.033, +0.025] | 0.826 | 1.23 | +0.084 |
| **X2 top 100 → stage 2 (chat-3)** | **no** | **0.484** | – | 0.822 | 1.35 | +0.068 |
| X2 top 100 → L2 (`chat-1`, OOF) | yes | 0.497 | +0.012 [−0.011, +0.036] | 0.826 | 1.30 | +0.120 |
| X2 top 100 → C2 + L2 (5d) | yes | 0.501 | +0.016 [−0.009, +0.041] | 0.840 | 1.18 | +0.118 |
| L2 top 30 → C2 + L2 (5b) | yes | 0.522 | +0.037 [+0.013, +0.061] | 0.849 | 1.17 | +0.117 |

- Among pipelines that never trained on these JDs' chat scores, stage 2 (chat-3) is the best. It is
  +0.106 over C2 alone and +0.021 over the best zero-shot pipeline (C2 + S1; CI crosses 0).
- C2 adds nothing to stage 2 (chat-3) (−0.005), as it added ≤ +0.02 to S1 and L2 in 5b-5d. It does
  cut chat No Fits (1.35 → 1.23).
- Only 5b's cascade is clearly above stage 2 (chat-3) (+0.037). It reranks with L2 trained on
  `chat-1`, and its top-30 cut was set from `chat-1` coverage on these JDs. The clean analogue,
  stage 2 (chat-3) top 30 → C2 + stage 2, was not computed: it would be one more arm chosen after
  seeing these figures.
- On the human labels, the `chat-1`-trained arms lead stage 2 (chat-3) (L2: AUC +0.052 [+0.011,
  +0.091]; pairwise +0.021, CI across 0). `chat-1` cannot explain that, since the humans never saw
  it. More likely causes are that L2's training JDs were drawn like the test JDs, or L2 on F9. This
  is exploratory.
