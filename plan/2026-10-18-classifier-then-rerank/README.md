# Two-stage screening: a learned stage-1 classifier, then a reranker trained on `chat-1`

Status: **approved by the owner, 18 Oct 2026. Steps 1-3 done (18 Oct 2026); D55 cold run and
step 4 open.** Decision D61 below. Branch
`exp/classifier-then-rerank`, from `exp/stage2-llm-rerank`.

## Vision (owner, 18 Oct 2026)

```
500-CV pool per JD ──stage 1──▶ top 100 ──stage 2──▶ top 10 ──▶ recruiter
                    classifier            reranker
                    (cheap, learned)      (trained on chat-1 scores)
```

- **Stage 1 is a classifier, not a hand-set scorer.** X2 (`2·role_match + experience_compatible
  + 0.999·tfidf_scaled`) works, but its rules are hard to manage and optimise. A learned model
  chooses the coefficients. Later, optionally: hard filter rules first, then a softer classifier.
- **Stage 2 reranks stage 1's top 100** and hands over the top 10. Its target is the `chat-1`
  score (0-100). Getting the right 10 in matters more than their order inside the 10.
- **Order of work:** a quick stage-1 test, then the overlap check, then the reranker. Stage-1
  tuning and hard filters only if time is left (about 10 days remain, report included).

## Decision D61 (in `docs/decisions.md`)

> Two-stage screening (owner, 18 Oct 2026). (1) D51 amended: a **stage-1 classifier** may be
> trained on the D50 batch's Fit / No Fit labels (run 11), as notebook 19 already did under D56.
> (2) **D59 and D60 superseded:** `chat-1` (the owner's claude.ai scoring of X2's top 100 for the
> 60 final-set JDs, `claude-haiku-5-5`, the listwise instrument `c64578dd`) is kept. For the
> stage-2 task only it is the **training target and the reference**. (3) **D46 does not apply to
> the stage-2 task**, a separate application from classification. Folds are by JD (5-fold), and
> its human-grade figures are labelled exploratory. The human grades stay the only reference for
> classification. `chat-1` never enters `judgements.csv` or the final set. No Claude subprocess is
> started from the repository.

Marked in the log: D59, D60 → *superseded by D61*; D51 → *amended by D61*. They are not deleted (D45).

## What is already known

- **Stage-1 AUC on the final set is about done.** Notebook 19, part 4 trained logistic base (4)
  on all 100 D50 JDs. Per-JD AUC on the A2 final set was **+0.004 [−0.006, 0.014]** against X2
  (0.850); logistic all (11) gave +0.010 and boosting +0.004. All three are well inside ±0.06.
  *Sign corrected 18 Oct 2026:* notebook 19 parts 3-4 call `paired_delta(hand, learned)`, so
  these are X2 − learned. Learned − X2 is **−0.004**, −0.010 and −0.004; S1's base-4 arm
  reproduces −0.0037 [−0.0138, 0.0060], the exact mirror. The ±0.06 reading is unchanged.
  The new questions are the 500-CV pools and the overlap.
- **Risk for the pools.** D50 pairs are 80% same-keyword and 61% Fit. A 500-CV pool is about
  1% Fit, mostly other roles. A classifier trained on D50 has rarely seen an off-role CV, so it may
  weight role less than X2 does. Step 1 measures this directly (pool recall).
- **`chat-1` covers X2's top 100 only** (6,000 pairs, every one scored). 57% are chat No Fit,
  so a top 100 is not a set of Fits.

## Feature set (checked 18 Oct 2026)

The same features feed stage 1 and stage 2. **F12** = the 11 of `a1_weights.ALL_FEATURES` plus
SBERT (owner, 18 Oct 2026):

| Feature | From | In F12 |
|---|---|---|
| `role_match`, `experience_compatible`, `tfidf_scaled` | X2's three terms | yes |
| `coverage` | entity-weighted concept coverage (seed-rule concepts, IDF weights) | yes |
| `coverage_unweighted`, `log_matched`, `log_jd_concepts`, `log_cv_concepts` | the same concepts | yes |
| `skill_overlap` | vocabulary matcher, no context filter | yes |
| `exp_gap`, `exp_known` | extracted years | yes |
| `sbert` | SBERT weak cosine (X4's text term, chunk-mean embeddings) | **added** |
| R+N tagger concepts (GLiNER / MiniLM) | gliner-concepts plan: passed the overlap gate, no learner gain | no (option) |
| soft coverage at 0.60 | soft-matching plan: no learner gain | no |
| coverage weighting W1-W3 | coverage-weighting-variants plan: not chosen | no |
| F1 negation drop, F2 no-experience flag | score-defects notebook: implementation not decided | no (option) |

See open question 5 for the options.

## Step 1. Stage-1 quick test (S1)

- **Model:** logistic regression on **F12**, trained on **all 1,000 D50 pairs** (100 JDs, Fit =
  Good ∪ Potential). Second arm: logistic on the base 4 (the notebook 19 reproduction). Nothing
  is tuned.
- **Stage-1 output = the CVs the classifier calls Fit** (probability ≥ 0.5, the untuned default).
  Tuning the threshold is a step 4 enhancement.
- **Measures**, beside X2, with generic / targeted / total and JD-resampled 95% CIs:
  1. per-JD AUC on the final set (`full` and `llm_free`), paired Δ against X2;
  2. **recall of the passed set** on the 500-CV final pools: judged Fits kept / judged Fits
     (unjudged count as not relevant). It is compared with **X2 cut at the same size per JD**
     (size-matched), so neither gains from passing more CVs;
  3. the passed-set size per JD (median, range), and the same on dev-select (role-family proxy).
- **Bar, "roughly similar to X2":** the paired Δ against X2 is **≥ −0.06** on both per-JD AUC
  (total) and size-matched recall. Better also passes (owner, 18 Oct 2026).
- **Learned coefficients** are reported (standardised), so the role weight is visible.
- If S1 fails, stop and report. Stage 2 still runs on X2's shortlist.

## Step 2. Overlap with X2's top 100 (S2)

- Per JD: the share of S1's passed set that lies in X2's top 100, so has a chat score. Mean,
  CI, by stratum.
- **Bar: mean overlap ≥ 0.80** (owner, 18 Oct 2026). Pass: the reranker trains and is scored on
  `chat-1` with no gap that matters.
- Also reported: the passed CVs per JD with no chat score, and the final set's judged Fits that
  each shortlist keeps.
- **Known risk:** D50 is 61% Fit, so at 0.5 the classifier may pass far more than 100 CVs per JD.
  The overlap then fails mechanically (a set of 200 is at most 50% inside X2's 100). See open
  question 6.
- If it fails, the options are written up for the owner (score the missing CVs in chat, or keep
  X2 as stage 1). Nothing is decided here.

## Step 3. The reranker on `chat-1` (S3)

**Data:** the 6,000 `chat-1` pairs, target = chat score. **5 folds by JD**, seed 0, stratified
(8 generic + 4 targeted JDs per fold). A model never trains on the JD it is scored on.
1,014 CVs appear under more than one JD. They may cross folds; the target is per pair.

**Candidates.** No training on `chat-1`:
- X2's order;
- the S1 classifier's probability;
- R1, R2, R3 as stored (trained on D50 run 11).

Trained on `chat-1`, per fold:
- **L1 linear:** ridge regression on F12, the engineered features held to non-negative
  coefficients;
- **L2 boosting:** `HistGradientBoostingRegressor` on F12, with monotonic increasing
  constraints on the engineered features;
- ablations of the better of L1/L2: without `tfidf_scaled`, without `role_match`, without `sbert`;
- **L3 cross-encoder** (owner, 18 Oct 2026): MiniLM fine-tuned per fold on the chat score.
  - **Truncation, measured on 600 `chat-1` pairs:** the current cross-encoder (`max_length` 384,
    longest-first truncation) cuts 90% of pairs. The median pair keeps 63% of its JD and 52% of
    its CV, and one CV in ten keeps under 22%. R1/R2 were trained and scored this way. That is
    stated beside their results.
  - **L3 chunks instead.** `max_length` 512. The JD keeps its first 192 tokens. The CV is cut into
    windows of 317 tokens, stride 256. The pair score is the max over windows (MaxP). In training,
    every window carries the pair's target.
  - Every pair scores every word of its CV. A JD beyond 192 tokens is still cut; the plan reports
    how often.
  - One fold is timed first. If all five come to more than about an hour, the owner is asked first
    (AGENTS testing rule 10).

Defaults stay fixed. Any tuning happens inside the training folds only.

**Metrics** (per JD, mean over 60 JDs, JD-resampled CI, generic / targeted / total, paired Δ
against X2's order):
1. **Recall@10 against the chat's top 10** (the bar metric). It is the share of the chat's top 10
   in the model's top 10. Order is ignored, and each outsider costs 0.1. Tie-aware: if *t* CVs tie
   at the chat's 10th score for *r* remaining slots, each counts *r/t*.
2. **nDCG@10, gain = chat score:** order, and how bad each outsider is.
3. **Chat No Fits in the top 10:** a diagnostic. It matters most for X2 as a sole model.
4. **Human check (exploratory):** out-of-fold scores on the 422 judged CVs inside the shortlists.
   Per-JD AUC and pairwise on the human grades, to see whether a model learned the chat's mistakes.
5. **Learning curve** for the best learner: 12, 24, 36 and 48 training JDs (inside each fold).

**Bar (owner, 18 Oct 2026):** Recall@10 Δ against X2's order **≥ +0.05** (half a CV per JD),
with the paired CI above 0.

**No noise ceiling (accepted, owner 18 Oct 2026).** `chat-1` was run once. Its batch drift (+3.6
in batch 1 to −3.7 in batch 5) and its multiples-of-5 scores cap every model's Recall@10 below
1. The report states this as a limitation.

**End to end, if S2 passes:** S1's passed set → the best reranker → top 10. Same metrics. CVs
with no chat score are counted and shown, never imputed.

## Step 4. Enhancements (only if time)

Stage-1 tuning (class weights, the extra features), hard filters (each filter's recall loss is
measured on dev-select first), and a threshold in place of a fixed K.

## Code and records

| What | Where |
|---|---|
| S1 classifier as a ladder system (fits on D50) | `matching/learned.py`, implements `matching.base.System` |
| S1 + S2 evaluation | `evaluation/stage1_classifier.py` → `output/matching/stage1-classifier.json` |
| `chat-1` targets, folds, tie-aware Recall@10 | `rerank/chat_targets.py` |
| S3 evaluation | `evaluation/rerank_chat.py` → `output/rerank/chat-rerank-eval.json` |
| Report | `notebooks/24-classifier-then-rerank.ipynb` |
| Tests (inline synthetic) | tie-aware Recall@10, JD-disjoint folds, monotonic/non-negative fits, the registry |

- Both `--check` modules go in the `checks` registry.
- `chat-1` replies (`output/rerank/chat-rerank-replies.*`) are committed with this plan.
- Verification: `uv run pytest`. These are new records, so D55's cold run for them is needed
  before the plan is marked complete. The owner is asked first, with the estimate (D58).
- **Cost to check first:** S1 features on 60 × 500 pool pairs plus dev-select's 150 × 500.
  `a1_weights.build_scorer` uses `dev_pools`, a > 4.5 GB check, so one JD is timed before the rest.

## Open questions

Resolved 18 Oct 2026 (owner):
1. The S3 bar: Recall@10 Δ ≥ +0.05, with the CI above 0.
2. "±0.06" means no worse than −0.06; better also passes.
3. L3 is included, with chunking (above).
4. No noise ceiling: accepted as a limitation.

5. Feature options: the R+N tagger concepts and the F1/F2 fixes stay out of F12. They are one
   ablation arm in S3 if time allows.
6. If the 0.5 threshold passes far more than 100 CVs (the S2 risk), S2 and S3 run on the
   classifier's top 100 by probability. The threshold is left to step 4.

Open: none.

## Implementation record (steps 1-2, 18 Oct 2026)

Code: `matching/learned.py` (`LearnedStage1`, `Features`), `evaluation/stage1_classifier.py`
→ `output/matching/stage1-classifier.json`, registered as check `stage1_classifier`
(`run_alone`, 440 s, peak 4.2 GB). Tests: `tests/test_stage1_classifier.py`. Guards in the
run: X2's final-set and pool scores match the stage-1 record's digests; X2's hand weights over
F12's first three columns reproduce X2 row by row on all three frames; X2's top 100 is exactly
the 6,000 pairs `chat-1` scored.

### S1: pass

| Arm | Per-JD AUC Δ vs X2, total (gen / tar) | Size-matched recall Δ, total (gen / tar) | Passed per JD, median [range] |
|---|---|---|---|
| **F12** (S1) | **+0.009** [−0.013, 0.031] (+0.022 / −0.016) | **−0.007** [−0.040, 0.027] (+0.006 / −0.032) | 23 [1, 81] |
| base 4 | −0.004 [−0.014, 0.006] (−0.004 / −0.003) | +0.004 [−0.011, 0.020] (+0.011 / −0.009) | 17.5 [0, 75] |

AUC n = 60 JDs, recall n = 59 (JDs with a judged Fit). F12 on `llm_free`: AUC Δ +0.003
[−0.035, 0.038]. Within-family AUC Δ, total: +0.028 [−0.015, 0.072]. Absolute AUC: X2 0.850,
F12 0.860, base 4 0.847.

- **Role still dominates.** F12's standardised coefficients: `role_match` 1.13, `sbert` 0.37,
  `log_matched` 0.35, `tfidf_scaled` 0.26, then ≤ 0.17. The pool risk above (D50 rarely shows an
  off-role CV) did not show as a weak role weight.
- **The 0.5 threshold passes few CVs.** Median 23 per JD (generic 27, targeted 10); 17 of 60 JDs
  pass ≤ 10, none passes 100. The passed set keeps **240 of 367** judged Fits (recall 0.59),
  where X2's top 100 keeps 332. Size-matched, X2 does no better (0.60), so the loss is the size,
  not S1's order.
- Dev-select (role-family proxy): passed median 21 [0, 110], 3 JDs over 100; size-matched proxy
  recall Δ −0.012 [−0.021, −0.003].

### S2: pass on the passed set (the registered shortlist)

| Shortlist | Mean overlap with X2's top 100, total (gen / tar) | CVs with no chat score |
|---|---|---|
| **Passed set** (governs: median 23 ≤ 125) | **0.987** [0.978, 0.995] (0.987 / 0.987) | 20 in all, max 4 per JD |
| Top 100 by probability (open question 6) | 0.691 [0.665, 0.717] (0.704 / 0.664) | 1,857, median 32 per JD |

The pass is the other face of the small passed set: 23 CVs nested in X2's top 100. S1's top 100
would fail the bar (0.69).

### Readings and assumptions (owner to confirm)

- Both bars were judged on **point estimates**, CIs stated beside (stage1.py's S2 convention). *Confirmed.*
- Open question 6's "far more than 100" was read as a **median passed-set size above 125**, the
  size past which even a passed set nested around X2's top 100 cannot reach 0.80. Not triggered. *Confirmed.*
- Implemented inline, not by sub-agents (AGENTS workflow step 3): the work needed the context
  already loaded. Step 4's review can be a sub-agent.

### Open questions from steps 1-2 (resolved 18 Oct 2026, owner)

7. **Stage 1 for step 3's end-to-end run = S1's passed set** (P >= 0.5), not a top 100. The 20
   passed CVs outside X2's top 100 (12 JDs, at most 4 per JD) get chat scores: one more batch
   in each JD's existing chat through the Streamlit app (`chat_rerank --extend-s1`; D61 amended).
8. Notebook 19 corrected: parts 3-4 now call `paired_delta(learned, hand)` and were re-executed.
9. Both readings above (point estimates; "far more" = median above 125) confirmed.

## Step 3 record (18 Oct 2026)

Owner, 18 Oct 2026: the L1/L2 constraint split, features from the pool frame and the "best"
rule were confirmed; **learning curves skipped**; L3 runs all five folds without a stop to ask.

| What | Where |
|---|---|
| `chat-1` targets, folds, tie-aware Recall@10, nDCG@10, No Fits | `rerank/chat_targets.py` |
| L1 (bounded ridge), L2 (monotone boosting), out of fold | `rerank/learners.py` *(new; not in the code table)* |
| L3 (MaxP windows), per-fold weights and OOF scores | `rerank/chunked.py` *(new; not in the code table)* |
| S3 evaluation, human check, end to end | `evaluation/rerank_chat.py` → `output/rerank/chat-rerank-eval.json` |
| Report | `notebooks/24-classifier-then-rerank.ipynb` |

Fitting code sits in `rerank/` so it never names the final labels (D46 guard tests); the
evaluation module reads them only for the human check.

- **L1/L2 constraints:** non-negative / increasing on the 8 match signals (`role_match`,
  `experience_compatible`, `tfidf_scaled`, `coverage`, `coverage_unweighted`, `log_matched`,
  `skill_overlap`, `sbert`); free on `log_jd_concepts`, `log_cv_concepts`, `exp_gap`,
  `exp_known`. L1 = ridge alpha 1 on fold-standardised features (bounded least squares, equal
  to sklearn `Ridge` when no bound binds); L2 = `HistGradientBoostingRegressor` defaults.
- **L3 cost, measured:** 12,607 windows (2.1 per pair); a fold trains on ~10,100 windows.
  0.53 s/window on one process (11 threads), so ~8 h for five folds; two processes of 5
  threads gave 2.4 windows/s against 1.9, so L3 runs as two workers (folds 0, 2, 4 and 1, 3),
  about 6.5 h. `threads: 5` is in L3's config (float sums depend on it). The earlier ~3.8 h
  estimate (scaled from R2) was low by about half.
- **Cold check cost:** `rerank_chat --check` retrains L3's folds in one process, about 9 h.
  `--warm` replays R1-R3 and L3. D55 needs the cold run before the plan is marked complete; the
  owner is asked first.


### S3 result: L2 passes the bar (best), L1 passes; every text model loses

Recall@10 against the chat's top 10, X2's top 100, 60 JDs, out of fold (5 JD folds):

| Candidate | R@10 total (gen / tar) | Δ vs X2 order, total [CI] | Bar |
|---|---|---|---|
| X2's order | 0.419 (0.421 / 0.416) | – | – |
| S1 probability | 0.454 | +0.034 [+0.005, +0.063] | fail |
| **L2 boosting** | **0.492** (0.502 / 0.472) | **+0.072 [+0.042, +0.104]** (+0.081 / +0.056) | **pass, best** |
| L1 linear | 0.483 (0.490 / 0.469) | +0.064 [+0.035, +0.094] (+0.068 / +0.054) | pass |
| R3 bi-encoder (D50) | 0.409 | −0.010 [−0.074, +0.057] | fail |
| R2 cross-encoder (D50) | 0.234 | −0.186 [−0.246, −0.125] | fail |
| R1 cross-encoder (D50) | 0.191 | −0.228 [−0.293, −0.162] | fail |
| L3 cross-encoder (chat-1, MaxP) | 0.189 | −0.231 [−0.302, −0.156] | fail |

- nDCG@10 (gain = chat score): X2 0.776, L2 0.824 (+0.048 [+0.019, +0.081]). Chat No Fits in the
  top 10: X2 1.70, L2 1.35 (−0.35 [−0.70, −0.07]).
- **Ablations of L2** (Δ vs X2): without `role_match` +0.002 (level with X2), without `sbert`
  +0.042, without `tfidf_scaled` +0.060. The learned gain is role.
- **L1 coefficients** (standardised, mean of folds): `role_match` 12.1, `sbert` 6.1, `tfidf_scaled`
  4.0, `log_matched` 3.6; `coverage`, `coverage_unweighted`, `skill_overlap` held at 0 by the bound.
- **L3 hardly learned.** Training MSE 0.065-0.071 per fold against a target variance of 0.072
  (one epoch, lr 2e-5, every window carrying its pair's score). A scratch diagnostic (not a
  record): within-JD Spearman of L3 with the chat score 0.27; with the CV's window count 0.17,
  where the chat score's is 0.03, so MaxP also favours long CVs. Not a code fault: the scoring
  loads report no missing weights. Options not run: R1's selected grid point (lr 5e-5, 3 epochs,
  ~3x the 5.6 h), mean instead of max over windows. Either is exploratory now (its result is seen).
- **Human check (exploratory, 422 judged CVs in the shortlists, 55-60 JDs):** L2 over X2, per-JD
  AUC +0.116 [+0.043, +0.198], pairwise vs yc +0.127 [+0.064, +0.193]; L1 +0.071 / +0.100. L2 did
  not merely learn the chat's mistakes.

### End to end (reference: chat's top 10 over every chat-scored CV)

| Pipeline | R@10 total (gen / tar) | Δ vs X2 alone | Chat No Fits in top 10 | JDs returning < 10 |
|---|---|---|---|---|
| X2 alone | 0.419 (0.421 / 0.416) | – | 1.70 | 0 |
| S1 passed set, by probability | 0.422 | +0.003 [−0.033, +0.038] | 0.75 | 14 |
| X2 top 100 → L2 | **0.492** | +0.072 [+0.042, +0.104] | 1.35 | 0 |
| **S1 passed set → L2** (the owner's stage 1) | 0.449 (0.470 / 0.406) | +0.030 [−0.010, +0.071] | **0.72** | 14 |

S1 passed set → L2 against X2 top 100 → L2: −0.043 [−0.070, −0.018] (generic −0.031, targeted
−0.065). The short passed set misses part of the chat's top 10 (14 JDs hand over fewer than 10
CVs); it shows half the chat No Fits, mostly because it shows fewer CVs. Its threshold (step 4)
is the lever.

### Deviations

- Learning curve skipped (owner). L3 ran as two parallel workers at 5 threads (`threads` in its
  config). `rerank/learners.py` and `rerank/chunked.py` added beside the planned modules.
- `stage1_classifier.fit_arms` / `passed_set` were factored out after the S1 record was written;
  the record's `--check` was re-run to prove it unchanged.

## Assumptions (to confirm)

- Stage 2 uses F12, the same features as stage 1 (owner, 18 Oct 2026). R1–R3 and L3 bring the
  text models.
- SBERT in F12 is SBERT weak (X4's term), not the untuned base model. SBERT weak was trained on
  weak pairs from the A2 train region. The D50 documents may be among them, so the D50 fit can
  see a mildly optimistic SBERT column.
