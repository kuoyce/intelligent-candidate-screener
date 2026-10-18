# Two-stage screening: a learned stage-1 classifier, then a reranker trained on `chat-1`

Status: **approved by the owner, 18 Oct 2026. Steps 1-3 done. Step 4 (revised) approved 18 Oct 2026,
in progress; step 5 is findings only. D55 cold run of `rerank_chat` deferred to the end of the branch.** Decision D61 below. Branch
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

*Superseded 18 Oct 2026 by "Step 4 (revised)" below; kept as approved (D45).*


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

## Analysis after step 3 (exploratory, 18 Oct 2026)

**End to end, corrected reading.** X2's top 100 → L2 *does* beat X2 alone (+0.072 [+0.042,
+0.104]). The pipeline that does not is S1's passed set → L2 (+0.030 [−0.010, +0.071]).

**Where the loss is: a ceiling check** (scratch script, not a record). *Ceiling* = the share of the
chat's top 10 (tie-aware, over every chat-scored CV) that stage 1 hands to stage 2, i.e. Recall@10
with a perfect stage 2.

| Stage 1 | Ceiling, total (generic / targeted) | L2 reaches |
|---|---|---|
| X2's top 100 | 1.000 | 0.492 |
| S1 passed set (P ≥ 0.5) | **0.671** (0.745 / 0.523) | 0.449 |

1. **Stage 1 is the main loss** for the S1 pipeline: a third of the chat's top 10 (half for
   targeted JDs) never reaches stage 2.
2. **Stage 2 recovers about half** of the chat's top 10 even inside X2's 100. Part is reference
   noise (a median of 4 CVs tie at the chat's 10th score; one run with batch drift), part is L2
   (it only reweights F12; the text models failed, step 5).
3. Judged Fits in the top 10 barely move (X2 alone 116, X2 top 100 → L2 113, S1 → L2 110); only 10
   of 500 CVs per JD are judged, so this is weak either way.

**Disclosure.** The same scratch run also scored "passed set with a floor of 10 / 20 CVs per JD"
on the final-set JDs (floor 20: R@10 0.488, ceiling 0.770). Those variants are now **seen**: a
floor rule in step 4 is post-hoc unless justified on development data.

**Why F12's concept terms get 0 or negative weights: collinearity, not a fault.** On the D50
training pairs, `coverage`, `coverage_unweighted` and `skill_overlap` are near-copies (Spearman
0.99-1.00; variance inflation factors 130, 177 and 38); `log_matched` correlates 0.95 with them.
Each one fitted alone beside X2's three terms gets a **positive** standardised weight (+0.20 to
+0.30). Together the model splits one signal four ways, so their individual signs mean nothing;
under L1's non-negativity three are held at 0 because the others already carry it.
`log_jd_concepts` is a count control: negative means more listed requirements make the same
matches worth less.

## Step 4 (revised; approved by the owner 18 Oct 2026)

**Goal.** Raise what stage 1 hands to stage 2 (the ceiling) without an unbounded list, then see
whether better features help either stage. Order: 4a, then 4b. If 4b changes S1, 4a's chosen rule
is re-applied mechanically and reported beside.

### 4a. Stage-1 threshold

**Sizes** (counted 18 Oct 2026, sizes only; no recall or ceiling looked at):

| t | Final pools: median passed [range] | JDs < 10 | JDs > 100 | New CVs to chat-score (≥ batches of 20) | Dev-select median |
|---|---|---|---|---|---|
| 0.5 (now) | 23 [1, 81] | 14 | 0 | 0 | 21 |
| 0.4 | 32 [1, 98] | 8 | 0 | 102 (27) | 31 |
| 0.3 | 56 [1, 142] | 2 | 9 | 702 (63) | 55 |
| 0.2 | 131 [30, 270] | 0 | 39 | 3,529 (207) | 122.5 |
| 0.1 | 372 [183, 462] | 0 | 60 | 15,720 (817) | 357 |

S1's probabilities are not calibrated for a pool: it was fitted with balanced classes on D50 (61%
Fit), a pool is ~1% Fit. Below 0.2 nearly the whole pool passes, so 0.1 and 0.05 are out.

- **Arms:** t ∈ {0.5, 0.4, 0.3, 0.2}, beside X2's top 100. Stage 2 is L2 as trained in step 3 (it
  scores any pair; no retraining).
- **Operating point, chosen before looking (proposal):** a stage-2 budget of a median ≤ 100 CVs
  per JD on **dev-select** (D46-clean: sizes only) → **t = 0.3** (dev-select median 55; 0.2 gives
  122.5). Ceiling and recall rise as t falls, so any "maximise the ceiling" rule picks the lowest t
  allowed; the real choice is the budget. The other arms are reported as a curve (exploratory).
- **Measures** (per JD, mean, JD-resampled CI, generic / targeted / total): ceiling; end-to-end
  Recall@10, nDCG@10 and chat No Fits with L2; list size and JDs under 10; judged Fits kept
  (stage-1 recall, human final set). Paired against X2 alone, X2's top 100 → L2, and t = 0.5 → L2.
- **Bar (proposal):** the S3 bar, end to end: Recall@10 Δ vs X2 alone ≥ +0.05 with the CI above
  0. Reported beside: Δ vs X2's top 100 → L2.
- **Chat scoring.** Every passed CV outside the CVs already scored, down to the lowest arm scored:
  702 CVs (≥ 63 batches) for t = 0.3, 3,529 (≥ 207) for t = 0.2. Pasted as further batches in each
  JD's existing chat (batch 7 onwards), same instrument, through the app (`--extend-s1` gains a
  threshold). At t = 0.2 a JD's chat grows by up to 170 CVs, which may near the chat's context
  limit; t = 0.3 adds at most 42 per JD.
- **Caveat.** Scoring more CVs enlarges the reference: a newly scored CV can enter the chat's top
  10, so X2's top-100 ceiling can drop below 1. That is the more complete reference, not an error.

### 4b. Features

- **4b-0 de-duplicate:** F12 → **F9** without `coverage_unweighted`, `skill_overlap`,
  `log_matched` (keep `coverage`, `log_jd_concepts`, `log_cv_concepts`). Expected: no loss, and
  coefficients that read correctly.
- **Candidates to add** (text only, D53; the JD title is allowed, D54). Each is one arm:

| Id | Feature | Why | Cost |
|---|---|---|---|
| N1 | JD-title role match: the CV's probability of the JD title's family (Z1L's lookup-or-text) | role carries all of L2's gain; Z1L was the dev-best role rule | minutes (role model cached) |
| N2 | R+N tagger concept coverage (GLiNER / MiniLM), *replacing* `coverage` | passed the overlap gate before; never tested in a pool | to measure first (tagging 12.6k pool + 14k dev CVs) |
| N3 | F1 negation drop, F2 no-experience flag | open options from score-defects | minutes |
| N4 | SBERT zero-shot cosine | a second text view; embeddings cached | minutes |
| N5 | log CV length | control (L3 favoured long CVs) | minutes |

- **Where each stage is judged** (never the final set for stage 1, D46): stage 1 by 5-fold JD CV on
  D50 (per-JD AUC on held-out JDs); stage 2 by `chat-1` out-of-fold Recall@10 (D61).
- **Adoption rule (proposal):** F9 is adopted if L2's out-of-fold Recall@10 Δ vs F12 is ≥ −0.005
  (no loss). An added feature is adopted if its paired Δ (vs F9) is > 0 with the CI above 0. Then
  one final arm with every adopted feature. Multiple arms: a single nominal pass is weak evidence.

**Cost.** 4a: about half a day of code and ~10 min per run, plus the owner's chat time (63-207
batches). 4b: about half a day, ~15 min per arm (N2 extra, to measure).

### Open questions (step 4)

Answered 18 Oct 2026 (owner):

10. Operating point: **t = 0.3**, from the dev-select budget (median ≤ 100). *Agreed.*
12. 4a bar: the S3 bar end to end (Recall@10 Δ vs X2 alone ≥ +0.05, CI above 0). *Agreed.*
13. Features: **N1-N4**; N5 (CV length) dropped. Adoption rule as above. *Agreed.*
14. New chat batches go in a **fresh chat** per JD (not the JD's existing chat).
15. D55 cold run of `rerank_chat`: **deferred** until this plan and branch are finished.

11. Chat scoring: **t = 0.3 only** (702 new CVs over 47 JDs; median 10, max 62 per JD); t = 0.2
    stays a size-only row. *Agreed.*
16. Anchors: yes. Each fresh chat is padded with already-scored CVs of the same JD to a multiple
    of 20 (5 new → 15 anchors). *Agreed.* Total: 558 anchors, 1,260 CVs, 63 batches.

17. At least 5 anchors per JD: **69 batches** (702 new + 678 anchors = 1,380 CVs). *Agreed.*
18. Anchor *selection* is fixed before scoring: spread evenly over the JD's `chat-1` score order
    (seeded), shuffled with the new CVs, never marked. How anchors are *used* (correct or not) is
    **decided after seeing the drift** (owner); whatever is chosen is labelled post-hoc.
19. Mixing `chat-1` and the fresh chats in 4a's reference is accepted (same model, same instrument;
    owner). The fresh chats get their own label (`chat-2`) so they can be told apart. Per pair the
    **latest score wins**, as `chat_rerank.latest()` does now: an anchor's fresh score replaces its
    `chat-1` score in step 4's reference. Step 3's record and L2's training stay on `chat-1` as
    committed, so they still reproduce.
20. A possible second, smaller chat round after 4b; N2 timed first and skipped if over ~1 h unless
    the owner says otherwise. *Agreed.*

**Why 4a needs chat scores.** L2 needs none: it is trained on X2's top 100 and scores any pair.
The scores are for the **reference**. At t = 0.3 a JD's passed set holds CVs outside X2's top 100
that the chat never saw. Unscored, (a) the ceiling cannot tell whether one of them belongs in the
chat's top 10, and (b) when L2 puts one in its top 10 it can only count as a miss, never a hit and
never a No Fit. Every unscored CV would count against the lower threshold and for X2's top 100
(whose CVs are all scored), so the comparison 4a exists for would be biased. Scoring the 702
removes that; t = 0.2 (3,529) would stay a size-only row.

## Step 5 (findings only, 18 Oct 2026; not planned)

The template the retrieval literature follows for cross-encoder rerankers, against R1-R3 and L3:

| Practice | Template | Ours |
|---|---|---|
| Starting checkpoint | fine-tune a model **already trained as a reranker** (MS MARCO cross-encoder); two-stage (MS MARCO, then LLM labels) was best in Rank-DistiLLM; a non-reranker base "requires more training data" | `all-MiniLM-L6-v2`, a sentence-embedding model, random score head: **the biggest gap** |
| Training candidates | the first-stage retriever's hard candidates, "the first-stage retriever matters most"; ~10k queries, depth 50 sufficed | R1-R3: D50, 100 JDs × 10, not X2's shortlists; L3: right distribution but 48 JDs per fold |
| Loss | LLM distillation: pairwise RankNet ≈ listwise; binary labels: BCE with mined hard negatives (`pos_weight`); LambdaLoss reported strong | R1, L3 pointwise MSE on 0-100 (learns the chat's batch drift; a within-batch pairwise loss cancels it); R2 RankNet on 800 pairs |
| Long documents | passages + MaxP / FirstP (Dai & Callan); effective **only with MS MARCO pre-fine-tuning** (Zhang, Yates & Lin); or long-context rerankers; person-job work decomposes résumé and job (Malt 2026) | R1/R2 cut at 384 tokens; L3 MaxP with no pre-fine-tuning: the setting that study found ineffective |
| Validation | cross-encoders "overfit rather quickly": reranking evaluator on the stage-1 top k, keep the best checkpoint | fixed epochs, no in-fold validation; L3 under-trained |
| Scale | 110M-600M (monoELECTRA, ModernBERT) | MiniLM-L6, 22M (CPU) |

Possible shape, to plan later: **5a** zero-shot MS MARCO and long-context rerankers on X2's top 100
against `chat-1` (inference only, ~30 min-3 h CPU); **5b** fine-tune the better one on `chat-1`
with a within-batch pairwise loss, 5 JD folds, early stopping on an inner fold (MiniLM-size: ~1.5-2
h per fold at L3's measured 0.5 s/window). Needs model downloads (the code loads local files only).

Sources: [Rank-DistiLLM](https://arxiv.org/html/2405.07920v1);
[Training reranker models with Sentence Transformers](https://huggingface.co/blog/train-reranker);
[sbert.net MS MARCO cross-encoder training](https://sbert.net/examples/cross_encoder/training/ms_marco/);
[sbert.net cross-encoder losses](https://sbert.net/docs/package_reference/cross_encoder/losses.html);
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2);
[Zhang, Yates & Lin, ECIR 2021](https://cs.uwaterloo.ca/~jimmylin/publications/978-3-030-72240-1_11.pdf);
[Pretrained Transformers for Text Ranking](https://arxiv.org/pdf/2010.06467);
[Malt person-job fit, 2026](https://arxiv.org/abs/2601.10321v2);
[TalentCLEF 2026](https://ceur-ws.org/Vol-4283/paper491.pdf).

## Assumptions (to confirm)

- Stage 2 uses F12, the same features as stage 1 (owner, 18 Oct 2026). R1–R3 and L3 bring the
  text models.
- SBERT in F12 is SBERT weak (X4's term), not the untuned base model. SBERT weak was trained on
  weak pairs from the A2 train region. The D50 documents may be among them, so the D50 fit can
  see a mildly optimistic SBERT column.
