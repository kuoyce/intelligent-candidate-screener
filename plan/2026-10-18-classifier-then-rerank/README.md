# Two-stage screening: a learned stage-1 classifier, then a reranker trained on `chat-1`

Status: **approved by the owner, 18 Oct 2026. Steps 1-4 done (18 Oct 2026); step 5 is findings;
step 5a (an instruction-tuned embedder) is proposed and awaits approval.
D55 cold run of `rerank_chat` deferred to the end of the branch.** Decision D61 below. Branch
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
21. Drift (owner, 18 Oct 2026, after seeing it: anchors +3.4 [+2.2, +4.6]): **headline left
    uncorrected**; no correction rule is chosen ("not enough data to make a decision"). The offset
    is reported beside every 4a figure.
22. The accidental retest (JDs 2-6 re-scored during the `chat-2` session): kept as its own run,
    `chat-1r`, through `output/rerank/chat-rerank-relabel.csv`; `chat-1` stays as committed (owner,
    option A). Reported as an exploratory noise check only.
20. A possible second, smaller chat round after 4b; N2 timed first and skipped if over ~1 h unless
    the owner says otherwise. *Agreed.*

**Why 4a needs chat scores.** L2 needs none: it is trained on X2's top 100 and scores any pair.
The scores are for the **reference**. At t = 0.3 a JD's passed set holds CVs outside X2's top 100
that the chat never saw. Unscored, (a) the ceiling cannot tell whether one of them belongs in the
chat's top 10, and (b) when L2 puts one in its top 10 it can only count as a miss, never a hit and
never a No Fit. Every unscored CV would count against the lower threshold and for X2's top 100
(whose CVs are all scored), so the comparison 4a exists for would be biased. Scoring the 702
removes that; t = 0.2 (3,529) would stay a size-only row.

## Step 4 record (18 Oct 2026)

Code: `rerank/chat_rerank.py` (`--fresh-s1`, fresh chats with anchors; `relabel`), `rerank/chat_targets.py`
(`load(run)`, `load_latest`, `anchor_pairs`), `evaluation/stage1_threshold.py` →
`output/rerank/threshold-ceiling.json`, `matching/learned.py` (F9, N1, N3, N4, `concepts="R+N"`),
`evaluation/feature_arms.py` → `output/rerank/feature-arms.json`, `output/rerank/chat-rerank-relabel.csv`,
notebook 25. Both checks registered. Guards in the runs: X2 reproduces the stage-1 record; step 3's L2
fold models reproduce step 3's scores; L2 on F9 reproduces 4b's A1 arm; every CV a list hands over is
chat-scored.

### 4a: the ceiling rises, the bar is missed by 0.0003

| Pipeline (stage 2 = step 3's L2) | List per JD (median) | Ceiling (gen / tar) | Recall@10 | Δ vs X2 alone |
|---|---|---|---|---|
| X2 alone | 10 | – | 0.417 | – |
| X2 top 100 → L2 | 100 | 0.964 (0.960 / 0.971) | **0.482** | +0.065 [+0.034, +0.099] |
| t = 0.5 → L2 | 23 | 0.655 (0.721 / 0.523) | 0.438 | +0.021 |
| t = 0.4 → L2 | 32 | 0.748 | 0.458 | +0.041 |
| **t = 0.3 → L2** (registered) | 56 | **0.858** (0.878 / 0.817) | 0.467 | **+0.0497 [+0.017, +0.084]: fail** (bar +0.05) |
| t = 0.3 → L2 on F9 (after 4b, option B) | 56 | 0.858 | 0.465 | +0.048 [+0.015, +0.082] |

- t = 0.3 against X2's top 100 → L2: −0.015 [−0.032, −0.001]. Stage-1 recall of judged Fits: 295 of
  367 at t = 0.3 (240 at 0.5; X2's top 100: 332). t = 0.2: size only (median 131; 2,827 unscored).
- **Drift:** the 678 anchors scored +3.4 [+2.2, +4.6] higher in the fresh chats (mean |Δ| 8.3, rank
  agreement 0.88, labels 79%). Headline uncorrected (owner, question 21). It favours the low-threshold
  lists, so 4a's gains are if anything optimistic.
- **Retest** (`chat-1r`, 5 JDs, exploratory): the chat reproduces 0.80 of its own top 10 (0.63-0.90
  per JD; offset +0.2, rank agreement 0.93, labels 84%): a rough noise ceiling.

### 4b: F9 adopted, no change passes

| Arm | L2 Recall@10 | Δ vs F9 [CI] | Stage-1 D50-CV AUC |
|---|---|---|---|
| A0 F12 | 0.492 | −0.005 (F9 vs F12: +0.005 [−0.005, +0.016]) | 0.868 |
| **A1 F9** (adopted) | **0.497** | – | 0.870 |
| A2 + JD-title role (N1) | 0.488 | −0.008 [−0.035, +0.020] | 0.869 |
| A3 + no-experience flag (N3 F2) | 0.492 | −0.005 [−0.015, +0.004] | 0.870 |
| A4 negation-stripped CVs (N3 F1) | 0.506 | +0.009 [−0.005, +0.023] | 0.867 |
| A5 + zero-shot SBERT (N4) | 0.478 | −0.019 [−0.040, +0.001] | 0.868 |
| A6 R+N concepts (N2) | 0.478 | −0.018 [−0.039, +0.001] | 0.877 |

The final arm is F9. Re-applying 4a's rule to an F9 S1 gives t = 0.3 (dev-select median 49), whose
list holds 134 CVs never chat-scored; the owner chose **option B**: stage 1 stays on F12, L2 on F9
(table above). No second chat round.

### Reading

The best pipeline is still **X2's top 100 → L2** (+0.065 over X2 alone). The learned stage 1 at
t = 0.3 comes within a hair of the bar but not past X2's top 100, and hand-made features are
saturated for stage 2. What is left is a stage-2 model that reads the text (step 5), against a
noise ceiling near 0.8.

### Deviations

- **N2 reinstated** (owner, 18 Oct 2026, before any 4b result): tagging 26,155 documents took ~60 min
  in the background (timed at 80).
- **App bug:** after a fresh chat's last batch the app moved to the next *position*, a saved `chat-1`
  batch, and JDs 2-6 were re-scored (501 replies). Fixed (next *unsaved* batch, red warning on saved
  ones); the replies are relabelled `chat-1r` (owner, option A), so `chat-1` stays as committed.
- 4a's record gained the F9 rows (option B) and the retest block after its first run; the registered
  row and bar are unchanged. `reporting.dot_ci` gained an optional label format (0.0497 printed as
  0.05 would read as a pass).

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

*Correction, 18 Oct 2026:* `paper491` is Skillberg's system paper, not the TalentCLEF overview; the
overview is [arXiv 2606.31692](https://arxiv.org/html/2606.31692v1).

### Step 5 findings, part 2: zero-shot reranking beyond cross-encoders (18 Oct 2026; not planned)

Owner, 18 Oct 2026: fine-tuning is hard for now; look for anything shown to rerank in a RAG
pipeline, zero-shot.

**Closest benchmark: TalentCLEF 2026 Task A** (a JD ranks a pool of ~470 CVs; 10 dev and 40 test JDs
per language; MAP; synthetic CVs with expert labels; no training data). Dev sets are 10 JDs, and
dev-to-test drops reached ~0.25 MAP, so every row is directional.

| Finding | Evidence |
|---|---|
| A zero-shot MS MARCO cross-encoder **hurts** | ui-nlp: `ms-marco-MiniLM-L-12-v2` on the top 50, dev MAP 0.578 against 0.866 for mE5-large alone. The same failure as R1, R2, L3 |
| A fine-tuned cross-encoder hurt in some languages | Skillberg (4th): reranker applied only where dev improved |
| **Instruction-tuned embeddings work zero-shot**, full text | ui-nlp: `multilingual-e5-large-instruct` 0.899 against MiniLM-L12 0.706. classum (1st): Nemotron-8B embedding 0.906, near its best component, Claude Opus listwise (0.913) |
| BM25 is a strong single route | CYUT: BM25 0.852, BGE-M3 dense 0.813 |
| **Fusion beats any single route** | CYUT: RRF of BM25 + HyDE routes 0.885. classum: an equal-weight sum of per-JD min-max scores beat every tuned weighting (0.934). Organizers: multiple sources of evidence fused |
| HyDE | CYUT: a hypothetical CV written from the JD ≈ dense (0.815); ESCO-grounded +0.013. A hypothetical JD per CV was the 2nd-best single route (0.842) |
| LLM-parsed structured views (title, seniority, years, domain, skills) | QTPride (2nd), classum (1st); not ablated |
| Learned sparse: small gain; zero-shot ColBERT: none | Palomino: JobBERT → SPLADE (RRF) +0.006; → ColBERT −0.002 (en) |
| LLM listwise reranking: small, unstable | CYUT: RankZephyr-7B +0.005 (en), −0.040 (es). Devoteam: GPT-5.2 on the top 50, +1.6 points [+0.6, +3.1] (20 dev JDs) |

**The general reranking literature**, by family:

| Family | Zero-shot options (size, context, licence) | For us |
|---|---|---|
| MS MARCO cross-encoders | `ms-marco-MiniLM-L6/L12` (512), monoT5 | weakest bet: topical relevance is not fit (above). Size helps out of domain (Rosa et al. 2022, monoT5-3B), but 3B is out of reach on CPU |
| **Instruction-following LLM rerankers** (pointwise, yes/no logit) | Qwen3-Reranker-0.6B (Apache-2.0, 32k, an instruction slot; loads as a sentence-transformers `CrossEncoder`); mxbai-rerank-base-v2 (0.5B, Apache-2.0, 8k); jina-reranker-v3 (0.6B listwise, 64 CVs per pass, CC BY-NC); zerank-2 (4B, CC BY-NC) | the main candidate: the instruction can ask for fit, and a whole JD + CV fits |
| Long-context encoder rerankers | gte-reranker-modernbert-base (149M, 8k, Apache-2.0); bge-reranker-v2-m3 (568M, 8k) | cheapest with no truncation; no instruction |
| **Instruction-aware embeddings** | Qwen3-Embedding-0.6B (32k, instruction); multilingual-e5-large-instruct (512); gte-modernbert-base (8k); bge-m3 (8k) | TalentCLEF's zero-shot winner; each document encoded once; also a stage-1 signal |
| Late interaction (token MaxSim) | GTE-ModernColBERT-v1; answerai-colbert-small-v1; jina-colbert-v2 | requirement-to-span matching suits the task, but no gain in Palomino; needs PyLate |
| Learned sparse | SPLADE-v3 | lexical with expansion; small gain at best (TF-IDF is in X2) |
| Expansion | HyDE (Gao et al. 2022); ConFit v2's hypothetical résumé (HyRe, but trained) | forward HyDE = 60 generations, needs an LLM (question 26) |
| LLM rerankers: pointwise with graded labels (Zhuang et al. 2024), pairwise (PRP), setwise, listwise (RankGPT, RankZephyr, FIRST), reasoning (Rank1) | 7B and up | strongest in the literature; days on CPU. The reference is itself an LLM listwise score, so this would partly measure agreement with the judge |
| Fusion and interpolation | RRF (Cormack et al. 2009); BM25 + dense interpolation (Wang, Zhuang & Zuccon 2021); PRF interpolation (Li et al. 2022) | the cheapest, best-supported use: combine a zero-shot text score with L2, not replace it |

**Already measured here.** Zero-shot MiniLM SBERT as an L2 feature (N4): −0.019. R3 (bi-encoder,
D50): −0.010. Every cross-encoder: −0.19 to −0.23. In the 500-CV pools, zero-shot MiniLM recall@100 is
0.633, TF-IDF 0.777, X2 0.902 (`output/matching/stage1-metrics.json`, 59 JDs). `all-MiniLM-L6-v2` reads
256 word pieces (chunk-mean here); the models above read 8k-32k.

**A possible 5a (proposal, not planned).** X2's top 100, scored against `chat-1`, nothing trained:
- **Arms:** (1) Qwen3-Reranker-0.6B with a fit instruction; (2) an instruction-aware embedder
  (Qwen3-Embedding-0.6B or multilingual-e5-large-instruct), cosine; (3) gte-reranker-modernbert-base,
  the cheap uninstructed control.
- **Uses:** alone; fused with L2 by the equal-weight per-JD min-max sum (classum) or RRF (k = 60),
  nothing tuned; as an extra L2 feature (L2 retrains in minutes; the text model stays zero-shot).
- S3's metrics and bar; the human check beside every arm (the guard against matching the judge).
- **Cost, to time on one JD first.** Scaling L3's measured speed by FLOPs: roughly 2-5 h for a
  150M reranker and 6-20 h for a 0.6B one over all 6,000 pairs. An embedder costs about half (each of
  ~5k CVs once). Reranking only L2's top 30 cuts a cross-encoder's cost by 3x; that cut is decided
  before scoring.

**Vector database (owner's question, 18 Oct 2026).** Not for the experiments. A vector database
stores embeddings and returns an approximate top-k fast over millions of vectors. Here every pool is
500 CVs, and exact cosine over 500 vectors is one matrix product: the index would add approximation
and change no score. The cost is encoding, and the embeddings are already cached in files. Quality
comes from the embedding model and from what is embedded. Stage 1 is where dense retrieval would
sit, but dense alone lost to X2 in the pools (above): a stronger embedder enters as an F12 column or a
fused route, not through a database. A database matters only for a deployed app searching a whole
CV corpus for a new JD (FAISS, pgvector, Qdrant): a scaling note for the report.

**Open questions (step 5a, for the owner)**

23. Which arms and uses, and the rule that picks the winner, fixed before scoring. Arms × uses
    multiply comparisons, as in 4b. *Owner, 18 Oct 2026: no answer yet. Context given: after stage 2
    a language model converses with the recruiter about which CV fits the JD and why (the CNI
    Workshop B pattern: retrieved evidence, typed tool calls, app-owned state). Open to suggestions.*
24. Models download from Hugging Face; the code loads local files only. Are the CC BY-NC models
    (jina-reranker-v3, zerank-2) acceptable for a university project? *Owner: yes to both; no
    restrictions so far.*
25. Is fusing a zero-shot score with L2, or adding it as an L2 feature, still "zero-shot"? (L2 is
    trained on `chat-1`.) *Owner: zero-shot is not a requirement; fine-tuning is allowed but CPU-
    and time-limited. Owner is sceptical of fine-tuning a small model and prefers pipeline building,
    or a larger model (Qwen 4B / 8B) without fine-tuning. No checks run yet.*
26. LLM-generated views (forward HyDE, structured JD parsing): a small local LLM (60 generations is
    feasible on CPU), or the owner in claude.ai like `chat-1` (a D42/D61 decision)? A hypothetical JD
    per CV (~5k generations) is out on CPU. *Owner: the `chat-1` method if it proves useful; low
    priority.*

Sources (part 2): [TalentCLEF 2026 overview](https://arxiv.org/html/2606.31692v1);
[classum](https://ceur-ws.org/Vol-4283/paper481.pdf); [VerbaNexAI](https://ceur-ws.org/Vol-4283/paper483.pdf);
[Palomino](https://ceur-ws.org/Vol-4283/paper484.pdf); [CYUT](https://ceur-ws.org/Vol-4283/paper485.pdf);
[ui-nlp](https://ceur-ws.org/Vol-4283/paper486.pdf); [Devoteam](https://ceur-ws.org/Vol-4283/paper487.pdf);
[QTPride](https://ceur-ws.org/Vol-4283/paper490.pdf); [Skillberg](https://ceur-ws.org/Vol-4283/paper491.pdf);
[ConFit v2](https://arxiv.org/abs/2502.12361);
[In Defense of Cross-Encoders for Zero-Shot Retrieval](https://arxiv.org/abs/2212.06121);
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B);
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B);
[jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3);
[mxbai-rerank-v2](https://www.mixedbread.com/blog/mxbai-rerank-v2);
[gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base);
[zerank-2](https://huggingface.co/zeroentropy/zerank-2);
[GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1);
[PyLate](https://arxiv.org/abs/2508.03555);
[PRP](https://arxiv.org/abs/2306.17563); [Setwise reproducibility](https://arxiv.org/abs/2504.10509);
[Beyond Yes and No](https://aclanthology.org/2024.naacl-short.31);
[UPR](https://arxiv.org/abs/2204.07496); [Rank1](https://arxiv.org/abs/2502.18418);
[Dense retrievers require interpolation with BM25](https://ir.webis.de/anthology/2021.ictir_conference-2021.36);
[To Interpolate or not to Interpolate](https://arxiv.org/abs/2205.00235).

## Step 5a (proposal, 18 Oct 2026; awaiting the owner's approval)

Owner, 18 Oct 2026: **stage 2 only**; test an instruction-tuned embedder **locally** first; Colab
only if the result is positive. Fine-tuning a small model is dropped (the owner is sceptical, and
gte-reranker fine-tuning measured 13.7 h per epoch, below). Stage 3 (the recruiter conversation)
is not started.

**No chunking is needed.** Tokens over all 6,020 `chat-1` pairs (Qwen tokenizer): JD median 311
(max 895), CV median 369 (p99 1,438, max 2,544), pair median 753 (p99 1,961, max 3,105). The models
below read 8k-32k tokens, so every JD and CV is read whole. The text R1/R2 and L3 lost came from
MiniLM's 384/512-token inputs. Requirement-to-passage matching remains possible as a later arm.

**Measured cost** (scratch timing, 18 Oct 2026, this CPU; `logs/step5-timing*.log`, not records):

| Model | Kind | Cost over the 6,020 pairs | Note |
|---|---|---|---|
| **Qwen3-Embedding-0.6B**, float32 | instruction-tuned embedder | **3.9 h** (4,781 CVs encoded once; 156 tok/s, batch 1, 6 threads, 4.4 GB) | the checkpoint's bfloat16 default is 3x slower on this AVX2 CPU (50 tok/s) |
| gte-reranker-modernbert-base | cross-encoder, no instruction | 3.7 h inference; 13.7 h per fine-tuning epoch | timed under memory pressure: an upper bound |
| Qwen3-Reranker-0.6B | instruction-tuned cross-encoder | 37 h (bfloat16, with swapping) | float32 not timed; out unless re-timed |

Batch 4 was slower than batch 1 (padding), and 11 threads no faster than 6.

**Design (proposal)**
- **E1:** Qwen3-Embedding-0.6B, float32, whole documents, max length 4,096. The instruction goes
  on the JD side only (the model card's convention), fixed before scoring: `Instruct: Given a job
  description, retrieve the CVs of candidates who fit its requirements` / `Query:` + JD text. CVs
  get no instruction. Score = cosine. Embeddings are cached with a fingerprint (model revision,
  dtype, threads, instruction, text variant v2). Nothing is trained.
- **Scope:** the 6,020 `chat-1` pairs (X2's top 100 + the S1 extension), as in S3.
- **Uses**, all on `chat-1`, out of fold wherever L2 is involved:
  - **U1 alone:** E1's order.
  - **U2 fused:** the equal-weight sum of per-JD min-max-scaled E1 and L2 (F9, out of fold).
    Nothing tuned (the classum rule).
  - **U3 feature:** F9 + E1's cosine as one more L2 column, retrained per fold (4b's harness).
- **Baselines:** X2's order; L2 on F9 (4b's A1, Recall@10 0.497).
- **Metrics:** S3's: Recall@10 against the chat's top 10, nDCG@10, chat No Fits in the top 10, and
  the human check on the 422 judged CVs. Generic / targeted / total, JD-resampled CIs.
- **Bars (proposal):** U1 against S3's bar (Δ vs X2's order ≥ +0.05, CI above 0). U2 and U3
  against 4b's adoption rule (Δ vs L2 F9 > 0, CI above 0). Three comparisons: one nominal pass
  is weak evidence, as in 4b.
- **Run:** one background job, encoding once into the cache: **~3.4 h**, one process, 4 GB. The
  uses then cost minutes. *Measured 18 Oct 2026 (`logs/step5-parallel.log`): splitting does not
  help.* Combined throughput was 178 tok/s for 1 process × 6 threads, 183 for 1 × 11, 172 for
  2 × 6, 174 for 2 × 5 and 182 for 3 × 4. The CPU is saturated by one process. The earlier
  "two processes, ~2 h" was an assumption, now withdrawn.
- **The instruction changes only the JD side.** The 4,781 CV vectors are encoded once; one more
  instruction costs 60 JD encodings (seconds). Several wordings are cheap to compute, but choosing
  among them on `chat-1` is selection on the reference (question 28).
- **Code:** `rerank/embedder.py` (encode, cache; never reads a human label) and
  `evaluation/rerank_embed.py` → `output/rerank/embed-eval.json`, registered in `checks` (warm
  replays the cache; cold re-encodes, ~2-4 h). Tests (inline synthetic): instruction format, cache
  fingerprint, per-JD min-max fusion, the registry. `HF_HUB_OFFLINE` stays on in the package: the
  model is downloaded once, outside it.
- **If positive:** the owner chooses among Qwen3-Embedding-4B/8B (Colab), gte-reranker locally
  (3.7 h), or requirement-level matching.

**Open questions (step 5a)**

27. Approve E1 with U1-U3 and the bars above? *Owner, 18 Oct 2026: OK.*
28. The instruction wording above, fixed before scoring, or another? *Owner: needs more
    information on what goes into the model.* Given: the JD is `Long Description` only (text v2,
    D53; the JD title is allowed by D54 but not in v2); the CV is `CV` + `Highlights` + `Moreinfo`
    + `Looking For`; the instruction is one sentence on the JD side. Options: (a) generic; (b) naming
    role, skills and years; (c) either with the JD title. Report a fixed set, or pick on one fold and
    score on four.
29. How many processes? *Measured: one; splitting does not raise throughput (above). ~3.4 h.*
    Ways to shorten: a pilot on 2 folds (~40%), int8 quantisation (ONNX/OpenVINO, untested), or a
    smaller long-context embedder (nomic-embed-text-v1.5, gte-modernbert-base; untimed).
30. gte-reranker as a second arm in the same run (+3.7 h), or only if E1 is positive? *Owner:
    pending the explanation of model types (embedder vs cross-encoder vs listwise), given 18 Oct.*
31. *(Owner, 18 Oct 2026)* Why not put all 100 CVs in one context (listwise)? Given: ~45k tokens
    per JD; at least the cross-encoder's cost plus attention that grows with the square of the
    length; listwise rankers use windows of ~20 (RankGPT), `chat-1` used batches of 20,
    jina-reranker-v3 takes at most 64 per pass. A later arm, timed on one JD first.

### Step 5a pilot (approved by the owner 18 Oct 2026; design fixed before any scoring)

Owner, 18 Oct 2026: the embedder first, Qwen3-Embedding-0.6B, **on a sample with ~10 min of
encoding**. The instruction wording is left to the agent, and several are tested (question 28).
Cross-encoders wait until after the embedder (question 30). Run it as parallel as the machine
allows: one process (measured above).

- **Sample:** fold 0's 12 JDs (8 generic, 4 targeted), X2's top 100 only. Per JD: every CV in the
  chat's tie-aware top 10 (`chat_weights > 0`), plus 8 other CVs drawn at random (seed 0) from the
  rest of its 100. That is about 240 CVs, ~10 min at the measured 180 tok/s. A system scores each
  pair alone, so the sample cannot change any score.
- **Instruction variants** (JD side only; CVs get none; the model card's `Instruct: …\nQuery:` form):
  - **I0:** none (the JD text alone);
  - **I1:** "Given a job description, retrieve the CVs of candidates who fit its requirements";
  - **I2:** "Given a job description, retrieve the CVs of candidates whose role, skills and years of
    experience match its requirements";
  - **I3:** I2 with the JD title first in the query (`Job title: …`, D54).
- **Model:** `Qwen/Qwen3-Embedding-0.6B` at revision `97b0c614`, float32, max length 4,096,
  batch 1, 6 threads; normalised vectors, score = cosine. CV vectors are cached by CV id and text
  hash, so a full run reuses them.
- **Pilot metric** (a full list is not scored, so Recall@10 is not defined): per-JD **AUC of the
  score for the chat's top 10 against the sampled others**, the quantity Recall@10 rewards. Mean
  over the 12 JDs, JD-resampled 95% CI, generic / targeted / total. Secondary: within-JD Spearman
  with the chat score on the sampled CVs.
- **Compared on the same pairs:** X2; L2 on F9 out of fold (4b's A1; its scores are guarded against
  4b's digest); **U2 fused** = the per-JD min-max sum of E and L2 over the sampled CVs (the full run
  scales over all 100). U3 (the L2 feature) needs every training pair, so it waits for a full run.
- **Status: exploratory.** Every variant is reported. **Rule for a full run, fixed now:** its
  instruction is the variant with the highest pilot AUC (total). The full run then reports folds
  1-4 as its held-out estimate (the instruction was chosen on fold 0) beside all five. Whether to
  run it is the owner's call. Proposal: go if that variant's AUC Δ vs X2 > 0, or U2's Δ vs L2 > 0
  (point estimates; 12 JDs make wide CIs).
- **Code:** `rerank/embedder.py` (encode, cache, instructions; never reads a human label) and
  `evaluation/rerank_embed.py --pilot` → `output/rerank/embed-pilot.json`. No `--check` until a
  full run. Tests (inline synthetic): query format, sample composition, per-JD fusion and AUC.

### Step 5a pilot record (18 Oct 2026; exploratory)

`output/rerank/embed-pilot.json`, `logs/rerank_embed_pilot.log`. 12 JDs (8 generic, 4 targeted),
239 pairs, 235 CVs, 143 pairs in the chat's top 10. Encoding took 671 s (181 tok/s, as timed);
16 min end to end; peak RSS 5.2 GB. The guards held: X2 matches the stage-1 record, and L2 on F9
reproduces 4b's A1.

| System | AUC, chat top 10 vs others (gen / tar) | Δ vs X2 [CI] | Δ vs L2 F9 [CI] |
|---|---|---|---|
| X2 | 0.802 (0.834 / 0.737) | – | −0.092 [−0.156, −0.035] |
| L2 F9 | **0.894** (0.905 / 0.870) | +0.092 [+0.035, +0.156] | – |
| E1 I0 (no instruction) | 0.812 (0.816 / 0.804) | +0.010 [−0.057, +0.081] | −0.081 [−0.117, −0.048] |
| E1 I1 (fit) | 0.779 | −0.023 | −0.115 |
| E1 I2 (role, skills, years) | 0.764 | −0.038 | −0.129 |
| E1 I3 (I2 + JD title) | 0.745 | −0.057 | −0.149 |
| U2 I0 (E1 + L2, min-max sum) | 0.892 (0.908 / 0.861) | +0.090 | **−0.002 [−0.023, +0.021]** |
| U2 I1 / I2 / I3 | 0.883 / 0.880 / 0.870 | – | −0.011 / −0.013 / −0.023 |

- **Alone, the zero-shot embedder is level with X2** and well below L2. On this metric L2 beats X2
  by +0.092, in line with S3's Recall@10 gain (+0.072). The pilot metric tracks the full one.
- **Every instruction made it worse**, in order: none > fit > criteria > criteria + title. A
  reading, not tested: Qwen's instructions are built for asymmetric query-to-passage search, while a
  JD and a CV are both long documents, so plain document-to-document similarity suits them better.
- **Fused with L2, it adds nothing** (−0.002). U3 (an L2 feature) is untested. 4b's nearest arm,
  zero-shot MiniLM as an L2 feature (N4), lost 0.019.
- **The pre-set rule chooses I0, and the proposal is technically met** (E1 I0 Δ vs X2 = +0.010 > 0),
  but by a margin far inside its CI. A full run (the remaining 4,546 CVs, ~3.2 h) would test only U3.
  The owner decides.

### Step 5a cross-encoder pilot (owner, 18 Oct 2026: "run the cross-encoders on the same sample")

Design fixed before scoring: the same 239 pairs, metric, CIs and comparisons as the embedder pilot.

- **C1:** `Alibaba-NLP/gte-reranker-modernbert-base` at revision `f7481e60` (no instruction).
- **C2:** `Qwen/Qwen3-Reranker-0.6B` at revision `e61197ed`, with I1's wording as its instruction
  ("Given a job description, retrieve the CVs of candidates who fit its requirements").
- **C3:** the same model with its own default instruction (web search). It tests whether the
  instruction helps, as I0 did for the embedder.
- **Settings for all three:** float32 (the Qwen checkpoint's bfloat16 default was the slow
  timing), max length 8,192 (the longest pair is 3,390 tokens, so nothing is cut), batch 1,
  6 threads. Score = the raw logit (Qwen: yes minus no). Scores are cached per pair.
- **Compared with:** X2, L2 F9 and E1 I0, each on the same pairs. U2 for each Cx = the per-JD
  min-max sum with L2.
- **Order:** C1, C2, then C3. The record is rewritten after each, so a long run still leaves
  results.
- **Cost:** ~203k tokens of pairs. C1 ~9 min; C2 and C3 ~19 min each at the embedder's measured
  181 tok/s (same 0.6B backbone); ~50 min in all. The earlier "~20 min for both Qwen variants"
  was low by half.
- `rerank/cross.py`, `evaluation/rerank_embed.py --pilot-cross` → `output/rerank/cross-pilot.json`.
  Exploratory.

### Step 5a cross-encoder pilot record (18 Oct 2026; exploratory)

`output/rerank/cross-pilot.json`, `logs/rerank_cross_pilot.log`. The same 239 pairs. 61 min end to
end; peak RSS 6.8 GB. Measured: C1 425 s (1.8 s/pair), C2 1,524 s and C3 1,482 s (6.3 s/pair), so
the ~19 min estimate for each Qwen variant was low by a third.

| System | AUC (gen / tar) | Δ vs X2 [CI] | Δ vs L2 F9 [CI] |
|---|---|---|---|
| X2 | 0.802 | – | −0.092 |
| L2 F9 | **0.894** | +0.092 [+0.035, +0.156] | – |
| E1 I0 (embedder) | 0.812 | +0.010 | −0.081 |
| C1 gte-reranker | 0.727 (0.711 / 0.758) | −0.075 [−0.203, +0.066] | −0.167 [−0.252, −0.076] |
| **C2 Qwen3-Reranker, fit instruction** | **0.846** (0.843 / 0.853) | +0.044 [−0.018, +0.110] | −0.047 [−0.095, −0.001] |
| C3 Qwen3-Reranker, default instruction | 0.823 (0.823 / 0.824) | +0.021 | −0.070 |
| U2 C1 + L2 | 0.860 | +0.058 | −0.033 [−0.066, +0.003] |
| **U2 C2 + L2** | **0.902** (0.907 / 0.890) | +0.100 [+0.037, +0.165] | **+0.008 [−0.024, +0.044]** |
| U2 C3 + L2 | 0.892 | +0.090 | −0.001 |

- **C1, a reranker trained on web search, is below X2**: the same failure as R1, R2 and L3.
- **C2 is the best text model so far.** Alone it beats X2 (+0.044) and the embedder (+0.034),
  but stays below L2 (−0.047). It is the only arm whose fusion with L2 has a positive point
  estimate (+0.008), with a CI across 0 on 12 JDs.
- **The fit instruction helps the reranker** (C2 0.846 against C3 0.823), unlike the embedder,
  where every instruction hurt. A reading, not tested: in the reranker the instruction states the
  yes/no question it answers; in the embedder it only shifts the query vector.
- **Cost of a full C2 run**: the remaining 5,781 pairs at 6.3 s each come to ~10 h on this CPU.

### Cascade cost: the chat's top 10 inside L2's top k (owner, 18 Oct 2026; exploratory)

`evaluation/rerank_embed.py --coverage` → `output/rerank/topk-coverage.json`. All 60 JDs, X2's
top 100. Coverage = the share of the chat's tie-aware top 10 inside a system's top k, i.e. Recall@10
with a perfect reranker on that list. No cross-encoder score is used. L2's k = 10 row reproduces
4b's A1 Recall@10 (0.497).

| k | L2 F9 (gen / tar) | X2 | C2 pairs to score (60 × k) | ~Time at 6.3 s/pair |
|---|---|---|---|---|
| 10 | 0.497 (0.503 / 0.485) | 0.419 | – | – |
| 20 | 0.711 (0.725 / 0.682) | 0.651 | 1,200 | 2.1 h |
| **30** | **0.819** (0.832 / 0.794) | 0.762 | 1,800 | 3.2 h |
| 40 | 0.874 (0.879 / 0.866) | 0.805 | 2,400 | 4.2 h |
| 50 | 0.910 (0.917 / 0.896) | 0.857 | 3,000 | 5.3 h |
| 100 | 1.000 | 1.000 | 6,000 | 10.5 h |

Times are before subtracting the pilot's cached pairs that fall inside each top k (at most 239).

### Step 5b: cascade L2 top 30 → Qwen3-Reranker (owner, 18 Oct 2026: k = 30; fixed before scoring)

- **Stage 2a:** L2 on F9, out of fold (4b's A1, digest-guarded), over X2's top 100 of each JD. Its top
  30 per JD (ties by pair id) go on.
- **Stage 2b:** C2 (Qwen3-Reranker-0.6B, the fit instruction, the pilot's exact config and cache)
  scores those 1,800 pairs. Nothing is trained.
- **Arms:**
  - **"L2 top 30 → C2 + L2"** (the decision arm): within the top 30, the sum of per-JD min-max L2 and
    min-max C2;
  - "L2 top 30 → C2": within the top 30, C2's order alone.
  - In both, ranks 31-100 stay in L2's order below the 30.
- **Metrics:** S3's Recall@10 against the chat's top 10, nDCG@10, chat No Fits; generic / targeted /
  total, JD-resampled CIs; paired against L2 F9 and against X2. The human check (judged CVs in X2's
  top 100: per-JD AUC on the final label, pairwise on yc's grades, `full` and `llm_free`) is
  exploratory.
- **Decision, on folds 1-4** (48 JDs; fold 0's C2 scores were seen in the pilot): 4b's adoption rule,
  i.e. the fused arm is adopted if its Recall@10 Δ vs L2 F9 (total) is > 0 with the CI above 0. All
  60 JDs are reported beside it.
- **Cost:** ≤ 1,800 pairs at 6.3 s, minus the pilot pairs already cached: ~3 h, one background
  process. Scores are cached per pair, so a crash after scoring costs only the ~5 min of features.
- `evaluation/rerank_embed.py --cascade` → `output/rerank/cascade-c2.json`. Exploratory: no
  `--check` until the owner decides to keep it.

### Step 5b record (18 Oct 2026): not adopted

`output/rerank/cascade-c2.json`, `logs/rerank_cascade.log`. 1,800 pairs rescored, 150 of them cached
from the pilot. The other 1,650 took 8,305 s (5.0 s/pair, faster than the pilot's 6.3); 2 h 22 min
end to end; peak RSS 5.2 GB. The L2 digest guard held.

| Pipeline | R@10, folds 1-4 (gen / tar) | Δ vs L2 F9 [CI] | nDCG@10 Δ vs L2 | Chat No Fits | R@10, all 60 (Δ vs L2) |
|---|---|---|---|---|---|
| X2's order | 0.410 | −0.079 [−0.114, −0.045] | −0.042 | 1.60 | 0.419 |
| L2 F9 | 0.489 (0.498 / 0.471) | – | – | 1.38 | 0.497 |
| **L2 top 30 → C2 + L2** | **0.502** (0.503 / 0.500) | **+0.013 [−0.011, +0.038]** | +0.019 [+0.001, +0.039] | 1.27 | 0.522 (+0.025 [+0.003, +0.046]) |
| L2 top 30 → C2 alone | 0.422 | −0.067 [−0.105, −0.027] | −0.034 | 1.48 | 0.439 |

- **Decision (folds 1-4, 48 JDs): not adopted.** +0.013 Recall@10 over L2, CI across 0. The
  all-60 row (+0.025, CI above 0) includes fold 0, whose C2 scores were seen in the pilot, so it
  does not decide.
- The fused arm improves the order a little: nDCG@10 +0.019, CI just above 0, and 0.1 fewer chat No
  Fits. Against X2 it clears S3's bar (+0.092 [+0.055, +0.132]), but L2 alone already does (+0.079).
- **C2 alone loses to L2 inside the same 30 (−0.067).** The reranker helps only as a second opinion
  beside L2's features.
- **Human check (exploratory, 422 judged CVs): no change.** Fused vs L2: per-JD AUC −0.004
  [−0.040, +0.028], pairwise on yc +0.007 [−0.024, +0.042]; `llm_free` is similar.
- **Reading:** a zero-shot 0.6B instruction reranker adds ~+0.01 to +0.02 Recall@10 at ~2.3 h of
  CPU per 60 JDs (top 30). That is within noise against the pre-set bar. The text gain over
  hand-made features stays small, as in steps 3 and 4b.

### Step 5c: S1 → zero-shot reranker, nothing trained on `chat-1` (owner, 18 Oct 2026; fixed before scoring)

Owner, 18 Oct 2026: L2 is suspect because it is trained on the reference it is scored against. Stage
2 should be zero-shot: less likely to overfit, and swappable by speed. Stage 1 should be S1: trained
and tested on labels, modular, with explainable coefficients, where X2's formula is hand-set. The
agent's take, recorded beside it: the human check (+0.116 AUC over X2) says L2 learned more than the
chat's habits. S1 is also trained on LLM labels, but from D50's other JDs, never the evaluation
judge's scores. So this pipeline uses `chat-1` only to score.

- **Stage 1:** S1 (logistic on F12, fitted on D50 as in steps 1 and 4a), the t = 0.3 list (4a's
  registered threshold), capped at its top 30 by probability (ties by pair id). 1,657 pairs, median
  30 per JD [1, 30]. Every one has a chat score. Counted 18 Oct 2026: 1,296 already have C2 scores,
  361 are new (~30 min).
- **Stage 2:** C2 (Qwen3-Reranker-0.6B, the fit instruction, the pilot's config and cache).
- **Arms on that list:** "S1 alone" (S1's probability order); "S1 → C2" (C2's order); **"S1 → C2 +
  S1"** (the decision arm: the sum of per-JD min-max S1 probability and min-max C2 within the list).
- **Beside them:** X2's top 100 → L2 F9 (what training on `chat-1` buys) and X2 alone.
- **Reference:** `ct.load_latest()` (`chat-1` + `chat-2`, the latest score per pair), as in 4a.
  `chat-2`'s +3.4 offset (4a's anchors) is reported beside; the headline is uncorrected (question
  21). Recall@10, nDCG@10, chat No Fits and the ceiling, tie-aware over every chat-scored CV of the
  JD. A list shorter than 10 hands over what it has.
- **Decision, on folds 1-4** (C2 scores from fold 0 were seen in the pilot): the zero-shot stage 2 is
  adopted over S1 alone if "S1 → C2 + S1" has a Recall@10 Δ vs "S1 alone" (total) > 0 with the CI
  above 0. All 60 JDs are reported beside.
- **Human check (exploratory):** judged CVs inside the S1 list. Per-JD AUC and pairwise on yc, the
  three S1 arms only, against S1 alone.
- **Guards:** X2's pool scores match the stage-1 record; the uncapped t = 0.3 list sizes match 4a's
  record; L2 F9 reproduces 4b's A1; every listed CV has a chat score.
- `evaluation/rerank_embed.py --s1-zero-shot` → `output/rerank/s1-zero-shot.json`. Exploratory.

### Step 5c record (18 Oct 2026): not adopted

`output/rerank/s1-zero-shot.json`, `logs/rerank_s1_zero_shot.log`. Lists: 1,657 pairs, median 30 per JD
[1, 30]; 253 of the listed pairs take their score from `chat-2`. Scoring the 361 new C2 pairs took
1,716 s; the run took 34 min end to end, peak RSS 6.1 GB. Every guard held.

| Pipeline (folds 1-4, 48 JDs) | Trained on `chat-1`? | Ceiling | R@10 (gen / tar) | Δ vs S1 alone [CI] | Δ vs X2 top 100 → L2 [CI] | Chat No Fits |
|---|---|---|---|---|---|---|
| X2 alone | no | – | 0.405 | −0.031 [−0.066, +0.004] | −0.068 | 1.65 |
| S1 alone (t = 0.3, top 30) | no | 0.724 | 0.436 (0.447 / 0.415) | – | −0.037 [−0.065, −0.008] | 1.21 |
| S1 → C2 | no | 0.724 | 0.384 | −0.052 [−0.096, −0.008] | −0.089 | 1.48 |
| **S1 → C2 + S1** | **no** | 0.724 | **0.446** (0.452 / 0.434) | **+0.009 [−0.016, +0.035]** | −0.027 [−0.056, +0.001] | 1.12 |
| X2 top 100 → L2 F9 | yes | 0.959 | 0.473 (0.477 / 0.466) | +0.037 [+0.008, +0.065] | – | 1.40 |

All 60 JDs: S1 alone 0.443, S1 → C2 + S1 0.454 (+0.011 [−0.012, +0.033]), X2 top 100 → L2 0.483, X2
alone 0.417. These figures use step 4a's reference (`chat-1` + `chat-2`), so they differ slightly from
step 5b's `chat-1`-only figures.

- **Decision: not adopted.** The zero-shot stage 2 adds +0.009 over S1's own order, with the CI
  across 0. Alone, C2 is worse than S1's order (−0.052).
- **The fully chat-free pipeline (S1 → C2 + S1) is 0.027 below X2 top 100 → L2** (CI just reaching
  0) and beats X2 alone by +0.040 [+0.001, +0.081]. About 0.03 Recall@10 is what training on `chat-1`
  buys.
- **The ceiling is where the loss is.** S1's capped list holds 0.72 of the chat's top 10; X2's top
  100 holds 0.96. Within its list, S1's own order reaches 0.436 / 0.724 = 60% of its ceiling; L2
  reaches 49% of a much larger one.
- Chat No Fits are lowest for the S1 lists (1.12-1.21 against L2's 1.40). Part of that is shorter
  lists: some JDs pass fewer than 10.
- **Human check (exploratory; 269 judged CVs; 27 JDs for AUC, 44 for pairwise): no detectable
  change.** S1 → C2 + S1 vs S1 alone: per-JD AUC +0.023 [−0.077, +0.110], pairwise on yc +0.038
  [−0.033, +0.116].

### Step 5d: C2 on all of X2's top 100 (owner, 20 Oct 2026: "start the full job for C2"; fixed before scoring)

5b let L2 pick the 30 that C2 saw (ceiling 0.819), and 5c let S1 pick them (0.724). Here C2 scores
every pair `chat-1` scored, so the ceiling is 1.0 and C2's order is tested on the whole list.

- **Pairs:** all of `chat-1`, 60 JDs × 100 = 6,000. Counted 20 Oct 2026: 2,181 already have C2
  scores (pilot, 5b, 5c), 3,819 are new.
- **Stage 2:** C2 (Qwen3-Reranker-0.6B, the fit instruction, the pilot's config and cache).
- **Arms, each over the full 100 of a JD:**
  - **"X2 top 100 → C2"** (replace: C2's order alone);
  - **"X2 top 100 → C2 + L2"** (enhance: the sum of per-JD min-max L2 F9 and min-max C2 over the
    100);
  - exploratory, nothing trained on `chat-1`: "X2 top 100 → S1" (S1's probability order) and "X2
    top 100 → C2 + S1" (min-max sum). They test whether 5c lost on S1's cap or on the order.
  - Beside them: X2's order and L2 F9.
- **Reference and metrics:** `chat-1` (as 5b). S3's Recall@10 against the chat's tie-aware top 10,
  nDCG@10, chat No Fits; generic / targeted / total, JD-resampled CIs; paired deltas vs L2 F9, X2,
  and S1's order. Human check exploratory, as 5b.
- **Decisions, on folds 1-4 (48 JDs), both fixed now:**
  - replace: "X2 top 100 → C2" is adopted over L2 if its Recall@10 Δ vs L2 F9 (total) is > 0 with
    the CI above 0;
  - enhance: the same rule for "X2 top 100 → C2 + L2".
  - The S1 arms carry no rule. Which decision leads the write-up is the owner's open question
    (review notes comment, 20 Oct 2026); both rules are fixed before scoring, so that choice cannot
    change an outcome.
- **What is not fresh:** folds 1-4 had C2 scores on L2's top 30 in 5b. The new information is C2 on
  the other 70 CVs of each JD. Expected from 5b, recorded before scoring: C2 alone loses to L2 (−0.067
  inside the same 30).
- **Guards (before scoring):** X2's pool scores match the stage-1 record; L2 F9 reproduces 4b's A1;
  S1's uncapped t = 0.3 list sizes match 4a's record; no JD has more than 100 chat pairs.
- **Cost:** 3,819 pairs at ~5.0 s ≈ 5.3 h, one background process, ~6 GB peak. `cross.score_pairs`
  now writes every 100 pairs and logs progress, so an interruption keeps what was scored (batch 1,
  so scores are unchanged).
- `evaluation/rerank_embed.py --full-c2` → `output/rerank/full-c2.json`. Exploratory: no `--check`.

### Step 5d record (20 Oct 2026): neither rule adopted

`output/rerank/full-c2.json`, `logs/rerank_full_c2.log`. 3,819 new pairs scored. All 6,000 of X2's
top 100 now have C2 scores. The run took 11 h 51 min of wall clock, but the machine slept for about
6 h 24 min. Active time was about 5 h 27 min, with stage 2 at about 5.1 s per pair. Peak RSS was
6.0 GB, every guard held, and the run exited 0.

| Pipeline (folds 1-4, 48 JDs) | Trained on `chat-1`? | R@10 (gen / tar) | Δ vs L2 F9 [CI] | Δ vs S1's order [CI] | nDCG@10 | Chat No Fits |
|---|---|---|---|---|---|---|
| X2's order | no | 0.410 | −0.079 [−0.114, −0.045] | −0.039 [−0.071, −0.006] | 0.776 | 1.60 |
| L2 F9 | yes | 0.489 (0.498 / 0.471) | – | +0.040 [+0.013, +0.067] | 0.819 | 1.38 |
| **X2 top 100 → C2** (replace) | no | 0.371 (0.376 / 0.362) | **−0.118 [−0.156, −0.077]** | −0.078 [−0.126, −0.030] | 0.737 | 1.77 |
| **X2 top 100 → C2 + L2** (enhance) | yes (L2 part) | 0.485 (0.488 / 0.479) | **−0.004 [−0.029, +0.023]** | +0.036 [+0.003, +0.071] | 0.828 | 1.29 |
| X2 top 100 → S1 (no rule) | no | 0.449 (0.463 / 0.421) | −0.040 [−0.067, −0.013] | – | 0.805 | 1.46 |
| X2 top 100 → C2 + S1 (no rule) | no | 0.452 (0.465 / 0.424) | −0.038 [−0.068, −0.007] | +0.002 [−0.033, +0.039] | 0.808 | 1.31 |

On all 60 JDs: C2 alone 0.379, C2 + L2 0.501 (+0.004 [−0.018, +0.026] vs L2), S1's order 0.454 and
C2 + S1 0.464 (+0.010 [−0.019, +0.040] vs S1's order).

- **Replace: not adopted.** Over the whole list, C2 alone is 0.118 below L2 and below X2's own order
  (−0.039 [−0.087, +0.008]). This was the outcome expected before scoring.
- **Enhance: not adopted.** C2 + L2 over all 100 CVs is level with L2 (−0.004). It is also below
  5b's top-30 cascade (0.502 on the same reference). Restricting C2 to L2's top 30 kept its errors in
  the tail from lifting weak CVs. That comparison is exploratory, because 5b's figure had been seen.
  nDCG@10 is +0.010 [−0.010, +0.033] and chat No Fits 0.09 fewer, both within noise.
- **The S1 arms correct 5c's reading.** Given the whole of X2's top 100 (ceiling 1.0), S1's own
  order still trails L2 by 0.040. That is the same gap 5c found with S1's capped list (0.037). So
  most of the gap comes from S1's order against L2's, not from S1's cap: the ~0.04 Recall@10 is what
  training stage 2 on `chat-1` buys. C2 adds nothing measurable to S1's order (+0.002).
- **Human check (exploratory; 422 judged CVs; 48 JDs for AUC on the final label, 55 for pairwise on
  yc).**
  - C2 alone vs L2: AUC −0.063 [−0.155, +0.031], pairwise −0.066 [−0.133, +0.006].
  - C2 + L2 vs L2: AUC −0.002 [−0.067, +0.061], pairwise −0.002 [−0.045, +0.044].
  - C2 + S1 vs S1's order: AUC +0.050 [−0.022, +0.120], pairwise +0.041 [−0.009, +0.094].
  - L2 vs S1's order: pairwise +0.061 [+0.021, +0.104]. The humans prefer L2's order too.
- **Reading:** a zero-shot Qwen3-Reranker-0.6B does not replace a learned stage 2, and adds nothing
  measurable to one. That holds with the top-30 cut (5b, 5c) and without it (5d). Every C2 score is
  now cached, so other fusions or cut-offs cost nothing to compute. Any of them would be chosen after
  seeing these figures, and so exploratory.

### D55 cold run (21 Oct 2026): 3 of 4 pass; `rerank_chat` reproduces every figure, not two score digests

`checks --cold rerank_chat feature_arms stage1_classifier stage1_threshold`, `logs/d55_cold.log`,
6 h 51 min, peak RSS 6.1 GB. `feature_arms`, `stage1_classifier` and `stage1_threshold` pass.
`rerank_chat` fails: its rebuild is not byte-identical to `chat-rerank-eval.json`.

- **Diagnosis (warm rebuild + field diff, `logs/rerank_chat_warm_diff.log`):** the cold run re-scored
  R1-R3 with their stored models and cached the new scores. A warm rebuild on those scores differs
  from the record in exactly two fields, the `scores_sha256` of R1 and of R2 (the cross-encoders).
  Every figure (Recall@10, nDCG@10, No Fits, CIs, deltas, the human check, the decisions) is
  identical. R3 (a bi-encoder) re-scores bit for bit. So re-inference of R1 / R2 differs in the last
  bits of the floats, too little to move a figure at the record's 6 decimals.
- **Not verified:** whether L3's retrain (5 folds, ~6 h) reproduces. The check reports only pass or
  fail, and it deletes the retrained scores.
- **Side effect:** the R1 / R2 caches now hold the cold run's scores, so `rerank_chat --check --warm`
  also fails, on the same two digests, until this is resolved.
- **L3 reproduces (21 Oct 2026, owner-approved check):** fold 0 retrained into a temporary root is
  bit-identical to the stored fold (1,204 pairs, max |Δ| 0.0, epoch loss 0.066877 both;
  `logs/l3_fold0_repro.log`, 1 h 26 min).
- **Fix (owner-approved):** R1-R3's fingerprint is now `top10_sha256`, a digest of each JD's top 10
  by the score (`rerank_chat._top10_sha`), the thing Recall@10 reads; every other candidate keeps
  `scores_sha256`. Rounding before hashing was not used: some of 6,000 scores always sit near a
  rounding edge. The record was rewritten from the cold run's R1-R3 scores; exactly three lines
  changed (those fingerprints), no figure.
- **Cold re-run after the fix (21 Oct 2026): PASS.** `checks --cold rerank_chat`,
  `logs/d55_cold_rerank_chat.log`, 6 h 51 min, peak RSS 6.2 GB. With today's other three passes,
  every record of this plan with a `--check` now reproduces from scratch (D55).

## Assumptions (to confirm)

- Stage 2 uses F12, the same features as stage 1 (owner, 18 Oct 2026). R1–R3 and L3 bring the
  text models.
- SBERT in F12 is SBERT weak (X4's term), not the untuned base model. SBERT weak was trained on
  weak pairs from the A2 train region. The D50 documents may be among them, so the D50 fit can
  see a mildly optimistic SBERT column.
