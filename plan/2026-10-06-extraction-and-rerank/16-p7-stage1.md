# 16 — P7: stage 1, text-only (X rungs, K1–K5, S2)

Record: `output/matching/stage1-metrics.json` (`uv run python -m
candidate_screener.evaluation.stage1 --check` reproduces it). Text: `text_v2` (D53).
Registration: [06](06-registration.md), amendments A1 and A2, committed before this ran.

## 1. What was built

| File | What |
|---|---|
| `matching/retriever.py` | `Retriever` protocol (02 §4) and `SystemRetriever`: any rung as `shortlist(jd, pool, k)`; ties keep pool order |
| `matching/extracted.py` | **X1** `ExtractedRule`: `2 × role_match + experience_compatible`, both from text. **X2–X4** `ExtractedPlus`: X1 plus a text rung, combined exactly as 1a+2 (`MetadataPlus` with X1 as the level). The extractors see a frame holding `jd_text` and `cv_text` only |
| `extraction/experience.py` | `experience_compatible_extracted(..., unknown_compatible=True)`: amendment A1. The default (unknown → 0) is unchanged |
| `matching/lexical.py` | `MetadataPlus._level`: the rule the text orders within, so X2–X4 reuse the 1a+2 combination. 1a+2 output unchanged (the v1 record reproduces; `scores_sha256` of the shared rungs equal P1's v2 record) |
| `evaluation/retrieval_pilot.py` | `build_pools(seed, text_variant)`: same pool members, CV text cut to `v2`; the `v1` cache name is unchanged |
| `evaluation/stage1.py` | AUC figures, K1–K5 with the §3 verdict, retrieval figures, S2, selection |

Status: X1–X4 `held-out`; random, TF-IDF, SBERT zero-shot and weak `exploratory` (P1);
1a and 1a+2 on the true fields `oracle` (upper bound, never a result).

## 2. AUC against the final set (per JD; `text_v2`)

Total *n* = 59 JDs (within-family: 48). 95% CI resamples JDs.

| Rung | Status | generic | targeted | **total** [CI] | within-family total | `llm_free` total |
|---|---|---|---|---|---|---|
| random | expl. | 0.558 | 0.552 | 0.556 [0.505, 0.606] | 0.580 | 0.587 |
| X1 extracted rule | held-out | 0.851 | 0.755 | 0.818 [0.765, 0.867] | 0.678 | 0.866 |
| X3 X1 + skill overlap | held-out | 0.851 | 0.761 | 0.821 [0.767, 0.869] | 0.704 | 0.863 |
| TF-IDF | expl. | 0.780 | 0.807 | 0.789 [0.741, 0.835] | 0.714 | 0.826 |
| **X2 X1 + TF-IDF** | held-out | 0.863 | 0.826 | **0.850** [0.809, 0.889] | 0.745 | 0.900 |
| SBERT zero-shot | expl. | 0.740 | 0.731 | 0.737 [0.686, 0.784] | 0.731 | 0.774 |
| SBERT weak | expl. | 0.853 | 0.796 | 0.834 [0.796, 0.870] | 0.752 | 0.852 |
| **X4 X1 + SBERT weak** | held-out | 0.891 | 0.785 | **0.855** [0.817, 0.890] | 0.735 | 0.892 |
| *oracle* 1a | oracle | 0.776 | 0.784 | 0.778 [0.732, 0.819] | 0.540 | 0.799 |
| *oracle* 1a+2 | oracle | 0.843 | 0.851 | 0.846 [0.799, 0.888] | 0.697 | 0.859 |

Residual title leak beside these figures (P1, unchanged): the removed `Position` still
appears in 22.7% [19.3, 26.0] of the 600 CVs' `v2` text (generic 20.3%, targeted 27.5%).

## 3. Registered comparisons (paired Δ AUC, B − A)

| # | B vs A | generic | targeted | **total** [95% CI] | `llm_free` total | Verdict |
|---|---|---|---|---|---|---|
| K1 | X2 vs TF-IDF | +0.082 | +0.019 | **+0.061** [0.029, 0.096] | +0.075 | **Justified** |
| K2 | X2 vs oracle 1a+2 | +0.020 | −0.025 | +0.005 [−0.034, 0.050] | +0.042 | Not resolved (half-width 0.042); decides nothing |
| K3 | X4 vs SBERT weak | +0.037 | −0.011 | +0.021 [−0.009, 0.050] | +0.040 | Not resolved (0.030) |
| K4 | X2 vs SBERT weak | +0.009 | +0.030 | +0.016 [−0.025, 0.058] | +0.048 | Not resolved (0.041) |
| K5 | X3 vs X1 | +0.001 | +0.006 | +0.002 [−0.020, 0.026] | −0.003 | Not resolved (0.023); decides nothing |

## 4. Retrieval (S2) on the 500-candidate pools

60 pools of 500 (59 with a fit). Relevant = final-set Fit among the 10 judged CVs; the 490
distractors are unjudged and assumed not fits, so recall is an upper bound.

| Rung | Recall@10 total [CI] | Recall@50 generic | targeted | **total** [CI] |
|---|---|---|---|---|
| random | 0.016 | 0.107 | 0.141 | 0.119 [0.088, 0.151] |
| X1 | 0.255 [0.195, 0.317] | 0.835 | 0.652 | 0.773 [0.691, 0.846] |
| X3 | 0.291 [0.227, 0.355] | 0.789 | 0.605 | 0.726 [0.649, 0.798] |
| TF-IDF | 0.244 [0.190, 0.299] | 0.588 | 0.726 | 0.635 [0.563, 0.702] |
| **X2** | 0.300 [0.238, 0.363] | 0.795 | 0.749 | **0.779** [0.706, 0.845] |
| SBERT zero-shot | 0.159 [0.113, 0.209] | 0.427 | 0.459 | 0.438 [0.373, 0.504] |
| SBERT weak | 0.224 [0.170, 0.282] | 0.602 | 0.573 | 0.592 [0.520, 0.663] |
| **X4** | 0.322 [0.253, 0.391] | 0.823 | 0.640 | **0.761** [0.683, 0.832] |
| *oracle* 1a | 0.426 [0.358, 0.496] | 0.904 | 0.962 | 0.924 [0.876, 0.961] |
| *oracle* 1a+2 | 0.479 [0.403, 0.553] | 0.909 | 0.969 | 0.929 [0.882, 0.966] |

| S2 group | Bar | Best | Recall@50 | Verdict | CI above bar |
|---|---|---|---|---|---|
| classical (TF-IDF, X2) | 0.80 | X2 | 0.779 | **miss** | no |
| advanced (SBERT weak, X4) | 0.90 | X4 | 0.761 | **miss** | no |

Recall@10 has no bar (registered).

## 5. Selection

**None selected**: no candidate meets its S2 bar, so §5 has no rung to adopt. The trail
is in the record. How §5 was read (written into the code before the first run,
`stage1.SELECTION_NOTE`): candidates are the four rungs S2 names; each qualifies on its
own Recall@50 point estimate; a qualifying rung is beaten only by a dearer qualifying rung
justified over it on K1, K3, or K4 with B and A swapped.

## 6. Reading

- **Extraction replaces the field on AUC.** X2 matches the true-field 1a+2 (K2 +0.005,
  CI ±0.04), and adds clearly to TF-IDF (K1, justified, both strata positive, holds on
  `llm_free`). Against the fine-tuned bi-encoder (K3, K4) it is not resolved.
- **It does not replace the field on retrieval.** Recall@50 drops from 0.93 (true fields)
  to 0.78 (X2). In a pool of 500 the role filter carries retrieval, and the soft role
  match is weaker where it matters most: `targeted` X1 Recall@50 0.65 vs oracle 0.96.
- **X1 is almost all role.** Experience is unknown on one side or both for 71% of the 600
  pairs, so under A1 `experience_compatible` is 1 on 95% of them. X1 orders by
  `role_match` (median 0.13, 90th percentile 0.90).
- **Skill overlap adds nothing to X1** (K5, +0.002), and lowers Recall@50 (0.73 vs 0.77).
- Within-family AUC of X1 (0.68) is above the oracle 1a's (0.54; inside a keyword only its
  experience term varies): the soft role score orders some candidates inside a keyword.
- Costs in `stage1-costs.json` include embedding-cache hits and are not comparable
  across rungs; latency stays with `evaluation/latency.py`.

## 6a. Why S2 misses: diagnosis (exploratory, 7 Oct, after scoring)

**Exploratory.** Run after the held-out figures were seen, and it reads the final-set
labels (relevance), so nothing here may tune a rung (D46). A fix it suggests is a new
rung: registered, tuned on train data only, and reported as post-hoc. Script was ad hoc
(scratchpad), on X2 and the oracle 1a over the same `v2` pools; it is not in the package.

**What changed from the metadata rule.** The formula has the same shape
(`2 × role + experience`, plus `0.999 ×` per-JD min-max TF-IDF), but three inputs differ:

| | Oracle 1a / 1a+2 | X1 / X2 |
|---|---|---|
| Role term | keyword equality, **0 or 1** | soft agreement `Σ p_jd p_cv`, **continuous in [0, 1]** |
| Label space | `Primary Keyword` (~45) | 60 role families (T-A split generic) |
| Experience | true bands; unknown → 0 | extracted bands (stated on 60% of JDs, 47% of CVs); unknown → 1 (A1), so 1 on 95% of pairs |

In 1a a keyword match (≥ 2) always outranks a mismatch (≤ 1.999), so the keyword acts as a
filter. With a continuous role term the text can overtake it (role 0.3 + top TF-IDF = 1.6 >
role 0.7 + bottom TF-IDF = 1.4). Extractor accuracy on the final-set documents (P3, P4):
role family JD 0.683 / CV 0.700 (top-3 0.850 / 0.877).

**Where the fits are lost** (367 fits over 59 JDs):

| Finding | Figure |
|---|---|
| Fits sharing the JD's true keyword | 0.965 (same true family 0.913) |
| Same-keyword CVs per 500-pool | median 26.5 (min 10, max 80) |
| Fits missed in top 50: oracle 1a / X2 | **5 / 71** |
| Missed fits that share the JD's true keyword | 0.873 (so: extraction error, not an atypical fit) |
| `role_match`, missed vs kept fits (median) | **0.051 vs 0.639** |
| X2 rank of missed fits (quartiles) | 73 / 95 / 187: well past the cut, not just below it |
| CVs outranking a missed fit (median): other-keyword / same-keyword non-fit | **75 / 17** |
| Share of top 50 with the JD's true keyword: X2 / oracle | 0.417 / 0.574 |
| `role_match` at rank 1 / rank 50 (median per JD) | 0.730 / 0.008 |
| X2 Recall@50 by whether the JD's family is predicted right | right 0.841 (41 JDs) / wrong 0.637 (18 JDs) |
| … generic: right / wrong | 0.828 (31) / 0.665 (8) |
| … targeted: right / wrong | 0.883 (10) / 0.615 (10) |

**Reading.** The scores are not flat; they are lopsided. The role model is confident about
the first ~20 CVs of a pool, then `role_match` is near 0 and the rest of the top 50 is
ordered by TF-IDF across all roles. Fits are lost where the extractor does not recognise
them as the JD's role (the JD's family is wrong for 18 of 59 JDs; many same-role CVs get a
low agreement), and other-role CVs with similar wording fill the shortlist. Where the JD's
family is right, X2 Recall@50 is 0.84.

**Caveat.** The 490 distractors are unjudged and assumed not fits. Some of the
other-keyword CVs that outrank a missed fit may be real fits, so the recall figures are
upper bounds on what a judged pool would show, and "crowded out by wrong roles" is an
upper bound too.

**Candidate fixes (not built, not registered):** a hard filter on the JD's top-k families
(top-3 holds the right family 85–88% of the time), or a larger role weight. Either needs a
registration entry and train-only tuning before it is scored.

## 7. Deviations and open questions

- **Deviation (workflow):** implemented inline rather than by sub-agents, to keep the
  session limit for the LLM batch (both draw on it).
- **Q-P7-1 (new):** confirm the reading of §5 in §5 above (`SELECTION_NOTE`). It was fixed
  before scoring but is an interpretation; it changes nothing here, since no rung qualifies.
- **Q-P7-2 (new, high impact for P8.5):** S2 misses, so stage 1 has no adopted retriever.
  Registration §5 does not say what stage 2's end-to-end figure (P8.5) runs on in that
  case. Options: (a) report the miss and run P8.5 on the best classical rung (X2, 0.779),
  labelled "below S2"; (b) on X4; (c) skip P8.5. Recommendation: (a), since X2 is the K1-
  justified rung and the cheaper. Needs a user decision before P8.5.
