# P5 tasks 5.1–5.2: the ordering evaluation (`evaluation/ordering.py`)

Status: implemented; `status: exploratory` in every record (the configurations were seen in
01 §4 before registration, task 5.3).

## What was built

- `src/candidate_screener/evaluation/ordering.py`: the scratchpad check from 01 §4,
  promoted into the package (D52: ordering figures on human grades are a secondary metric;
  classification stays Fit / No Fit).
- `tests/test_evaluation_ordering.py`: 13 tests on inline synthetic data (no `data/`).
- Records: `output/matching/ordering-metrics.json` (`--text-variant v2`, the committed
  record of task 5.2) and `output/matching/ordering-metrics-v1.json` (`v1`, to show the
  module reproduces 01 §4).

```bash
uv run python -m candidate_screener.evaluation.ordering                    # v2 (default)
uv run python -m candidate_screener.evaluation.ordering --check
uv run python -m candidate_screener.evaluation.ordering --text-variant v1 [--check]
```

## Semantics

| Item | Rule |
|---|---|
| Grades | Raw human labels from `a2-final-labels.csv`: Good Fit = 2, Potential Fit = 1, No Fit = 0. Any other label raises |
| References | `yc` (600 pairs), `van` (326), `mean` = mean of both, NaN unless both exist (326), `yc_single` = yc on the 274 pairs van did not label. Anything else (the LLM, `final_label`) raises `ValueError` |
| nDCG | Per JD over the CVs the reference grades; linear gain (the grade), log2 discount; tied scores share their block's mean gain, so a tie is worth exactly the expectation of breaking it at random. Scores are rounded to 9 decimals first so float noise is not a tie-break |
| Pairwise | Per JD, over CV pairs whose reference grades differ: 1 if the score orders them the same way, 0.5 on a score tie |
| Exclusion | A JD with no two different grades (which includes zero ideal DCG) is excluded from **both** metrics; each figure carries `n_jd_excluded` |
| Aggregation | Mean over JDs; 95% percentile CI from `resample.mean_ci` (10,000 JD resamples, seed 0). `generic`, `targeted`, `total` always together (`assert_total_has_strata`), each with `n_jd` and `n_pairs` |
| Paired deltas | Mean per-JD difference B − A over the JDs both are scored on, JD-resampled CI. Comparisons: every system vs random; SBERT zero-shot and SBERT weak vs TF-IDF; the LLM vs TF-IDF and vs SBERT weak; for every reference, subset, metric and scope |
| Subsets | `full`; `llm_free` (tiers in `final_set.LLM_FREE_TIERS`); `fit_only` (final label Fit) |
| Random | Analytic: expected nDCG = mean(gain) × Σ discounts / ideal DCG; pairwise = 0.5. No sampling |
| Systems | random (expected), TF-IDF, SBERT zero-shot, SBERT weak fine-tuned (from `matching`, built with `text_variant`), and the LLM 0–100 score (`LLM_AGENT_SHA256`, `claude-haiku-4-5`, runs 9 and 10; asserted to be exactly 600 unique pairs). SBERT rungs are skipped, with the reason in `skipped`, when `sentence_transformers` is absent |
| Context | `inter_annotator`: van's grades scored as an order against yc's and the reverse, on the 326 doubly-graded pairs (D44, no target) |

## Named invariant tests, each shown to fail against a naive version

Checked with a scratch harness that monkeypatches the guarded function with the naive
version and runs the one test (all six failed, as required):

| Test | Naive version it fails against |
|---|---|
| `test_score_ties_give_no_tie_break_luck` | plain `argsort` nDCG: 6 distinct values over permutations of a tie |
| `test_a_jd_with_all_equal_grades_is_excluded_and_counted` | "informative" = at least two CVs (all-No-Fit JD kept) |
| `test_a_constant_scorer_scores_exactly_one_half_pairwise` | a score tie counted as 0 |
| `test_a_perfect_scorer_scores_one_on_both_metrics` | pairwise over all ordered pairs, equal grades included |
| `test_the_mean_reference_is_nan_where_only_one_human_exists` | `mean(axis=1)` with the default `skipna=True` |
| `test_the_llm_is_rejected_as_a_reference` | `reference` that returns any column it is asked for |

## Reproduction of 01 §4 (`v1`)

Total, full set; value [95% CI], n JDs. Read from `ordering-metrics-v1.json`.

| System | nDCG vs `yc` (n=56) | pairwise vs `yc` (n=56) | nDCG vs mean (n=53) | pairwise vs mean (n=53) |
|---|---|---|---|---|
| random (expected) | 0.736 [0.704, 0.766] | 0.5 | 0.831 [0.809, 0.854] | 0.5 |
| TF-IDF | 0.873 [0.841, 0.902] | 0.785 [0.746, 0.824] | 0.916 [0.892, 0.939] | 0.726 [0.654, 0.790] |
| SBERT (zero-shot) | 0.871 [0.840, 0.900] | 0.761 [0.724, 0.797] | 0.916 [0.889, 0.941] | 0.706 [0.630, 0.778] |
| SBERT (weak fine-tuned) | 0.905 [0.876, 0.931] | 0.819 [0.789, 0.848] | 0.919 [0.891, 0.945] | 0.725 [0.641, 0.805] |
| LLM score | 0.896 [0.856, 0.931] | 0.843 [0.804, 0.879] | 0.924 [0.895, 0.949] | 0.770 [0.695, 0.839] |

Differences from 01 §4, and why:

1. **nDCG vs `mean` is lower** (TF-IDF 0.916 vs 0.921, SBERT 0.916 vs 0.921, LLM 0.924 vs
   0.928, random 0.831 vs 0.839) and its n is 53, not 56. The scratch script excluded only
   JDs with zero ideal DCG from nDCG, so three JDs whose doubly-graded CVs all share one
   non-zero grade entered with nDCG = 1 for every system. The spec excludes every JD with no
   differing pair from both metrics, so nDCG and pairwise now share one n per reference.
   Against `yc` no such JD exists, and the `yc` figures are identical.
2. **random nDCG vs `yc` is 0.736, not 0.737**: the expectation is now exact rather than a
   200-draw average.
3. **CIs move in the third decimal** (LLM pairwise vs `yc` [0.804, 0.879] vs [0.805,
   0.880]): 10,000 resamples through `resample.mean_ci`, against 2,000 with another RNG.

Every other point value in 01 §4 that this module computes is reproduced to three decimals.
(Keyword match and metadata rules are not in this module's system list, per the spec's
baselines; they are metadata rungs.)

## `v2` figures (the committed record)

Read from `ordering-metrics.json` (`text_variant: v2`, seed 0, nothing skipped).
`exploratory`. The LLM score is the committed series, read on the text the LLM saw; only
the text rungs change with `v2`. Value [95% CI], n JDs.

**Pairwise vs `yc`, full set**

| System | generic | targeted | total |
|---|---|---|---|
| random (expected) | 0.5, n=38 | 0.5, n=18 | 0.5, n=56 |
| TF-IDF | 0.774 [0.719, 0.828] | 0.792 [0.754, 0.831] | 0.780 [0.741, 0.819] |
| SBERT (zero-shot) | 0.736 [0.697, 0.774] | 0.749 [0.676, 0.817] | 0.740 [0.705, 0.775] |
| SBERT (weak fine-tuned) | 0.828 [0.794, 0.861] | 0.767 [0.729, 0.807] | 0.809 [0.781, 0.835] |
| LLM score | 0.874 [0.836, 0.910] | 0.778 [0.699, 0.847] | 0.843 [0.804, 0.879] |

**nDCG vs `yc`, full set**

| System | generic (n=38) | targeted (n=18) | total (n=56) |
|---|---|---|---|
| random (expected) | 0.763 [0.723, 0.799] | 0.679 [0.630, 0.727] | 0.736 [0.704, 0.766] |
| TF-IDF | 0.893 [0.858, 0.924] | 0.820 [0.754, 0.876] | 0.869 [0.837, 0.899] |
| SBERT (zero-shot) | 0.873 [0.831, 0.909] | 0.852 [0.777, 0.915] | 0.866 [0.829, 0.898] |
| SBERT (weak fine-tuned) | 0.919 [0.890, 0.944] | 0.843 [0.769, 0.906] | 0.895 [0.862, 0.923] |
| LLM score | 0.938 [0.906, 0.964] | 0.807 [0.717, 0.889] | 0.896 [0.856, 0.931] |

**Pairwise, totals across references and subsets**

| System | vs `mean`, full (n=53) | vs `yc`, `llm_free` (n=55) | vs `yc`, `fit_only` (n=49) |
|---|---|---|---|
| TF-IDF | 0.724 [0.652, 0.791] | 0.785 [0.741, 0.828] | 0.679 [0.607, 0.748] |
| SBERT (zero-shot) | 0.692 [0.613, 0.765] | 0.749 [0.706, 0.791] | 0.666 [0.595, 0.732] |
| SBERT (weak fine-tuned) | 0.732 [0.648, 0.808] | 0.803 [0.764, 0.841] | 0.690 [0.619, 0.758] |
| LLM score | 0.770 [0.695, 0.839] | 0.866 [0.829, 0.900] | 0.719 [0.635, 0.797] |

**Paired deltas, pairwise vs `yc`, total** (B − A [CI], n JDs)

| Comparison | full (n=56) | `llm_free` (n=55) | `fit_only` (n=49) |
|---|---|---|---|
| SBERT zero-shot − TF-IDF | −0.040 [−0.090, 0.008] | −0.036 [−0.094, 0.021] | −0.014 [−0.100, 0.070] |
| SBERT weak − TF-IDF | +0.028 [−0.016, 0.074] | +0.018 [−0.035, 0.070] | +0.011 [−0.081, 0.100] |
| LLM − TF-IDF | +0.063 [0.016, 0.112] | +0.081 [0.032, 0.131] | +0.040 [−0.056, 0.133] |
| LLM − SBERT weak | +0.035 [0.000, 0.068] | +0.063 [0.021, 0.107] | +0.029 [−0.062, 0.124] |

Deltas vs random: of the 288 (system × reference × subset × metric × scope), 24 are empty
(`yc_single` × `targeted`: van graded every targeted pair, so no targeted JD has a
yc-only pair; the figure is emitted with `value: null`, `n_jd: 0`, beside its generic and
total). Of the other 264, 236 have a CI above 0, including every `full` and `llm_free`
cell. The 28 that do not are all `fit_only`,
mostly on `van` / `mean` (few JDs) or `targeted` (e.g. LLM pairwise vs `yc`, `fit_only`,
targeted: +0.066 [−0.062, 0.185]). The `fit_only` totals vs `yc` all clear 0 (e.g. TF-IDF
+0.179 [0.107, 0.248]).

Readings (exploratory):
- `v2` costs the text rungs a little against `v1` (TF-IDF pairwise vs `yc` 0.785 → 0.780,
  SBERT zero-shot 0.761 → 0.740, SBERT weak 0.819 → 0.809), consistent with `Position`
  having carried some of the signal.
- Ordering among Fit CVs (`fit_only`) is much weaker for every system (0.67–0.72 total);
  the LLM's `fit_only` targeted figure is 0.566 [0.438, 0.685], not separable from random.
- Inter-annotator context (D44, `v1` and `v2` identical): van's grades as an order against
  yc's, pairwise 0.726 [0.656, 0.794] (n=42 JDs); yc's against van's, 0.674 [0.613, 0.736]
  (n=51).

## Deviations and open items

- **Default `--text-variant` is `v2`**, so the bare command and `--check` address the
  committed record `ordering-metrics.json`; `v1` writes `ordering-metrics-v1.json`.
- **The ordering bar is not decided here.** Deltas vs random are recorded; whether "beats
  random" means the system's lower CI bound above random's point value or the paired delta's
  lower bound above 0 is for `05-registration.md` (task 5.3).
- `fit_only` and `yc_single` figures exist for every system but were not in 01 §4, so
  nothing to reproduce.
