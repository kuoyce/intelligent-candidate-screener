# Implementation record

Status: **Parts A and B ran; Stage 2 refits and A2 scoring not run** (Part A found no variant to
freeze). Exploratory (D56); nothing is adopted. Records: `output/extraction/coverage-variants.json`,
`output/extraction/match-keys.json` (figures only; the sampled phrases are in the git-ignored `data/processed/extraction/match-keys-samples.json`).

## Built

| File | Role |
|---|---|
| `matching/entity_weighted.py` | `DocumentFrequency(unknown, min_df, top_k)` and `.variant(...)`: W0-W3 as views over one fit; defaults are W0 and unchanged |
| `evaluation/coverage_variants.py` | Part A: 70/30 D50 JD split (seed 0), coverage-only per-JD AUC, paired deltas, the plan's tie rule |
| `evaluation/match_keys.py` | Part B: match keys K0, K1, K2, K3 (0.7 / 0.8 / 0.9), gains, Fit rates, redacted samples |
| `tests/test_coverage_variants.py`, `tests/test_match_keys.py` | 11 synthetic tests; the unknown-weight, top-k ordering and K2 length-rule tests were shown to fail when the code they guard is broken |

`uv run pytest`: 824 passed (see the commit). No new extraction; both parts read the caches.

## Part A: weighting variants (coverage-only per-JD AUC, delta vs W0)

R+N (R is unchanged by W1 and W3 because the seed rules emit no unknown phrases):

| Variant | D50 selection (70) | D50 held-out (30) | A1-LLM (639) |
|---|---|---|---|
| W0 | 0.669 | 0.704 | 0.756 |
| W1 | -0.011 [-0.034, 0.006] | -0.012 [-0.031, 0.003] | -0.015 [-0.045, 0.009] |
| W2 | -0.124 [-0.168, -0.082] | -0.106 [-0.155, -0.057] | -0.033 [-0.107, 0.031] |
| W3 | -0.011 [-0.034, 0.006] | -0.012 [-0.031, 0.003] | +0.002 [-0.016, 0.020] |

Nothing meets the plan's rule (>= +0.005 on selection and positive on both held-out sets), so
the guidance is **no change**; no variant is frozen and A2 final is not scored.

Notes for the reader:
- Per-JD AUC is invariant to the per-JD denominator, so this table cannot show a dilution fix;
  that effect only appears in a learned score that compares coverage across JDs.
- W2 as registered breaks ties among weight-5 concepts alphabetically (most unknown phrases have
  df 0 and hit the cap), so "top 10" is an arbitrary slice. A fairer tie-break would be a new
  variant and needs registering before it runs.
- W3's recovery on A1-LLM is the floor flattening df-0 phrases (A1 text is not in the IDF
  population), not a gain.

## Part B: what exact matching misses (R+N; diagnostic, labels read only to report)

Coverage-only per-JD AUC and share of pairs with no overlap (A2 final / D50 / A1-LLM):

| Key | No overlap, A2 | AUC A2 | AUC D50 | AUC A1-LLM | Pairs gained vs K0 (A2) | Fit rate gained / still-empty (A2) |
|---|---|---|---|---|---|---|
| K0 exact | 47.5% | 0.718 | 0.665 | 0.747 | 0 | - / 0.44 |
| K1 punct. + plural + .js | 46.8% | 0.724 | 0.666 | 0.737 | 4 | 1.00 / 0.43 |
| K2 K1 + containment | 29.7% | 0.741 | 0.707 | 0.798 | 107 | 0.57 / 0.37 |
| K3 @ 0.7 | 33.2% | 0.776 | 0.705 | 0.840 | 86 | 0.65 / 0.35 |
| K3 @ 0.8 | 42.2% | 0.734 | 0.687 | 0.811 | 32 | 0.69 / 0.41 |
| K3 @ 0.9 | 46.0% | 0.725 | 0.672 | 0.759 | 9 | 0.89 / 0.43 |

- Surface-form mismatch (punctuation, plurals, `.js`) is **small**: K1 gains 4 A2 pairs.
- Partial-phrase and semantic mismatch is **large**: K2 recovers 107 of 600 A2 pairs (18%) and
  K3@0.7 86, with coverage-only gains of +0.02 to +0.06 AUC on A2 and +0.04 to +0.09 on A1-LLM
  and D50. Gained pairs are mostly Fit (57-69% on A2, against 37% for the pairs that stay empty).
- Sampled K1 matches are correct (`code reviews` / `code review`, `express js` / `express`).
  K2 and K3 samples mix correct matches (`ci/cd tools` / `ci/cd`, `crm systems` / `crm system`,
  `communication` / `communications`) with loose ones (`tableau development` / `development`,
  `programming` / `object-oriented programming principles`), so some of the gain is generic
  overlap.
- K3 improves monotonically as the threshold falls; 0.7 is the lowest tried, so the best
  threshold is not located. Thresholds are not chosen here.

## Reading

Exact matching **does** lose signal, but it is partial and semantic phrasing, not typography.
That is a different lever from weighting. It is a coverage-feature result only: the earlier
plan showed that a better coverage feature did not carry into the learned score, so this does
not show a better ranking. Testing it in the score is a follow-up plan, not part of this one.

## Deviations

- Stage 2 refits and the A2 final score were not run (nothing to freeze).
- Part B reads A2 final labels to report Fit rates and coverage-only AUC, as the plan states;
  no choice is made from them.

## Amendment 1 results (Part C threshold search; W2b and the in-score check)

Registered in [02-amendment-threshold-and-w2b.md](02-amendment-threshold-and-w2b.md) before the
runs. A2 final is not read. Records: `output/extraction/match-threshold.json`, and the
extended `output/extraction/coverage-variants.json` (now with W2b and an `in_score` block).

### Part C: embedding threshold (R+N, coverage-only per-JD AUC; delta vs exact K0)

| Threshold | D50 selection (70 JDs) | D50 held-out (30) | A1-LLM (639) |
|---|---|---|---|
| K0 exact | 0.655 | 0.690 | 0.747 |
| K2 containment | 0.706 (+0.051) | 0.708 (+0.018) | 0.798 (+0.051) |
| 0.50 | 0.715 (+0.060) | 0.726 (+0.036) | 0.845 (+0.098) |
| 0.55 | 0.723 (+0.068) | 0.723 (+0.034) | 0.856 (+0.108) |
| **0.60** | **0.724 (+0.070)** | 0.714 (+0.024) | 0.854 (+0.107) |
| 0.65 | 0.701 (+0.047) | 0.728 (+0.038) | 0.853 (+0.105) |
| 0.70 | 0.697 (+0.042) | 0.726 (+0.036) | 0.840 (+0.093) |
| 0.75 | 0.691 (+0.036) | 0.714 (+0.024) | 0.825 (+0.077) |
| 0.80 | 0.678 (+0.023) | 0.709 (+0.019) | 0.811 (+0.063) |
| 0.85 | 0.663 (+0.008) | 0.702 (+0.012) | 0.752 (+0.004) |
| 0.90 | 0.659 (+0.005) | 0.704 (+0.014) | 0.759 (+0.011) |

- The optimum is **interior**: selection AUC peaks at 0.60 (0.55 and 0.65 are within 0.023),
  then falls as the threshold rises; 0.50 is lower than 0.55. A broad plateau spans 0.50-0.70.
- The rule's pick is **0.60** (ties within 0.002 go to the higher threshold). It is above K0 on
  both held-out sets. The held-out sets would pick 0.65 (D50) and 0.55 (A1-LLM), inside the plateau.
- Every threshold from 0.50 to 0.80 beats containment (K2) on the selection set.

### W2 repaired (W2b) and the in-score check

W2b (ties by JD mention count) is **no better than W2**: R+N D50 selection -0.126 against
-0.124. The alphabetical tie-break was not the main cause; truncating the JD to 10 concepts
drops informative ones. Per-JD coverage-only AUC and the in-score check (base-4 logistic fitted
on the 70 D50 selection JDs with coverage replaced, scored on held-out data):

| R+N variant | In-score A1-LLM AUC (delta vs W0) | In-score D50 held-out AUC (delta) |
|---|---|---|
| W0 | 0.897 | 0.868 |
| W1 | 0.888 (-0.009 [-0.024, +0.004]) | 0.865 (-0.003 [-0.011, +0.006]) |
| W2 / W2b | 0.883 (-0.014 [-0.032, +0.001]) | 0.863 (-0.005 [-0.013, +0.004]) |
| W3 | 0.893 (-0.004 [-0.013, +0.003]) | 0.865 (-0.003 [-0.011, +0.006]) |

No variant is above W0 on either metric, so the guidance stays **no change** for weighting and
dilution. For R, W1 and W3 are identical to W0 (no unknown phrases).

### Reading

The lever is the match key, not the weights: soft matching at about 0.6 gains +0.07 coverage-only
AUC on D50 selection, +0.02 to +0.04 on D50 held-out and +0.10 on A1-LLM. As before, this is a
coverage-feature result; carrying it into the learned score is untested.
