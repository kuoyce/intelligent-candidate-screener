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
