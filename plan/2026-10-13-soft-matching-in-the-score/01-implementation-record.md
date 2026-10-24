# Implementation record

Status: **complete (exploratory).** Soft matching at 0.60 does not let any learner beat hand-set X2,
and does not beat exact matching inside the score. By the registered rule, **coverage is not the
bottleneck**; no further coverage-feature work in this line. Record:
`output/extraction/concept-stage2-R_N-soft0.60.json` (R+N exact and soft, side by side).

## Built

| File | Role |
|---|---|
| `evaluation/soft_coverage.py` | `SoftSims` (best-cosine per JD concept, IDF weights as W0) and `replace_columns` for `coverage`, `coverage_unweighted`, `log_matched` |
| `evaluation/concept_stage2.py` | Refactored into `evaluate` / `context` / `exact_features`; `--soft` runs exact and soft side by side with paired comparisons and the sensitivity block |
| `tests/test_soft_coverage.py` | 3 synthetic tests; the threshold-inclusive test fails when `>=` is mutated to `>` |

Refactor check: re-running `--arm R+N` left `concept-stage2-R_N.json` byte-identical. `--soft`
takes about 6 min and 5 GB (63k distinct phrases embedded).

## Result: A2 final, delta vs hand-set X2 (0.850), soft 0.60 (R+N)

| Labels | Learner | Δ generic | Δ targeted | Δ total [CI] | Bar |
|---|---|---|---|---|---|
| A1-LLM | logistic base (4) | +0.003 | +0.003 | +0.003 [-0.017, 0.024] | no |
| A1-LLM | logistic all (11) | -0.011 | +0.001 | -0.007 [-0.043, 0.023] | no |
| A1-LLM | boosting all (11) | -0.010 | +0.005 | -0.005 [-0.031, 0.021] | no |
| D50 | logistic base (4) | -0.004 | +0.010 | +0.001 [-0.016, 0.016] | no |
| D50 | logistic all (11) | -0.000 | -0.027 | -0.009 [-0.044, 0.023] | no |
| D50 | boosting all (11) | +0.011 | -0.008 | +0.005 [-0.027, 0.035] | no |

None of the 12 grid fits (feature set × A / B / C labels) meets the bar either. Disclosure: A2 had
already been seen for this feature in Part B; only the threshold is clean of it.

## Soft against exact (same features, same labels; paired)

- **A2 final, final fits:** between -0.005 and +0.007, every CI crossing 0 (best: D50 boosting
  +0.007 [-0.008, 0.023]).
- **Inside the score on D50 held-out** (fit on the 70 selection JDs, per-JD AUC on the 30
  held-out; A2 not used), soft minus exact at 0.55 / 0.60 / 0.65:
  - logistic base (4): -0.013 / -0.005 / -0.005, CIs include 0;
  - logistic all (11): -0.016 / -0.014 / -0.008, CIs include 0.
  On A1-LLM the soft versions are higher (+0.014 to +0.022 for base-4) but only one of the six
  intervals (base-4 at 0.55: +0.022 [0.001, 0.047]) clears 0, which is one nominal pass in six.
- The 11-feature curves are still rising on A1-LLM labels (boosting +0.073 from 25 to 70 JDs),
  but stay at or below hand-set X2 at 70 JDs; the pass rule is not met, so the label decision rule
  does not open.

## Decision (registered rule)

No learner passes **and** the soft-versus-exact delta on D50 held-out has no CI above 0 (all are
negative or cross zero) → **coverage is not the bottleneck.** The binary 0.60 stays as chosen;
neither the sensitivity thresholds nor the EDA gave a reason to change it. The +0.07 gain of the
feature on its own (Part C) does not become ranking gain: role and TF-IDF already carry what
the extra matches add.

## Next

[error-analysis-2](../2026-10-13-error-analysis-2/README.md) (draft): size the other levers (role
signal, experience coverage, label noise) before choosing the next plan.
