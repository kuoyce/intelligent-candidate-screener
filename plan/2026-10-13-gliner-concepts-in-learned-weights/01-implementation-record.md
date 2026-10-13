# Implementation record

Status: **complete (exploratory).** Both stages ran; the Stage 2 bar was not met by any learner.
Nothing here is adopted, and no result selects a rung (D56). Numbers are in
`output/extraction/concept-overlap.json` and `concept-stage2-{R,R_N}.json`.

## What was built

| File | Role |
|---|---|
| `extraction/concept_sources.py` | Disk-cached concept sources (GLiNER, MiniLM tagger, keyed by text SHA-256), `UnionConcepts`, the five arms R / G / N / R+G / R+N |
| `evaluation/concept_overlap.py` | Stage 1 statistics and gate (`--extract`, `--extract-idf`, `--stage1`) |
| `evaluation/concept_stage2.py` | Stage 2 refits and learning curves for one arm (`--arm`), plus the registered pass rule |
| `evaluation/a1_weights.py` | `build_scorer(extractor=...)`: default unchanged |
| `tests/test_concept_sources.py` | 7 synthetic tests; the context test and the gate-boundary test were shown to fail when the code they guard is broken |

`uv run pytest`: 813 passed. No `--check` module was added (not in the checks registry).

## Deviations from the plan

1. **Extra extraction for Stage 2.** The IDF population (18,721 train CVs) and A1 train (245 JDs,
   446 CVs) had to be tagged too. This was not in the first cost estimate. MiniLM only; GLiNER
   did not pass the gate. Wall time: 19.5 min (Stage 1 documents, both taggers), 32 min (IDF CVs),
   2 min (A1 train).
2. **GLiNER 0.2.29 breaks `load_model`**; `gliner==0.2.21` is pinned in the `ml` group. Smoke test
   and a 200-document rerun matched the saved run within ~1% (CV 1,082 vs 1,077; JD 747 vs 755).
3. The MiniLM tagger returns spans only, so its rows get label SKILL and the rules-v1 context.
4. Notebook 21 was not written; the records above are the evidence.

## Stage 1 (label-free)

Reference: seed rules R on A2 final, 53.3% of pairs without any JD-CV overlap (reproduces
notebook 20). Gate: some other arm cuts it by >= 5 pp.

| Arm | No-overlap, A2 final | Cut vs R |
|---|---|---|
| G (GLiNER) | 77.7% | -24.3 pp |
| N (MiniLM) | 64.5% | -11.2 pp |
| R+G | 53.0% | +0.3 pp |
| **R+N** | **47.5%** | **+5.8 pp** |

Only R+N passed (D50: +6.3 pp; A1-LLM: +8.0 pp). It tags about 30 concepts per JD against 6.

## Stage 2 (A2 final, delta against hand-set X2 = 0.850)

Nothing met the bar. Final fits on all labelled JDs, total delta [95% CI over JDs]:

| Labels | Learner | R+N | R (reproduction) |
|---|---|---|---|
| A1-LLM | logistic base (4) | +0.005 [-0.012, 0.022] | +0.006 |
| A1-LLM | logistic all (11) | -0.002 [-0.024, 0.019] | -0.004 |
| A1-LLM | boosting all (11) | -0.006 [-0.033, 0.021] | -0.001 |
| D50 | logistic base (4) | +0.000 [-0.009, 0.009] | -0.004 |
| D50 | logistic all (11) | -0.009 [-0.031, 0.015] | -0.010 |
| D50 | boosting all (11) | -0.003 [-0.031, 0.024] | -0.004 |

- All 12 notebook 18 Part 2 fits for R+N miss the bar. `targeted` is negative for every
  11-feature learner (-0.015 to -0.037).
- R reproduces notebooks 18-19 exactly, so the new code path matches the old one.
- The 11-feature curves are still rising on A1-LLM labels (+0.017 logistic, +0.043 boosting,
  25 to 70 JDs) but remain below hand-set X2 at 70 JDs (0.844, 0.840).
- **Decision rule:** the bar is met nowhere, so "more A1 LLM labels worth scoping" is **no**
  on this evidence.

## Diagnosis after the fact (exploratory; A2 labels read, nothing fitted)

- Coverage-only per-JD AUC on A2 final: R 0.693, G 0.609, N 0.649, R+G 0.692, **R+N 0.718**. The
  extraction helps the feature on its own; the full score does not move, so the combination is
  the suspect.
- MiniLM does extract terms the seed rules miss in CVs (`ci/cd` 26/37 vs 0, `crm` 27/33 vs 0,
  `api` 106/133 vs 4, `mssql` 14/16 vs 0). GLiNER finds fewer and its long phrases rarely match
  across JD and CV: of 320 R-no-overlap pairs it adds overlap to 2, MiniLM to 35 (71% Fit vs 44%).
  The added shared concepts are often generic ("english" in 12 pairs).
- 84% of MiniLM's concepts (52% of GLiNER's) are unknown to the vocabulary and carry a flat
  IDF weight of 1, so "english" counts as much as "kafka".
- Concept frequency over the 18.7k train CVs: no concept is in more than 30% of CVs (max
  `html`, 15-30%); generic words ("english", "testing") sit in the same 4-15% band as real
  skills (Python, SQL, JavaScript), so a frequency stoplist cannot separate them. 86% of
  MiniLM's distinct phrases occur in exactly one train CV.

## Follow-up

Weighting and dilution fixes: [coverage-weighting-variants](../2026-10-13-coverage-weighting-variants/README.md).

## Open questions (carried)

- GLiNER accuracy against gold spans is still unmeasured (no validated gold adapter).
- Whether exact-string matching (`ci/cd` vs `cicd`) is a larger loss than weighting is untested.
