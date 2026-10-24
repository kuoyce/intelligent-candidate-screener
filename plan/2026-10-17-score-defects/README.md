# Score defects found by the manual review

Status: **notebook version run 17 Oct 2026 at the owner's direction; implementation not decided.** Exploratory (D56). Follows
[error-analysis-2](../2026-10-13-error-analysis-2/README.md) step 4, which named three concrete
defects in the hand-set X2 score (`2·role + experience + 0.999·TF-IDF`) in about 12 of 45
reviewed mistakes.

## Question

Does fixing each defect lift ranking quality, or are they too rare to matter? A fix is cheap; the
question is whether it is worth keeping.

## The three defects (from the review notes)

| ID | Defect | Seen in | Where it lives |
|---|---|---|---|
| F1 | **Negation counted as a match.** "No Angular/Vue projects" scores as Angular evidence | 3 cases, one CV | TF-IDF text and the rules/concept matchers |
| F2 | **No professional experience scored as compatible.** A CV with no work history passes the experience term | 5 cases | `experience_compatible` (and the unknown-band rule) |
| F3 | **Role model misses a same-role pair.** JD and Fit CV are the same role, `role_match` says out of role | 3-4 cases | the role-family model |

## Steps

1. **Measure the prevalence first** (no scoring): how many A2 and D50 CVs contain a negated skill
   phrase (F1), state no professional experience (F2). If a defect touches under 1% of pairs, stop
   it there and report the count.
2. **F1:** drop skill mentions inside a negation window ("no", "not", "without", "never") before
   TF-IDF and concept matching. A rule in `extraction/`, with tests.
3. **F2:** detect "no (commercial | professional | work) experience" and the absence of any dated
   role, and score it as experience-incompatible when the JD asks for years. A rule in
   `extraction/experience.py`, with tests.
4. **F3:** diagnostic only in this plan: list the role model's errors on the reviewed cases. A fix
   to the role model is a training change and gets its own plan if F3 is worth it.
5. Score X2 with F1, F2, F1+F2.

## Measures and the rule (proposed; **deferred by the owner** until after the notebook, so the results are exploratory, not registered)

- Per-JD AUC, paired delta against unchanged X2, JD-bootstrap CI, `generic`, `targeted` and
  `total` together.
- **Design on the D50 selection JDs (70); the bar is checked on the D50 held-out JDs (30)**, then
  confirmed on A2 final (scored on, never tuned on, D46).
- A fix is **kept** if its paired total delta on D50 held-out is at least **0.02 with a CI above
  0** and A2 final's delta is not below 0. Same bar as error-analysis-2.
- The negation window and the F2 phrase list are fixed on D50 selection only.

## Constraints

- Text only (D53); no new labels, no dispatch (D42); the reviewed cases and override flags are
  not used to tune or fit anything (D57).
- Minutes of compute. Warm checks for the records the extractors feed; cold (D55) before marking
  the plan complete, since the extractors feed the frozen rungs.

## Open questions

1. **Does a fix change the frozen rungs?** F1 and F2 alter shared extractors used by the
   selected pipeline (D48). Assumed: the fixes go behind a flag, default off, so no committed
   record moves. Please confirm.
2. **F2 for unknown experience:** what does X2 do today when the CV band is unknown (0 or 1)?
   To be checked in step 1; it decides whether F2 is a new rule or a change to the unknown band.
3. **Is the 1% prevalence floor right?** Proposed; please confirm or change.

## Owner direction, 17 Oct 2026

Run the fixes as a notebook first, view the results, and decide later whether to implement them.
Prevalence floor and bar: decided after the notebook. Open questions 1 and 3 are therefore moot
for now: nothing touches the pipeline (the fixes live in `evaluation/score_defects.py` only).
Open question 2 answered: X2 counts an unknown band as **compatible** (`unknown_compatible=True`,
X1's registered variant), which is why a no-experience CV passes.

## Results (record `output/extraction/score-defects.json`, [notebook 22](../../notebooks/22-score-defect-fixes.ipynb))

Paired ΔAUC against X2 (A2 final: generic / targeted / total; D50: total):

| Fix | Prevalence (A2) | D50 selection | D50 held-out | A2 final |
|---|---|--:|--:|---|
| F1 negation | 6.7% of CVs | +0.000 | −0.005 | −0.005 / 0.000 / −0.003 |
| F2 no experience | 0.5% of CVs | +0.001 | +0.004 | +0.003 / 0.000 / +0.002 |
| F2b unknown = 0 | 71% of pairs | −0.017 | −0.009 | −0.010 / **−0.041** / −0.020 |
| F1 + F2 | | +0.000 | −0.000 | −0.001 / 0.000 / −0.001 |

None helps. F2 is correct but rare and never moved a JD down; F1 is imprecise and neutral; F2b
hurts (it punishes missing data). F3: the role model is confident on the reviewed misses (median
0.04 on the six A-tagged Fit CVs), so a fix is retraining. The parked idea of raising the
experience and coverage weights is not run here.
