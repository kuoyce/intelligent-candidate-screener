# Ceiling checks: experience extraction and must-have requirements

Status: **draft, to run later (owner, 17 Oct 2026).** Exploratory (D56). The bar is fixed now,
before anything is scored. The role model idea is dropped: the role oracle in error-analysis-2
showed no headroom (−0.002 / +0.012).

## Question

Two of the owner's ideas could still be worth building. Each gets a cheap check of the most it
could add **before** any build:

- **C1. Experience extraction.** The experience band is unknown on about 71% of pairs, and X2
  counts unknown as compatible. Would a *perfect* extractor help?
- **C2. Must-have vs nice-to-have.** The manual review notes say generic overlap (SQL, AWS) beats
  a missing core requirement. Does scoring against the JD's required section only help?

## C1: experience oracle (diagnostic, never a system)

- Replace `experience_compatible` with the rung-1a rule on the **metadata** bands (`jd_exp_band`
  from the JD `Exp Years`, `cv_exp_band` from the CV `Experience Years`). Metadata is forbidden at
  inference (D53) and allowed as an oracle, as the role oracle was.
- Two arms: unknown metadata band = compatible (as X2), and = incompatible.
- Also report how often the text extractor and the metadata band agree, where both are known.
- If the oracle does not reach the bar, a better extractor cannot either: stop C1.

## C2: must-have signal

1. **Prevalence first.** Share of JDs (D50 selection and A2) with a separable optional section
   ("nice to have", "will be a plus", "would be a plus", "optional", "bonus"). If under 30% of
   JDs, stop C2 there: the signal cannot move enough JDs.
2. **Split** each JD into required text and optional text by those headings (a rule in
   `extraction/`, tested on inline examples; written on D50 selection only).
3. **Two arms, fixed now:** (a) X2 with TF-IDF computed against the required text only;
   (b) X2 + 1.0 · coverage of required-section concepts. No weight is tuned.

## Measures and the bar (registered 17 Oct 2026)

- Per-JD AUC, paired delta against X2, JD-resampled 95% CI, `generic`, `targeted` and `total`
  together, and the same on `llm_free`.
- **Design on D50 selection (70 JDs); the bar is checked on D50 held-out (30 JDs)** for C2, and
  on A2 final for the C1 oracle (the metadata bands are the A2 fields).
- A lever **passes** if its paired total delta is at least **0.02 with a CI above 0**, and A2
  final's delta is not below 0. A pass earns a build plan (amendment); a fail is reported and
  stops there.
- Nothing is fitted or tuned on A2 final (D46).

## Cost

Minutes of compute each (features cached from the earlier runs). Code in
`evaluation/ceiling_checks.py` with tests, reported in a thin notebook. pytest only (D58); no
record checks unless a shared extractor changes.

## Open questions

1. C2's 30% prevalence floor is proposed; confirm or change before the run.
