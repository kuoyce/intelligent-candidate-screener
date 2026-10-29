# Evidence-led alignment summaries

29 October 2026. Owner-requested correction to the approved v3 presentation flow.

## Requirement

For “Summarise why candidate 1 is a good fit to this JD”, lead with the supported CV statements and finish with a promising-candidate conclusion instead of a repeated list of “I can't confirm” messages for generic extraction categories.

The supplied failure example contains quoted manual-QA activities, Jira bug reporting and test-case design, followed by unsupported-category boilerplate. It demonstrates a presentation failure; the example alone does not establish the actual JD's requirements.

## Design

1. Preserve complete source-owned CV quotations, including material qualifications and negation.
2. Do not turn generic extraction categories (Knowledge, Skill, Title, Degree, Experience Years, Certification) into six apparent JD requirements. Omit these repetitive category-only missing messages from a broad alignment summary.
3. Preserve unresolved concrete JD requirements and source-unavailable/partial limitations when relevant. Place them before the closing statement.
4. Retain the qualified meaning with the owner's requested closing: “Hence, Candidate 1 is a promising candidate for further review, subject to confirming the remaining requirements.” Its purpose is further review, not a new Fit classification or hiring decision; source-owned JD/CV overlap remains required.
5. Permit verified literal JD/CV overlap to support that conclusion when entity mentions are absent. Match only meaningful source-owned content; generic role words, unsupported inferences, negated/qualified-only evidence and foreign candidate facts cannot establish the closing.
6. Use the same registered conclusion for accepted generation and deterministic fallback. No model training, runtime migration or UI edits are required.

## Validation

Use invented JD/CV content reproducing the quoted QA activities. Show the missing closing and boilerplate regressions failing before the correction. Test absent entity mentions, concrete requirement gaps, negation, qualifiers, unrelated JD/CV evidence and correct source ownership. Run the affected summary/context/v3 tests and inspect the registered record-check selection.

Full-suite time approval remains pending from the evaluation implementation request; do not imply focused tests constitute a completed full-suite run.

## Open questions and limits

- No additional clarification is needed for the presentation change. The actual example's JD text was not supplied, so a positive closing must still depend on verified overlap at runtime.
- No broad model-quality claim follows from this correction. Preserve the independent evaluation gates.
