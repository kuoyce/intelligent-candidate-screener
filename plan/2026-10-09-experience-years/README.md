# Years-of-experience extraction

Status: approved by the user on 9 October 2026; implemented and validated.
Branch: `feat/experience-years`; plan committed before implementation as `53e1ec9`.

## Requirement and scope

Add explicit years-of-experience mentions to notebook 13's GLiNER experiment
alongside KNOWLEDGE, SKILL, TITLE and DEGREE. This provides evidence for future
CV/JD fit classification; this change does not train or change a fit classifier.

## Design

- Add `EXPERIENCE_YEARS` to the shared entity schema and a descriptive GLiNER
  prompt for years of professional experience, including ranges and minimums.
- Preserve exact source spans, field, document type and raw model confidence.
  Examples include `5 years of experience`, `3–5 years` and `at least 2 years`.
- Extract explicit duration mentions only. Do not infer total experience from
  employment dates, sum overlapping jobs, or treat age, graduation years and
  course duration as professional experience.
- Keep CV experience claims distinguishable from JD requirements through their
  existing document type and source context. Do not silently turn `5+` into an
  exact value or choose a total from skill-specific mentions.
- Update adapter version/provenance so the five-label experiment cannot reuse
  four-label cached inference. Keep the rules comparator's four-label scope
  explicit; it has no years-of-experience extractor in this change.
- Update notebook 13's explanation and synthetic example, and README. Inspect
  analysis consumers for fixed four-label assumptions and accommodate the new
  label where necessary without rewriting historical results.

## Implementation and validation

After approval, create a branch and commit this plan. Use a sub-agent for the
implementation, then review its changes. Add a regression test shown to fail
against the old adapter/schema, covering EXPERIENCE_YEARS span preservation.
Run `uv run pytest`, applicable committed-record checks, and a five-CV/five-JD
smoke run before executing notebook 13's bounded 100-CV/100-JD run. Execute the
notebook before committing; retain PII redaction and output caps. Record runtime,
results and any deviations in an implementation record.

## Open questions and limitations

- Resolved by approval: interpret "number of years" as explicit professional
  experience duration, rather than inferred employment tenure.
- Numerical normalization and CV/JD duration comparison are deferred. Extracted
  spans alone do not establish that a candidate meets a requirement.
- No claim of extraction accuracy or improved fit classification without labelled
  evaluation. A shortfall in years should be assessed with profession and career
  stage; the annotation guide permits Potential Fit for small minimum-year gaps.
