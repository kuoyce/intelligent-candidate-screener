# Validate saved entity extractions against source data

Status: proposed; awaiting approval. Scope: the saved GLiNER outputs discussed in
notebooks 13–15, with rules as a comparator. Preserve unrelated working changes.

## Design and deliverables

1. Inventory saved runs and their label prompts, source/sample hashes and completion.
   Keep the historical four-label run and later experience-enabled runs separate;
   never silently substitute a newer model output. Report which run is validated.
2. Reconstruct the exact source fields using each run's sample manifest. Check every
   saved entity's document/field identity, offset bounds and exact surface slice,
   duplicates, score validity and field coverage. Source agreement establishes
   traceability, not whether the label is correct. Report mismatches explicitly;
   never repair source text or regenerate predictions silently.
3. Create a thin notebook and reusable validation helpers that show a saved entity
   next to bounded, redacted real source context, with label, score, field and offsets.
   Include deterministic examples per label and CV/JD, disagreements with rules and
   suspicious degree/title cases. Cap displays at five rows/2,000 text characters.
   No full resumes or new human annotation campaign.
4. Inventory existing DataTurks/SkillSpan gold and existing evaluation records.
   Score GLiNER only where matching gold and compatible labels exist, using the
   frozen prompt/threshold, without fitting or tuning. Report strict and partial
   span precision/recall/F1, unit counts and document/sentence bootstrap CIs.
   Describe dataset boundary conventions and partial label coverage. Existing
   learned-tagger scores must not be attributed to GLiNER. If assets or compatible
   gold are unavailable, mark this portion unavailable rather than infer accuracy.
5. Save aggregate validation records and an implementation report; link the executed
   notebook from documentation. Real contextual exports remain in ignored data.

## Implementation and verification

After approval, create a dedicated branch and commit this plan without including
unrelated modifications. Delegate source-integrity helpers and notebook/reporting,
then conduct independent review. Use uv only. Test synthetic offset corruption,
wrong fields, incomplete coverage and misleading gold-label mappings; demonstrate
regression tests fail against their guarded defects. Execute the notebook, run the
full suite once with a saved log, and use the checks runner for affected records.
Cold checks are required before marking the plan complete (D55); estimate runtime
before any model-inference job. No training, threshold search, final-set tuning,
annotation dispatch or modification of collected labels.

## Open questions

- V1: Does the user approve this source inspection plus available-gold evaluation
  scope? Pending. It includes inference on existing labelled extraction datasets
  only if the required assets are already available.
- V2: Which saved run should be the primary report? Proposed: the run referenced
  by notebook 13's committed transformer record; inventory later runs separately.
- V3: Are compatible gold assets locally present and usable for this frozen GLiNER
  configuration? Resolve through inventory, without treating old availability
  entries in a historical metrics file as a current filesystem check.
- V4: Djinni Fit labels do not label entity spans. Entity-label accuracy and missing
  entities in Djinni cannot be established from text slices or Fit labels alone.
  New annotation remains outside this plan under D42.
