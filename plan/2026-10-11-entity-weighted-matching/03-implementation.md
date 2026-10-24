# Implementation progress

Status: in progress; no adoption or held-out score.

The new scorer preserves X2 at alpha zero and adds bounded, deduplicated JD concept coverage at other registered weights. IDF uses the first 2,000 sorted train-region CV IDs after excluding every development CV; this operational population is fixed before scoring. Unknown phrases receive weight 1. Experience remains the tested text rule.

The frozen GLiNER EDA contains 100 CVs and 100 JDs, retaining zero counts and exporting anonymous numerical summaries only. It describes extraction predictions, not accuracy or fit classification.

A ten-document GLiNER runtime pilot completed in 24.92 seconds including startup. Linear extrapolation is about 2.49 seconds per document, with substantial length and startup uncertainty; this raw-field pilot is not a text_v2 retrieval comparison. Record: output/extraction/entity-weighted-runtime-pilot.json.

Missing inherited role-model weights and keyed development pools currently prevent an X2 comparison. A guarded selected-classical restoration path verifies the committed configuration, labels, validation and calibration before publishing weights. First JD-side timing runs once in the background, logs/role-first-side.log, before budgeting the second side. No neural grid or final-set scoring is started.

Validation so far: 27 focused scorer, EDA and notebook tests pass; meaningful scorer mutations fail their invariant tests. Full suite and affected cold checks remain outstanding; model implementation is not committed or declared complete.

Open questions: full GLiNER development inference budget; independent training labels for a three-class classifier. Neither is resolved by this bounded ranking experiment.

## Record regenerated (22 Oct 2026, on the owner's instruction)

The committed `output/matching/entity-weighted-dev.json` (11 Oct 2026, `714fdcb`) had status
`blocked` with `missing_prerequisites: role_latest`: the machine that wrote it lacked the role-family
model and the dev pools. Both exist on the owner's machine, so `entity_weighted --check` ran the pilot
and failed against the placeholder (cold run of 22 Oct, `logs/d55_cold_rest.log`). The matching code
has also changed since 11 Oct (its dependency digest differs), so a rerun anywhere would now differ.

Regenerated with `uv run python -m candidate_screener.evaluation.entity_weighted`
(`logs/entity_weighted_write.log`); `checks --warm entity_weighted` then passes.

- Status `pilot_complete`, **not adopted** (the config forbids adoption from this four-query pilot).
- 4 JDs, 2 per dev split; `selected_alpha_exploratory` = 0.5. Recall deltas against X2 are 0 to
  +0.045 at n = 2 JDs per split: no evidence either way.
- The GLiNER comparison is still not run ("complete registered pool extraction cache required").
- No other record or plan reads this file; the reported system (D63, D64) does not use it.

The original author may want to rerun it on their machine to confirm; the check now compares
against this record.
