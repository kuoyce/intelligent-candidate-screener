# Implementation progress

Status: in progress; no adoption or held-out score.

The new scorer preserves X2 at alpha zero and adds bounded, deduplicated JD concept coverage at other registered weights. IDF uses the first 2,000 sorted train-region CV IDs after excluding every development CV; this operational population is fixed before scoring. Unknown phrases receive weight 1. Experience remains the tested text rule.

The frozen GLiNER EDA contains 100 CVs and 100 JDs, retaining zero counts and exporting anonymous numerical summaries only. It describes extraction predictions, not accuracy or fit classification.

A ten-document GLiNER runtime pilot completed in 24.92 seconds including startup. Linear extrapolation is about 2.49 seconds per document, with substantial length and startup uncertainty; this raw-field pilot is not a text_v2 retrieval comparison. Record: output/extraction/entity-weighted-runtime-pilot.json.

Missing inherited role-model weights and keyed development pools currently prevent an X2 comparison. A guarded selected-classical restoration path verifies the committed configuration, labels, validation and calibration before publishing weights. First JD-side timing runs once in the background, logs/role-first-side.log, before budgeting the second side. No neural grid or final-set scoring is started.

Validation so far: 27 focused scorer, EDA and notebook tests pass; meaningful scorer mutations fail their invariant tests. Full suite and affected cold checks remain outstanding; model implementation is not committed or declared complete.

Open questions: full GLiNER development inference budget; independent training labels for a three-class classifier. Neither is resolved by this bounded ranking experiment.
