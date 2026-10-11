# Merge repair and JD/CV entity inspection

## Authorization and current state

User explicitly approved resolving all merge conflicts on 11 October 2026.
The merge of `feat/stage2-rerank` into `feat/experience-years` was active; preserve
the user's staged incoming changes. All five conflicts are now resolved. This repair
continues the active merge rather than creating another branch before it ends.
The user also requested GLiNER confirmation, additional entities, a one-JD/one-CV
EDA view, and inspection of the incoming reranker. New entity scope and the
unfinished "check if" sentence remain pending clarification.

## Approved merge repair design

1. Preserve both rules APIs: local `extract_document` and context helpers, and
   incoming `SkillMatcher`, vocabulary loading/tokenization, without changing
   their extraction policies.
2. Preserve both metrics APIs: field/label/layer exact scoring and incoming
   strict/partial unit scoring with bootstrap intervals.
3. Combine package exports and keep both test suites. Resolve imports/docstrings
   cleanly; remove all conflict markers and prove both callers import/run.
4. The two extraction-metrics JSON files describe different campaigns and schemas.
   Keep the incoming evaluated skill-extraction record at its existing path for
   incoming callers. Preserve the local rule-inspection record under a distinct
   path and update only its consumers. Never combine figures across campaigns.
5. Review automatic merge results for semantic/API collisions, including notebook
   numbering and reconstruction docs; preserve incoming data/annotation records.
6. Run focused compatibility tests, then the full suite once with a retained log.
   Use the check runner for applicable records and record missing dependency/data
   blockers. Changes to cached-code provenance must be recorded, without changing
   historical measured figures or claiming historical inference was rerun.
7. Stage the five resolved paths and commit the reviewed merge when checks permit;
   no push requested. Record changes, validation and outstanding questions here.

## Findings from read-only inspection

- The five local neural labels, including EXPERIENCE_YEARS, are passed to GLiNER
  `predict_entities`; original field spans are preserved.
- Incoming stage 1 uses `extraction.experience` rules and a role-family classifier.
  It does not consume the GLiNER entity parquet.
- Incoming R1/R2/R3 rerankers train/score JD and CV text, not GLiNER entity fields.
- Incoming classification is binary Fit/No Fit (Good and Potential collapse into
  Fit); graded human labels are used for ordering. A single pair can illustrate
  evidence, but cannot demonstrate an accuracy improvement.

## Proposed next design, superseded by the expanded user request

Create a separate, thin notebook and package helper for one frozen JD/CV pair:
per-label counts, exact shared versus unmatched mention evidence, and experience
claim/requirement comparison with explicit extractor provenance. Keep raw real
text local, redact displays and limit saved previews; a synthetic authored pair
is an alternative if the user chooses it. No classifier training or held-out
tuning. Any performance claim requires a separately approved baseline-versus-
entity ablation across multiple JDs with uncertainty estimates.
The user's subsequent distribution/model request supersedes this one-pair scope;
see `plan/2026-10-11-entity-weighted-matching/README.md` for the concrete proposal.

## Open questions

- Merge repair authorization is resolved; no conflicts remain.
- Existing five types remain the scope of the new distribution request.
- Model budget and three-class supervision are tracked in the new model plan.
