# Resume evidence and themes: preliminary execution plan

19 September 2026. The user authorises this plan and a new preliminary notebook.
Production extraction, model training and data restoration remain proposed. This extends
[the extraction design](../2026-09-13-entity-extraction/README.md) rather than replacing it.

## Basis and design

Proposal v3 §§4.2–4.7 separates extraction from retrieve-then-rank, starts with rules,
compares BERT on extraction labels, and makes clustering optional. Follow that architecture.
Notebook 08 already audits readiness; notebook 09 will demonstrate the **output contract**
for evidence and themes, not present manually specified examples as model predictions.

Useful first outputs: skills/tools, job titles, degrees/fields, and explicit activities.
Next: certifications, role-linked dates, projects and quantified outcomes. Themes such as
pipeline engineering, cloud operations or experimentation must cite activity evidence;
a tool mention alone cannot establish a theme, proficiency or seniority. Leadership means
an explicit activity such as mentoring, never inferred personality. Employer and school
names are context, not prestige scores. Exclude identity/protected traits from scoring.

An entity has entity_id, doc_id, source, text_version, start/end (half-open), surface,
label, canonical_id (nullable), method, extractor_version, confidence (nullable).
Validate text[start:end] == surface. Relations link evidence to roles and themes.
The illustration uses readable `JOB_TITLE` and generic `SKILL` labels; production
adapters must map these to the existing `TITLE` and `SKILL`/`KNOWLEDGE` scheme.
`ACTIVITY` is a proposed extension, without a claim of gold supervision.
Separate applied, learning, negated and unknown evidence; a missing requirement is
not evidenced, not absent ability. Required/preferred status comes from JD evidence.

## Delivery scope and acceptance

Create notebooks/09-resume-evidence-and-themes.ipynb with:
1. source-registry presence audit and metadata-only local parquet inventory;
2. information priorities and source roles;
3. synthetic, manually specified evidence with checked offsets;
4. a theme linked to activity evidence, plus learning/negation counterexamples;
5. a synthetic requirement/evidence table and explicit limitations;
6. executable integrity checks and staged implementation checklist.
No real document text is displayed. At most five records or 2,000 output characters per
cell. No extractor/model is defined in the notebook. No dependencies, labels or frozen
records change. Execute and inspect saved outputs; run repository validation and record
failures. Implementation is delegated under AGENTS.md. No branch/commit is requested
for this preliminary delivery; future production follows the approved-plan workflow.

## Production sequence and exit criteria

1. Restore prerequisites: B1 DataTurks and B2 SkillSpan; D1 ESCO/D2 data-jobs for
   vocabulary. Current local raw files are A1 and A2 only. Verify provenance before use.
2. Freeze inputs: repair inclusive B1 offsets with ambiguous repairs quarantined;
   mask PII without invalidating offsets; group duplicate documents before CV. Keep
   original Skills blobs as gold; automatically segmented skills are silver. Retain
   B2 published splits and independent BIO layers. Audit duplicate leakage. Protect
   A2's committed train/eval partition. Exclude the 44 C1/A1 overlaps if using C1/C2.
3. Implement extraction/ in the package: schema, source adapters, versioned aliases,
   phrase/rule baseline. Handle C++, C#, .NET, ambiguous Go/R, negation and learning.
   Synthetic regression invariants must fail against their guarded defects.
4. Evaluate exact span+label precision/recall/F1 and class support: B1 document-group
   CV mean/std, B2 per split/layer separately. Evaluate canonical mapping separately.
   A2 coverage is descriptive, not accuracy. Freeze rules before held-out evaluation.
5. Compare BERT only after baseline under identical evaluation; document subword
   alignment, long-document windows, latency, budget and useful-gain threshold.
6. Link roles, dates, activities and outcomes; merge overlapping employment periods
   before totals. Theme output includes supporting entity IDs and taxonomy version.
   Measure unsupported claims and theme precision/recall only with suitable labels;
   no new annotation campaign without revisiting the one-session budget.
7. Integrate with ranking/evidence UI. Measure ranking benefit separately, using
   Recall@10 and separate generic/targeted results. Precision retains its soft ceiling.
   A1 fit labels and deferred option D labels are not extraction gold.

## Open questions

- EE1: First production scope remains proposed: skills/tools, titles, qualifications.
- EE2: Data restoration and source-use scope need agreement before supervised work.
- EE3: BERT compute budget and minimum useful gain remain undecided.
- EE4: Theme taxonomy and independent evaluation budget remain undecided; examples
  are illustrative, not labelled data or performance evidence.
- EE5: Role/date linking policy remains undecided before experience totals are exposed.

## Validation record

Notebook 09 executed successfully with nbconvert; all eight code cells completed.
Reviewed every saved output: metadata and synthetic text only, under 2,000 characters
and at most five records per cell. Notebook schema validation and offset/reference
assertions passed. README links the notebook and this plan. No package code, dependencies,
source data or benchmark records changed. No production extractor was implemented.

`uv run pytest`: 191 passed, 2 failed in 110.26 seconds. Both failures are in
`test_baseline_golden.py` and reflect the baseline mismatch described below. Existing
notebook-thin tests only cover notebook 07; notebook 09 was reviewed directly.

Current check results (19 September): `data.verify --derived` fails its queue resume
check (200 already-judged pairs); `baselines.run --check` reproduces split/config/floor
but differs in vocabulary digest and 19 model fields, with environment reported unchanged.
These match the issues recorded for notebook 08; no frozen records were refreshed.
