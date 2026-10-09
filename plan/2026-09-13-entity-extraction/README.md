# Entity extraction: evidence first

Date: 13 September 2026. Status: preliminary planning and notebook explicitly requested;
production implementation and model training remain proposed, pending plan approval.

## Basis and scope

Proposal v3 §§4.3–4.5 separates information extraction from matching. Follow that separation:
extract auditable evidence first, and evaluate any ranking benefit in a separate experiment.
This plan extends the unfinished DataTurks repair (3.1) and vocabulary (3.5) work in
[the derived-artefacts implementation record](../2026-08-23-derived-artefacts/05-implementation.md).
Notebook 08 follows notebooks 04 (extraction sources) and 06 (vocabulary); it is a readiness
and exploration notebook, not a trained extractor or an extraction quality benchmark.

## Useful information to extract

| Priority | Information | Screening use | Evidence and limitation |
|---|---|---|---|
| First | Skills, languages, tools, frameworks | Compare explicit JD requirements with resume evidence | Preserve exact mention and context; a mention does not prove proficiency |
| First | Job titles and core activities | Explain relevant work: built pipelines, deployed services, analysed experiments | Titles alone do not establish seniority or capability |
| First | Degrees and fields of study | Show explicit qualifications when relevant to the JD | B1 Degree/College labels support part of this; institution prestige is not a scoring feature |
| Next | Certifications | Surface named credentials | No dedicated gold certification class in the adopted sources; do not imply certification validity |
| Next | Employment date ranges and stated durations | Explain relevant experience and recency | Link dates to roles; merge overlapping intervals before computing totals; distinguish stated from computed years |
| Next | Projects, responsibilities and quantified outcomes | Ground explanations in what someone actually did | Requires relations/events beyond flat NER; metrics need their unit, activity and context |
| Later | Explicit domain themes and behavioural evidence | Group evidence into data engineering, backend, cloud, etc. | Themes are derived summaries with supporting spans, not additional gold entities or inferred personality |

Do not turn names, contacts, age, gender, nationality or inferred protected traits into ranking
features. Keep employer/college mentions only as evidence context. Missing evidence means
“not evidenced”, not “candidate lacks skill”. A resume mention is not a JD requirement;
required/preferred status comes from the actual JD, not an ESCO occupation relation.

## Data readiness and source roles

Local inspection on this date found raw `fit`, `djinni-cv` and `djinni-jd` parquet files only.
`data/interim` and `data/vocab` are empty. This is local availability, not a revision of the
historical acquisition measurements in `docs/data/profile-metrics.json` or the source cards.

- B1 DataTurks: resume-native entity supervision; restore before repair and evaluation.
  Inclusive ends, broken offsets and multi-skill blobs need explicit treatment. See its card.
- B2 SkillSpan: skill/knowledge supervision on JDs; retain its independent BIO layers and
  published split. Report JD results separately from resume results.
- D1 ESCO plus D2 data-jobs: proposed canonical vocabulary and tool aliases, not gold spans.
- A2 Djinni: currently available, unlabelled for NER. Respect the committed train/eval
  partition before inspecting examples to develop rules. Fit judgements are not NER labels.
- A1: matching benchmark, not extraction gold. A13 fails; do not validate extraction with
  fit labels or use option D's deferred judge output.
- C1/C2: later parsing checks only after restoring data and the 44-document contamination
  exclusion. C3 ResumeAtlas is unsuitable for evidence offsets. B3 stays in reserve.

## Design specification

Proposed reusable package: `src/candidate_screener/extraction/`. Notebooks call package code;
they do not define models. Preserve immutable source text locally and explicitly version any
normalised text. An entity record contains `doc_id`, `source`, `text_version`, `start`, `end`
(half-open), `label`, `surface`, nullable `canonical_id`, `method`, `extractor_version` and
nullable `confidence`. Require `text[start:end] == surface`. Do not invent probability scores
for rules. Context relations reference entity IDs and keep role/date/skill associations explicit.

Initial mapping follows notebook 04: SKILL and KNOWLEDGE, TITLE (Designation), ORG
(Companies worked at), DEGREE, EDU (College Name), YEARS. Certifications, activities and
theme relations are extensions with no claim of existing gold supervision. B1's Skills blobs
cannot be scored as individual skill gold after automatic segmentation: retain original span
evaluation and label segmentation as silver until independently validated.

Keep annotation redaction inside the display path. The existing regex redactor does not remove
all names and changes offsets; never apply it and then reuse original coordinates. For B1,
mask annotated PII with equal-length replacements or maintain an explicit offset map, then
validate surviving spans. Commit aggregates and synthetic examples only.

## Execution sequence and acceptance criteria

1. **Readiness (this delivery).** Execute notebook 08: registry-based presence checks,
   schema/count summaries of local parquet, and an existing-keyword-helper demonstration on
   synthetic text. Separate recorded historical counts from current observations. No downloads,
   new labels, training, source mutations or benchmark changes are needed.
2. **Restore prerequisites after plan approval.** Fetch B1/B2/D2 via the registered fetch
   command; restore ESCO via the documented manual process. Verify files and provenance.
   Resolve licence/usage and compute scope questions before adoption beyond existing decisions.
3. **Repair and freeze inputs.** Implement B1 repair with bounded offset correction, ambiguous
   cases quarantined, and a reasoned repair log. Deduplicate/group documents before fixed
   document-level CV. Validate PII handling, offsets, token alignment and BIO legality. Freeze
   splits before vocabulary tuning; retain B2 split and audit duplicate leakage across it.
4. **Build the classical extractor.** Implement spaCy phrase/rule matching in the package,
   version the ESCO/tool alias map, preserve punctuation (C++, C#, .NET), and handle ambiguous
   aliases (Go, R, AI Engineer), negation and aspirational mentions. Add spaCy with `uv add`
   only in this approved implementation phase. Keep the frozen matching helper unchanged.
5. **Evaluate and inspect errors.** Report exact entity span-and-label precision/recall/F1,
   per-class support and document-group CV mean/std for B1; B2 results by BIO layer and split.
   Report normalisation accuracy separately on independently labelled mappings. Coverage on
   unlabelled A2 is descriptive, never accuracy. Log boundary, omission, alias and context errors.
6. **Compare BERT NER only after baseline.** Reuse the same frozen evaluation protocol;
   choose settings on training/validation only. Handle subword alignment and long-document
   windows explicitly. Compare F1, latency and effort; agree a minimum useful gain before tuning.
7. **Integrate evidence and themes.** Add role-linked skills, durations and supported thematic
   summaries only after extraction validation. Assess ranking changes separately on the frozen
   pools and per in-domain stratum; Recall@10 remains the headline and precision carries its
   soft judged-supply ceiling. No extra annotation wave without re-budgeting D25.

Future implementation follows AGENTS.md: approved plan, branch and committed plan, sub-agent
implementation, review, `uv run pytest`, `uv run python -m candidate_screener.data.verify
--derived`, and `uv run python -m candidate_screener.baselines.run --check` where runnable.
New regression tests must demonstrably fail against the guarded defect. Record missing-data
check failures rather than regenerating benchmark records to make checks pass.

## Open questions

| ID | Question | Status / gate |
|---|---|---|
| EE1 | Confirm skills/tools, titles and qualifications as first implementation scope? | Proposed; does not block preliminary notebook |
| EE2 | Restore B1/B2/D2 and ESCO here, and resolve any source use questions? | Required before supervised extraction/vocabulary work |
| EE3 | What compute/time budget and minimum useful gain justify BERT? | Required before advanced comparison |
| EE4 | Can certifications, activities and theme relations be evaluated within the existing annotation budget? | Unresolved; descriptive only until supervision is agreed |
| EE5 | Which role/date linking and overlapping-employment policy should the UI expose? | Required before computed experience is used |

## Preliminary delivery record

Notebook construction is delegated as required by the repository workflow. The user's request
authorises this preliminary notebook; it does not approve the future production plan. No branch
or commit is made before that approval.

Notebook 08 executed successfully with `uv run jupyter nbconvert --to notebook --execute
--inplace notebooks/08-entity-extraction-readiness.ipynb`; all nine code cells completed.
Reviewed saved outputs: compact metadata/aggregates and synthetic examples only, each below
2,000 characters. The audit reports the four missing prerequisites and reads no real document
values. Synthetic probes reproduce the helper's C++, Go and negation limitations. README links
the notebook and this plan. `git diff --check` passed. No package code or dependencies changed.

Required checks exposed existing issues without changing package code or benchmark records:

- `uv run pytest`: 191 passed, 2 failed (126.38 seconds). Both failures are in
  `test_baseline_golden.py` and arise from the same baseline-record mismatch below.
- `data.verify --derived`: one failed check — the committed queue includes 200 already-judged
  pairs. Other available checks passed, with missing local dispatch checks explicitly skipped.
- `baselines.run --check`: split, configuration and floor reproduced; vocabulary SHA-256 and
  19 model fields differed. The check reported an unchanged environment. Cause is not established
  by this preliminary extraction task; no baseline record was refreshed.

These failures are separate from notebook execution and remain unresolved repository issues.
