# Entity extraction notebooks on available local data

Date: 5 October 2026. Status: approved by the user on 5 October; implemented and validated.

## Requirement and relationship to the existing design

The user requests a plan and notebooks that carry out entity extraction using the
repository's existing `data/`, based on
[the 13 September design](../2026-09-13-entity-extraction/README.md).
Deliver real predictions rather than another synthetic-only demonstration. Preserve
notebooks 08 and 09. Reusable extraction and evaluation code belongs in
`src/candidate_screener/extraction/`; notebooks configure, invoke and report it.

## Local inventory and constraints

Read-only file inspection on 5 October found:

- `data/raw/djinni-cv/default-train-0000.parquet`;
- `data/raw/djinni-jd/default-train-0000.parquet`;
- `data/raw/fit/default-train-0000.parquet` and `default-test-0000.parquet`;
- `data/processed/fit/train.parquet` and `test.parquet`.

No local B1 DataTurks, B2 SkillSpan, D1 ESCO or D2 data-jobs files were found.
The committed `docs/data/manifests/a2-partition.csv` and its report exist; use them
without rebuilding. The report records seed 0 and evaluation fraction 0.25.
Raw and processed A1 fit labels are not extraction labels. A13 still fails.
Historical source-card counts do not imply current local presence.

The approved design expects ESCO-backed spaCy phrase/rule extraction. To make a
working local-data delivery possible, this plan proposes a deliberately limited,
versioned seed vocabulary and standard-library rules first. This is an explicit
deviation requiring approval: no claim of ESCO coverage, comprehensive NER, or
measured real-world accuracy. Do not modify the frozen baseline keyword helper.
No downloads, external inference, new annotation, BERT training or dependencies
are included in this delivery. Those remain later stages of the original plan.

## Proposed notebook sequence

| Notebook | Runs | Reports |
|---|---|---|
| `10-entity-extraction-inputs.ipynb` | Registry audit, parquet schema checks, A2 partition validation, deterministic sampling and source hashing | Actual available/missing inputs, eligible and sampled document counts, selected text fields, provenance and limitations |
| `11-rule-based-entity-extraction.ipynb` | Package extractor over real sampled CV/JD text; writes evidence records | Entity counts and document coverage by document type and label, context status counts, synthetic evidence examples with valid offsets |
| `12-entity-extraction-evaluation.ipynb` | Synthetic invariant probes, reproducibility/offset checks; labelled-source evaluation only when prerequisites pass | Operational checks separately from accuracy; explicit blocked gold-evaluation status with current local data; exact span/label metrics when supported |

Each notebook is executable in reading order and can rebuild its own required local
artifacts through package calls. It must not silently rely on kernel state from another
notebook. Missing mandatory A2 data or inconsistent partition raises a concise error;
missing optional NER data produces a visible status, never fabricated zero F1.

## Extraction design specification

### Input selection and leakage controls

Load only needed columns using PyArrow/Pandas. Resolve paths from project config.
Validate unique raw IDs, partition membership and the expected partition coverage
before sampling; refuse unknown or multiply assigned IDs. Use only A2 `train`
documents for this development delivery. Stable sort by document ID then select
with seed 0; propose 1,000 CVs and 1,000 JDs as the default CPU-sized run, configurable
up to the full training region. Report the realized size and sampling rule.

Use CV `Position`, `Moreinfo`, `Looking For`, `Highlights` and JD `Position`,
`Long Description`, subject to schema confirmation before implementation. Keep each
field independently identified so evidence never crosses artificial field boundaries.
Distinguish a profile title in `Position` from employment history; aspiration in
`Looking For` is not applied experience. Do not treat `Primary Keyword` or structured
experience categories as extracted entities or gold annotations.

### Rules and supported output

Start with explicit technology/tool mentions, job-title phrases and degree/field
phrases (`KNOWLEDGE`, `SKILL`, `TITLE`, `DEGREE`). Document the distinction between
skill/competence and tool/knowledge labels in the seed vocabulary. Use conservative
title and degree patterns, preserving the matched surface rather than claiming a
complete credential interpretation. Unknown or ambiguous vocabulary mappings remain
null; seed canonical IDs have their own namespace and never masquerade as ESCO IDs.

Punctuation-sensitive matching must support C++, C#, .NET and Node.js, longest-match
overlaps and aliases. Bare Go/R require conservative programming context; do not
extract ordinary verbs. Context rules record negated, learning/aspirational and
unqualified mention status using bounded clauses and source-field context. These
statuses are heuristic, not proof of proficiency; positive keyword occurrence alone
must never be labelled applied experience. No computed experience totals, institution
prestige features, inferred personal traits, employer extraction or theme generation
are proposed for this first implementation.

Versioned records include `entity_id`, `doc_id`, `source`, `doc_type`, `field`,
`text_version`, half-open `start`/`end`, `label`, `surface`, nullable `canonical_id`,
`method`, `extractor_version`, nullable `confidence`, and `context_status`.
Offsets are into immutable original field text; require `text[start:end] == surface`.
Rule confidence is null. Entity IDs are deterministic and unique within a run.
Keep role/date linking for the original plan's later stage.

### Storage, provenance and privacy

Write local sample manifests, entity parquet and run metadata under
`data/processed/extraction/`. Include input SHA-256, partition SHA-256, seed,
selected IDs/fields, sample size, vocabulary/configuration digest, package version
and extractor version. Writes are confined to these new artifacts; refuse incompatible
existing runs unless a new run ID is chosen. Source bytes and all evaluation/judgement
manifests remain untouched. Raw text is not copied into committed artifacts.

Notebook displays contain aggregates and synthetic examples only. Redaction and
`assert_clean` run inside the notebook before any synthetic text display. Real
entity surfaces and real excerpts stay local and are never printed. Redacting text
for display must not change the text against which offsets are validated. Committed
`output/extraction/` records contain configuration, input digests and aggregate results
only; saved outputs are capped at five records or 2,000 characters per cell.

### Evaluation contract

Unlabelled A2 supports coverage, runtime and extraction integrity only. Do not quote
precision, recall, F1, normalisation accuracy or ranking gains from it. Synthetic
tests establish named implementation invariants, not corpus accuracy.

Notebook 12 checks B1/B2 availability. With today's files it reports supervised
evaluation as blocked and names the missing prerequisites. If the sources are later
restored, adapters must pass provenance/schema checks before scoring: B1 inclusive
offset conversion, bounded unambiguous repair/quarantine, PII-safe handling and
document-group folds; B2 BIO legality, independent layers, original published splits
and duplicate leakage audit. Freeze vocabulary/rules before held-out scoring.
The implementation delivery need not build untestable adapters for absent data;
it must supply the exact span-and-label scorer with synthetic tests and a clear
adapter interface, documenting actual absent-source support as pending.

Do not score individual predicted skills against automatically segmented B1 blobs
as individual gold. Preserve original-span evaluation and explicitly state its
granularity mismatch with individual skill extraction. Report B1/B2 separately,
per class/layer with support and B1 document-group CV mean/std when executable.
Canonical mapping evaluation needs independently labelled mappings and stays blocked.

## Implementation and validation sequence after approval

1. Create a new branch from the current branch. Commit this finalized plan only;
   preserve unrelated existing edits and untracked files.
2. Delegate package schema/loader and extraction work, notebook work, and independent
   review with clear ownership under the repository's sub-agent requirement.
3. Implement the bounded offline pipeline, versioned seed vocabulary and local CLI.
   Add extraction tests on inline synthetic data; no local-data test dependency.
4. Demonstrate regression tests failing against controlled defects for punctuation,
   ambiguous Go, negation, offsets and train/eval leakage, then passing with the fix.
   Test stable sampling, duplicate IDs, null fields, overlapping matches, missing
   optional sources, source text preservation and exact span/label scoring.
5. Execute all three notebooks with `uv run jupyter nbconvert --execute`, validate
   notebook schemas and review every saved output for privacy and output limits.
6. Run `uv run pytest`, `uv run python -m candidate_screener.data.verify --derived`
   and `uv run python -m candidate_screener.baselines.run --check`. Record existing
   failures separately; do not refresh frozen records to hide them. Earlier plans
   recorded two golden-test failures and an already-judged queue check failure;
   current status must be measured again.
7. Update README and data reconstruction instructions for new local artifacts;
   add an implementation record here. Update catalog/source cards together if
   measured source findings change, preserving superseded figures in place.
8. Final independent review against this plan; report actual sample size, supported
   labels, executed checks, missing gold evaluation and any deviations.

## Acceptance criteria

- Notebook 11 produces nonempty, deterministic evidence records from actual local
  A2 training text, not manually supplied examples.
- Every saved entity has valid field-relative offsets and provenance; source bytes
  and frozen matching/annotation records are unchanged.
- C++, C#, .NET, ambiguous Go and negation/learning regressions are checked explicitly.
- All notebooks run with the current files, report missing NER prerequisites clearly,
  and contain only bounded aggregate/synthetic outputs.
- Extraction logic lives in the package and tests run without downloaded corpora.
- Coverage is labelled descriptive. No production-readiness or ranking-benefit claim
  is made from unlabelled documents.

## Open questions and proposed resolutions

| ID | Question | Proposal / gate |
|---|---|---|
| EN1 / EE1 | Which first entities? | Approve skills/tools, conservative titles and qualifications; defer relation-heavy entities |
| EN2 / EE2 | Use only current data or restore sources? | Use current A2 only; missing B1/B2/ESCO/D2 visibly block gold/vocabulary stages; no downloads |
| EN3 | May the first extractor use a seed vocabulary without spaCy/ESCO? | Explicit deviation proposed above; approval required |
| EN4 | Default extraction size? | 1,000 training CVs + 1,000 training JDs, configurable to full train; approval confirms this default |
| EE3 | BERT budget and minimum useful gain? | Deferred; no BERT in this delivery |
| EE4 | Certification/activity/theme supervision? | Deferred; no additional annotation or claims |
| EE5 | Role/date linking and overlapping employment? | Deferred; no computed durations |

The user's approval resolves EN1–EN4 using the proposed choices. Other original
open questions remain explicit later-stage gates. Implementation uses branch
`feat/entity-extraction-notebooks-2026-10-05`; the plan was committed before code work.
