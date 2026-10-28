# Comparison entity diagnostic

## Authorized scope and design

The owner requests investigation of GLiNER-medium evidence behind
“Compare the candidates 1 and 9.”, continuing the authorized v3 comparison
repair. Stay on the current branch and preserve staged work and the owner's
happy-flow document edits. Inspect cached extraction coverage and labels without
displaying raw applicant documents. Distinguish extraction failures from routing,
retrieval and rendering failures; do not train, download or tune a model.

Confirmed parsing defect: `_topic` returns `the and 9` for the reported wording,
but an empty topic for `Compare candidate 1 and 9.`. Repair the optional article
in participant scaffolding, retaining genuine criterion terms and numeric
criteria. Delegate parser repair and invariant tests, demonstrate failure before
the fix, run focused tests and inspect affected record checks. Review extraction
status and source/cache ownership separately; record measured diagnostics.

## Open questions

The cached review has complete extraction coverage, as measured below. Semantic
entity accuracy has not been evaluated against gold spans. Full-suite timing
approval from the preceding fixes remains pending.

## Measured findings

Read-only audit of the saved Ruby-role review, using the installed configuration
and cached records, found full coverage for JD (468 characters), candidate 1
(572 characters) and candidate 9 (984 characters). Source hashes were matched to
the current portal snapshot and records revalidated through `_normalise`.
Entity counts: JD 14 (TITLE 1, KNOWLEDGE 10, SKILL 2, DEGREE 1); candidate 1
30 (KNOWLEDGE 17, TITLE 4, SKILL 9); candidate 9 19 (EXPERIENCE_YEARS 6,
KNOWLEDGE 11, TITLE 2). Metadata-only audit is retained in
`logs/comparison-entity-cache-audit.json`. Full coverage and valid offsets are
not gold-label precision/recall; no extraction accuracy claim is made.

Saved comparisons with completed extraction nevertheless retrieved only the JD
and marked both CV cells not_found: a prior “What role does this JD require?”
was carried into the comparison criteria. This is a separate confirmed context
leak from the optional-article parsing bug. Repair only JD-question carryover,
retaining genuine comparison criteria and numeric terms. Recover recognizable
legacy frames so the user need not delete saved history.

Focused testing additionally exposed a fallback runtime error: mixed loop
implementations reference undefined `owners` and `choices`. Restore per-cell
selection of checked statements while preserving the owner's raw JD-source
presentation, rather than allowing rejected Qwen output to raise an exception.
No model training, new model inference, downloads or cache changes are required.

## Implementation and verification

Delegated routing tests failed 11 times before repair
(`logs/v3-comparison-routing-before.log`). The optional participant article is
now removed as scaffolding; a fresh explicit comparison discards JD-only topic
carryover, including recognizable legacy frames. Genuine candidate criteria and
numeric criteria remain covered. All 56 service tests passed in isolation.

Restored the previously tested per-cell fallback loop after reproducing 21
undefined-variable failures (`logs/v3-comparison-routing-focused.log`). Two
remaining failures exposed removed Bachelor/Master possessive normalization;
restoring that narrow grammatical normalization keeps completion qualifiers.
Final focused verification: 145 passed in 10.85 seconds across conversation
service, v3, prose and v3 Streamlit tests
(`logs/v3-routing-fallback-repair-final.log`). The affected check runner selects
no records (`logs/v3-comparison-routing-checks.log`); diff whitespace checks pass.

A read-only replay against the current Ruby-role snapshot and revalidated cache
uses the exact question and a prior JD-role frame. It clears the stale criterion,
retrieves four CV citations and two JD citations, and sets table visibility true
(`logs/comparison-entity-replay.json`). This tests routing and evidence delivery
without invoking Qwen or re-extracting entities. It does not establish live prose
quality, exhaustive comparison criteria, or semantic entity precision/recall.
No staged work was overwritten, no branch switched, no service restarted, no
cache mutated and no new dataset acquired. Full-suite timing approval remains
pending; prior broader suite failures are recorded in the preceding plans.
