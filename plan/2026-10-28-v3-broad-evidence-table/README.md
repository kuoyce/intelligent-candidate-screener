# Preserve v1 evidence coverage with v3 explanations

## Approved scope

The owner explicitly requests v1's broad evidence table alongside v3's clear
explanatory answers. Implement on the existing branch; preserve all staged and
owner edits. No training, downloads, model changes or branch switching.

## Design

Keep the current source-checked, JD-linked `comparison` as the explanation input.
For multi-candidate comparison turns, also obtain the original broad comparison
from the existing `PortalExtractiveService` (GLiNER-supported six categories,
no broad RoBERTa invocation). Present those broad rows first, followed by any
distinct JD-specific rows, clearly distinguished. Do not replace broad rows
with the parsed subset of JD criteria. Single-candidate/JD answers retain their
existing paths. Accepted Qwen prose and the checked explanatory fallback remain.

Use `comparison_rows` for the displayed table if appropriate; do not feed extra
broad passages into the explanation prompt unnecessarily. Reauthorize every
added citation against the current snapshot. Keep missing/unavailable/partial
states, qualifications and participant ownership; an entity mention is not
verified competence or skill-specific tenure. No duplicated table in saved JSON
when the existing comparison is already the broad table. Retain the 23,900-byte
storage guard: discard whole records with explicit partial provenance if needed,
never clip qualification text or leave stale source mappings.

## Verification

Delegate implementation and regression tests. Demonstrate failure before the
change. Verify a JD with only one recognized criterion still displays all six
broad categories and GLiNER-supported CV evidence, with the criterion alongside.
Cover explicit comparisons (including ranks 1 and 9), focused and explanatory
follow-ups, accepted/rejected/unavailable generation, table UI and source
ownership/storage bounds. Run focused tests, inspect affected check selection,
review the implementation and update README. Full-suite timing approval remains
pending: the previous full run took 289.68 seconds with documented existing
failures; do not repeat the long run without owner confirmation.

## Open questions

None for the approved implementation. General semantic answer quality and
gold-label entity accuracy remain outside this coverage repair.

## Implementation and review

V3 materializes broad display rows after synthesis. Each displayed cell and JD
requirement is reconstructed from currently authorized source quotations, keeping
full qualifiers and explicit missing/partial/unavailable states. The table shows
broad rows first, then distinct JD criterion or focused rows, with Scope and
Criterion columns. It retains one table when the explanatory rows already are
the broad table. Qwen inputs and accepted prose are unchanged; the original
checked comparison remains available to the explanatory fallback.

Review checked source ownership, absence of a broad QA call, prompt independence,
friendly participant labels, clear criterion scope, complete context and the
storage guard. Oversized records discard whole rows/citations/model mappings
with an explicit partial fallback. The owner's unrelated UI edits remain intact.

The read-only cached Ruby-role replay for ranks 1 and 9 now displays seven rows:
all six broad categories plus one JD criterion, with 13 citations (seven CV),
19,336 bytes and table visibility true (`logs/v3-broad-table-cache-replay.json`).
The replay uses source-revalidated cached GLiNER output and no Qwen generation;
it verifies display coverage, not live conversational quality or gold-label
entity accuracy. Affected checks select no committed records
(`logs/v3-broad-table-checks.log`). No branch switch, commit, model change,
cache mutation or running-service restart was performed.

## Regression evidence

Before implementation, five coverage/UI/storage invariants failed and one passed
(`logs/v3-broad-table-before.log`). After implementation, 170 focused service,
facts, broad-table and real Streamlit tests passed in 10.68 seconds
(`logs/v3-broad-table-final.log`), and 31 prose tests passed in 3.46 seconds
(`logs/v3-broad-table-prose.log`). A final explicit partial-table status change
passed its storage invariant (`logs/v3-broad-table-storage-final.log`). Covered
accepted/rejected/unavailable generation, numeric criteria, owner-matched
citations, tampered sources, no duplicated broad table, focused/why follow-ups,
complete qualifications and byte bounds. `git diff --check` passes. The long
full-suite run has not been repeated pending the owner confirmation documented
above; the existing broader failures are not claimed resolved.
