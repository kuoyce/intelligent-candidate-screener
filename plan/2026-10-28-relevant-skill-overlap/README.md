# Source-owned relevant skill overlap

## Approved scope and diagnosis

The owner requests JD skill overlap across the current candidates rather than
the generic single-candidate missing-skills reply. The saved exact question
“Any relevant skills for any of these candidates?” inherited candidate 1 only,
became a literal criterion and timed out. Continue on the current branch with
all prior work preserved; no weights, training, downloads or ranking changes.

## Design

Add a v3 relevant-skills/skillset route. Explicit ranks remain scoped; “any/all
of these candidates” means the current authorized review's candidate set, not
a stale single selection. Scoped plural follow-ups can retain an active pair.
Never search across another JD or workspace. Unknown explicit references ask
for clarification rather than widening scope.

Use current source-validated GLiNER SKILL/KNOWLEDGE mentions from the JD and each
requested CV. Normalize case/whitespace only; no fuzzy Java/JavaScript matches
or invented synonym/tenure/proficiency claims. Compare mentions across extraction
labels because technologies may be labelled knowledge in one source and skill
in another. Reauthorize exact spans, hashes, owner, qualification context and
current source on each read. Qualified/negative evidence cannot become an
unqualified positive match. Preserve pending, partial and unavailable states.

Write concise application-owned statements about the overlap, with citations
to both JD and CV, explicit participant coverage and no-absence caveats. This
lookup does not require generation and supports the full current top-ten review
without raising Qwen's three-participant budget. Metadata, source evidence and
the answer stay within the existing storage bound; omit whole matches with an
explicit partial state if necessary. Missing/unfinished extraction never means
the candidate lacks the skill. Preserve ordinary comparisons, focused QA,
single-candidate summaries and one-line directed explanations.

## Validation

Delegate entity overlap implementation and UI/route invariant tests separately.
Prove the reported question fails before repair. Cover all-review vs explicit
rank/pair resolution, cross-label overlap, qualified/negative/partial results,
foreign/stale/tampered spans, exact word matching, source citations and byte
bounds using invented documents. Run focused tests and a metadata-only cached
replay; no new live model inference is required. Inspect record selection
without execution; long full-suite/research timing approval remains pending.

## Open questions

None blocking the scoped lookup. Absence of an extracted mention and semantic
skill proficiency are not established by this feature.
