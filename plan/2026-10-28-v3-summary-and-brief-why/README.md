# Broad single-candidate summaries and brief comparative explanations

## Approved scope

The owner requests the single-candidate coverage fix and one concise answer line
for directed “why is candidate 1 better than candidate 2?” questions. Preserve
the current branch and all preceding metadata-first work. Do not change models,
thinking settings, generation budgets, ranks or Fit decisions.

## Design

Single-candidate good-fit/alignment summaries retain broad evidence before JD
criteria enrichment and receive broad matrix statuses before synthesis. Their
fallback must include source-supported strengths where present as well as
unresolved JD criteria, rather than only the first parsed criterion. Mentions
and source overlap describe documented relevance, never verified competence.
Keep qualifications, source ownership and missing-evidence caveats. Avoid extra
broad extraction calls and preserve focused QA and two-candidate table behavior.

Directed explanations use a brief matrix/evidence check: compare documented
strengths and counterstrengths, use employer priorities only if explicitly
established, and use experience quantities only if source-linked. Output one
concise line; never display the intermediate Reasoning Steps. “Not Found” is
not proof that a candidate lacks a skill. Preserve an explicit tradeoff or
unsupported-premise statement if an overall superiority claim is not justified.
The application fallback must also be concise when Qwen times out or is rejected.

## Validation

Delegate summary pipeline and brief explanation/prompt work separately. Prove
regressions fail before edits. Test a JD recognizing only MySQL but with other
CV strengths; single-candidate matrix and summary must retain those strengths.
Test directed forward/reverse advantages, missing priorities, absent evidence,
qualified credentials, genuine numeric durations and both model/fallback
outcomes. Retain comparison tables, source controls, citations, deadlines,
qualification boundaries and storage guard. Run focused tests and log results.
The prior live prompt trial exceeded the existing 60-second limit; no successful
live quality claim is implied. Full-suite/long research-run timing approval is
still pending. No runtime downloads, training, branch switch or commits.

## Open questions

None blocking this approved quick fix. Explicit employer weighting and general
semantic entailment remain unavailable unless supported by the supplied evidence.

## Implementation and review

Single-candidate alignment/relevant-work flows preserve the original broad
comparison before JD enrichment mutates it, reuse that result once for display
metadata and construct broad matrix/authorized summary sources before generation.
No single-candidate comparison table is shown. `summary_fallback` retains complete
short CV statements, checked summaries or source-exact entity mentions for longer
passages, qualifications and unresolved JD gaps. Mentions explicitly do not
establish proficiency or skill-specific tenure. It never authors a Fit verdict.

`brief_explanation` writes the application fallback for directed comparisons from
source-checked JD advantages, retaining forward/reverse direction, counterpoints,
qualified credentials, missing-evidence caveats and rank noncausality. Accepted
model conclusions normalize whitespace to one line without changing factual
clauses or their validation. Tables remain expandable for directed explanations.
The prompt requests up to three brief evidence checks (matrix, explicit priority,
source quantities where established) and one concise directed conclusion. Only
Conclusion is displayed. Prompt version is `hiring-desk-v3-summary-brief-6`.

Review checked source ownership, retained qualifiers, no invented numeric tenure
or priorities, no extra broad extraction call, one-line fallbacks, unchanged
generation limits and preservation of the preceding matrix/table behaviors.

## Verification

Seven summary invariants failed before the repair (`logs/v3-summary-before.log`);
eight brief explanation/prompt invariants failed before implementation
(`logs/v3-brief-why-before.log`). Earlier directed-format assertions were updated
to require the same completion/counterevidence caveats in one line, rather than
v2's multi-paragraph section labels. Accepted multiline model conclusions also
have an explicit whitespace-normalization invariant with unchanged source claims.

Final combined focused run: 187 passed in 12.76 seconds
(`logs/v3-summary-brief-final.log`). Whitespace checks pass. The read-only cached
Ruby-role replay with a simulated generation deadline now has seven matrix rows,
both CV source statements and the MySQL gap in a 373-character summary, plus an
89-character directed response with the comparison table retained
(`logs/v3-summary-brief-cached-replay.json`). This is not a new live Qwen quality
or latency measurement.

Check selection was inspected without execution
(`logs/v3-summary-brief-check-selection.json`); prompt-resource changes select
all research records conservatively. No long checks or full suite were run;
timing approval remains pending. No staging, commits, branch switches, downloads,
training, extraction-cache mutation, runtime configuration changes or service
restarts were performed.

## Approved concluding statement refinement

The owner requests a concluding statement connecting documented CV strengths
to the JD. Add a concise, conditional relevance conclusion to alignment
summaries when source-checked JD/CV evidence establishes a shared skill, role
or requirement. Do not upgrade a mention into acquired proficiency, confirm
missing MySQL experience, assert all requirements are met or author a formal
Fit decision. Preserve unresolved requirements and qualifications. No positive
conclusion for unrelated, unavailable, partial-only or negative-only evidence.
Register the conclusion with supporting source IDs so it can pass the existing
validator when generated, and include it at the end of the checked fallback.
Delegate implementation/tests and demonstrate the invariant fails first; update
the summary prompt and run focused checks. Keep current branch and staged work.

### Concluding statement implementation and verification

`summary_conclusion` requires source-owned substantive JD/CV overlap, current
alignment intent and one selected candidate. It returns a registered source
assertion: “The documented overlap with the JD makes Candidate 1 a promising
candidate for further review, subject to confirming the remaining requirements.”
Generic overlap and negative, learning, aspirational, incomplete, expired,
conditional, conflicting, partial or unavailable evidence cannot create it.
MySQL mentions remain distinct from established MySQL development experience.

The fallback appends this conditional statement last. Accepted model summaries
that omit it receive the same validated closing, with source mappings and
`summary_conclusion_composer=application` provenance. No duplicate is appended
when the model already returned the registered sentence. Prompt version is
`hiring-desk-v3-summary-conclusion-7`; the closing is restricted to single-person
alignment summaries, leaving comparison examples and one-line why replies intact.

Nine app invariants failed before the delegated repair
(`logs/v3-summary-conclusion-before.log`). The new accepted-model closing
invariant also fails with its guarded block removed in memory
(`logs/v3-accepted-summary-conclusion-before.log`). Final combined focused run:
201 passed in 12.95 seconds (`logs/v3-summary-conclusion-integration-final.log`).
Coverage includes the Rails/MySQL example, conclusion last, unconfirmed tenure,
source ownership, unrelated/negative-only evidence and application provenance.
Whitespace checks pass. No model run, training, download, branch switch, staging,
commit, service restart or long check/full-suite run was performed for this
refinement; timing approval for long verification remains pending.

### Approved readability refinement

The owner requests a single “Candidate 1's CV states” attribution for grouped
source quotations and a separate paragraph for the closing statement. Group
complete quotations by owner without clipping or rewording qualification text;
keep explicit gaps outside the quotation group. Separate registered conclusions
with a blank line in fallback and accepted model summaries (including a model
that already included the registered conclusion). Preserve one-line directed
comparison replies, source mappings, provenance and byte guards. Delegate the
small formatting change and test complete quote preservation and paragraph
separation; no model/runtime changes, branch switches or commits.

Readability implementation groups complete short quotations by owner with one
CV attribution and places the registered conclusion after a blank line. Evidence
gaps remain outside the quotation group. Accepted model summaries receive the
same final paragraph even if the registered conclusion was already present;
formatting is revalidated and application provenance changes only when content
was added. The guard measures the actual formatted text including newlines.
The three new readability invariants failed before the change
(`logs/v3-summary-readability-before.log`). Focused verification passed 78 tests;
the final combined run passed 204 tests in 14.66 seconds
(`logs/v3-summary-readability-final.log`). Complete in-progress/expired
qualification text, source order, conclusion uniqueness and directed one-line
behavior are covered. Diff whitespace checks pass. No prompt/model/runtime
change, branch switch, commit, staging or long verification run was performed.
