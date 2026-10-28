# Metadata-first comparison context

## Approved scope

The owner requests pipeline inversion: calculate the broad six-category table
and checked relations before Qwen; inject a compact Evidence Matrix immediately
before Selected Passages in the final prompt. Keep the current branch, staged
work, original redaction/ownership rules and broad table UI. No model downloads,
training, change of ranking, thinking-mode enablement or larger runtime budgets.

## Design

1. Materialize and reauthorize broad plus focused/JD rows before generation.
   Build the matrix from exactly those displayed rows. Preserve explanation
   inputs without duplicating full CVs or all broad passages in the prompt.
2. Use source-derived flags, including documented/mentioned, qualified, negative,
   conflicting, Not Found, partial and unavailable. Do not invent High/Medium
   ratings, external verification or skill-specific tenure from entity mentions.
   Not Found means not established in reviewed evidence, never absence.
3. Pass compact matrix entries and broad checked relations in the context bundle;
   preserve owners, current source references and qualification constraints. Add
   only authorized source-backed paraphrases to allowed model claims. History
   remains discourse context and never sets matrix flags.
4. Implement the requested prompt layout: HR Data Analyst, examples, Job Priority,
   Evidence Matrix, Selected Passages, Reasoning Steps and Conclusion. Keep the
   illustrative Python/cloud example with corrected missing-evidence language;
   add a qualification/status example. Reasoning Steps are a brief evidence-check
   summary, not instructions to expose private chain of thought. Only Conclusion
   is presented/persisted as the final answer; existing prose remains readable.
5. Enforce source/matrix consistency on final claims. Matrix metadata cannot
   create new facts or override source qualifiers. Handle malformed structured
   replies explicitly. Keep shared deadlines, token limits, provenance and
   storage guards; drop history before evidence, and fallback on context overflow.

## Delivery and validation

Delegate pipeline/matrix and prompt work to separate agents with separate files.
Show invariant failures before edits. Verify pre-generation availability, flags
matching displayed cells, no second broad calculation, source ownership/tamper
handling, prompt interpolation/order, malformed replies, source qualifiers and
matrix-inconsistent claims. Run focused app/prose/UI tests and affected check
selection. Retain logs. Full-suite timing approval remains pending because the
last run took 289.68 seconds with known existing failures.

## Open questions

None blocking the authorized design. Actual model answer quality/latency needs
a later bounded development trial; unit tests alone do not establish it.

## Implementation and review

`ConversationV3Service._finish` now builds the broad source-owned display table
before `build_context` and before its one composition attempt. `build_context`
derives `evidence_matrix` and compact `matrix_relations` from those exact rows;
the prompt formats each criterion as a bullet with owner-labelled flags. Only
explanation-linked passages are included as Selected Passages. Broad source
assertions remain available internally to validation without duplicating entire
CVs in the prompt. Matrix cell identities are namespaced separately; partial and
unavailable source states are normalized before deriving relations.

The system prompt retains the requested section layout and adds two examples,
correcting Not Found/absence and source verification language. An explicit
employer priority is not invented from the reviewer's question. A brief evidence
check may precede Conclusion; only the parsed final conclusion enters the
existing prose and source validation and is displayed/persisted. Runtime thinking
and generation limits remain unchanged. Prompt version is
`hiring-desk-v3-metadata-first-5`.

Review found and corrected shared-relation identity double-prefixing, stale
relations after source-state changes, and numeric criterion row loss. Tests
cover tampered citations through both paths, cross-owner evidence borrowing,
qualifiers, matrix/source consistency, absent priorities, malformed response
sections, current-question order and UI comparison tables.

## Verification and remaining limits

Baseline failures: three app invariants (`logs/v3-matrix-before.log`) and seven
prompt/parser invariants (`logs/v3-metadata-prompt-before.log`). Final combined
focused run: 170 passed in 17.15 seconds
(`logs/v3-metadata-integration-final.log`); diff whitespace checks pass.

A single invented-source live development comparison first returned
provider_unavailable within the sandbox (`logs/v3-metadata-live-invented.log`).
The authorized retry outside the sandbox reached the current 60-second turn
deadline and returned the checked fallback with all seven table rows
(`logs/v3-metadata-live-invented-unsandboxed.log`). Thus pipeline/prompt behavior
is regression-verified, but live natural-prose quality and acceptable CPU latency
are not established. No applicant text, weights or labels were modified.

Changing the prompt resource makes the check runner conservatively select all
research records. The `checks --explain` invocation also began execution; it was
interrupted once this broad selection was detected. `feature_arms` failed on
missing `data/interim/chat-rerank-plan.csv`; the interruption and process audit
confirmed no child checks remained. Log: `logs/v3-metadata-checks.log`. No passing
research reproduction is claimed, and no further long runs or full pytest run
were started without the still-pending timing approval. No branch switch,
commit, service restart, model download or training was performed.
