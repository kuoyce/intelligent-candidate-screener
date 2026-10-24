# Hiring Desk v1 — JD-first review and department handoff

Status: approved by the user on 23 October 2026; application implemented and
validated 24 October. Live-model binding remains pending the teammate's actual
interface. The plan is not declared complete under D55.
On 23 October the user
confirmed new-JD candidate ranking through a teammate's live-model integration
and requested a lightweight demonstration of email procedure.

## Outcome

Make the selected job and its requirements the focus of Hiring Desk. Reviewers
browse job cards, read the full JD, review its ten ranked candidates and continue
using the existing evidence-aware chat. Add a separate guided JD-creation page,
a useful alert inbox, live-ranking integration for newly published JDs, and a
lightweight top-ten email preview/confirmation flow captured locally.

The user requested a plan before execution. Approval covers the design below
and resolved choices in `03-open-questions.md`. Then create a branch, commit the
approved plan and implement through scoped sub-agents. Preserve existing work,
including the 40-day reviewer account and bottom-sidebar logout.

## Documents

- `01-design-spec.md`: screens, state, data and delivery contracts.
- `02-implementation-and-validation.md`: ordered work packages and acceptance.
- `03-open-questions.md`: unresolved decisions and proposed defaults.
- `04-ui-audit.md`: code-based findings and required visual audit.
- `05-live-ranking-contract.md`: proposed boundary for the teammate's model.
- `06-implementation.md`: delivered behavior, verification and integration limits.

## Scope boundaries

The existing CSV-based JD/CV review flow is a required v1 capability, explicitly
confirmed by the user. Continue loading `a2-jd-extraction.csv`,
`a2-cv-extraction.csv` and `a2-x2-scores.csv` with the current source-text joins,
backend ranks and scores. Existing jobs remain browsable and reviewable through
the upgraded cards, review actions and chat even when live modeling is absent
or unavailable. Live integration adds a path for new JDs; it does not replace
the CSV adapter or rerank existing exports automatically.

Existing export rankings, scores and evidence remain authoritative. Summaries
and drafting assistance cannot invent requirements, ranks or Fit calls. New JDs
are operational application records, separate from the frozen A2 benchmark and
annotation instruments. No fitting, threshold tuning or research scoring is
part of this upgrade. The teammate owns model implementation; this upgrade owns
the UI, job/request lifecycle and adapter boundary. Email is demonstrated with a
locally captured message; a production mail deployment is outside v1 scope.

This proposal replaces the current eight-CV handoff with a versioned ten-CV
handoff for new packages. Historical reports and packages retain their original
meaning. The change requires documentation of the superseded top-eight product
contract, rather than rewriting historical records.
