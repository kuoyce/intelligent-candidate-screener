# Conversational summaries and exact negative replies

27 October2026. Owner clarification: normal happy-flow answers should reason and
summarize without extensive source-excerpt references. Negative cases should
return only: “I am not able to answer this question. It is out of my scope.”
This is an amendment to the existing approved v3 conversation scope; keep the
current branch and local Qwen3.5 deployment.

## Changes

- Both ownership/05 negative cases return the exact sentence with `out_of_scope`
  status and no model call, citations, source controls or diagnostic captions.
- Normal v3/JD-search replies put all source controls behind one explicit
  “Show supporting sources” checkbox. Opening a source request still shows its
  passage. Unique saved turn identities prevent duplicate widget keys.
- Technical generation details remain in the diagnostics expander. Normal prose
  avoids repetitive source-prefixed fallback wording while preserving qualification
  uncertainty, negation, ownership and relevant comparison counterpoints.
- The system prompt uses the same exact scope sentence; its prompt version is
  `hiring-desk-v3-grounded-prose-4` with the actual prompt hash recorded.

## Limits and verification

Qwen3.5 can compose summaries and has demonstrated correct degree-status reasoning,
but previous real tests also showed unsupported comparison claims. The application
accepts only a bounded factual grammar and falls back on other wording. These UI
and fallback changes do not establish general semantic entailment or claim all
freeform happy-flow questions now succeed.

The exact-negative/source-rendering tests failed before changes
(`logs/v3-conversation-presentation-before.log`), then14transport/scope tests passed
(`logs/v3-conversation-presentation.log`). Test data and model stubs are synthetic;
no new real-model benchmark is claimed. Old saved reply text is historical; new
questions receive the updated behavior.

Final combined verification:80tests passed (`logs/v3-conversation-final.log`).
Normal fallback summaries cover relevant criteria and preserve reverse advantages,
ongoing/completed qualifications and missing-information caveats in up to three
sentences where practical. Positive degree alignment requires the requested degree
to match. Three summary regressions failed before changes
(`logs/v3-human-summary-before.log`), then35focused summary tests passed
(`logs/v3-human-summary-tests.log`). No arbitrary-prose validation was relaxed.

The portal was found stopped after testing and restarted on8503 with v3 and the
existing Qwen3.5 manifest selected. The model service on8001 remained running.
