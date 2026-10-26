# V2 enhancement: directed comparisons and red/black presentation

25 October 2026. Approved by the owner's implementation request. Continue on
`feat/hiring-desk-v2-conversation`; **do not create another branch**. Preserve all
existing v2 and unrelated report work. This extends the existing release scope.

## Outcome

1. “Compare 1 and 2” provides a clear comparison table. “Why is 1 better than 2?”
   uses that same evidence to explain which stated job criteria have stronger
   documented support, including the relevant skill/knowledge excerpts. It does
   not collapse to a one-candidate extract, repeat an unexplained table, or invent
   an overall hiring judgement. Follow-ups retain the active pair and criterion.
2. Apply red and black accents with a predominantly white background throughout
   the app, including native Streamlit controls. Retain responsive layouts and
   readable source/table/disabled/focus states.

## Comparison design

- Resolve directed numeric/prefixed references before existing ranking shortcuts.
  Preserve requested order, reject unknown ranks, and never mistake experience
  quantities for people. Explicit references override prior context.
- Distinguish a superiority question from an immutable-rank explanation. Saved
  rank alone is not a reason someone has stronger evidence. “Why ranked above”
  reports backend ordering and separately discusses supporting source evidence.
- Reuse the active criterion when specified; broad role questions use recognized
  literal JD criteria. Keep a visible comparison table alongside the explanation
  even when only one criterion is discussed. Label nonexhaustive coverage.
- Build directional, application-validated relations from the comparison cells:
  explicit positive/negative contrast; positive evidence versus not established
  in the other CV; explicit comparable skill tenure; completion-status differences.
  Role-relevance wording requires the criterion to be supported by JD evidence.
  Entity-table presence alone cannot prove proficiency, qualification or absence.
- Positive versus missing means “clearer documented evidence”, never “the other
  candidate lacks the skill”. Unavailable/partial sources remain explicit and
  cannot establish a reliable comparative advantage. No new score, Fit call,
  overall ranking or presumed causal explanation of model scores is introduced.
- Lead with a direct scoped conclusion. If evidence is mixed, say so and explain
  the tradeoff; if it favors candidate 2, correct the premise. If insufficient,
  state the limitation rather than affirming that candidate 1 is better.
- Qwen receives directed intent, the requested pair, checked relation IDs and
  complete supporting/counterevidence. Refine its prompt/plan contract to order
  explanatory points and retain counteradvantages. The application still writes
  factual clauses. No relaxed arbitrary-prose validator or weight training.
- Byte-budget trimming must rebuild directional conclusions and invalidate any
  stale generated-success provenance. Every claim stays tied to current sources.

## Example response contract

For an invented JD requiring Python and deployment, candidate 1 explicitly lists
both; candidate 2 lists Python but deployment is not established in the retrieved
CV. A suitable response is: “Candidate 1 has clearer documented support for the
deployment requirement: their CV states [exact excerpt]. Candidate 2's retrieved
CV evidence does not establish deployment experience. Both have Python evidence,
so the documented difference here is deployment; this is not proof that candidate
2 lacks it.” The comparison table shows both people, JD criterion and sources.

If candidate 2 explicitly reports more Python tenure, preserve that counterpoint
when the question is broad. If the question is specifically about Python tenure,
say that the evidence favors candidate 2 rather than endorsing the question's premise.

## Theme design

Use white main canvas, near-black body/headings, deep red interactive accents,
light neutral surfaces and subtle borders. Keep large surfaces light rather than
turning the app into a dark theme. Update both `.streamlit/config.toml` and the
portal CSS so native controls and custom containers agree. Check text/link/button
contrast, keyboard focus and narrow-screen layout. No authentication/content changes.

## Validation and delivery

Use scoped sub-agents for comparison facts, dialogue/Qwen integration, and theme.
Show named comparison regressions failing on the previous v2 implementation;
cover standalone/follow-up bare ranks, reversed premise, missing versus explicit
negative evidence, mixed advantages, unknown/partial source, context changes and
unchanged backend ranking. Test table visibility with explanatory responses.
Exercise a bounded invented-source real-Qwen comparison after integration.

Inspect theme/native configuration and synthetic UI rendering; keep existing
desktop/mobile/logout behavior. Run focused tests, one full suite once stable,
and affected checks through the central runner. Preserve known baseline-golden
failures separately. No training, new data collection, release push or new branch
is authorized by this amendment. Existing v2 human-quality gates remain pending.

## Open questions

No new blocking implementation question. This request authorizes prompt/contract
refinement rather than model-weight fine-tuning; weight adaptation retains the
approved v2 dataset/resource gates. Human reviewers for the broader v2 study
remain unassigned and do not block implementing these two requested features.

## Implementation and verification

Implemented on the existing `feat/hiring-desk-v2-conversation` branch. No new
branch, model-weight training, checkpoint download or remote push occurred.

Directed questions now resolve bare/prefixed/compact references, retain the pair
and active criterion, and show the same table/source evidence as comparison chat.
Application-validated advantage records carry JD/CV evidence, direction, basis and
counteradvantages. Qwen prompt version3 receives directed intent and must retain
counteradvantages; only checked relation IDs are accepted. Explanations correct
reverse/mixed premises and distinguish coverage from absent qualifications.
Canonical citation ownership is unchanged; the UI uses short S1/S2 labels.

Literal required-skill lists and knowledge requirements are recognized within
the bounded three-criterion grammar. The two retained passages are combined,
including overlapping-context unions, so contradictory source statements do not
silently disappear when explaining an advantage. Partial/unavailable sources do
not establish directional advantages. Broader semantic/coverage limits remain
explicit; the pending independent v2 study is not claimed complete.

Theme changes update both native Streamlit configuration and portal CSS. White
main/header, near-black text, deep-red primary/focus/link accents and a pale-red
sidebar retain the logout/footer and mobile layout behavior.

Validation logs (git-ignored):

- `v2-directed-focused-final.log`: **131 passed**, including the final required-list
  and readable-citation changes after full-suite execution.
- `v2-directed-pytest-final.log`: **1,808 passed, 11 skipped, two existing baseline
  golden failures**, 222.39 seconds. The same vocabulary/TF-IDF drift predates v2;
  no baseline records or assertions were changed.
- `v2-directed-checks.log`: central runner selects zero affected research records,
  exit 0. No training/cold benchmark jobs were launched.
- `v2-directed-native.json`: all fifteen assertions pass in the extended invented
  walkthrough. The three directed questions returned accepted Qwen plans with
  tables: broad tradeoff, false Python-tenure premise, and reversed candidate pair.
  These are diagnostic turns from one invented family, not a quality estimate or
  a population latency result. Post-inference source-label formatting is separately
  verified by the final focused tests.
- `v2-red-black-theme-tests.log`: four synthetic UI tests pass.
- `v2-theme-visual-check.log`: fully rendered desktop1440px/mobile390px previews
  have a white app canvas, pale-red sidebar and no horizontal overflow. Screenshots
  are `logs/v2-theme-desktop.png` and `logs/v2-theme-mobile.png`. The synthetic
  preview server/headless Chrome were stopped afterward; real accounts/app state
  were not altered.
- Fail-before logs: `v2-directed-why-before.log`, `v2-directed-qwen-before.log`,
  `v2-directed-explanation-before.log`, `v2-broad-conflict-before.log`,
  `v2-directed-partial-before.log`, `v2-required-lists-before.log`.

The implementation remains reviewable in the working tree with the earlier v2
work. Unrelated report files and Finder metadata were preserved.
