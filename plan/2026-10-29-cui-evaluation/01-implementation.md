# Implementation checkpoint

29 October 2026. Owner approved implementation. **Study remains pending independent review, configuration confirmation and live-run approval.** No evaluation-set outputs or live model results were generated during this work.

## Delivered

- Packaged CLI under `candidate_screener.evaluation.cui`: fixture validation, freeze, paired execution, injected recovery, blinded review export and reporting.
- Agent-authored draft fixture set: two development families and ten evaluation families; 30 standalone evaluation questions and eight five-turn evaluation dialogues (70 evaluation turns per arm, 68 generation-eligible).
- Validation checks source offsets/owners, global IDs, active scope, partition lineage and identical source leakage.
- Controlled and v3 arms reuse the actual `ConversationV3Service` and extraction service. Standalone pairs freeze QA outputs and compare prepared-evidence digests; dialogues use separate histories and reset on family-local scope switches.
- Explicit stub/live provenance, failure accounting, synthetic attempted-output diagnostics and immutable run writes. Extraction wrappers retain absolute deadlines.
- Freeze binds reviewed fixtures, protocol, package content, prompt, selected generator configuration and local model/QA manifests. It refuses unreviewed drafts. Live evaluation rejects changed identities.
- Blinded answer packets, reviewer templates and separate private mappings. Human metrics require two complete reviewer submissions, retain disagreement and permit explicit adjudication. No automatic factual correctness is inferred from answer wording.
- Family-clustered confidence intervals and paired quality deltas; independent question and dialogue summaries; accepted generation separated from fallback and outages.
- Seven injected recovery probes across five requested failure categories, including real controller post-inference source freshness and access checks.

## Scope precision and deviations

This harness evaluates the conversation **service**, not browser interaction or the asynchronous GLiNER cache lifecycle. Live mode uses pinned local RoBERTa QA and Qwen3.5; stub mode uses explicit unavailable-QA and synthetic-generation doubles. It does not claim full deployed-stack accuracy. Adding GLiNER preparation or browser timing requires an explicit extension and cache policy before execution.

Standalone arm timings include different first-use/replay extraction costs. They are labelled paired-cache service times, not production end-to-end latency. Dialogue timings exclude portal UI. Reader preparation and generator wall time are recorded; context building/validation are not falsely reported as separately measured stages. Shared generator coldness is unknown unless externally established. Infrastructure latency conclusions must retain these limitations.

Two agent-authored development families and ten evaluation candidates exist; independent human semantic/lineage review is still required. Schema validation does not establish that scenarios are truly independent. The preliminary study does not fulfill the earlier v3 promotion gates.

## Owner-requested role-title correction

During implementation the owner reported: “What is the role required for this JD?” returned “Required experience and skills:”. The owner requested using the same saved `jd_title` displayed by the UI.

V3 now handles narrowly matched title questions from the current authorized snapshot, returning a redacted title directly without extraction or generation. Skills, responsibilities, candidate roles and compound requests retain the existing evidence route. Missing titles fall through; no title is invented. Provenance identifies `jd_title` rather than fabricating passage offsets. The title is included in the scope digest so a changed title invalidates old factual dialogue context. Prior candidate/evidence selection is cleared for the JD-title answer. No UI edits were made for this correction.

## Verification evidence

- Fixture tests: 20 passed (`logs/cui-fixtures-tests.log`); exact-source guard removal reproduced failure (`logs/cui-fixtures-mutant.log`).
- Metrics/review tests: 16 passed (`logs/cui-metrics-review-tests.log`); paired coverage and reviewer coverage guards reproduced failures in `logs/cui-paired-coverage-before.log` and `logs/cui-review-coverage-before.log`.
- Recovery tests: four passed (`logs/cui-recovery-tests.log`); removal of post-inference authorization reproduced failure (`logs/cui-recovery-before.log`).
- Initial combined evaluation/title tests: 62 passed (`logs/cui-focused.log`). Subsequent tests are recorded in the final verification log below.
- Title correction reproduced six failures before the fix (`logs/cui-role-title-before.log`), including the owner's exact bad reply. Initial title/v3/scope verification: 68 passed (`logs/cui-role-title-after.log`).
- Development-only stub execution produced 32 paired records with zero execution errors (`logs/cui-evaluation/dev-offline-01.json`). These are harness results, not model quality measurements.
- Final combined evaluation/title/v3/scope/prose verification: **151 passed** (`logs/cui-final-focused.log`). All seven injected recovery probes passed (`logs/cui-evaluation/recovery-01.json`); blinded export and operational report CLI smoke passed (`logs/cui-review-cli.log`, `logs/cui-report-cli.log`). Affected warm record checks selected zero records (`logs/cui-affected-checks.log`).
- The subsequent development-only `--pilot` selector passed all 12 runner tests (`logs/cui-pilot-tests.log`); it restricts initial timing to one paired question and rejects evaluation subset selection.
- Full-suite execution requested from the owner because recent runs take about four minutes; permission is pending. No live generation has run.

## Remaining gates

1. Name scenario author/reviewers and review the candidate fixtures independently.
2. Confirm target hardware and selected generator/QA manifests.
3. Complete approved full-suite verification and any applicable record checks.
4. Authorize and time live development execution; present measured total estimate before evaluation.
5. Freeze and execute the complete study, collect independent ratings, report results.

Do not mark this plan complete or present quality/latency targets as achieved.
