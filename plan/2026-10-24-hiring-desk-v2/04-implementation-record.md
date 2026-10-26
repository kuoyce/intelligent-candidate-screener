# Implementation checkpoint

Status: **opt-in application implemented; independent quality acceptance pending**.
Resumed on 25 October at the owner's explicit request after the report-generation
pause. This is not completion of the entire study/training plan.

Branch: `feat/hiring-desk-v2-conversation`. Approved plan committed as `320c3fa`.
Implementation is uncommitted and remains in the working tree. Unrelated report
files, `.DS_Store` and the historical scratch note have not been changed/staged.

Implemented so far: v2 frame/service, comparison facts and controlled rendering,
strict Qwen relation-plan composer, environment feature flag (`HR_PORTAL_CHAT_VERSION=v2`),
UI presentation and an invented-source walkthrough module. Default remains v1.

Narrow verification logs:

- `logs/v2-conversation-ui-final.log`: 24 service/UI tests passed.
- `logs/v2-comparison-facts-final.log`: 24 facts/render tests passed.
- `logs/v2-qwen-tests.log`: 22 composer tests passed.
- Fail-before proofs: `v2-conversation-before.log`, `v2-comparison-facts-before.log`,
  `v2-comparison-semantics-guards-removed.log`, `v2-qwen-critical-before.log`.

## Delivered behavior

- Snapshot-bound comparison state retains participants, criterion switches,
  ordered references, pending clarifications and exact cited-source coordinates.
  Explicit IDs override prior plural references; adding/removing candidates keeps
  the current criterion. Saved-rank explanations state rank without invented causality.
- Broad comparisons use up to three literal recognized JD criteria instead of
  contradictory generic missing-category rows. Parsing is deliberately limited;
  unsupported JD wording falls back to the existing category matrix. The UI
  labels recognized-criterion coverage as nonexhaustive.
- Facts preserve full qualifier-bearing contexts and revalidate owner, digest,
  source coordinates and snapshot. Registered rules support qualified contrasts,
  explicit skill-linked tenure, degree-completion differences and pairwise
  three-person relations. Unrecognized/ambiguous meanings remain unclear.
- Controlled prose leads with meaningful comparisons, uses friendly labels and
  keeps critical qualification/missing-source statements. No unchecked generated
  factual prose is rendered. Source controls retain the full source context.
- Qwen selects/orders validated relation IDs through short request-local aliases.
  The old excerpt-only Qwen contract is unchanged. Empty/unbalanced relation sets
  skip generation truthfully; timeout/invalid plans retain controlled evidence.
- Native QA uses an absolute monotonic deadline including lock wait and worker
  exchange. Configured Qwen uses a per-turn subprocess killed/reaped at the same
  remaining deadline. A client timeout does not cancel computation already running
  in the separately managed shared HTTP model server. Its own runtime bounds remain.
  Arbitrary injected test clients do not acquire hard cancellation guarantees.
- Oversized answers omit whole evidence cells with explicit partial status;
  qualifications are not sliced away. Re-rendering after budget changes clears
  stale Qwen-success provenance. Original controller authorization, post-inference
  source checks, saved-turn idempotency and action confirmations remain.

## Verification on 25 October

- Full suite: **1,775 passed, 11 skipped, two existing baseline golden failures**
  in 205.51 seconds, `logs/v2-pytest-final.log`. Failure hashes and TF-IDF differences
  match prior v1/merge logs. Baseline records/tests/code were not changed.
- Additional review-driven fixes are covered by the final combined focused run
  `logs/v2-focused-final.log`: **120 passed** in 10.16 seconds, including facts,
  service/UI, composer and QA deadlines, after the last review-driven changes.
- Central runner: `logs/v2-checks.log`, zero affected research records, exit 0.
  Read-only import-closure selection is in `logs/v2-check-selection.log`.
  No benchmark scoring, model training or cold research grid was launched.
- Native local Qwen and RoBERTa walkthrough: `logs/v2-native-walkthrough-final.json`.
  Assertions check actual JD deployment contrast, Python-tenure comparison,
  qualifications, exact citation reopening, participant edits and storage bound.
  Earlier exploratory native run is retained separately, not presented as final code.
  Final mode records show accepted Qwen plans for broad comparison, explicit
  Python tenure, qualification follow-up, adding candidate 3 and removing candidate 2;
  citation reopening uses the original source, and cloud comparison skips generation
  when balanced relations cannot be formed. All eleven walkthrough assertions passed.
- Regression proofs include `v2-qa-absolute-before.log`,
  `v2-qwen-deadline-before.log`, `v2-deadline-service-before.log`,
  `v2-broad-criteria-before.log`, `v2-rank-explanation-before.log`,
  `v2-composition-skip-before.log`, `v2-semantic-final-review-before.log`,
  `v2-degree-future-before.log` and the original v1 proofs above.
- Separate agents reviewed source semantics, state/ownership and the process
  boundary. Their findings were repaired and covered by named regressions.
  Final independent review confirmed future/pending/conditional MSc examples
  remain qualified rather than completed evidence, with no remaining blocker.

The walkthrough is one invented JD family. Per-turn timings/modes are diagnostics,
not a population quality/latency estimate; no meaningful family-level CI can be
estimated from that single family. Do not claim the planned 120-family/human
evaluation or quality gates passed. Reviewers were requested; identities remain
unassigned. No fine-tuning, larger-model download or threshold tuning occurred.
Conditional refinement is still gated on reviewed data and measured failures.

## Running and rollback

Use `HR_PORTAL_CHAT_VERSION=v2` with the existing Streamlit command. V1 remains
the default and `HR_PORTAL_CHAT_VERSION=v1` rolls back. Setting
`SCREENER_QWEN_DISABLED=true` retains v2 controlled conversation without Qwen.
The already-provisioned local Qwen service was restarted for verification using
the existing pinned 0.5B CPU checkpoint; no app service or real review data changed.

No implementation commit, push or default-v2 release has been made. Unrelated
technical-report files and pre-existing metadata/scratch changes remain untouched.

The same-branch [directed-comparison and theme enhancement](06-directed-comparison-and-theme.md)
is also implemented. Its record contains the later full-suite/focused results,
native directed-Qwen walkthrough, conflict-handling regressions and desktop/mobile
red/black theme verification. Those results supersede earlier implementation-only
counts above; broader human-quality acceptance remains pending.
