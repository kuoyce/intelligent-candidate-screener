# Answer-first acceptance correction

21 October 2026. User feedback: the assistant displays the CV rather than
extracting evidence to answer the question. This is an unmet part of the approved
chatbox outcome, not a request to replace the dropdowns or change rankings.

## Cause verified in code

portal_qa.SYSTEM requests only evidence IDs. _model_claims accepts text only when
it equals the whole retrieved quote. answer() produces a generic pointer to source
passages, and portal_ui expands the first two passages. Thus successful Qwen
inference still does not produce a question-specific answer in the transcript.
The previous completion record overstated user-facing QA completeness.

## Acceptance walkthrough frozen before the correction

Use synthetic sources containing a skill, an ongoing degree, an expired certificate,
and total experience unrelated to skill tenure. Do not display actual applicant
documents in logs.

| Question | Required visible behavior |
|---|---|
| What degree does candidate 1 have? | Direct short source-backed education answer, retaining ongoing/incomplete wording; not a whole CV or pointer to passages |
| What Python experience does candidate 1 have? | Relevant skill/experience statement; exclude unrelated education and hobbies |
| Does candidate 2 have AWS certification? | Preserve expired wording; do not call it a verified active qualification |
| Compare candidates 1 and 2 on Python | Separate relevant answer for each participant, including explicit missing evidence where applicable |
| What does this role require? | Short JD-backed requirements in the answer body |
| Does candidate 1 have five years of Python? | Do not infer Python tenure from five total years of software work |
| Does candidate 1 know an unmentioned technology? | Explicit insufficient evidence rather than presenting unrelated skills as an answer |

Source citations are collapsed by default and may show fuller surrounding context
on request. Every extracted statement must resolve to the correct authorized
document and snapshot. Source text, selected answer spans and generated model
selection remain untrusted until validated. No arbitrary model paraphrase is
accepted as established fact solely because it carries a citation.

## Verification record

Implementation delegated to QA/extraction and UI agents with independent review.
The implementation now retrieves complete, short answer-bearing source statements
rather than 650-character blocks. It preserves adjacent credential status and
degree abbreviations, validates exact owner/source offsets, and renders statements
directly per applicant or JD. Qwen selects these short statement IDs under the
existing validated schema. The deterministic fallback renders the same direct
source-backed answers. Supporting source controls are collapsed by default.

Two direct-answer regressions failed against the original pointer-only response
(logs/chatbox-direct-answers-before.log). A separate isolated UI regression failed
against expanded-by-default citations (logs/chatbox-collapsed-ui-before.log).
The earlier logs/chatbox-answer-ui-before.log captured a transient missing-import
error during simultaneous edits and is not evidence of the UI regression.

Focused tests: 31 QA/context tests passed (logs/chatbox-direct-answers-after.log),
17 UI tests passed (logs/chatbox-answer-ui-final.log). Independent review checked
negation, credential qualifiers, question-specific retrieval and candidate scope.
Scoped cold registered checks selected zero affected research records for the two
changed portal modules (logs/chatbox-answer-checks.log). No training or benchmark
record changed. The first broad run passed 1,283 tests with 10 skipped and the
same two baseline reproduction failures (logs/chatbox-answer-full-pytest.log).
After the final credential guard, full pytest finished with 1,285 passed,
10 skipped and the same two baseline reproduction failures in 193.06 seconds
(logs/chatbox-answer-final-pytest.log). Both failures remain in the untouched
test_baseline_golden.py vocabulary/model reproduction checks, matching the earlier
portal and chatbox runs. All answer-extraction and portal tests passed.

Synthetic live smoke (logs/chatbox-answer-live-smoke.log): all seven scripted
questions produced direct, relevant source statements or an explicit missing-
evidence response. These are inspected walkthroughs, not a measured accuracy
claim. Qwen/Qwen2.5-0.5B-Instruct selected the degree answer successfully in 7.06
seconds. Its skills response failed validation after 28.27 seconds; deterministic
extraction still supplied the correct short Python/software-testing statement.
No checkpoint was changed. Live model reliability remains limited.

Final review additionally found that the technology focus gate could accept a
skills-only sentence for a certification question. Credential/degree questions
now require their corresponding evidence type as well as the requested subject;
a technology mention alone cannot answer a credential question. Its named
regression failed before the guard (logs/chatbox-credential-relevance-before.log)
and passed afterward. Singular year/month statements are supported too.

Known retrieval limits: lexical matching can miss paraphrases, and complete source
statements longer than 400 characters are omitted to avoid truncating a qualifier.
Missing evidence is therefore not an assertion that a qualification is absent.
This delivery extracts concise, attributed facts; it does not accept unrestricted
model paraphrases as verified hiring conclusions.

## Open questions

An optional actual failing question was requested from the user. Synthetic
walkthroughs above were used because no actual question was supplied during work.
Previously saved replies retain their original text. Verification of the change
requires a new question after refreshing/restarting the local app.

Status: answer-first correction implemented, independently reviewed and verified
with the recorded baseline failures and retrieval/model limitations above.
