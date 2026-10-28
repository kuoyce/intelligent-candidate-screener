# JD-only quotation fallback

## Authorization and design

The owner requested implementation of the proposed quick fix: when v3 cannot
accept model prose for a JD-only question, quote an authorized supporting JD
passage instead of claiming there is insufficient checked information.

Use only the current JD evidence reauthorized by `build_context`; never use
history, candidate sources or model output as the quotation. Return one complete,
nonempty passage, without clipping qualifications, within the existing 2,400
character answer budget. If none fits, retain the existing fallback. Keep
`qwen_fallback` and the original failure reason. Candidate replies are unchanged.
The same quotation is useful when generation is unavailable.

## Delivery and verification

1. Create a scoped branch and commit this authorized plan.
2. Delegate implementation and invariant regression tests to a sub-agent.
3. Demonstrate the new invariant fails before the fix; run focused tests after.
4. Review ownership, redaction, size and provenance; run the full pytest suite
   once with a retained log. Inspect affected check selection before running any
   record checks; obtain approval for long checks if selected.
5. Record results here. No model, training, benchmark or rollout changes.

## Open questions

None within the approved quick-fix scope. General paraphrase acceptance is deferred.

## Implementation and review

`fallback_text` now quotes one complete current JD passage before selecting the
generic fallback, including when the provider is unavailable. It requires no
selected candidates or candidate cells, matching JD ownership, nonempty text,
unchanged redaction and a full answer within 2,400 characters. Current-source
reauthorization remains in `build_context`; model acceptance and provenance are
unchanged. README documents the quotation fallback.

The delegated implementation was reviewed for source ownership, complete
qualifiers, answer budget and preservation of candidate behavior. Six regression
cases cover rejected/unavailable generation, tampered citations, complete
qualifiers/oversize passages and mismatched source ownership. Three fail on the
original implementation (`logs/jd-quotation-before.log`); all 41 v3 tests pass
afterward (`logs/jd-quotation-focused.log`). The affected check runner selected
zero records (`logs/jd-quotation-checks.log`); no research cold run is applicable
to this fallback-only change.

Full pytest (`logs/jd-quotation-pytest.log`): 2,088 passed, 56 skipped,
19 failed and 3 errors in 289.68 seconds. Two failures concern baseline vocabulary
and model-record drift, outside the changed path. The other 17 failures are
extractive-answer/UI assertions: all 17 reproduce with the original committed
`fallback_text` restored in memory (`logs/jd-quotation-original-comparison.log`).
The three annotation UI socket setup errors pass on that comparison run outside
the sandbox. These existing failures were not modified as part of this fix.
`git diff --check` passes. No running model service was changed.
