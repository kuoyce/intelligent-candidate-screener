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
