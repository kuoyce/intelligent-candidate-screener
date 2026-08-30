"""The annotation instrument — one guide, one queue, one session *(decision D25)*.

Manual labelling is capped at **~250 judgements in one sitting**: task 3.4b's 200
in-domain Djinni pairs, plus ~50 already-judged A1 pairs mixed in blind as a recheck.
The judging wave Phase 5 designed (D22/D23, 4–6 team-days over the union of every
system's top-k) is deferred, because Q18 closes as a standing limitation instead —
see `plan/2026-08-29-unified-judging-wave/` for what was traded and why.

Three modules, in the order they run:

- `redact` — strip PII. Nothing reaches an annotator without passing through it.
- `sample` — draw the 200 Djinni pairs from A2's **eval** region (D19), banded so the
  set is not degenerate.
- `queue` — mix in the A1 recheck, shuffle, blind, and write the dispatch file.
"""
