# Full-fit extraction campaign

Approved plan commit `3290e27`, branch
`feat/fit-gliner-comparison-2026-10-14`. The user expanded the single-pair proposal
to every pair in train and test, then approved and resumed work after interruption.
Campaign helpers and notebook reporting were delegated, with independent review.
Unrelated existing changes are preserved.

## Measured inventory and pilot

Train: 3,990 pairs, 446 CVs, 245 JDs. Test: 659 pairs, 158 CVs, 100 JDs.
Combined 949 unique documents and 4,551,699 text characters. No conflicting
same-ID source texts. Documents are inferred once and evidence is reused across
their original labelled pairs; this is not resampling pairs.

Four deterministic quartile-length documents were timed with the installed
five-label GLiNER configuration and threshold 0.5. Initialization11.09s,
inference9.72s across16,358chars; all four completed. Length-scaled full-run
estimate2,716s (~45minutes), communicated before launch. This estimate is not
a bound for atypically long documents. `logs/fit-gliner-pilot.json` records timings.
Darwin's `/usr/bin/time -l` replaces the GNU-specific `-v` example for peak RSS.

## Campaign status

Full inference launched once in the background with the frozen configuration.
The log is `logs/fit-gliner-full.log`; `/usr/bin/time -l` records peak RSS at exit,
and the shell appends an exit marker. Checkpoints live under ignored
`data/processed/extraction/fit-campaigns/`. The expected duration is about45min.
No foreground polling or repeated launch is needed. Notebook execution and final
validation follow completion, so this plan is not yet marked complete.

Prelaunch verification:14 focused synthetic tests passed (1.56s), independent
review passed. A controlled disabled same-ID conflicting-text guard made its
regression fail with DID NOT RAISE. Logs: `logs/fit-extraction-tests.log` and
`logs/fit-extraction-regression.log`. Identity-hashed code is frozen during run.
No extraction accuracy claim is inferred from Fit labels or prediction counts.
Source-level entity gold is absent. Every A1 label-conditioned display carries
the standing A13 limitation. No training, tuning, annotation or download is used.
