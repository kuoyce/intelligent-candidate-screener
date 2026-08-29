# The labelling UI *(decision D30)*

**Status:** implemented, 29 August 2026, branch `feat/annotation-ui`.

Three screens over the in-domain set: the batch list, a new-batch form, and a labelling
screen showing one job description with its candidates. Built after the operator walk-through
of `queue --build` turned up three gaps that only appear when someone tries to *run* the
session rather than design it.

## What this closes

| Gap | Before | Now |
|---|---|---|
| **No ingest path** | The dispatch CSV had no `label` column and nothing joined labels back to `pair_id`. The join needed `judging-queue-full.parquet`, a git-ignored file, in a pandas step nobody had written | The UI appends directly to `judgements.csv`, which is already the resume mechanism |
| **The shortlist question was unanswerable** | Asked "after you have labelled all the candidates for one JD", but the queue is shuffled precisely so a JD's five candidates are scattered, and the dispatch file carries no JD key | The screen *is* the JD. The question is asked once, over a complete field |
| **A mistyped job title drew nothing** | `--keywords "Data science"` matched nothing, `choose_jds` returned empty, and the batch was appended contributing **zero pairs** — no exception, no warning | `validate_keywords` fails with the valid list. The UI offers the 41 titles as checkboxes, so the CLI is the only way to type one |

## The question that prompted it

The generic batch's 22 job titles are **not a designed scope**. The raw Djinni corpus has
**45** distinct `Primary Keyword` values; 41 survive the eval-region and English filters; batch
1 is an unstratified draw of 40 JDs (D14) and 22 titles is simply what fell out. The 19
reachable titles it misses include `Data Science`, `Data Analyst`, `Business Analyst`,
`Security`, `iOS` and `C++`. `indomain-report.json`'s `titles` array is an observation, and any
title a report needs covered has to arrive as a targeted batch. `sample.available_titles()`
now names the 41, and the UI is where a batch gets built from them.

## Decisions

**D30 — the labelling instrument is a local UI over the in-domain set, and the A1 recheck
stays on the flat dispatch CSV.**

The split is forced by a measurement, not a preference. The in-domain set is uniformly 5
candidates per JD; the 50 A1 recheck pairs spread over 32 A1 JDs as 1, 2, 3 or 4 each. On a
screen that shows one query and its candidates, a group of two is visibly not an in-domain
group — and the recheck's entire value is that an annotator cannot tell a rechecked pair from
a fresh one (**A13**), which is what the ceiling figures in
`plan/2026-08-29-pool-precision-bias/` rest on. A UI that grouped both would leak that
distinction through its layout. So the recheck keeps the shuffled, blind CSV it was designed
for, and this UI does not touch it.

**Standard library only, no UI framework.** Adding one puts it in the dependency closure of
`uv sync` for everyone, including whoever only wants to re-run the baseline. More concretely:
creating a batch replays the whole campaign — 141,897 JDs and 210,250 CVs loaded, then a
TF-IDF fit per JD, ~6 s — and a framework that re-executes the script on every widget
interaction is the wrong shape for that. Here it is one POST. Cost: ~330 lines of vanilla JS
in `ui.html`, which is why every rule about what an annotator may see lives in `session.py`
where a test can reach it without a socket.

**The rules live in `session.py`, the transport in `ui.py`.** Redaction runs on `serve_group`'s
*output* and `assert_clean` runs after it, so a UI bug cannot bypass it. A group carries
`jd_id`, `batch`, `stratum`, `jd_text`, `complete` and the candidates' text — no
`lexical_band` (the similarity third the sampler drew from, which would tell an annotator the
sampler's guess before they made their own), no `selection_reason`, no existing label.

## Deviations found while building

**Y1 — the second annotator was being asked the shortlist question over a partial field.**
Found in an end-to-end run, not in review. 30% of pairs are double-labelled; when the second
annotator reaches a JD the first has finished, only that shared subset is left — 2 of 5, say.
The screen was still asking *"of the candidates you marked Good or Potential, which would you
shortlist first?"*, and the resulting pick would have been indistinguishable in
`judgements.csv` from one made over the whole field. A disagreement between the two annotators'
picks would then be unattributable between the people and the truncation, which is the one
thing the double-labelling exists to measure.

Fixed by `Group.complete`. A partial group hides the question, `record` refuses a pick on one,
and the screen says why. Two named tests, both shown to fail against the unfixed code.

**Y2 — the guide's "judge each pair on its own" needed rewording, not deleting.** Five
candidates on one screen is comparative context, and the guide previously forbade comparison.
It cannot simply be dropped: the *labels* must still be assigned against the standard rather
than against each other, or the three classes stop meaning the same thing across JDs. What
changed is that the comparison is now confined to the final pick, and the guide says so.

## Verification

```bash
uv run pytest                                              # 168 tests
uv run python -m candidate_screener.data.verify --derived  # + 6 collected-label checks
uv run python -m candidate_screener.baselines.run --check
uv run python -m candidate_screener.annotation.ui --annotator <name>
```

Twelve mutations were run against the new tests and each was caught; they are named in the
docstrings in `tests/test_annotation_ui.py`. One of them, M6, **passed on the first attempt** —
serving "the next 5 owed pairs" instead of grouping by JD is indistinguishable from correct
behaviour when every JD has exactly 5 candidates, so the fixture was changed to 3-per-JD, where
the mutation necessarily straddles two JDs. A test that passes against the bug it names is not
a test.

`verify --derived` gained six checks over collected labels, because the UI writes to
`judgements.csv` with no import step to inspect rows on the way in — deliberately, since an
import step is a second copy of the truth that can disagree with the first. The one that earns
its place is stratum agreement: `stratum` is stamped at labelling time (D29) so a batch spec
edited later cannot re-stratify collected labels, and the failure mode of stamping is a stale
value quietly disagreeing with the pair it names.

## Not built

- **Adjudication.** Disagreements are reviewed in a session per 5.6 step 4; both original rows
  are retained by construction, since `record` only ever appends. No screen for it.
- **κ.** Computed at report time from `judgements.csv`, not by the UI.
- **Editing a label.** A correction is a new row. Nothing rewrites history.
- **Multi-user serving.** Two annotators run two local copies against the same working tree, or
  one at a time. `judgements.csv` is written with append-mode single-line writes, which is
  atomic enough for two people on one machine and is not a claim about anything larger.
