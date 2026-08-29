# The labelling UI *(decision D30)*

**Status:** implemented, 29 August 2026, branch `feat/annotation-ui`. Revised the same day by **D31** and **D32**, after the first 30 pairs were labelled and the set turned out to be measuring nothing.

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


---

# D31 — 10 candidates per JD, 8 of them on-category

**The pilot found the defect that no amount of design review had.** Six JDs were labelled
through the UI and **28 of the 30 pairs came back `No Fit`**, with 2 `Potential Fit` and no
`Good Fit` at all. Those labels are kept at
[`pilot-judgements-superseded.csv`](pilot-judgements-superseded.csv); the re-cut orphaned them,
and they are the evidence for it.

The cause is one line of the original sampler. Each JD's 5 candidates were banded out of a
200-CV pool drawn at random from the whole eval region — 52,562 CVs spread over 41 role
families. A random draw therefore lands in the JD's own family about 1 time in 20, and the
measured rate was **5.0%**. Lexical banding spread the draw across the similarity range, which
is what it was for, but the range itself was almost entirely wrong-domain: a Python JD's "high"
band was the most Python-sounding of 200 mostly-unrelated CVs.

What that costs is the whole instrument. A set where nearly every pair is an obvious negative
gives no system anything to be right or wrong about — every ranker puts the same two candidates
on top and Precision@5 separates nothing — while costing exactly the same human hours as a set
that discriminates.

**Now:** 10 candidates per JD, **8 sharing the JD's `Primary Keyword`**, 2 from another family.
Measured after the re-cut: **80.0% on-category, exactly 8 of 10 on all 40 JDs**, up from 5.0%.

Three details are load-bearing:

- **The 2 off-category candidates come from the top of their own pool, not at random**
  (`OTHER_BANDS = ("high", "mid")`). A random off-category CV is a trivial `No Fit` that every
  system already ranks last, so it separates nothing and wastes 2 of the 10 slots. A lexically
  similar CV from the wrong role family is the mistake a screening system actually makes.
- **The quota degrades visibly.** Three of the 41 families have fewer than 100 CVs in the eval
  region — Rust has 40 — and a targeted batch can name one. When the on-category pool cannot
  fill 8, the shortfall is taken off-category, those pairs carry `keyword_match=False`, and
  `summarise` reports `jds_short_of_quota` per batch. The alternative is a batch quietly
  reverting to the uniform draw this decision replaced.
- **`sample.assert_labels_survive` now refuses any rebuild that orphans a collected
  judgement.** This re-cut changed every `pair_id` in batch 1 and was safe *only* because
  `judgements.csv` held nothing but the pilot. "Nothing has been labelled yet" is exactly the
  kind of precondition that is true when someone writes the change and false when someone
  repeats it a fortnight later, so it is a check rather than a sentence in a plan file.

**Stated consequence, and it must travel with every in-domain precision figure:** this pool is
now *deliberately* enriched for on-category candidates and is further from a uniform sample of
the corpus than it was before. In-domain absolute precision was never an unbiased estimate of
production precision; it is now emphatically not one. It is a comparison instrument between
systems, which is what the proposal asks it to be.

# D32 — the A1 recheck moves into the UI, reversing D30

D30 kept the recheck on the flat dispatch file so that group size could not betray which pairs
were rechecks. **That reasoning was too strong, and the operator overruled it.**

The two corpora were already distinguishable before any of this: A1 resumes run to a median of
5,134 characters against Djinni's 1,525, and the guide names both corpora and says outright
that some pairs carry existing labels. Group size adds little to what document length already
announces, and the cost of the split was a second, separate workflow with no ingest path — the
exact defect the UI was built to remove.

**What is still protected.** No `a1_label` and no `selection_reason` reach the page. Which
pairs carry an existing label, and what that label says, remains invisible. A1 groups are
interleaved with in-domain groups in each annotator's own order, and a short group gets **no
banner** — naming it would be the disclosure this decision does not make.

**Residual limitation, to be reported with the A13 figure:** an annotator who notices that
short groups are the long US resumes can infer that those are the rechecked pairs. They still
cannot infer the label, so the anchoring risk is on the existence of a prior judgement, not on
its value. Verified live: over 30 consecutive screens the tool served 17 in-domain groups of 10
and 13 A1 groups of 1-4, and asked the shortlist question on the 17 only.

The shortlist question is gated by `Group.asks_shortlist`, which is false for two kinds of
group: a partial in-domain re-serve (deviation Y1 above) and **any** A1 group, since 1-4
resumes drawn from a 193-deep A1 pool is not a field anyone can pick a top candidate from.

## What this does to the budget — unresolved

| | Pairs | Judgements incl. 30% double | Effort |
|---|---|---|---|
| D25 as approved | 250 | ~310 | ~2.5 team-days |
| **Now** | **450** (400 in-domain + 50 recheck) | **588** | **~5-6 team-days** |

D25 capped manual annotation at one session of ~250 judgements out of a ~30 team-day project.
At 10 candidates per JD across 40 JDs the set is 400 in-domain pairs, and the session is
roughly double its approved size. The operator was shown this arithmetic and chose not to
reduce the JD count, so **the figure stands and D25's budget line is superseded**; it is
recorded here rather than absorbed silently, because D25 was itself the outcome of a
deliberate scoping exercise.

Halving it is one edit — `n_jds: 40` to `20` in `indomain-batches.json` — but **only while
`judgements.csv` is empty**. Once labelling starts, `assert_labels_survive` will refuse, and
correctly: shrinking the campaign would orphan every label collected against the JDs removed.
