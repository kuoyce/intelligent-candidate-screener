# 05 — Implementation record (shortlist recall)

Branch `feat/shortlist-recall`. Every figure below is read from a committed record.

## Progress

| Phase | Status | Commit | Record |
|---|---|---|---|
| Q0 D54, `AGENTS.md` rules 5 and 6, plan committed | done | | `docs/decisions.md` |
| Q1 development set | done | (this commit) | `output/matching/shortlist-dev-set.json` |
| Q2 C1–C3 | code + tests; figures in the Q3 record | (this commit) | `output/matching/shortlist-dev.json` |
| Q3 Y1, Y2, Y2t, Y4 on dev | done: **early stop** (no Y beats X2 on dev) | (this commit) | `output/matching/shortlist-dev.json` |
| Q4–Q9 | waiting on the Q-R4 review (and now on Q-R6) | | |

Checks at this commit: `uv run pytest` 588 passed; `dev_pools --check`, `shortlist_dev --check`, `stage1 --check` all PASS.


## Q3 dev rules, committed before the first dev run

Written into `evaluation/shortlist_dev.py` (`RULES`, `choose`) and tested
(`tests/test_shortlist_dev.py`) before any Y rung was scored:

- **Options.** For each Y rung, the (`k`, role context) with the highest dev-select proxy
  Recall@50; ties at 6 decimals go to the cheaper option (smaller `k`, then context off).
- **Y2 vs Y2t.** The higher dev-select Recall@50 at each one's chosen option; ties to Y2.
- **Early stop (02 Q3).** The dev-best classical Y rung (Y1, Y2, Y2t) must beat X2's
  dev-select Recall@50 strictly (point estimate). Otherwise stop, record, and go to Q8.
- **Proxy relevance** is the plan's: same oracle role family, over all 500 CVs of a pool.
  A pilot-shaped figure (relevant only among the 10 seeded slots, the final set's shape)
  is reported beside it and decides nothing.

## Implementation choices not spelled out in the plan

| # | Choice | Why |
|---|---|---|
| I1 | Dev JDs: drawn seeded from dev JDs whose `Primary Keyword` has ≥ 8 dev CVs (so the 8 same-keyword slots can be filled) | The plan's pool design needs it; it removes JD-only keywords |
| I2 | The 490 random CVs come from **all** dev CVs (any keyword, the JD's included) | As the pilot drew distractors; relevance is labelled on all 500, so a same-family random CV counts |
| I3 | C1 returns probability 1.0 for a lookup hit. The classifier's probabilities are overconfident (count-weighted rows) and are **not used** by any rung | Plan asks for "one family, with its probability"; only the family enters a score |
| I4 | Role-context rule (C2): (a) line starts with the title (bullets, digits, punctuation allowed before), (b) a cue before it on the line (`position`, `role`, `title`, `occupation`, `worked/working/work as`, `as a(n)`, `i am a(n)`, `i'm a(n)`, optional `:` or dash), (c) a work-history header line (≤ 150 chars with a date range, P4's pattern) | The plan names the three cases; the exact wording is fixed here, before any dev figure |
| I5 | `title_similarity` compares `normalise_title(jd_title)` (lower-cased raw title if that is empty) with the CV spans' normalised alias titles, under the same role-context option as the role set | Seniority words would otherwise lower the similarity of the same role |
| I6 | Y2t's title term is `title_similarity` as is (already in [0, 1]); TF-IDF is min-maxed per JD, then the two are averaged | The plan's formula |
| I7 | TF-IDF's IDF is the shared A2 vectoriser (a seeded 20k train-region sample). It may include dev documents; IDF is unsupervised | Same TF-IDF as X2, so the comparison isolates the role term |

## Q2 component figures (record `components`)

**C1, JD title → family** (accuracy against the oracle family):

| Set | n | C1 | lookup share / its accuracy | classifier fallback accuracy | P3 text classifier (argmax), same JDs |
|---|---|---|---|---|---|
| dev JDs | 400 | 0.768 [0.725, 0.808] | 0.625 / 0.900 | 0.547 | 0.778 [0.738, 0.818] |
| final-set JDs, generic | 40 | 0.800 [0.675, 0.925] | 0.675 / 0.852 | 0.692 | 0.775 |
| final-set JDs, targeted | 20 | 0.650 [0.450, 0.850] | 0.550 / 0.909 | 0.333 | 0.500 |
| final-set JDs, total | 60 | 0.750 [0.633, 0.850] | 0.633 / 0.868 | 0.545 | 0.683 [0.567, 0.800] |

Final-set rows are measurement only (oracle = the documents' metadata; no final label is
read). On dev the title is **no more accurate** than P3's JD text classifier. The lookup is
right 90% of the time, but the title classifier fallback (37.5% of JDs) is right only 55%.

**C2, CV role set** on the 14,170 dev CVs (any span 0.931):

| k | context | recall (oracle ∈ set) | mean set size | empty |
|---|---|---|---|---|
| 0 | off | 0.730 | 2.64 | 0.069 |
| 0 | on | 0.325 | 0.65 | 0.541 |
| 1 | off | 0.858 | 2.84 | 0 |
| 1 | on | 0.809 | 1.30 | 0 |
| 2 | off | 0.907 | 3.41 | 0 |
| 2 | on | 0.893 | 2.20 | 0 |
| 3 | off | 0.932 | 4.14 | 0 |
| 3 | on | 0.926 | 3.15 | 0 |

The role-context rule (I4) keeps under half the CVs' spans. With `k ≥ 1` it narrows
the set a lot (2.84 → 1.30 at k = 1) for a small recall cost.

## Q3 dev results: early stop

Proxy Recall@50, dev-select (150 JDs; CI resamples JDs). dev-train beside it.

| Rung | Chosen option | dev-select R@50 [CI] | R@10 | dev-train R@50 | pilot-shaped R@50 |
|---|---|---|---|---|---|
| TF-IDF | — | 0.459 [0.420, 0.498] | 0.200 | 0.481 | 0.466 |
| SBERT weak | — | 0.501 [0.463, 0.539] | 0.195 | 0.531 | 0.503 |
| **X2** (P7, registered) | — | **0.727** [0.685, 0.767] | 0.350 | 0.713 | 0.734 |
| Y1 | k = 2, context on | 0.605 | | 0.582 | |
| **Y2** | k = 1, context on | **0.664** [0.618, 0.706] | 0.296 | 0.657 | 0.658 |
| Y2t | k = 2, context on | 0.647 | | 0.651 | |
| Y4 | k = 1, context on | 0.678 [0.632, 0.721] | 0.287 | 0.673 | 0.676 |

Every option of every rung is in the record. Y2 beats Y2t (`Y2_vs_Y2t`).

**Early stop (rule fixed beforehand):** the dev-best classical Y is Y2 at 0.664. It does
not beat X2's 0.727, so the decision is **stop**. Per 02 Q3 the next step would be to
record and go to Q8. Q8 is held for the review (Q-R4, Q-R6). No Y rung has been scored on
the final set.

### Why the hard filter loses (exploratory, dev only, decides nothing)

This is an ad hoc script (scratchpad, not in the package) run after the early stop. It uses
dev-select only and reads no final-set data. It splits the 150 dev-select JDs by whether C1
gave the right family:

| C1 | JDs | X2 R@50 | Y2 R@50 |
|---|---|---|---|
| right | 116 | 0.780 | 0.769 |
| wrong | 34 | 0.546 | **0.305** |

| C1 / P3 text classifier | JDs | X2 | Y2 |
|---|---|---|---|
| both right | 100 | 0.808 | 0.771 |
| title right, text wrong | 16 | 0.601 | **0.755** |
| title wrong, text right | 19 | 0.760 | 0.406 |
| both wrong | 15 | 0.273 | 0.177 |

When C1 is wrong, the median pool has **0** relevant CVs in the matched level (vs 18 of 22
when right). The hard filter then actively ranks the right role below the wrong one. The
soft `role_match` degrades gracefully instead. Where the title is right and the text
classifier wrong, the title helps a lot (0.755 vs 0.601). But the fallback title
classifier is wrong often enough that this is outweighed. Even where C1 is right, Y2 does
not beat X2 (0.769 vs 0.780), because the CV set misses the family for ~19% of CVs at
k = 1 with context on.

Reading: **the JD title helps only where the lookup hits.** A hard match punishes every C1
error, and the fallback errors are frequent.

## Deviations

- **Workflow.** Implemented inline rather than by sub-agents. Each phase was short, and
  most of the time went into one long dev run, which sub-agents would not have shortened.
  The plan allows this (02, last paragraph).
- **Q8 not run.** By Q-R4 (review after this phase), and because the early stop moves the
  decision to the user (Q-R6).

## Open

| ID | Question | Why it matters | Recommendation |
|---|---|---|---|
| Q-R6 | The early stop fired (Y2 0.664 < X2 0.727 on dev). Options: (a) **stop as planned**: freeze 03 §2 with the dev choices, score once on the final set at Q8, and report post-hoc Y beside the registered P7 miss; (b) **stop and skip Q8**: record the dev result only, keep X2 for P8.5 (Q-P7-2 (a)), return to P8; (c) **one new dev variant**, post-hoc and registered first, that follows the diagnosis: use the title family only on a **lookup hit**, else P3's JD text distribution, and keep the CV side soft (`Σ p_jd p_cv` with the JD side one-hot on a hit). Dev only; continue only if it beats X2 on dev-select | (a) spends a final-set look on rungs already known to be worse on dev. (c) is the one change the diagnosis points to, but it is a further post-hoc step | **(b)**, with (c) only if you want one more try. Q4/Q5 (main tech, learned combination) are not worth starting while the role term is the bottleneck |

## Amendment Z: three explorations (user, 7 Oct), rules fixed before the run

The user asked for three explorations: (1) the soft alternative to the hard filter, read
as P7 §6a's "larger role weight"; (2) a hard filter on the JD side's top 3; (3) the title
only on a lookup hit, else the full JD text. They are tested as five families, in
`matching/role_filter.py` (`SoftRoleWeight`, `JdSetFilter`), scored by
`shortlist_dev --variants` into `output/matching/shortlist-dev-variants.json`:

| Family | JD side | Match | Options (chosen on dev-select R@50, ties to cheaper) |
|---|---|---|---|
| Z1 | text classifier probabilities | soft `w × Σ p_jd p_cv` | w ∈ {2, 4, 8, 16} |
| Z1L | one-hot title family on a lookup hit, else text probabilities | soft | w ∈ {2, 4, 8, 16} |
| Z2 | text top 3 | hard, vs CV role set | k ∈ 0–3 × context |
| Z2L | title family on a hit, else text top 3 | hard | k × context |
| Z3 | title family on a hit, else text top 1 | hard | k × context |

A family "beats X2" when its chosen option's dev-select Recall@50 is strictly above X2's
(0.727). Recall@50 split by lookup hit / miss is reported for each choice and for X2
(exploratory). **Assumption to confirm:** item (1) read as a larger soft weight.
