# 05 — Implementation record (shortlist recall)

Branch `feat/shortlist-recall`. Every figure below is read from a committed record.

## Progress

| Phase | Status | Commit | Record |
|---|---|---|---|
| Q0 D54, `AGENTS.md` rules 5 and 6, plan committed | done | | `docs/decisions.md` |
| Q1 development set | done | (this commit) | `output/matching/shortlist-dev-set.json` |
| Q2 C1–C3 | code + tests; figures in the Q3 record | (this commit) | `output/matching/shortlist-dev.json` |
| Q3 Y1, Y2, Y2t, Y4 on dev | rules committed before the first run | | `output/matching/shortlist-dev.json` |
| Q4–Q9 | waiting on the Q-R4 review | | |

## Open

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
