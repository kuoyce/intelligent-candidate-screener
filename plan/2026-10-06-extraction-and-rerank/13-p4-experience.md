# 13 — P4 implementation record: E3 experience years, E4 seniority

Tasks 4.1–4.4 of `03-work-plan.md` P4 (design: `02-design-spec.md` §1, §3.1, §3.4).
Measurement only. Nothing here is a rung or a held-out result: X1 (§3.4) is scored in P7.

| File | What |
|---|---|
| `src/candidate_screener/extraction/experience.py` | The rules: `jd_experience`, `cv_experience`, `cv_ranges`, `merged_years`, `jd_band`, `cv_band`, `experience_compatible_extracted`, `position_seniority`, `text_seniority` |
| `src/candidate_screener/extraction/experience_eval.py` | The record and its CLI (`--evaluate`, `--check`) |
| `output/extraction/experience-metrics.json` | The committed record |
| `tests/test_extraction_experience.py` | 46 synthetic tests, incl. the named invariants |

```bash
uv run python -m candidate_screener.extraction.experience_eval --evaluate
uv run python -m candidate_screener.extraction.experience_eval --check
uv run pytest tests/test_extraction_experience.py
```

## 1. Rules (full text in the module docstring)

Input text is `text_v2` throughout: JD `Long Description`; CV `CV`, `Highlights`, `Moreinfo`,
`Looking For` (no `Position`). The metadata fields are read only as labels.

**JD requirement → (min_years, max_years | None, spans).** Mention shapes, longest first:
range `2-5 years` → (2, 5); prefix `at least / minimum / from / over N years` → (N, None);
`up to N years` → (0, N); plus `5+ years` / `5 years+` → (5, None); plain `3 years` → (3, None).
Months become fractions of a year; numbers 1–2 digits, decimals (`0,5`) or words one..twenty.
A mention counts only in an experience context (a cue such as *experience*, *commercial*,
*with*, *as a* within 40 chars before / 60 after on the line, or a range/plus/prefix mention
opening a bullet line). Company-context mentions (*we, our, company, market, days* … in the
window) are dropped unless the line addresses the candidate (*you, someone, looking for,
need* …). **Aggregation: the largest minimum among required mentions**; mentions on a
nice-to-have line or under a nice-to-have header are used only if nothing else is stated.
No number but "no experience required" → 0.

**Rule for a year range in a JD (invariant 4):** a number of 21 or more is never a
requirement, and a mention needs a `year`/`month` unit, so `2019-2022` (a date) is not one.

**CV years → (years, source, spans).** The largest explicit statement (same shapes, always
with a cue, never `N years old` / `ago`) wins. Without one, date ranges (`2019 – 2022`,
`Jan 2020 - present`, `03.2018–05.2021`) are merged where they overlap and summed. A year alone
is mid-year (so `2019 - 2022` = 3 years); a month runs from its first to its last day. A range
on or next to an education line is skipped; one starting before 1970 or after the reference,
or longer than 45 years, is dropped; an end after the reference is capped.
**`present` = `REFERENCE_DATE` = 2023-09-01**, the latest `Published` date of an A2 JD (the
corpus spans 2020–2023), passed as a parameter; nothing reads the wall clock.

**Bands.** JD years snap down to Djinni's `Exp Years` scale (<1 `no_exp`, <2 `1y`, <3 `2y`,
<5 `3y`, else `5y`; there is no 4-year option, so 4 → `3y`) and go through `JD_EXP_TO_BAND`;
CV years go through `cv_experience_band`. Both are imported from `data.a2_finetune`.

**Missing data.** `experience_compatible_extracted(jd_texts, cv_texts)` builds the two band
columns and calls `matching.corpus.experience_compatible` itself, so the semantics are the
1a rule's by construction: **unknown on either side → 0**. That is what 1a does with a
missing field; "unknown = compatible" would credit a document for saying nothing. The cost
of this choice is large (§3) and is the main open item (§5, Q-P4-1).

**Seniority (E4).** Levels intern < junior < middle < senior < lead. Tokens are
`title_profile.SENIORITY` (asserted equal), `strong` is a modifier, not a level. Ground
truth: the highest level among `Position`'s tokens (`lead generation` is not a level);
documents whose `Position` has no token are excluded. From text: a token that heads a title
(a singular role noun within four tokens on the same line, or a standalone title token:
*team lead, head of, intern, trainee*); plural nouns (`senior engineers`) are colleagues.
Fallback from years: <2 junior, <5 middle, else senior.

**Variant selection (A2 train only).** JD aggregation `max` vs `first`, and seniority span
`first` vs `max`, were compared on the train sample by `accuracy_all`; the winner is applied
to the final-set documents (ties keep the a-priori default, `max` / `first`). Chosen: JD
`max` (0.554 vs 0.552), seniority `max` on both sides (JD 0.466 vs 0.455; CV 0.346 vs 0.335).
One rule change was made after the first train run and before the final-set documents were
scored: the company-context filter was narrowed (it had dropped "We need someone with 3+
years …"; JD coverage 0.640 → 0.659 on train).

## 2. Named invariants (task 4.3), each shown failing against a naive implementation

Shown by a throwaway pytest plugin that monkeypatches the module (not committed) and runs
`tests/test_extraction_experience.py`:

| Invariant | Test | Naive implementation | Result |
|---|---|---|---|
| Overlapping ranges not double-counted | `test_overlapping_date_ranges_are_not_double_counted`, `test_nested_range_adds_nothing` | `merged_years` = sum of range lengths | both fail (5.0 ≠ 4.0) |
| `5+ years` is a minimum, no maximum | `test_plus_and_at_least_parse_as_minimum_without_maximum` (5 cases) | one `(\d+)\s*\+?\s*years?` regex returning (N, N) | all 5 fail (plus 6 other JD tests) |
| Open range uses the reference date | `test_open_range_uses_the_reference_date_not_today`, `test_range_ending_after_the_reference_is_capped` | "present" = `date.today()` | both fail |
| A JD date range is not experience | `test_dates_and_company_age_are_not_a_requirement` (2 date cases) | `(\d+)\s*[-–]\s*(\d+)` → (2019, 2022) | both date cases fail |
| Bands match the metadata | `test_jd_four_years_snaps_down_to_3y` (+ parametrised `JD_EXP_TO_BAND` / `cv_experience_band` tests) | `jd_band = cv_experience_band` (one numeric banding) | fails (4 → 4-6) |
| Unknown → 0, as 1a | `test_experience_compatible_extracted_is_the_1a_rule_with_unknown_as_zero` | `experience_compatible` with unknown = compatible | fails |

## 3. Results (`output/extraction/experience-metrics.json`)

Document shares: Wilson 95%. Pair figures: percentile bootstrap over JDs, 10,000 resamples,
seed 0. `accuracy_all` counts a document with no extraction as wrong; `accuracy_covered`
is over documents with an extraction.

### 3.1 A2 train (seeded sample, 2,000 JDs and 2,000 CVs, D19)

| Side | Coverage | Band accuracy (all) | Band accuracy (covered) | Within one band (covered) |
|---|---|---|---|---|
| JD (`Exp Years`) | 0.659 [0.637, 0.679] | 0.554 [0.532, 0.575] | 0.841 [0.820, 0.859] (n=1,317) | 0.995 |
| CV (`Experience Years`) | 0.444 [0.422, 0.466] | 0.301 [0.281, 0.322] | 0.678 [0.647, 0.708] (n=888) | 0.899 |

CV by source: explicit statement 0.741 [0.706, 0.772] (n=686); date ranges 0.465
[0.398, 0.534] (n=202). Errors on statements are mostly genuine statements of non-IT years
("14 years of work experience from sales") where the Djinni field counts IT years.

Seniority (documents whose `Position` carries a token: JD 837/2,000, CV 567/2,000):

| Side | Title-token coverage | Token accuracy | Years-fallback accuracy | Coverage with fallback | Accuracy (all) |
|---|---|---|---|---|---|
| JD | 0.280 | 0.735 [0.675, 0.788] (n=234) | 0.517 (n=422) | 0.784 | 0.466 [0.432, 0.500] |
| CV | 0.351 | 0.658 [0.590, 0.721] (n=199) | 0.458 (n=142) | 0.601 | 0.346 [0.308, 0.386] |

### 3.2 Final-set documents (60 JDs, 600 CVs of the 600 pairs; no labels read)

| Side | n | Coverage | Band accuracy (all) | Band accuracy (covered) | Within one (covered) |
|---|---|---|---|---|---|
| JD | 60 | 0.600 [0.474, 0.714] | 0.417 [0.301, 0.543] | 0.694 [0.531, 0.820] (n=36) | 0.917 |
| CV | 600 | 0.467 [0.427, 0.507] | 0.335 [0.298, 0.374] | 0.718 [0.663, 0.767] (n=280) | 0.918 |

Pair-level `experience_compatible`, extracted vs oracle (true fields):

| Scope | Pairs (JDs) | Agreement [95% CI] | κ [95% CI] | Compatible rate oracle / extracted | Both bands extracted | Agreement where both extracted |
|---|---|---|---|---|---|---|
| generic | 400 (40) | 0.465 [0.368, 0.563] | 0.151 [0.074, 0.249] | 0.763 / 0.253 | 119 | 0.924 |
| targeted | 200 (20) | 0.375 [0.255, 0.495] | 0.028 [−0.048, 0.132] | 0.765 / 0.230 | 56 | 0.732 |
| total | 600 (60) | 0.435 [0.357, 0.513] | 0.109 [0.050, 0.181] | 0.763 / 0.245 | 175 | 0.863 |

Of the 339 disagreements, 325 are extracted 0 / oracle 1: a side with nothing extracted
scores 0 under the 1a rule. Where both bands are extracted (175 pairs) agreement is 0.863.
The "both extracted" columns were added after the first final-set build, as description;
no rule changed after it.

Seniority (final-set documents with a `Position` token: JD 20/60, CV 152/600): JD accuracy
(all) 0.50 [0.30, 0.70], token accuracy 5/6; CV accuracy (all) 0.329 [0.259, 0.407], token
accuracy 0.625 [0.484, 0.748] (n=48).

## 4. Deviations

- Rules and evaluation are two modules (`experience.py` pure, `experience_eval.py` the
  record), rather than one; the CLI is on the evaluation module.
- Two aggregation variants per entity were compared on A2 train and one selected by a
  pre-stated rule (§1). The work plan did not list this; it uses train only (D46).
- Train sample uses `corpus.a2_train_documents(2000, 2000, seed=0, "v2")`, the same sampler
  the ladder uses (its cache lands in `data/processed/matching/`).

## 5. Open items

- **Q-P4-1 (high impact for X1).** Under the unchanged 1a semantics, unknown → 0 makes
  `experience_compatible(extracted)` 0 on ~75% of pairs (oracle: 24%), so X1's experience
  half mostly says "unknown". Options for the user: keep 1a unchanged (spec §3.4 says
  "unchanged"); or register a variant where an unknown side is compatible (agreement where
  both are extracted is 0.86). Must be decided and registered before P7 scores X1.
- CV coverage (0.44–0.47) is bounded by the text: Djinni's `Experience Years` is a form field
  that many CVs never restate. Date ranges are rare (≈7–10% of CVs) and the weaker source.
- JD 4-year snapping (`3y`) is a priori; 4-year requirements are rare enough that it barely
  moves the confusion.
- Seniority from text is weak (token coverage ~0.3); as a reranker feature it should be
  used with its source (`token` / `years`), not as a hard filter.
