# 11 — P2 data checks: the evidence for Q-B1 (tasks 2.1–2.3)

*6 Oct 2026. Measurements only. The taxonomy is the user's decision (P2.4, 05 Q-B1). The
span-vs-classifier read in §6 is a **recommendation**.*

Reproduce:

```bash
uv run python -m candidate_screener.extraction.title_profile            # writes the record
uv run python -m candidate_screener.extraction.title_profile --check    # rebuild and diff
uv run python -m candidate_screener.data.profile --check titles         # 2.1 into profile-metrics.json
```

Record: `output/extraction/title-profile.json` (deterministic, no timestamps). Code:
`src/candidate_screener/extraction/title_profile.py`. Tests: `tests/test_title_profile.py`
(synthetic only). The 2.1 figures also go into `docs/data/profile-metrics.json` under
`checks.titles` (a partial `profile --check` now merges rather than overwrites, and records
an `updated` timestamp per check).

**Region.** Everything on A2 uses the D19 **train** region only: 106,423 JDs and 157,688 CVs.
`load_split` reads the whole parquet and eval-region rows are filtered out by id before any
field is used.

## 1. The normalisation rule (defined here, used for titles, aliases and text alike)

1. Lower-case. `.net` → `dotnet`, `x.js` → `xjs`. Tokens are `[a-z0-9]+` with an optional
   `++`/`#` suffix (`c++`, `c#` survive).
2. Phrase merges: `front end`→`frontend`, `back end`, `full stack`, `ui ux`/`ux ui`→`uiux`,
   `react native`, `quality assurance`→`qa`, `machine learning`→`ml`. Token synonyms:
   `golang`→`go`, `js`→`javascript`, `reactjs`→`react`, `programmer`/`dev`→`developer`, ….
3. `engineer`/`developer`/`programmer` after a technology or a development qualifier
   (`frontend`, `software`, `web`, `mobile`, `ios`, …) becomes `developer`.
   `data engineer`, `qa engineer`, `devops engineer` keep their noun.
4. **Titles only:** cut at the first `,` `|` ` - ` `(` `[` `@` ` / ` (what follows is stack,
   location or company); drop seniority and level tokens (junior, jr, middle, mid, senior,
   sr, strong, lead — except in `lead generation` —, team/tech lead, head (of), principal,
   chief, trainee, intern, internship) and trailing level markers (`i`–`iv`, `1`–`5`,
   `l1`–`l5`, `level`).

Examples: `Middle+ Python Engineer (Django)` → `python developer`; `UX/UI designer` →
`uiux designer`; `Senior Software Engineer - Fintech` → `software developer`;
`Head of Sales` → `sales`; `Team Lead .NET` → `dotnet`; `Trainee` → `` (empty).

- **T-B1** = the normalised title, language and framework tokens **kept**.
- **T-B2** = T-B1 with language/framework tokens **dropped** (`java developer` → `developer`),
  shown only to say what a function-only taxonomy would look like.
- A T-B family is a normalised title with **≥ m train JDs and ≥ m train CVs**. Headline
  m = 20; m = 10 and 50 are reported.

**Known weakness (not fixed, deliberately simple):** word order is not normalised, so
`qa automation engineer`, `automation qa engineer` and `qa automation` are three families;
`devops` and `devops engineer` are two. A T-B taxonomy would need a merge step (alias
table, token-set key, or a short hand-made list) before use. Fan-out figures below are
therefore **upper** estimates of real role diversity.

*Note:* P1's `evaluation/text_leak.py` (another agent, same day) has its own
`normalise_title` (seniority tokens only). The two rules differ; P3 should use one.

## 2. Task 2.1 — title and role fields in the other sources

| Source / field | Rows | Fill | Distinct | IT coverage | PII in the field | Usable for |
|---|---|---|---|---|---|---|
| ESCO `occupations` `preferredLabel` + `altLabels` | 3,043 | 100% / 100% | 3,039 preferred; **30,417** alt labels; 31,116 normalised strings | ICT (ISCO 25/35/133): **110** occupations, 749 alt labels | none (0 email/phone-like) | alias table, part; **not** a taxonomy (EU occupations, thin on IT) |
| `data_jobs` `job_title_short` | 785,741 | 100% | **10** | all 10 are data/software roles | none | coarse label for data roles only |
| `data_jobs` `job_title` | 785,741 | 99.9999% | 234,666 (94,401 normalised) | mostly data/software | **1 email-like**, 428 long digit runs (requisition ids) | title recogniser / alias table |
| livecareer (C1 = C2) `Category` | 2,484 | 100% | **24** | **1** value (`INFORMATION-TECHNOLOGY`, 120 rows) | none in field; documents are real resumes | **not usable** for IT role labels |
| resume-atlas `Category` | 13,389 | 100% | **43** (47–410 rows each) | ~16 IT values (`Java Developer`, `DotNet Developer`, `Python Developer`, `React Developer`, `DevOps`, `Testing`, `Data Science`, …) | none in field | coarse classifier pre-training at most; text is pre-normalised, so no spans |

Example values (titles/categories only): ESCO ICT `ICT security administrator`,
`software analyst`, `integration engineer`; `data_jobs` `Data Engineer`, `Data Scientist`,
`Data Analyst`; resume-atlas `Education`, `Java Developer`, `DevOps`.

**Alias-table coverage of normalised A2-train `Position`** (exact string match; share of
documents, 95% Wilson CI; distinct-string share in brackets):

| Alias source | JDs (n = 105,628 non-empty) | CVs (n = 154,958 non-empty) |
|---|---|---|
| ESCO, all occupations | 24.6% [24.4, 24.9] (2.6%) | 42.6% [42.3, 42.8] (4.0%) |
| ESCO, ICT only | 12.7% (0.7%) | 17.9% (1.0%) |
| `data_jobs`, all titles | 68.4% [68.1, 68.7] (11.6%) | 79.9% [79.7, 80.1] (13.2%) |
| `data_jobs`, titles in ≥ 5 postings | 57.8% (5.1%) | 71.6% (6.4%) |
| **ESCO ∪ `data_jobs`, all** | **70.5%** [70.2, 70.8] (12.9%) | **83.8%** [83.6, 84.0] (15.2%) |
| ESCO ∪ `data_jobs` ≥ 5 | 62.1% (6.9%) | 77.8% (9.1%) |

The lookup recognises the common titles (most documents) and misses the long tail (most
distinct strings). Recognition is not family assignment: `data_jobs` maps a title to one of
10 data/software classes and ESCO to an EU occupation, neither of which is an IT role
taxonomy. **Neither external source can supply the taxonomy; A2-train `Position` must.**

## 3. Task 2.2 — T-A (`Primary Keyword`) vs T-B (normalised `Position`)

### 3.1 Normalisation effect

| | JDs | CVs |
|---|---|---|
| Raw distinct `Position` (lower-cased) | 50,661 | 36,817 |
| Normalised distinct (T-B1) | 21,197 | 12,532 |
| Function distinct (T-B2) | 18,643 | 10,780 |
| Empty after normalisation (title was only seniority, e.g. `Team Lead`) | 795 | 2,730 |
| Title carries a seniority token | 43.2% | 30.0% |

### 3.2 The comparison table

| | T-A JD | T-A CV | T-B1 m=20 JD | T-B1 m=20 CV |
|---|---|---|---|---|
| Classes | 45 (**4 with 1 JD**: `Block-chain`, `SAP`, `React`, `Product Owner`) | 41 | **298** (shared by construction) | 298 |
| Min / median / max class size | 1 / 1,395 / 13,390 | 120 / 2,321 / 25,448 | 20 / 76 / 3,582 | 20 / 78 / 14,663 |
| Documents with a class | 100% (`Other` 5.3%) | 99.98% (30 null; `Other` 2.5%) | **65.3%** [65.0, 65.6] | **82.3%** [82.1, 82.5] |

Sensitivity of T-B to *m*:

| m | T-B1 classes | JD / CV coverage | median size JD / CV | T-B2 classes | T-B2 JD / CV coverage |
|---|---|---|---|---|---|
| 10 | 479 | 68.9% / 84.8% | 39 / 40 | 371 | 70.8% / 84.5% |
| 20 | 298 | 65.3% / 82.3% | 76 / 78 | 217 | 67.6% / 82.3% |
| 50 | 159 | 59.3% / 77.5% | 162 / 180 | 120 | 63.8% / 78.9% |

Dropping language tokens (T-B2) removes few classes and adds almost no coverage: the
uncovered third of JDs is a long tail of distinct titles, not language variants.

### 3.3 Keyword → family fan-out (m = 20; share of the keyword's documents)

| Keyword | JD n | in a family | families | families for 80% | top families (JD) | CV: families / for 80% / top |
|---|---|---|---|---|---|---|
| JavaScript | 13,390 | 0.77 | 99 | 9 | frontend dev .26, react dev .09, fullstack dev .06, angular dev .06 | 103 / 5 / frontend dev .57 |
| Java | 6,568 | 0.74 | 65 | 5 | java dev .47, java software dev .04, backend dev .04 | 80 / 3 / java dev .65 |
| DevOps | 5,956 | 0.67 | 36 | 2 | devops engineer .50, devops .15 | 40 / 2 / devops engineer .54 |
| .NET | 5,806 | 0.69 | 60 | 8 | dotnet dev .32, fullstack dev .07, c# dev .04 | 68 / 6 / dotnet dev .49 |
| **Other** | 5,623 | **0.35** | **193** | **55** | technical writer .02, fullstack dev .02, salesforce dev .02, account manager .01 | 145 / 40 / project manager .04 |
| QA Automation | 5,268 | 0.67 | 32 | 7 | qa automation engineer .15, automation qa engineer .14, qa automation .08 | 34 / 4 |
| Marketing | 5,242 | 0.50 | 72 | 20 | marketing manager .06, digital marketing manager .03, copywriter .03 | 82 / 18 |
| QA | 5,126 | 0.75 | 31 | 3 | qa engineer .29, manual qa engineer .21 | 56 / 2 / qa engineer .64 |
| Node.js | 4,768 | 0.79 | 52 | 6 | nodejs dev .27, fullstack dev .16, backend dev .11 | 52 / 6 |
| PHP | 4,366 | 0.73 | 46 | 8 | php dev .31, fullstack dev .07, wordpress dev .05 | 53 / 6 |
| **Python** | 4,232 | 0.71 | 57 | 5 | python dev .43, backend dev .05, fullstack dev .04 | 69 / 4 / python dev .52, python django dev .11 |
| Project Manager | 3,454 | 0.72 | 28 | 3 | project manager .49, it project manager .06, scrum master .03 | 76 / 1 |
| **Lead** | 1,386 | **0.38** | 82 | 19 | solution architect .07, cto .04, engineering manager .03, architect .03 | 91 / 16 / cto .14 |
| **Data Science** | 1,719 | 0.59 | 26 | 4 | **data scientist .24, data engineer .14, ml engineer .08, data analyst .06** | 41 / 2 / data scientist .55, ml engineer .11 |
| Data Engineer | 1,085 | 0.60 | 27 | 2 | data engineer .48, big data engineer .07 | 19 / 2 |
| Data Analyst | 593 | 0.49 | 18 | 2 | data analyst .34, product analyst .05, bi analyst .02 | 27 / 1 |

All keywords: median 26 families per keyword (JD), 34 (CV); 41 keywords have ≥ 5 families
on each side. **Purity** (share of a family's documents in its modal keyword, document
weighted): **0.83 JD, 0.89 CV**; 105 JD / 88 CV families fall below 0.8, the least pure
being `software developer` (0.15 JD), `fullstack` (0.24), `bi developer` (0.24),
`mobile developer` (0.28). T-B mostly nests inside T-A, with a set of cross-cutting
families (generic developer, fullstack, mobile, BI) that T-A splits by language.

**Is `Python` too generic?** Less than expected on the JD side: python developer is 43% of
its JDs and the remainder is backend/fullstack, not data. The keyword that *is* generic is
**`Data Science`** (scientist, data engineer, ML engineer and analyst together), plus
**`Lead`** (architect, CTO, engineering manager) and **`Other`** (193 families, 35% mapped).

## 4. Task 2.3 — is the title in the text?

Seeded sample (seed 0 for JDs, 1 for CVs; ids sorted before permuting), **n = 300 per side**,
train region. JD text = `Long Description`; CV text = `CV`, `Highlights`, `Moreinfo`,
`Looking For` (`text_v2`, `Position` removed). Matching is token-bounded on the shared rule
(§1). "First 200" = the earliest match starts in the first 200 characters. 95% Wilson CI
over documents.

| Measure | JD: found | JD: first 200 | JD: later only | CV v2: found | CV v2: first 200 | CV v2: later only |
|---|---|---|---|---|---|---|
| Own normalised `Position`, exact (n = 300) | **32.3%** [27.3, 37.8] | 18.7% [14.7, 23.5] | 13.7% | **27.0%** [22.3, 32.3] | 16.3% [12.6, 20.9] | 10.7% |
| Own T-B1 family title (JD n = 201, CV n = 250 with a family) | 36.3% [30.0, 43.2] | 21.4% | 14.9% | 29.2% [23.9, 35.1] | 18.0% | 11.2% |
| Any alias title, ESCO (≥ 2 tokens) | 31.0% | 12.7% | 18.3% | 43.0% | 21.0% | 22.0% |
| Any alias title, `data_jobs` ≥ 5 postings (≥ 2 tokens) | 73.3% | 27.7% | 45.7% | 62.0% | 37.0% | 25.0% |
| Any alias title, A2-train families (≥ 2 tokens) | 50.3% | 22.7% | 27.7% | 54.7% | 30.3% | 24.3% |
| **Any alias title, union** | **78.3%** [73.3, 82.6] | 34.0% | 44.3% | **68.7%** [63.2, 73.7] | 42.7% | 26.0% |
| Earliest multi-token family title **is** the own family (JD n = 94, CV n = 132) | **58.5%** [48.4, 67.9] | | | **42.4%** [34.3, 50.9] | | |

Caveats. Exact matching is a **lower** bound for span finding: a paraphrase (`backend
engineer with Python` against `Python Developer`) is not counted. "Any alias" is an
**upper** bound on usefulness: precision of those matches is not measured, and JDs name
other roles ("you will work with the project manager"). The alias table has 39,233
multi-token strings (ESCO 29,986, `data_jobs` 9,419, A2 families 246).

## 5. What each taxonomy option implies (for the user's decision, P2.4)

**T-A — `Primary Keyword` as given.**
- 45 JD / 41 CV classes, effectively **41** (4 JD classes are singletons); every document
  labelled; no extra construction; the oracle (1a) already uses it, so X1 vs oracle is a
  like-for-like comparison.
- Classes are language/stack-centric. `role_match` on T-A says a Python data-engineer JD
  matches a Python backend CV, and puts `Data Science` scientists, data engineers and ML
  engineers in one class; `Lead` and `Other` are heterogeneous (§3.3).
- A classifier has ample data per class (CV min 120).

**T-B — normalised `Position` families (m = 20).**
- ~298 classes before any merge step, fewer after word-order/synonym merging (which is
  still to be built: §1 weakness). Covers **65% of JDs and 82% of CVs**; the rest need an
  "unmapped" class or a nearest-family rule, which is a new design choice.
- Finer where T-A is generic (Data Science, Lead, Other), but also splits roles T-A
  correctly joins (`qa automation engineer` / `automation qa engineer`), and produces
  cross-cutting families (`software developer`, `fullstack`) with purity 0.15–0.3.
- Exact `role_match` gets much harsher with ~300 classes; a classifier gets a median of ~76
  train JDs per class. The oracle would have to be redefined on T-B families (the 1a
  oracle's meaning changes) and the hand-check (Q-B2) becomes the only ground truth for
  the merge.

A middle option exists and is **only noted, not proposed**: T-A with the generic keywords
(`Data Science`, `Lead`, `Other`) split by their top T-B families. It would need its own
definition and a decision.

## 6. Span extraction vs classifier — recommendation (the decision is the user's)

**Recommendation: a text → family classifier carries E2; title spans are a feature and the
evidence shown in the UI, not the label source.**

- With `Position` removed, the document's own title is in the text in only **32% of JDs and
  27% of CVs** (same-family: 36% / 29%). A span extractor's recall is capped near those
  figures under exact matching.
- Where the text does contain family titles, the **earliest** one is the document's own only
  **59%** (JD) and **42%** (CV) of the time: CVs name past roles, JDs name colleagues. A
  "first title wins" rule would be wrong about half the time it fires.
- Fewer than one in five documents have their own title in the first 200 characters
  (19% JD / 16% CV), so a "headline" rule is not enough either.
- A classifier reads the whole text, is trained on A2-train labels for free (D19), and can
  take span hits as features. This holds for either taxonomy.

## 7. Deviations and open items

- **Deviation (minor):** `profile.main` now merges a partial `--check` into the existing
  `profile-metrics.json` instead of overwriting it (otherwise `--check titles` would have
  deleted the other eight sources' figures). Tested (`test_profile_merges_partial_runs`).
- **Open:** the word-order/synonym merge for T-B families (only needed if T-B is chosen).
- **Open:** two title normalisers now exist (`extraction.title_profile`, P1's
  `evaluation.text_leak`); reconcile in P3.
- **Open:** precision of alias matches in text is unmeasured (it would need a hand-check,
  cf. Q-B2).
- **Found:** 30 train CVs have a null `Primary Keyword`; 4 JD keywords are singletons.
