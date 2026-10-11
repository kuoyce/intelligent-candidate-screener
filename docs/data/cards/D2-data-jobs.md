# D2 · `lukebarousse/data_jobs`

| | |
|---|---|
| Catalog entry | [D2](../data-catalog.md) — Tier 2, skill vocabulary only |
| Source | https://huggingface.co/datasets/lukebarousse/data_jobs |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/data-jobs/` — 75.3 MB, 1 file |
| Licence | **Apache-2.0** |
| PII | None *Superseded 6 Oct 2026:* no personal fields, but **1 `job_title` holds an email-like string** and 428 a long digit run (requisition-number shaped). Never display raw `job_title`; the P2 title normalisation cuts both away |
| Verified rows | 785,741 postings, 17 columns |

## Schema

`job_title_short`, `job_title`, `job_location`, `job_via`, `job_schedule_type`,
`job_work_from_home`, `search_location`, `job_posted_date`, `job_no_degree_mention`,
`job_health_insurance`, `job_country`, `salary_rate`, `salary_year_avg`, `salary_hour_avg`,
`company_name`, **`job_skills`**, `job_type_skills`.

## Correction to the dataset's framing — verified

**There is no job-description text column.** No column carries long-form text at all. It
cannot serve as a JD corpus; A2 Djinni does that job.

## Verified figures (19 Aug 2026)

**252 distinct skills** over 785,741 postings. Top 12 by posting count:

`sql` 384,849 · `python` 380,909 · `aws` 145,381 · `azure` 132,527 · `r` 130,892 ·
`tableau` 127,213 · `excel` 127,018 · `spark` 114,609 · `power bi` 98,147 · `java` 85,612 ·
`sas` 83,404 · `hadoop` 64,842

## Title fields — verified 6 Oct 2026 (P2)

- `job_title_short`: 100% filled, **10 classes**, all data or software roles: `Data Analyst`
  196,075 · `Data Engineer` 186,241 · `Data Scientist` 172,286 · `Business Analyst` 49,063 ·
  `Software Engineer` 44,929 · `Senior Data Engineer` 44,563 · `Senior Data Scientist`
  36,957 · `Senior Data Analyst` 29,216 · `Machine Learning Engineer` 14,080 ·
  `Cloud Engineer` 12,331.
- `job_title`: 785,740 of 785,741 filled, **234,666 distinct**, 94,401 after title
  normalisation; 9,419 multi-token normalised titles occurring in at least 5 postings form
  its part of the P2 text-search alias table.
- Exact-match coverage of normalised A2-train `Position`: **68.4% of JDs, 79.9% of CVs**
  (57.8% / 71.6% using only titles in ≥ 5 postings).

Read: a good title *recogniser* for IT titles, but its classes are 10 data/software roles,
so it cannot be the role taxonomy. See [`11-p2-data-checks.md`](../../../plan/2026-10-06-extraction-and-rerank/11-p2-data-checks.md).

## Handling trap

`job_skills` ships as a **stringified Python list**, not a list. Iterating a cell directly
yields *characters* and produces a silent vocabulary of 37 punctuation marks. Always parse
through `candidate_screener.data.profile.parse_skill_cell` /
`skill_frequencies` — this bug was found and fixed during the acquisition phase.

## How it is used

Frequency weighting for the skill vocabulary (action plan §3.5) and the coverage test that
quantifies ESCO's tech-tool gap — see [D1](D1-esco.md).

Explored in [`notebooks/06-skill-vocabulary.ipynb`](../../../notebooks/06-skill-vocabulary.ipynb).
