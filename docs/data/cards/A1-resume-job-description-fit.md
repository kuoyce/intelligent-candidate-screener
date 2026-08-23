# A1 · `cnamuangtoun/resume-job-description-fit`

| | |
|---|---|
| Catalog entry | [A1](../data-catalog.md) — Tier 1, core benchmark |
| Source | https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit |
| Retrieved | 19 Aug 2026 (HF parquet conversion, no token) |
| On disk | `data/raw/fit/` — 12.3 MB, 2 files |
| Licence | **None declared.** Accepted by decision **D7**; record the status in the report |
| PII | Livecareer-style resumes; personal names largely stripped, employers and schools remain |
| Verified rows | 6,241 train + 1,759 test = 8,000 pairs |

## Schema

`resume_text` (str) · `job_description_text` (str) · `label` ∈ {`No Fit`, `Potential Fit`, `Good Fit`}

## Verified figures (19 Aug 2026)

| | Train | Test |
|---|---|---|
| Pairs | 6,241 | 1,759 |
| **Unique resumes** | **642** | **477** |
| **Unique JDs** | **280** | **71** |
| Label mix (No / Potential / Good) | 50.4 / 24.9 / 24.7 % | 48.7 / 25.2 / 26.0 % |
| Resumes per JD (mean / median / max) | 22.3 / 14 / 111 | 24.8 / 20 / 89 |
| JDs with ≥1 Good Fit | 115 | 28 |
| Median chars, resume / JD | 5,134 / 2,384 | 5,080 / 2,401 |

Good Fit resumes per test JD: min 1, Q1 3, **median 18**, Q3 24, max 48.

## Known defects

1. **Effective sample size is ~640 resumes and ~280 JDs**, not 8,000. The rows are a
   near-complete cross-product. Every confidence interval must be computed against the
   document counts.
2. **Resume leakage.** 476 of 477 test resumes (99.8%) also appear in train, paired with
   different JDs. The shipped split is JD-disjoint, **not** candidate-disjoint.
3. **6 pairs carry conflicting labels** across the corpus — these are annotations, not
   authoritative ground truth.
4. **General-industry, not IT-specific.** In-domain evaluation comes from A2 instead.

## How it must be used

- Do **not** report on the shipped split. Use the doubly-disjoint re-split
  (action plan §3.2, hold-out 25%) and state that results are not comparable to published
  numbers on the shipped split.
- Retrieval pools are constructed from it (action plan §3.3), padded to N = 100 with
  distractors; report **Recall@50, Precision@10, nDCG@10** with *n* and a bootstrap CI.
- **Do not report Recall@10** — with a median of 18 relevant resumes per query it is capped
  at 0.56 for the median query and unreachable above 0.90 for 57% of queries.

Explored in [`notebooks/02-fit-benchmark-eda.ipynb`](../../../notebooks/02-fit-benchmark-eda.ipynb).
