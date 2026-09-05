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
5. **The labels do not reproduce — A13 fails** *(added 30 Aug 2026, D33)*. On a recheck of
   100 pairs (machine) / 50 (human), two independent judges working from
   `docs/annotation-guide.md` and blind to A1's answer both fail to recover it:
   kappa(A1, human) = **0.034** (n=50, CI [-0.162, 0.225]) and
   kappa(A1, `llm:claude-sonnet-5`) = **0.125** (n=100, CI [0.013, 0.241]), while the two
   blind judges agree with each other at kappa = **0.275** (CI [0.127, 0.457], n=50). Under
   binary collapse (`Good`∪`Potential` vs `No Fit`) the A1 legs are **0.031** and **0.113**
   — and exactly **0.000** over the original 50-pair draw — against **0.425** between the
   blind judges. The LLM judge's test-retest reliability is 0.857-1.000 over three runs, so
   this is not a noisy judge. **The breakdown is in A1's positives**: against A1's `Good
   Fit` (n=30) the judge agrees **0.067** and reads 26 of the 30 as `No Fit`; over the
   original 50 the human confirms 3 of A1's 15 `Good Fit` and the judge none. Agreement on
   A1's `No Fit` is 0.60 human / 0.90 judge — the whole of the small positive kappa.
   Evidence:
   [`llm-recheck.csv`](../manifests/llm-recheck.csv),
   `output/annotation/llm-recheck-report.json`,
   `plan/2026-08-30-llm-recheck/01-findings.md`.

   *Superseded 5 Sep 2026 (D34).* The human leg closed at 100 pairs and the judge's guide
   was rewritten to define `No Fit` positively, which raised kappa(human, llm) to **0.352**
   (**0.614** collapsed, n=100). Against A1 both blind judges stayed at chance:
   kappa(A1, human) = **0.081** [-0.052, 0.224] and kappa(A1, `llm:claude-haiku-4-5`) =
   **0.085** [-0.038, 0.206], n=100 each. **The defect stands** — a judge that agrees far
   better with the human after being made to apply the guide as written, and no better with
   A1, is further evidence about A1, not less. Independently: A1's three classes are
   **non-monotone** in JD-CV TF-IDF cosine (0.060 `No` / 0.072 `Potential` / 0.068 `Good`),
   where both blind judges are monotone (rho 0.494 human, 0.365 judge); and one resume,
   `r_e741e1ca6ee8` — a network-security engineer — carries A1 `Good Fit` against four
   unrelated software JDs that both blind judges reject.
   Evidence: `plan/2026-09-05-prompt-calibration/02-calibration-run.md`.
6. **Label provenance is unrecorded** *(Q28)*. The publisher does not say how the labels
   were produced. It no longer blocks reading defect 5 — shared machine bias would have
   *inflated* kappa(A1, llm), and the observed value is at chance — but it is still unknown.

## How it must be used

- **Quote no ceiling off these labels without the A13 caveat** *(defect 5)*. D26's
  attainable Precision@5 of **0.7806** strict / **0.7188** graded and Recall@10 of 0.983 are
  computed against pools whose A1 leg carries labels that do not reproduce. The figures have
  not been restated — that is a separate change, superseding in place — but every use of
  them now travels with the finding, not only with the ceiling.
- Do **not** report on the shipped split. Use the doubly-disjoint re-split — **built, at a
  30% hold-out, seed 0**: [`docs/data/manifests/fit-split.csv`](../manifests/fit-split.csv),
  rebuilt with `python -m candidate_screener.data.build --task fit-split`.
  **Results on it are not comparable to any published number on the shipped split.** State
  that wherever Stage 1–4 results appear; it is the accepted price of removing the leakage.
- **The evaluation is 100 test JDs, 31 of them carrying a Good Fit, over 659 pairs** — *n* is
  31, not 659, for anything reported per query. 3,990 pairs remain for training; 3,338 cross
  pairs are discarded by construction. The predecessor plan's 25% recommendation was
  superseded once the yield was measured across seeds: see
  [`fit-split-survey.json`](../manifests/fit-split-survey.json).
- **Pooling the shipped splits is what the re-split starts from** (D16): 642 train + 477 test
  unique resumes dedupe to 643, so there was no partition worth preserving. Pooling also
  surfaced 7 duplicate pairs — 6 labelled two ways — all inside the shipped *train* set;
  they are dropped and logged in
  [`fit-pair-conflicts.csv`](../manifests/fit-pair-conflicts.csv), leaving 7,987 usable.
- Retrieval pools are **built**: [`docs/data/manifests/pools.csv`](../manifests/pools.csv),
  100 queries over a 193-resume candidate universe, nested variants N20 ⊂ N100 ⊂ Nfull with
  N100 primary. *n* is **31** queries under the strict definition and **64** under the graded
  one — report which, always.
- **Recall@10 is reinstated; Recall@50 is retired** (Q17, 23 Aug 2026). The "median 18
  relevant" figure is the *shipped* split's. On the leak-free split the test-side median is
  **6**, Recall@10 reaches 0.90 for 93.5% of queries, and Recall@50 saturates at a ceiling of
  1.0 for every query. Adopted: **Recall@10, Precision@5, nDCG@10**, each with *n* and a
  bootstrap CI.
- **Pool distractors are unjudged.** A1 judges a median of 4 resumes per test JD, so a 100-deep
  pool is ~96% assumed non-relevant and precision is biased downward by construction. The pools
  compare systems; they do not estimate production precision.

Explored in [`notebooks/02-fit-benchmark-eda.ipynb`](../../../notebooks/02-fit-benchmark-eda.ipynb).
