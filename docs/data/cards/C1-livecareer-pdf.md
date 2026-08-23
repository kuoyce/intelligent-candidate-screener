# C1 · Kaggle `snehaanbhawal/resume-dataset` — the PDF corpus

| | |
|---|---|
| Catalog entry | [C1](../data-catalog.md) — Tier 1, raw-format corpus (decision **D6**) |
| Source | https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset |
| Retrieved | 19 Aug 2026 (Kaggle account + API token) |
| On disk | `data/raw/snehaanbhawal-resume-dataset/` — 118.3 MB, `Resume.csv` + 2,484 PDFs in 24 category folders |
| Licence | Kaggle-hosted, publisher terms — verify on the page before redistributing anything derived |
| PII | Livecareer-scraped resumes; treat as personal data even though names are largely stripped |
| Verified rows | 2,484 CSV rows · **2,484 PDFs** across 24 categories |

## Schema

`ID`, `Resume_str`, `Resume_html`, `Category` — plus `data/<CATEGORY>/<ID>.pdf` on disk.
Three views of the same document: **PDF**, publisher-extracted text, and HTML.

## Verified figures (19 Aug 2026)

- Median HTML 15,025 chars; median publisher text 5,886 chars; median PDF 23 KB.
- Probe of 40 PDFs: median **2 pages**, ~5,800 extracted chars, ~3,000 chars/page,
  **0 pages with an empty text layer** — these are digital-born PDFs, so OCR is not needed
  for this corpus.
- Reference extraction differs from ours: on ID 16852973, `Resume_str` is 5,442 chars against
  5,179 from `pypdf`. The publisher column is a **reference extraction to diff a parser
  against**, not ground truth.

## Parsing failure modes to design for

1. No section structure — `Summary` / `Skills` / `Experience` arrive as ordinary lines.
2. Multi-column and table layouts emit in draw order, interleaving a skills sidebar into the
   experience narrative; sentence boundaries are unreliable.
3. Dates and bullets lose their anchoring — the worst case for a `Years of Experience`
   extractor.
4. Pages with no text layer must be detected and routed to OCR or failed loudly, never
   returned as a short resume.

## Contamination — must be excluded

**44 of these resumes (6.8% of A1's 643 unique resumes) also appear in the A1 benchmark**,
measured by ≥0.70 8-gram containment (2 at ≥0.90; **exact matching finds none** — the two
scrapes format differently). The list is written to
`data/interim/c1_a1_contamination.csv` by `profile --check livecareer` and must be excluded
from any NER training set or distractor pool built off this corpus.

Explored in [`notebooks/05-raw-formats.ipynb`](../../../notebooks/05-raw-formats.ipynb).
