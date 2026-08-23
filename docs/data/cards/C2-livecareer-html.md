# C2 · `opensporks/resumes` — the HTML mirror

| | |
|---|---|
| Catalog entry | [C2](../data-catalog.md) — Tier 1, HTML fallback for C1 |
| Source | https://huggingface.co/datasets/opensporks/resumes |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/livecareer/` — 20.0 MB, 1 file |
| Licence | **None declared** |
| PII | As C1 |
| Verified rows | 2,484 |

## Schema

`ID`, `Resume_str`, `Resume_html`, `Category` — the same corpus as [C1](C1-livecareer-pdf.md)
**without the PDFs**, and with no Kaggle account required.

## Verified figures (19 Aug 2026)

Median HTML 15,025 chars. Tag inventory of a representative record: ~129 `span`, 48 `div`,
30 `li`, 11 `p` — nested layout `div`s with inline styles and **no semantic sectioning**.

## How it is used

Real markup for the ingestion component. A naive tag strip concatenates words across cells —
the same interleaving failure as the PDF path, so both routes need the same whitespace and
segmentation repair. The C1 contamination exclusion list applies here identically: it is the
same 2,484 documents.

Explored in [`notebooks/05-raw-formats.ipynb`](../../../notebooks/05-raw-formats.ipynb).
