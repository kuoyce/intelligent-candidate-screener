# A2 · Djinni Recruitment Dataset (`lang-uk/recruitment-dataset-*`)

| | |
|---|---|
| Catalog entry | [A2](../data-catalog.md) — Tier 1, in-domain evaluation |
| Source | [job descriptions](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) · [candidate profiles](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/djinni-jd/` 145.9 MB · `data/raw/djinni-cv/` 237.4 MB |
| Licence | **MIT** |
| PII | **Pre-anonymized by the publishers** — the stated purpose of the release |
| Provenance | Djinni IT hiring platform, 2020–2023. Drushchak & Romanyshyn (2024), *Introducing the Djinni Recruitment Dataset*, UNLP @ LREC-COLING 2024 |
| Verified rows | 141,897 JDs · 210,250 CVs (English splits) |

## Schema

**JD** `Position`, `Long Description`, `Company Name`, `Exp Years`, `Primary Keyword`,
`English Level`, `Published`, `Long Description_lang`, `id`

**CV** `Position`, `Moreinfo`, `Looking For`, `Highlights`, `Primary Keyword`,
`English Level`, `Experience Years`, `CV`, `CV_lang`, `id`

## Verified figures (19 Aug 2026)

- `Primary Keyword`: 45 JD-side values, 41 CV-side, **41 of 41 shared** — the join works.
- Largest shared technical families: `JavaScript`, `Java`, `DevOps`, `.NET`,
  `QA Automation`, `Node.js`, `PHP`, `Python`.
- Median length: **JD 1,629 chars, CV 751 chars.**
- Banding variables: `Experience Years` (CV, 0–11) and `Exp Years` (JD, `no_exp`…`5y`),
  plus `English Level` on both sides.

## Known defects and limitations

1. **No fit labels and no interaction table.** Pairs must be constructed and adjudicated by
   the team — 200 pairs, 60 double-labelled with κ reported (decisions **D4**, **D12**).
2. **Ukrainian / Eastern-European IT labour market** (decision **D11**). The report must
   state that in-domain generalisation was demonstrated on that market, not the deployment
   market — §5.4 scope boundaries and §10 risks.
3. **CVs are ~7× shorter than A1 resumes** (751 vs 5,134 median chars). A model tuned on A1
   sees far less evidence per candidate here; treat this as a transfer risk in §10.

## How it is used

The in-domain end-to-end evaluation set (action plan §3.4). It replaces the "200+
self-collected job descriptions" plan, which had no lawful collection path.

Explored in [`notebooks/03-djinni-in-domain.ipynb`](../../../notebooks/03-djinni-in-domain.ipynb).
