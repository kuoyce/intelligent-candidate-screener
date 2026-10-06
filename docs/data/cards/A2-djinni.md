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

## Title fields on the train region — verified 6 Oct 2026 (P2)

Measured on the D19 **train** region only (106,423 JDs, 157,688 CVs); the eval region is
filtered out before any field is read. Figures in `output/extraction/title-profile.json`.

- `Position`: 100% filled on both sides. Distinct values: 50,661 JD / 36,817 CV, falling to
  21,197 / 12,532 under the P2 normalisation rule. 43.2% of JD and 30.0% of CV titles carry
  a seniority token.
- `Primary Keyword`: 45 JD classes, of which **4 have a single JD** (`Block-chain`, `SAP`,
  `React`, `Product Owner`); median class 1,395, max 13,390 (`JavaScript`). 41 CV classes,
  min 120 (`Rust`), median 2,321, max 25,448. **30 train CVs have a null keyword.**
- Normalised titles with ≥ 20 train JDs and ≥ 20 train CVs: **298**, covering 65.3% of JDs
  and 82.3% of CVs.
- With `Position` removed, the document's own normalised title appears in the text of
  **32.3%** of sampled JDs and **27.0%** of sampled CVs (n = 300 each).

See [`11-p2-data-checks.md`](../../../plan/2026-10-06-extraction-and-rerank/11-p2-data-checks.md).

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

*Added 6 Oct 2026 (D50).* The train region (D19) also supplies the one LLM scoring batch:
100 JDs × 10 CVs (`docs/data/manifests/llm-train-batch.csv`), none in the final set or the
eval region, scored by the unchanged instrument `5d986c82` to train the stage-2 reranker
(D51). See `plan/2026-10-06-extraction-and-rerank/15-p6-batch.md`.

Explored in [`notebooks/03-djinni-in-domain.ipynb`](../../../notebooks/03-djinni-in-domain.ipynb).
