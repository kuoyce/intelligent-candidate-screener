# A2 · Djinni Recruitment Dataset (`lang-uk/recruitment-dataset-*`)

20 October 2026 derived-input audit: the guided HR portal can read the supplied
600CV/60JD extraction rows and600saved X2-ranked pairs (10perJD). No new source,
labels or ranking experiment is introduced. The adapter excludes `final_label`
and all oracle metadata from its source-text reads. Extraction cells are unverified
table mentions without source offsets; predicted role family is not a verified
job title. Producer rebuild lineage/timestamps and original PDFs remain unavailable.
See the [implementation record](../../../src/llm-plan/2026-10-19-hr-conversation-design/05-implementation.md).

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

## Local extraction use (5 Oct 2026)

Notebooks 10–12 additionally use a deterministic sample of the committed **training
region** for rule-based skill/tool, title and qualification mentions. Evidence offsets
are relative to immutable source fields; `Position` is a profile/vacancy title rather
than an employment-history claim, and `Looking For` is aspirational context.
`Primary Keyword` and experience categories are not extraction gold or model outputs.
The seed vocabulary is limited and is not an ESCO mapping. A2 supplies descriptive
coverage and operational validation only; extraction accuracy needs separate labelled
sources. Local evidence stays under git-ignored `data/processed/extraction/`.
See the [implementation plan](../../../plan/2026-10-05-entity-extraction-notebooks/README.md).

**Added 6 Oct 2026:** notebook 13 runs pinned GLiNER zero-shot inference on a
separate training-only sample (100 CVs and 100 JDs), retaining original field spans.
Custom model descriptions map to the same four project types (*Superseded 9 Oct
2026*: notebook 13 now adds `EXPERIENCE_YEARS` as a fifth neural label). Neural scores are
uncalibrated; context is still heuristic. Rule comparison uses exactly the same
completed documents and measures disagreement, not accuracy. Local neural records
are separate from rule artifacts and never enter fit or human annotation labels.
See the [transformer plan](../../../plan/2026-10-06-transformer-entity-extraction/README.md).

**Added 9 Oct 2026:** explicit professional experience duration spans retain their
original wording, including ranges and minimums. This does not infer tenure from
employment dates or supply numeric fit features. Rules still support four labels;
historical four-label predictions remain separate from the new five-label run.
See the [experience-years plan](../../../plan/2026-10-09-experience-years/README.md).

**Branch integration, 11 Oct 2026:** the seed-rule inspection aggregate is retained
unchanged at `output/extraction/rule-inspection-metrics.json`; the incoming skill
rules/NER evaluation retains `output/extraction/extraction-metrics.json`. GLiNER
duration spans are distinct from the incoming stage-1 rule-based experience
features. Stage-2 rerankers read text and do not directly consume GLiNER spans.

Notebook 14 explores these saved predictions with visualizations and bounded real
entity/context examples. Its rule comparison is recomputed on exactly the neural
run's training IDs. Frequency, score and disagreement analysis does not provide
gold extraction accuracy or constitute another annotation campaign.

**Source validation, 14 Oct 2026:** notebook 17 reconstructs the historical
GLiNER run's recorded fields and sampled training IDs rather than using today's
input-field defaults. It checks source/artifact hashes, exact entity slices and
processing coverage. Bounded redacted contexts support qualitative inspection;
Djinni Fit labels are not entity-span gold and cannot establish NER accuracy.
