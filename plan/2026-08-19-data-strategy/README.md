# Data Strategy — Intelligent Candidate Screening System

**Date:** 19 August 2026
**Status:** Draft for team review
**Inputs:** `docs/proposal/PLP_Group2_Intelligent_Candidate_Screening_Proposal-v3.md`

## Contents

| File | Purpose |
|---|---|
| [`01-requirements-and-findings.md`](01-requirements-and-findings.md) | Data requirements derived from the proposal, and what empirical profiling of the three named sources actually found |
| [`docs/data/data-catalog.md`](../../docs/data/data-catalog.md) | Catalog of all candidate sources — moved to `docs/` during the acquisition phase, where it is maintained alongside the per-source dataset cards |
| [`03-acquisition-action-plan.md`](03-acquisition-action-plan.md) | Sequenced action plan to download, view and validate each source |
| [`04-acquisition-implementation.md`](04-acquisition-implementation.md) | What the acquisition phase actually built, and where it deviated from this plan |

## Method note

Every figure in these documents marked **Verified** was measured directly by downloading the
data and profiling it on 19 Aug 2026, not read from a dataset card. Several dataset cards were
found to be wrong (see `01-requirements-and-findings.md` §3), so card claims are marked
**Card claim** and treated as unconfirmed until profiled.

## Team decisions already taken

These were confirmed before this document was written and are treated as settled:

| # | Decision | Consequence |
|---|---|---|
| D1 | Retrieval will be evaluated on **synthetic candidate pools** built from the fit dataset | Pool-construction protocol becomes a documented deliverable |
| D2 | §7 success targets are **rebased on the measured Stage-1 baseline** as relative deltas | Requires a proposal amendment; the absolute 85% figure survives only for in-domain human agreement |
| D3 | In-domain data comes from **public JD dumps + public resume corpora**, supplemented by **LLM-generated synthetic resumes** | No scraping; every source must be licence-traceable |
| D4 | In-domain fit labels are **team-adjudicated with inter-annotator agreement reported** | ~1–2 team-days of annotation effort must be budgeted |
| D5 | DataTurks is **kept as the resume-native gold seed and supplemented**, not replaced | Must be de-identified before use; needs a second skill-span source |
| D6 | A **real PDF/HTML corpus** is sourced so parsing robustness is genuinely exercised | Kaggle account required for the PDF variant |

Confirmed 19 Aug 2026, closing open questions Q1–Q7:

| # | Decision | Consequence |
|---|---|---|
| D7 | The undeclared licence on the core benchmark (A1) is **accepted**; proceed | Note the licence status in the report's data section |
| D8 | Re-split **at candidate level**, accepting that results are **not comparable** to any published numbers on the shipped split | Implemented as a *doubly-disjoint* split — see `03-acquisition-action-plan.md` §3.2 |
| D9 | Pool size and retrieval metrics **decided from the data**, not by preference | §7's Recall@10 target is arithmetically unreachable — see `01-requirements-and-findings.md` §2.7 |
| D10 | Kaggle access is **available** | Not on the critical path; PDFs feed the ingestion demo only |
| D11 | Djinni's Ukrainian/EE IT market is **acceptable**, stated clearly as a limitation | Add to §5.4 scope boundaries and §10 risks |
| D12 | **200 pairs** hand-labelled by business-analytics master's students with software/data-science backgrounds | Annotation design in `03-acquisition-action-plan.md` §3.4 |
| D13 | **ESCO v1.2.1 acquired** and in place at `data/raw/esco_dataset-v1.2.1-classification/` | Q7 closed; Phase 1.10 complete |

## Open questions

All seven opening questions are resolved. Remaining items are tracked in
[`01-requirements-and-findings.md` §6](01-requirements-and-findings.md#6-open-questions).

**Update 23 Aug 2026.** Q8–Q11, Q13 and Q15 were closed by decisions D14–D18 in
[`plan/2026-08-23-derived-artefacts/`](../2026-08-23-derived-artefacts/README.md), which
supersedes this plan for Phase 3. Q12 is closed by measurement (length carries no label signal in
A1; concatenating Djinni's CV fields halves the gap). Q14 and Q16 are new and tracked there.
