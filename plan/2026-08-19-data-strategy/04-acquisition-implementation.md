# Acquisition Phase — Implementation Record

**Date:** 19 August 2026 · **Branch:** `feat/data-acquisition` ·
**Covers:** Phases 0–2 and Phase 4 of [`03-acquisition-action-plan.md`](03-acquisition-action-plan.md)

## 1. Scope — and what is deliberately *not* in it

**In scope, and complete:**

| Phase | Deliverable |
|---|---|
| 0 | Package layout, dependencies, importable `candidate_screener` |
| 1 | All eleven adopted sources downloaded — 676 MB in `data/raw/` |
| 2 | Integrity verification, profiling, and six executed notebooks covering the plan's viewing checklist |
| 4 | Catalog moved to `docs/data/`, ten dataset cards written, manifests committed |

**Explicitly out of scope — Phase 3 derived artefacts are untouched:** DataTurks BIO
conversion and de-identification (§3.1), the doubly-disjoint re-split (§3.2), retrieval pool
construction (§3.3), the 200-pair in-domain annotation set (§3.4) and the skill vocabulary
build (§3.5). Nothing under `data/interim/`, `data/processed/` or `data/vocab/` exists yet
except the one contamination list described in §4.2 below.

## 2. What was built

```
src/candidate_screener/
  config.py              canonical paths — one definition of the data layout
  data/
    sources.py           registry: 12 sources x locator, licence, tier, PII, expected artefacts
    fetch.py             download        (python -m candidate_screener.data.fetch --all)
    verify.py            integrity check (python -m candidate_screener.data.verify --all)
    profile.py           the Verified figures (python -m candidate_screener.data.profile --all)
notebooks/               01-06, executed, outputs committed
docs/data/               catalog, ten cards, acquisition manifest, profile metrics
```

The registry is the single source of truth: it drives what `fetch` downloads, what `verify`
asserts, and what the cards record. Adding a source means adding one `Source(...)` entry.

### Notebooks against the plan's Phase 2 checklist

| Plan row | Notebook |
|---|---|
| — | [`01-acquisition-overview.ipynb`](../../notebooks/01-acquisition-overview.ipynb) — inventory, licence/PII posture, effective sample size |
| A1 | [`02-fit-benchmark-eda.ipynb`](../../notebooks/02-fit-benchmark-eda.ipynb) — label mix, leakage, relevance density, sample pairs read end to end |
| A2 | [`03-djinni-in-domain.ipynb`](../../notebooks/03-djinni-in-domain.ipynb) — `Primary Keyword` on both sides, banding variables, pool feasibility |
| B1 + B2 | [`04-extraction-sources.ipynb`](../../notebooks/04-extraction-sources.ipynb) — span inventory, offset defect, rendered spans, label-scheme mapping |
| C1/C2/C3 | [`05-raw-formats.ipynb`](../../notebooks/05-raw-formats.ipynb) — real PDFs opened, HTML tag inventory, normalisation check, contamination list |
| — | [`06-skill-vocabulary.ipynb`](../../notebooks/06-skill-vocabulary.ipynb) — ESCO aliases, essential/optional, tech coverage test |

## 3. Deviations from the approved plan

| # | Deviation | Why |
|---|---|---|
| V1 | Scripts moved out of `plan/…/scripts/` into an installed package `src/candidate_screener/data/`, run as `python -m …` | Acquisition code is project code with a long life, not plan documentation. The plan directory is a record of a decision; it should not also be a runtime |
| V2 | The single `profile_sources.py` was split into **`verify`** (bytes: files, rows, SHA-256) and **`profile`** (meaning: the Verified figures) | They answer different questions and fail for different reasons. `verify` is now a hard acceptance test with expected row counts in the registry — it caught a wrong ESCO figure during development |
| V3 | `docs/data/data-catalog.md` moved out of `plan/` | Directed by the team; it is living documentation, and it now sits beside the cards it indexes |
| V4 | Two runtime dependencies added (`matplotlib`, `pypdf`) and three dev (`ipykernel`, `nbconvert`); `pyproject.toml` gained a `build-system` | Plotting and PDF probing are needed for Phase 2; the build-system entry is what makes the package importable |
| V5 | B3 `green` downloaded although it is "adopt only if B2 proves insufficient" | 1.1 MB and already scripted. Downloaded, **not adopted** — see its card |
| V6 | A4 `ats` registered in the source registry but **not downloaded** | It is rejected. `fetch --all` skips tier 3; reproducing the rejection evidence needs an explicit `fetch --source ats` |
| V7 | Notebook 04 redacts `Name` / `Email Address` spans at display time | Notebook outputs are committed. The first execution put a real name and profile URL into the notebook; the renderer now substitutes `[NAME]` / `[EMAIL]`, which also demonstrates the §3.1 de-identification |

## 4. Findings the acquisition phase added

Each of these changes something downstream; all are folded into
[`docs/data/data-catalog.md`](../../docs/data/data-catalog.md) and the cards.

### 4.1 SkillSpan is 20× DataTurks, not 26× — and the exact figure matters

Measured: **11,543 sentences, 4,381 skill spans + 5,236 knowledge spans (9,617 total)**
against DataTurks' 472 `Skills` spans. The plan's "~26×" came from the card claim
">12.5K spans". The measured number is what the extraction budget should be built on.

### 4.2 The C1↔A1 contamination figure needed a criterion, and is 44 rather than 40

**Exact matching finds zero overlap** — the two scrapes format the same documents
differently. Under 8-gram containment: **44 resumes at ≥0.70** (6.8% of A1's 643 uniques),
65 at ≥0.50, 2 at ≥0.90. The criterion is now explicit and the list is materialised at
`data/interim/c1_a1_contamination.csv` by `profile --check livecareer`, so the exclusion can
actually be applied rather than remembered.

### 4.3 ESCO does not know the tools this domain hires for — measured

Of D2's 100 most frequent skills, ESCO matches **33 exactly** (47% of posting volume), 55
under relaxed word matching. **Absent entirely:** `aws`, `azure`, `tableau`, `snowflake`,
`databricks`, `gcp`, `kafka`, `docker`, `kubernetes`, `mongodb`, `pytorch`, `terraform`,
`scikit-learn`, `jira`, `github` and ~30 more.

This converts the plan's qualitative caveat ("tech-tool coverage is thinner") into a number,
and settles the design of §3.5: the vocabulary is ESCO + aliases **left-joined** with D2
frequencies, with unmatched high-frequency tools added as project-local entries carrying a
`source = data_jobs` provenance flag.

### 4.4 Djinni CVs are ~7× shorter than A1 resumes

Median **751 chars (Djinni CV)** against **5,134 (A1 resume)**; JDs 1,629 against 2,384. A
model tuned on A1 sees far less evidence per candidate in the in-domain evaluation. This is a
transfer risk distinct from the market limitation (D11) and belongs in the §10 risk table.

### 4.5 The PDF corpus needs no OCR, and ships its own reference extraction

A 40-file probe: median 2 pages, ~5,800 extracted chars, **0 pages with an empty text
layer** — digital-born PDFs throughout. The corpus carries three views of each document
(PDF, publisher `Resume_str`, `Resume_html`), so the publisher's extraction is a reference to
diff our parser against. Failure modes to design for are listed in the
[C1 card](../../docs/data/cards/C1-livecareer-pdf.md).

### 4.6 A data trap worth one line of code

D2's `job_skills` is a **stringified** Python list. Iterating a cell yields characters and
produces a silent 37-symbol "vocabulary" — this was in the first draft of the vocabulary
notebook. Both consumers now go through `parse_skill_cell` / `skill_frequencies`.

## 5. Assumptions made (flagged for confirmation)

| # | Assumption | If wrong |
|---|---|---|
| A1 | "Data acquisition phase" = Phases 0–2 + Phase 4 documentation. Phase 3 derived artefacts are a separate phase | Phase 3 work starts on the next branch; nothing here needs undoing |
| A2 | Package name `candidate_screener` under a `src/` layout, modules invoked with `python -m` | A rename is a mechanical change; imports are all internal |
| A3 | `acquisition-manifest.json` and `profile-metrics.json` may be committed. They contain counts, licences, digests and aggregate figures — **no document text** | Remove both from git; they are regenerable |
| A4 | Executed notebook outputs may be committed. They include short **A2 (MIT, pre-anonymized)** and **C1** text samples, redacted B1 spans, and small ESCO label examples — this last touches **Q11** | Strip outputs with `nbconvert --clear-output` before committing |

## 6. Open questions after this phase

Q1–Q7 remain closed. Of the four carried in, none is resolved by acquisition — they are
Phase 3 decisions — but two now have evidence attached:

| # | Question | Status |
|---|---|---|
| Q8 | Is the 25% doubly-disjoint split's yield (~560 test pairs / 30 JDs with Good Fit) accepted? | **Open** — Phase 3.2 |
| Q9 | Which Djinni role families are in scope for the 200-pair set? | **Open, evidence attached** — notebook 03 sizes each candidate family and shows every one has thousands of CVs per posting, so the choice is not supply-constrained. Proposed: `JavaScript, Java, Python, DevOps, .NET, QA Automation, Node.js, PHP` |
| Q10 | Do the 200 in-domain pairs use A1's 3-class scheme? | **Open** — annotation guide |
| Q11 | May derived ESCO extracts be committed? | **Open, now urgent** — notebook 06's committed outputs already contain small ESCO label samples (see assumption A4), and §3.5's `data/vocab/skills.csv` is a derived extract |
| **Q12** | *New.* Does the 7× length gap between Djinni CVs and A1 resumes need a mitigation (e.g. length-normalised scoring), or is it reported as a limitation only? | **Open** — affects §3.4 and §10 |

**Update 23 Aug 2026.** Q8–Q12 are all closed — see
[`plan/2026-08-23-derived-artefacts/README.md`](../2026-08-23-derived-artefacts/README.md) for
decisions D14–D18 and the questions (Q14, Q16) that replaced them.

## 7. Next step

Phase 3.4 remains the critical path: it is the only step gated on human effort rather than
compute, and notebook 03 has now established that the pools it needs are constructible.
Q9 is its first blocker.
