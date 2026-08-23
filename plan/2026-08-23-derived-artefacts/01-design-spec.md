# Design Specification — Derived Artefacts

## 1. Target layout

```
data/
  interim/                       git-ignored — repaired / de-identified intermediates
    dataturks/                     BIO conll + repair log
  processed/                     git-ignored — the artefacts themselves
    fit/                           A1 splits
    pools/                         retrieval pools
    indomain/                      Djinni evaluation set
  vocab/                         COMMITTED — derived ESCO extracts (D15)
docs/data/manifests/             COMMITTED — identity-only manifests, no document text
docs/annotation-guide.md         COMMITTED — the 3.4 labelling protocol
```

The split between `data/processed/` (ignored) and `docs/data/manifests/` (committed) is the
mechanism that makes the artefacts reproducible without committing corpora. A manifest names
documents by ID and records the seed; running the builder on the same `data/raw/` reproduces the
artefact byte-for-byte.

## 2. Document identity

A1's parquet files carry **only** `resume_text`, `job_description_text`, `label` — no publisher
ID (verified). Every downstream manifest needs stable document identity, so it is minted:

```
doc_id = "r_" | "j_" + sha256(normalise(text)).hexdigest()[:12]
normalise(t) = re.sub(r"\s+", " ", t).strip()
```

Rationale: content-addressed IDs are stable across re-download, make duplicate detection free
(the 643/351 unique-document counts fall straight out), and carry no text into the manifest.
Collision risk at 48 bits over ~1,000 documents is negligible; the builder asserts zero
collisions and fails loudly otherwise.

Djinni does not need this — both tables ship a publisher `id` column (verified: 141,897 JDs,
210,250 CVs). Use it directly and record the table and row count alongside.

## 3. Artefact inventory

### 3.1 `data/interim/dataturks/` — repaired NER seed *(decision D5)*

| | |
|---|---|
| Input | `data/raw/dataturks/` — 220 docs, 3,570 spans, 6.2% mis-offset |
| Output | `train.conll` style BIO + `repair-log.csv` (one row per dropped or repaired span, with the reason) |
| Label scheme | Merged with SkillSpan's `tags_skill` / `tags_knowledge`, mapped to the project scheme in a committed `label-map.yaml` |
| De-identification | `Name` and `Email Address` entity classes dropped; residual 5 emails / 17 phone strings masked in the text. **Non-negotiable before any commit or sharing** |
| Evaluation | Cross-validation over all 220 docs, mean ± std. The 200/20 split is unusable and is not built |

### 3.2 `data/processed/fit/` — leak-free A1 splits *(decision D8)*

Doubly-disjoint: hold out a resume set *and* a JD set; test takes only pairs where **both** sides
are held out, train only pairs where **neither** is; cross pairs are discarded.

| File | Committed? | Schema |
|---|---|---|
| `docs/data/manifests/fit-split.csv` | **Yes** | `doc_id, doc_type {resume,jd}, split {train,val,test,discarded}` |
| `data/processed/fit/{train,val,test}.parquet` | No | `resume_id, jd_id, resume_text, job_description_text, label` |
| `docs/data/manifests/fit-split-yield.json` | **Yes** | The measured yield table, seed, and the discarded-pair count |

Seed fixed at 0 and recorded. The builder is a pure function of (`data/raw/fit/`, seed,
hold-out fraction) — no interactive steps.

**Q13 is settled here, on measured numbers.** Task 3.2 produces the yield table for both a two-way
25% split and a three-way split (val carved from the train side by a second doubly-disjoint
hold-out) before the choice is made.

### 3.3 `data/processed/pools/` — retrieval pools *(decisions D1, D9)*

Per JD *q* in the evaluation split: relevant = its `Good Fit` resumes, plus a second graded pass
over `Good ∪ Potential`; distractors sampled from the split's unique-resume pool carrying no
label against *q*.

| File | Committed? | Schema |
|---|---|---|
| `docs/data/manifests/pools.csv` | **Yes** | `query_jd_id, candidate_resume_id, relevance {good,potential,none}, source {labelled,distractor}, pool_variant` |
| `data/processed/pools/pools.parquet` | No | The above joined to text |

- `pool_variant` ∈ {`N20`, `N100`, `N477`}; **N100 is primary**, the others are sensitivity runs.
- Metrics: **Recall@50, Precision@10, nDCG@10**. Recall@10 is not reported — with a median of 18
  relevant resumes per JD it is capped at 0.56 for the median query.
- Every reported figure carries *n* (≈30 scored queries) and a bootstrap CI. At this sample size
  the interval is the finding.
- **Stated limitation:** unlabelled distractors are *assumed* non-relevant.

### 3.4 `data/processed/indomain/` — Djinni evaluation set *(decisions D3, D4, D11, D12, D14)*

| File | Committed? | Schema |
|---|---|---|
| `docs/data/manifests/indomain-pairs.csv` | **Yes** | `pair_id, jd_id, cv_id, primary_keyword, exp_band, assigned_to, is_double_labelled` |
| `data/processed/indomain/labels.csv` | **Yes** (labels are our own work, no source text) | `pair_id, annotator, label {good,potential,no}, notes` |
| `data/processed/indomain/pairs.parquet` | No | The above joined to JD/CV text for the annotation UI |
| `docs/annotation-guide.md` | **Yes** | Class definitions, worked examples, tie-breaking rule |

**Eval holdout region (D18) — reserve this before any pretraining run.** A2 may be used for
unlabelled domain-adaptive pretraining, but excluding only the 200 selected pairs is **not
sufficient**, for two reasons:

1. Exclusion must be at **document** level, not pair level — the 40 JD IDs and every CV ID in the
   set, not the 200 `(jd, cv)` combinations.
2. Wave 2 of the annotation (see [`02-work-plan.md` §Q14](02-work-plan.md#q14--pool-depth-and-whether-to-collect-rankings))
   judges *additional* CVs for the same 40 JDs, and it runs **after** pretraining. Any CV that
   wave 2 might reach must already be excluded, or it will have been pretrained on before it is
   ever judged.

So the builder writes `docs/data/manifests/indomain-holdout.csv` — `doc_id, doc_type, reason` —
covering the 40 JDs **and the entire banded candidate pool each was sampled from**, not just the 5
drawn per JD. That whole region is subtracted from the pretraining corpus. It is a few thousand
documents out of 210,250 CVs and 141,897 JDs, so the cost to pretraining is negligible and wave 2
stays valid. `verify --derived` asserts the pretraining corpus and the holdout region are disjoint.

Per **D14** the sample is drawn without role-family stratification. `primary_keyword` is still
*recorded* on every pair so the realised family mix can be reported as an observation — the
constraint removed is on sampling, not on measurement.

Double-label 60 of 200 (30%) across annotators; report Cohen's κ (2 annotators) or Fleiss' κ
(3+); adjudicate disagreements in a review session; single-label the remaining 140. Freeze the
set and isolate it from all training and tuning. Synthetic resumes (D3) may be added **only** as
extra pool distractors, flagged, never mixed into human-labelled evidence.

### 3.5 `data/vocab/` — skill and occupation vocabulary *(R5)*

| File | Committed? | Schema |
|---|---|---|
| `data/vocab/skills.csv` | **Yes** (D15) | `skill_id, preferred_label, alias, skill_type {skill,knowledge}, reuse_level, tech_frequency, source {esco,data_jobs}` |
| `data/vocab/occupations.csv` | **Yes** (D15) | `occupation_id, preferred_label, alias, isco_group` |
| `data/vocab/requirement-relations.csv` | **Yes** (D15) | `occupation_id, skill_id, relation {essential,optional}` |
| `data/vocab/VERSION` | **Yes** | ESCO release, D2 snapshot date, build date, row counts |

- ESCO `skills_en.csv` preferred labels + ~86,694 `altLabels`, **left-joined** with D2
  `job_skills` frequencies.
- Acquisition measured that ESCO matches only **33 of D2's 100 most frequent skills** exactly
  (47% of posting volume); `aws`, `azure`, `docker`, `kubernetes`, `pytorch`, `terraform`,
  `snowflake` and ~30 more are **absent entirely**. Unmatched high-frequency tools are added as
  project-local rows carrying `source = data_jobs`.
- `occupationSkillRelations_en.csv` (126,051 links: **67,600 essential / 58,451 optional**) seeds
  the hard-vs-preferred requirement distinction directly — it is not invented.
- **Known trap, already cost us once:** D2's `job_skills` is a *stringified* Python list.
  Iterating a cell yields characters and silently produces a 37-symbol "vocabulary". Both
  consumers go through `parse_skill_cell` / `skill_frequencies`.
- **Known trap from the rejected O\*NET join:** string similarity alone is unsafe against any
  taxonomy — `AI Engineer` matches ESCO's `animal artificial insemination technician` at 1.000
  via the genuine alt label `ai engineer`. Any alias-based occupation lookup keeps a review gate.

## 4. Determinism and validation

Every builder is a module under `src/candidate_screener/data/`, invoked with `python -m`, taking
a seed and writing a manifest. Two commands must pass at the end of the phase:

```bash
uv run python -m candidate_screener.data.build   --all --seed 0
uv run python -m candidate_screener.data.verify  --derived
```

`verify --derived` is an acceptance test, extending the existing `verify --all`:

1. Every manifest ID resolves to exactly one document in `data/raw/`.
2. **Split disjointness**: zero resume IDs and zero JD IDs shared across train/val/test.
3. **Pool integrity**: every `query_jd_id` is in the evaluation split; no distractor carries a
   label against its query; pool sizes match `pool_variant`.
4. **De-identification**: no `Name`/`Email Address` class survives in the DataTurks output, and a
   regex sweep for emails and phone numbers over `data/interim/` returns zero.
5. **Contamination guard**: no document in any split or pool matches the rejected
   `data/processed/vm-structured-onet/` set — 7 of its 344 resumes have a `career_objective`
   appearing verbatim inside an A1 `resume_text` (see the C4 entry in the catalog). Cheap
   insurance against a future re-import.
6. Re-running `build --seed 0` reproduces every manifest byte-for-byte.

## 5. Commit policy (D15)

**Committed:** manifests (IDs, splits, seeds, counts), `data/vocab/*` (derived extracts),
`data/processed/indomain/labels.csv` (our own labels), the annotation guide, all code and plans.

**Never committed:** anything under `data/raw/`, `data/interim/`, or `data/processed/` other than
the labels file. `.gitignore` already enforces this with `data/*`; the vocab and labels
exceptions are added as explicit `!` rules so the intent is visible in the file.

**Notebook and script output cap (D15):** no cell may print or embed bulk source text. Samples
only — **at most 5 records or 2,000 characters per cell**, and derived aggregates (counts,
frequencies, label lists) in preference to raw records. Applies to `stdout` and to committed
notebook outputs alike; `nbconvert --clear-output` is the fallback if a cell exceeds it.
