# Hiring Desk on the inference API: merge `feat/inference-api` and rank new JDs through it

Status: **approved by the owner (24 Oct 2026, draft v2); implementation in progress.** Branch
`feat/hiring-desk-inference-2026-10-24`, off `feat/hiring-desk-v1-2026-10-23`; step 1 (the merge)
is `f6a1dc8`.

Builds on `plan/2026-10-23-hiring-desk-v1/` (the app and its live-ranking contract,
`05-live-ranking-contract.md`) and `plan/2026-10-23-inference-api/` (the `inference.Screener` API).

## Owner's answers (24 Oct 2026)

1. **Pool:** a new JD ranks against **the 600 A2 final-set CVs** (the applicants already in the
   benchmark reviews), reference date 2023-09-01. No labels are read.
2. **Benchmark JDs: unchanged.** The 60 CSV reviews stay on `phase2-rerank-output/` and keep working
   with the model offline. Only new JDs use the API.
3. **System: D63 by default, D64 as an option** (C2 reranks L1's top 50, ~2 min per JD), so the
   adapter needs a background worker and the app a status refresh.
4. **Explanations: yes**, a "Why this rank" expander for live-ranked candidates.

## Why

- **What the app shows today.** `PortalDataset.from_folder` reads
  `data/processed/phase2-rerank-output/`, whose three CSVs are byte-identical to
  `output/matching/a2-{cv-extraction,jd-extraction,x2-scores}.csv` on `feat/inference-api` (the
  `a2_export` hand-off tables). They are **X2 (the baseline) scores of the 10 labelled CVs** of each
  of the 60 final-set JDs: not D63, and not a top 10 out of a pool.
- **Why a new JD gets no candidates.** New JDs rank through `app/portal_ranking.RankingService`,
  which needs `HR_PORTAL_RANKING_ADAPTER=module:function` and `HR_PORTAL_CANDIDATE_POOL`. Neither is
  set and no adapter exists. The owner's published JD ("Data Scientist", revision 11, in
  `data/processed/hr-portal/workspace.sqlite`) has **no ranking request at all**: the page stops at
  "Live ranking is not connected yet" (reason: "Live ranking adapter is not configured").
- **What `feat/inference-api` provides.** The reported system as an in-process Python API,
  `inference.Screener` (`rank`, `extract_jd`, `extract_cv`, `explain`, `a2_pool`). There is no HTTP
  service; "API" below means this object.

## Findings: where the API falls short of what Hiring Desk needs

Measured 24 Oct 2026 on a throwaway trial merge (scratch worktree, deleted afterwards) against the
owner's published JD and local `data/`. No labels were read; IDs, scores, timings and validator
counts only, no CV text printed.

### On the API side

| # | Gap | Evidence | Handling |
|---|---|---|---|
| A1 | **No candidate pool for a new JD.** `rank(jd_text, cvs)` ranks only the CVs passed in; the API is "uploads only, no CV store" (inference plan, answer 9) and Hiring Desk has no upload. `a2_pool(jd_id)` covers only the 60 final-set JDs. | The hiring-desk contract: "Never silently choose benchmark CVs"; the pool is the model owner's call. | Answer 1: the adapter's pool registry, `a2-final-cvs` |
| A2 | **No adapter for the app's contract**: `adapter(operation, payload)` with `submit`/`status`, dedupe by `request_id`, status recovery without a handle, every identity field echoed, the ready schema. | `portal_ranking.py` docstring; `05-live-ranking-contract.md`. | New module `app/portal_inference.py` |
| A3 | **Synchronous only, no job handle.** | `Screener.load` 24.6 s cold (4.4 s in the inference plan, warm file cache). `rank` D63: 600 CVs 11.0 s cold, 2.5 s warm; a 500-CV A2 pool 8.4 s. D64: ~2.5 s per uncached C2 pair (~2 min per JD). Peak RSS 1.1-1.3 GB. | The adapter's worker queue (below) |
| A4 | **D64 holds the `Screener` lock for its whole C2 pass.** `rank` is serialised (`_serialised`), and C2 runs inside it. | `inference/screener.py`. | Every job runs on the adapter's single worker thread, in order; a D63 request behind a D64 job reports `queued` instead of blocking a page. No API change |
| A5 | **No run, model or pool provenance, no score meaning.** `Screening` has `system`, `scaling`, `pool_size`, `reference_date`; the app requires `run_id`, `model_id`, `candidate_pool_version`, `score_semantics`, and `source_ref` per candidate. | L1 scores are ~39-51 for this JD: an ordering score, not a probability. | Adapter derives them (L1 record SHA-256, pool manifest digest) |
| A6 | **The JD's years are lost.** The app renders `Minimum years: 2` / `Maximum years: 6`; `experience.jd_experience` parses neither (min = max = None), so `explain` says "The job states no years of experience". | Parsed correctly: `Experience: 2-6 years`, `Experience: at least 2 years` (min 2), `Experience: up to 6 years` (0-6). Parsed **wrongly**: `No more than 6 years of experience` (min 6). | Change `portal_jobs.render_jd`'s experience line. Text stays the only input (D53; the contract: structured fields do not become inference metadata). Stored revisions keep their text: re-edit and re-publish to benefit |
| A7 | **Vocabulary coverage.** Skills outside ESCO ICT + data_jobs are invisible; required vs preferred is not distinguished (text only). | Of "excel, tiktok, abacus", "aerospace", "Computer science", the API recognises `excel` and `computer science` only. | Inherent (no retraining). The JD's recognised skills go in `jd_entities`, so the recruiter sees what the model read |
| A8 | **Entity categories differ.** API: vocabulary skills, years, role family, seniority. App: KNOWLEDGE / SKILL / TITLE / DEGREE / EXPERIENCE_YEARS. The live contract has no slot for role / top roles / seniority (`table_aggregates` is `{}` for live candidates). | `portal_ranking._entities` accepts mention lists only. | Map SKILL, EXPERIENCE_YEARS (as text), TITLE (JD title); KNOWLEDGE / DEGREE stay "unavailable" (as in the CSV path today; GLiNER covers chat). Role and reasons go in `evidence_provenance` |
| A9 | **`reference_date` defaults to today**; A2 CVs must use 2023-09-01 (`experience.REFERENCE_DATE`), or their years differ from every record. | Inference plan, answer 6. | Adapter sets it per pool |
| A10 | **`Screening` carries no CV text**; the app needs it for preview, chat, compare and the ten-CV package. | `schema.py`. | Adapter attaches the pool's text |

### On the app side (found while connecting it)

| # | Issue | Evidence | Handling |
|---|---|---|---|
| B1 | **Real CV text fails the app's validator, so every ranking would be recorded as failed.** `portal_jobs._text` rejects any control character other than `\n` / `\t`. | 530 of the 600 final-set CV texts contain `\r`; 9 of 10 in this JD's top 10, 10 of 10 in a benchmark JD's top 10. After CRLF → LF: 0 of 10 fail. No CV hits the 60,000-character or `Bearer` rules; all 600 IDs pass `_opaque_identity`. | Adapter normalises line endings; the digest is computed on the text it returns |
| B2 | **Adapter errors are invisible.** By design the app persists one generic message ("failed or returned an invalid result"), never the exception. | `RankingService._invoke`. | Adapter logs the exception type and traceback, never document text, to `logs/hiring-desk-adapter.log` |
| B3 | **No way to choose the system.** The request payload (schema 1) has no model field; `RankingService.rank(rerank=…)` means "a new run", not C2. | `portal_ranking.py`. | Optional payload field `ranking_system` (`D63` default, `D64`), a checkbox by the Rank button |
| B4 | **Explanations have nowhere to go.** `evidence_provenance` is validated and stored but never displayed. | `grep evidence_provenance app/`. | Answer 4: "Why this rank" expander |
| B5 | Benchmark JDs: D63's top 10 (500-CV pool) for the first benchmark JD shares 2 CVs with the CSV's 10 labelled CVs. | Probe. | Answer 2: unchanged |
| B6 | Minor: the skills answer "excel, tiktok, abacus" on one line is stored as one requirement (the UI asks for one per line). The model still reads the text. | `operational_job_revisions`, revision 5. | None (noted) |

### The merge itself

Code merges cleanly. Conflicts: one table row each in `AGENTS.md` ("Where code goes") and
`README.md` (layout); keep both sides. Since the merge base (`b300943`) the hiring-desk side changed
no research module (only `app/streamlit_app.py`, plus new files), so the merged research code equals
`feat/inference-api`'s, whose 32 affected checks passed warm on 24 Oct 2026. Trial-merge
`uv run pytest`: pass (exit 0).

## Design

### Adapter: `app/portal_inference.py`

```
HR_PORTAL_RANKING_ADAPTER=candidate_screener.app.portal_inference:adapter
HR_PORTAL_CANDIDATE_POOL=a2-final-cvs
```

- **One `Screener`**, loaded on the worker thread on the first request, shared by every session.
  Loads, never fits.
- **Pool registry** keyed by `candidate_pool_ref`: `a2-final-cvs` = the 600 final-set CVs, `cv_id`
  and `cv_text` only from `corpus.a2_pairs("v2")` (no labels in it, D46; its metadata columns are
  dropped, D53), reference date 2023-09-01. An unknown reference gives `service_unavailable`.
- **One worker thread, one FIFO queue**, jobs keyed by `request_id`:
  - `submit` (new `request_id`): enqueue. D63 waits up to 45 s for its result and returns `ready`
    when done (the first request includes the model load, ~36 s); otherwise, and always for D64,
    it returns `queued` / `running` with a handle at once.
  - `submit` (known `request_id`): the job's current state, never a second run.
  - `status`: the job's state; `ready` carries the result. An unknown `request_id` (the process
    restarted) is re-enqueued from the payload, which carries the full request: D63 and D64 are
    deterministic for a fixed pool and date.
  - Finished results are kept in memory (last 200); the app persists its own snapshot.
- **The model sees** `jd_text` and `jd_title` only (D53, D54), and the pool's `cv_id` / `cv_text`.
  D63: `rank(…, top_k=10, reference_date=2023-09-01)`; D64: the same with `rerank=True` (k = 50,
  w = 0.25 from `c2-validation-select.json`), C2 scores cached by text digest under
  `data/processed/rerank/`.
- **Errors:** missing parts or weights (`FileNotFoundError`) give `service_unavailable`; anything
  else `failed`; both logged (B2).

| App field | From |
|---|---|
| echoed identity (7 fields) | the payload |
| `run_id` | `d63-` / `d64-` + digest(request, pool version, L1 record) |
| `model_id` | `D63 frozen L1 (l1-chat3-model.json <sha12>)`; D64 adds `+ C2 top 50, w 0.25` |
| `candidate_pool_version` | `a2-final-cvs:` + digest of the sorted (`cv_id`, text digest) pairs |
| `score_semantics` | "D63 frozen-L1 ordering score over the configured pool; higher ranks first; not a Fit probability" (D64: the fused min-max score within L1's top 50) |
| `eligible_candidate_ids` | every `cv_id` of the pool |
| `candidates[i]` | `Screening.top(10)`: `candidate_id` = `cv_id`, `rank`, `score`, `source_ref` = `a2-final-cvs/<cv_id>`, `cv_text` (CRLF → LF, B1), `entities` = SKILL (`CVProfile.skills`), EXPERIENCE_YEARS (`"<years> years (<source>)"`), `evidence_provenance` = `explain()`: reasons, roles, matched / missing skills, years, L1 shares |
| `jd_entities` | SKILL (`JDProfile.skills`), TITLE (`jd_title`), EXPERIENCE_YEARS (`"2-6 years"` when parsed) |

No `fit` field: the API makes no Fit call, and the app accepts one only with decision provenance.

### App changes (Hiring Desk)

- **`portal_ranking.RankingService.rank(…, system="D63")`**: `system` in {`D63`, `D64`} goes into
  the payload as `ranking_system` (schema 1, optional field; a stored payload without it means
  D63). The module docstring's contract gains the field.
- **`portal_authoring_ui._ranking_controls`**: a checkbox "Refine the top 50 with the reranker
  (D64, about 2 minutes)" by the Rank / Re-rank button. The existing "Refresh ranking status" button
  serves `queued` / `running`. "Ranking details" already shows `model_id`, so the system shows.
- **`portal_jobs.render_jd`** (A6): the experience block becomes one line the rules parse:
  `Experience: 2-6 years`, `Experience: at least 2 years`, `Experience: up to 6 years`, or
  `Experience: Not specified`. `tests/test_portal_jobs.py` (`"Minimum years: 0" in text`) changes
  with it.
- **`portal_ui._review`** (answer 4): in "Top10 ranked CVs", for live snapshots, one "Why #k"
  expander per candidate listing `evidence_provenance.reasons` (through `redact_text`). CSV reviews
  are unchanged.

## Steps

1. Branch `feat/hiring-desk-inference-2026-10-24`; merge `feat/inference-api`, resolve the two doc
   rows, commit the merge.
2. Commit this plan, approved.
3. Adapter and its tests (sub-agent).
4. App changes and their tests: `ranking_system`, the checkbox, `render_jd`, "Why #k" (sub-agent).
5. Review against this plan; `uv run pytest`.
6. End to end in the browser, on an isolated copy of the app state (a scratch worktree whose
   `data/processed/hr-portal/` is a copy, with a scratch test account): create, publish and rank a
   JD with D63, then re-rank with D64 (~2 min, via "Refresh ranking status"); top 10 and "Why #k"
   shown; open CV, compare and package work. Nothing is written to the real workspace; C2 and SBERT
   caches under `data/` gain the test JD's entries (keyed by text, harmless).
7. Docs: `AGENTS.md` (the adapter's environment variables in the Hiring Desk paragraph),
   `README.md` (how to launch with the adapter), hiring-desk plan `03-open-questions.md` (Q1: the
   pool chosen) and `06-implementation.md` (live connection bound), this plan's implementation
   record.

## Verification

Unit tests, no `data/` (an injected fake `Screener`), each shown to fail against the code it guards:

- A CV text with `\r\n` yields a ready snapshot through the real `RankingService` (fails without
  the normalisation).
- The adapter's ready response passes `RankingService` validation: ten candidates, contiguous
  ranks, every ID in the pool manifest, `source_ref` present.
- The same `request_id` submitted twice runs once; `status` for an unknown `request_id` re-runs it
  and returns the same result.
- D64 `submit` returns `queued` / `running` with a handle without waiting; `status` returns `ready`
  once the worker finishes; a D63 request behind it reports `queued` within the wait budget.
- `rank` receives only `cv_id`, `cv_text` and the payload's `jd_text` / `jd_title` (D53, D54).
- An unknown pool reference and missing parts give `service_unavailable`, not an exception.
- `RankingService.rank(system="D64")` puts `ranking_system` in the payload; an unknown system is
  refused; a stored payload without the field is treated as D63.
- `render_jd`: for each of the four experience cases, `experience.jd_experience` reads back the
  entered minimum and maximum (fails against the current rendering).

Then `uv run pytest` (full suite) and the browser run (step 6). **Checks:** none required: the new
code is app code, and the merged research code equals `feat/inference-api`'s (Testing rule 3).
Optional: `inference --check` (~9 min) after the merge, on the owner's say-so.

## Open questions (owner)

| ID | Question | Answer |
|---|---|---|
| Q1 | Which CVs does a new JD rank against? | The 600 A2 final-set CVs (owner, 24 Oct 2026) |
| Q2 | Do the 60 benchmark JDs move from the CSV folder to the API? | No, new JDs only (owner, 24 Oct 2026) |
| Q3 | D63 only, or D64 as an option? | D63 default, D64 option (owner, 24 Oct 2026) |
| Q4 | Show `explain()`'s reasons in the review? | Yes, "Why this rank" (owner, 24 Oct 2026) |

None open.

## Assumptions (stated, to confirm with the plan)

- "API" means `inference.Screener` in process. An HTTP wrapper is out of scope unless asked.
- The 45 s wait for D63 and the queue of one worker are defaults for a local demonstration with a
  few reviewers, not measured requirements.
- The pool's A2 CVs use the A2 reference date, 2023-09-01, as the benchmark does.

## Out of scope

Retraining or retuning; extending the skill vocabulary; an HTTP service; hosting and the portable
bundle (`plan/2026-10-22-model-bundle-and-hosting`); CV upload; moving the benchmark JDs to the API;
Fit calls; progress bars for D64 (the contract has no progress field).
