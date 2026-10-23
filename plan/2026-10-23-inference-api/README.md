# Inference API: the reported system as plain functions a Streamlit app can call

Status: **draft v3, 23 Oct 2026, owner's answers folded in (below); awaiting approval.** Nothing
implemented. Branch to be created on approval (`feat/inference-api`, off `exp/chat3-holdout`).
Follows D63/D64 (`docs/decisions.md`); covers the entry-point half of the on-hold plan
`2026-10-22-model-bundle-and-hosting`.

## Owner's answers (23 Oct 2026)

1. **Local only for now**, on a machine with `data/`. Portability stays in the bundle plan.
2. **D63 by default** (L1 alone), C2 (D64's rerank of the top 50) as an option.
3. Pool size / scaling: **undecided**. Step 2 measures it before anything is fixed (§ Pool scaling).
4. The UI shows **the model's top 10**; extraction, features and scores are backend.
5. **New JDs and CVs must work with inference only, no retraining** (§ New documents).
6. `present` in a CV date range: a `reference_date` argument, **default 2023-09-01 for A2 documents,
   today for uploads**, stated in the UI.
7. Freezing L1's existing fit is **an artifact, not training code**: warm checks suffice (D55).
8. Return types: **dataclasses** (with `as_dict()` / `to_frame()`).
9. **CVs come from uploads only.** No CV store or search; the document cache is all that persists.
   Pools are therefore small, which makes § Pool scaling matter.
10. No decision-log entry for the API.
11. **`explain` is for the recruiter**, through the chatbot ("why is candidate 3 here?"). Primary
    fields are plain: role match, matched / missing JD skills, years required vs stated. L1 feature
    shares are secondary (for the teammate, debugging, or for the chatbot to phrase). Quoted CV
    snippets are optional; any display passes `annotation.redact`.
12. **`docs/project-overview.md` is updated once the API is complete** (step 7).

## Why

A teammate is rebuilding the pipeline and the chatbot and needs to call the reported system from
Streamlit. Today they cannot:

- the system lives inside evaluation modules (`evaluation/chat3.py::l1_pool`,
  `evaluation/c2_validation.py::fused_order`) that read chat-3, the pools and the final set;
- **L1 is refitted from chat-3 every time** (`chat3.fit_stage2`); its coefficients are in no record;
- the feature builder (`matching.learned.Features` → `a1_weights.build_scorer`) refits the concept
  document frequencies via `dev_pools.build()` (> 4.5 GB, minutes) at every start-up;
- `pipeline.screen` and `app/streamlit_app.py` run a draft TF-IDF ranker (deprecated, untouched);
- the owner's first cut, `output/matching/a2-*.csv` (`evaluation/a2_export.py`, uncommitted), is
  a table over the 600 labelled pairs, not something to call, and not the model's top 10 of a pool.

## New documents: what changes when a JD or CV was never seen

Every fitted part is applied, never refitted, so a new JD or CV needs **no retraining**:

| Part | Fitted on | On a new document |
|---|---|---|
| Role-family model (60 classes) | A2 train JDs/CVs | predicts as usual; a non-IT JD is still forced into one of the 60 IT families (low `role_prob` is the tell) |
| Vocabulary matcher (ESCO ICT + data_jobs) | nothing | skills outside the vocabulary are not seen |
| Concept document frequencies | 20k A2 train CVs | unseen concepts get the flat default weight (W0), as in evaluation |
| TF-IDF vocabulary + IDF | 20k + 20k A2 train docs | unseen words are dropped |
| Experience rules | nothing | `present` in a date range = `reference_date`: 2023-09-01 for A2, today for uploads (answer 6) |
| SBERT weak | weak A2 train pairs | encodes as usual |
| L1 | chat-3 | frozen coefficients (below) |
| C2 | nothing (zero-shot) | as usual, ~5-7 s per pair |

What a new document *does* change:

- **The other CVs' scores**, through `tfidf_scaled`, which is min-max scaled within the pool
  (§ Pool scaling). Every other feature is per pair.
- **Distribution shift.** The models saw Djinni text (`text_v2` = `CV`, `Highlights`, `Moreinfo`,
  `Looking For`, joined). An uploaded PDF CV is longer and laid out differently; nothing measured
  says how the role model or L1 behaves on it. The API passes the text through unchanged and the
  plan does not claim accuracy on such documents.

### Document-level vs pair-level work (so a new document is processed once)

The live path splits work so each document is processed once and reused across JDs:

- **per document, cached by SHA-256 of its text:** role probabilities, vocabulary concepts and
  skills, years, SBERT weak embedding (SBERT already caches by text), TF-IDF vector;
- **per pair, cheap, never cached:** `role_match` (dot product), coverage and overlap terms,
  `exp_gap`, TF-IDF cosine, SBERT cosine, then the pool's min-max, then L1;
- **per pair, expensive, cached:** C2, by the existing `rerank/cross.py` cache (pair id + text digest).

Cache: `data/processed/inference/docs/` (git-ignored). A new CV against an existing JD costs one
document pass plus a pool re-score; a new JD against CVs already uploaded costs one document pass
plus pair features, no CV re-extraction.

## Design

### Module layout

| File | Contents |
|---|---|
| `inference/__init__.py` | the public names below, nothing else |
| `inference/schema.py` | frozen dataclasses `JDProfile`, `CVProfile`, `Ranked`, `Screening`, each with `as_dict()` (JSON-safe) and, for lists, `to_frame()`. Field names = the `a2-*` CSV columns |
| `inference/model.py` | `L1Model`: loads the frozen L1 JSON; `predict(X)`, `contributions(X)` |
| `inference/docs.py` | per-document extraction and its cache |
| `inference/screener.py` | `Screener`: holds every fitted part, loaded once; the public calls |
| `inference/__main__.py` | `--freeze`, `--cache-parts`, `--check`, `--demo JD_ID` |

### Public API

```python
from candidate_screener import inference

s = inference.Screener.load()                              # once; st.cache_resource in the app
jd  = s.extract_jd(jd_text, jd_title="QA Engineer")        # -> JDProfile
cv  = s.extract_cv(cv_text, cv_id="c1")                    # -> CVProfile
res = s.rank(jd_text, cvs, jd_title=..., top_k=10)         # D63 -> Screening
res = s.rank(jd_text, cvs, jd_title=..., rerank=True,      # D64: C2 on L1's top 50
             progress=lambda done, total: ...)
res.top(10)                                                # what the UI shows
why = s.explain(res, cv_id="c1")                           # -> dict
X2  = s.rank(jd_text, cvs, system="X2")                    # the baseline, for comparison
```

- `cvs`: `cv_id` and `cv_text` **only**; any other column raises (D53). `jd_title` is the one
  allowed field (D54).
- `Screening` holds every CV of the pool (rank, `score`, `l1_score`, `x2_score`, `c2_score` or
  `None`, features, profiles), `system`, `pool_size`; `top(k)` is the UI's view.
- D64 reuses `c2_validation.fused_order` (k = 50, w = 0.25), not a copy.
- `explain`: role (JD's and CV's top families), matched and missing JD skills (canonical names),
  years required vs stated, and each feature's exact share of the L1 score
  (`coef × standardised value`; L1 is linear, so the shares sum to `score − intercept`).
  Skill offsets refer to the text the caller passed; the app redacts before display (AGENTS).
- `s.a2_pool(jd_id)` returns a final-set JD's text, title and 500-CV pool as `rank`'s inputs, for
  the demo (needs `data/`). `labels(jd_id)` exposes final labels separately, never in a score (D46).

### Frozen L1 (the one fitted object this plan adds)

`output/rerank/l1-chat3-model.json`: columns (F12, in order), `mu`, `sd`, `coef`, `intercept`,
`alpha`, the SHA-256 of the sorted chat-3 pair ids, and the source commit. The model
`chat3.fit_stage2` already fits, written once: nothing is retrained or retuned. A few KB of numbers,
a committed record of a thing we fit (D23).

### Fitted parts cache (start-up)

`Screener.load` must not call `dev_pools.build()`. `--cache-parts` writes the concept
document-frequency table and the fitted TF-IDF vocabulary + IDF to
`data/processed/inference/parts/<fingerprint>/` (parquet/npz, no pickle). The role model and SBERT
weak load from where they already live.

### Pool scaling (question 3, decided after step 2's measurement)

`tfidf_scaled` = min-max of the TF-IDF cosine within the pool. L1 was fitted on rows of 500-CV
pools. On a pool of 5 uploads, scaling stretches the same cosine range over [0, 1], so TF-IDF weighs
differently against the other features (within a pool, its *order* is unchanged); a pool of 1 gets 0.

Options:

- **A. Within-pool (as tested).** Simple; scores of a small pool are not comparable with a 500 pool.
- **B. Background pool.** Score the uploads together with a fixed sample of A2 CVs filling the pool
  to 500, show only the uploads. Reproduces the tested condition; costs one pass over the
  background CVs per JD (document-level parts cached once, so pair features only). Works locally,
  needs `data/` (fine, answer 1).
- **C. Fixed scale.** Scale by fixed bounds taken from chat-3's training distribution. New,
  untested variant.

**Step 2 measures it, on A2 train-region dev pools only (never the final set, D46):** for pools of
500, take random sub-pools of 10, 25, 50, 100 that contain a pool's L1 top 10, and report how often
the order of those CVs differs under A vs the 500-pool order (B equals the 500-pool order by
construction, so A's disagreement is the cost of choosing A). Exploratory; the owner then picks A or
B. C only if both are unacceptable.

## Steps

0. Commit the owner's first cut (`evaluation/a2_export.py`, its test, four `a2-*.csv`) as its own
   commit on the new branch.
1. `inference/model.py` + `--freeze` → `l1-chat3-model.json`; `--cache-parts`.
2. **Measure:** load time per part; wall time of `rank` on a 500-CV pool cold and with the document
   cache warm, and on a 10-CV pool; C2 on 50; the pool-scaling comparison above. Recorded in the
   implementation record. Owner picks A or B.
3. `inference/schema.py`, `inference/docs.py` (document cache).
4. `inference/screener.py` with the chosen scaling.
5. `--check` (below), registered in `checks.py`; `--demo JD_ID` prints a final-set JD's top 10.
6. Docs: `AGENTS.md` (command table, "Where code goes"), `README.md` ("Using the system from an
   app": the snippet above and a ~20-line Streamlit example in the docstring, not a second app),
   this plan's implementation record.
7. **After the API is complete and verified:** update `docs/project-overview.md`: the status line,
   §2 (how the reported system is called: `inference.Screener`, D63 default, C2 opt-in, uploads
   only, the chosen pool scaling and `reference_date`), §7 "Still open" (close the uncommitted-work
   and entry-point rows; the hosting row stays with the bundle plan), §8 "Where to look" (this plan,
   `inference/`, the frozen L1 record). Committed with the plan's completion.

Sub-agents per AGENTS step 3: one for 1, one for 3-4, one for 5.

## Verification

**Unit tests (no `data/`, `uv run pytest`)**, each shown to fail against a broken version:

- `L1Model.predict` from the JSON equals `BoundedRidge.predict` on a synthetic fit; contributions
  sum to `predict − intercept`.
- `rank` refuses any `cvs` column beyond `cv_id`, `cv_text` (D53).
- Ties break by `cv_id`, as `a2_export.rank_within_jd` does.
- A pool of 1 and a pool of identical CVs give finite scores.
- D64 fusion reorders only the top 50 and never lifts a CV from below it.
- The document cache: same text → same profile without recomputation; changed text → recomputed
  (keyed by digest, not `cv_id`).
- A new document never triggers a fit: with fitting functions patched to raise, `rank` on unseen
  texts succeeds.
- `as_dict()` is JSON-serialisable.

**`python -m candidate_screener.inference --check` (needs `data/`, warm):**

- The API on the 600 final-set pairs reproduces `a2-features.csv`, both extraction CSVs and
  `a2-x2-scores.csv` exactly (6 dp).
- L1 refitted from chat-3 equals the JSON (1e-9).
- The API over the 60 pools reproduces `chat3-l1-pool.json` (Recall@10 0.484413, same top 10s).
- D64 through the API with C2 cached only reproduces `c2-validation-test.json`'s winner.

Runtime estimated in step 2 and announced before any run over a few minutes. D55: the frozen L1
is an artifact, not training code (owner), so warm is enough.

## Open questions (owner)

1. **Pool scaling A or B**, after step 2. With uploads only, A means scaling over a handful of CVs;
   B pads them with background A2 CVs (needs `data/`, which the local setup has).

## Out of scope

Hosting and the portable bundle (bundle plan); the chatbot and its intents (teammate);
`pipeline.py`, `app/`, the evidence layer (deprecated, untouched); any retraining or retuning;
measuring accuracy on non-Djinni documents.

## Cost (estimate)

About a day of agent work. `--freeze` needs chat-3's F12 features (as `chat3 --select`); the pool
measurement needs dev-pool features; both timed in step 2 before the full run.
