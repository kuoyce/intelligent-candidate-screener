# Project overview: results, design, data, extraction, methodology

Written 22 Oct 2026, updated after the c2-validation work (D64) and, on 23 Oct 2026, after the inference API (§2.6), from `docs/decisions.md`, the `plan/` records and `output/` records.
A reading aid, not a source of truth: every figure lives in a committed record.
Live pipeline / UI code (`pipeline.py`, `app/`) is deprecated and out of scope here; apps call the reported system through `candidate_screener.inference` (§2.6).

**Status: reported system = D64.** L1 on F12 (trained on `chat-3`) ranks the 500-CV pool; its top 50 are reordered by
L1 + 0.25 × C2 (zero-shot reranker). X2 is the baseline. D64 was the owner's post-hoc decision and **overrides a registered bar that the test
missed** (§2.2). The report write-up is still to do (§7). Cold suite: 29 of 30 checks passed, then the last (a teammate's placeholder record)
was regenerated and all pass warm; plans classifier-then-rerank and chat3-holdout are marked complete.
**Inference API (23 Oct 2026, branch `feat/inference-api`):** D63 / D64 / X2 callable from an app on new, uploaded JDs and CVs
with no retraining (§2.6); its `--check` reproduces the D63 and D64 records exactly.

**Length guide (for the report):** the sections marked 🔎 are the ones to expand;
the rest can be cut to a line or a table.

---

## 1. Headline numbers

All on the A2 final set: 60 JDs, 500-CV pools, 10 labelled CVs per JD. CIs resample JDs.

### 1.1 Stage 1 and the text-only ladder, one table (`text_v2`, no metadata at inference)

Per-JD AUC against the final Fit / No Fit labels, 59 JDs in total (generic 39-40, targeted 20). CVs have `Position` removed.
Recall@50 is on 500-CV pools (S2 bars 0.80 classical / 0.90 advanced).

| System | generic | targeted | **total** [CI] | Recall@50 total |
|---|---|---|---|---|
| random | 0.558 | 0.552 | 0.556 [0.505, 0.606] | 0.119 |
| BM25 | 0.610 | 0.673 | 0.632 [0.579, 0.682] | – |
| TF-IDF | 0.780 | 0.807 | 0.789 [0.741, 0.835] | 0.635 |
| SBERT zero-shot | 0.740 | 0.731 | 0.737 [0.686, 0.784] | 0.438 |
| SBERT weakly fine-tuned | 0.853 | 0.796 | 0.834 [0.796, 0.870] | 0.592 |
| X1 extracted role + experience | 0.851 | 0.755 | 0.818 [0.765, 0.867] | 0.773 |
| X3 = X1 + skill overlap | 0.851 | 0.761 | 0.821 [0.767, 0.869] | 0.726 |
| **X2 = X1 + TF-IDF** (stage 1) | 0.863 | 0.826 | **0.850** [0.809, 0.889] | **0.779** [0.706, 0.845] |
| X4 = X1 + SBERT weak | 0.891 | 0.785 | 0.855 [0.817, 0.890] | 0.761 |

- **BM25 and TF-IDF:** cheap word-overlap signals do most of the work, and BM25 is clearly worse than TF-IDF (0.632 vs 0.789).
- **SBERT** is a bi-encoder: it embeds the JD and each CV separately, and the score is the cosine between the two vectors. Zero-shot, it is below TF-IDF (0.737).
- **SBERT weakly fine-tuned** is the same model trained on (JD, CV) pairs labelled automatically from metadata (keyword and experience match). It is the one semantic step that paid:
  +0.097 over zero-shot SBERT here, and +0.057 over TF-IDF in the earlier text.
- **X1, X3, X2, X4** add rules extracted from the text (role and experience; then skill overlap, TF-IDF or SBERT weak). Role is what X1 adds: within a role family the gap closes
  (X2 0.745, SBERT weak 0.752, TF-IDF 0.714).
- **X2 is stage 1.** It beats TF-IDF by +0.061 [0.029, 0.096] (registered K1, justified); other X-rung comparisons are not resolved (±0.02-0.05).
- **S2 is missed:** X2 Recall@50 is 0.779, below the 0.80 bar. Its top 100 of 500 still holds 0.90 of the fits.
- *Footnote (metadata rungs, oracle only):* rules that read the Djinni `Primary Keyword` at inference, the earlier selected pipeline, score 0.778 (keyword rule) and 0.846 (rule + TF-IDF).
  They cannot be used on open-world documents, so X1/X2 replace them by extracting the role from text. Cross-encoder rungs are omitted.

### 1.2 Stage 2: reranking into a top 10 (reference: an LLM's scores of the top-10 candidates¹)

| Pipeline | Recall@10 | Δ vs L1 alone [CI] | nDCG@10 | "No Fit" CVs in top 10 | Trained on |
|---|---|---|---|---|---|
| X2's order (baseline) | 0.417 | −0.058 [−0.088, −0.029] | 0.775 | 1.73 | nothing |
| **L1 on F12 alone** | 0.475 | – | 0.820 | 1.38 | `chat-3` |
| **L1 top 50 → L1 + 0.25 C2 (D64, reported system)** | **0.480** | **+0.005 [−0.010, +0.021]** (nDCG +0.012 [+0.005, +0.019]) | **0.832** | **1.28** | `chat-3` (L1); C2 none |

60 JDs, 500-CV pools. The earlier rungs (S1, C2 alone) and the first-pass `chat-1`-only figures are in §5-D.

**What the models are.**
- **L1 on F12:** a bounded ridge regression (a linear model, not a logistic one). It predicts the LLM's 0-100 score for a (JD, CV) pair from the 12 F12 features, then ranks by that.
  F12 = `role_match`; `experience_compatible`, `exp_gap`, `exp_known`; `coverage`, `coverage_unweighted`, `log_matched`, `log_jd_concepts`, `log_cv_concepts`, `skill_overlap`
  (the skill-vocabulary matcher); `tfidf_scaled`; `sbert` (§2.5). Fitted once on `chat-3`: 48 train-region JDs, 4,800 pairs, with no final-set JD or `chat-1` score used.
- **C2:** Qwen3-Reranker-0.6B, used as it comes with a "fit" instruction. It reads the JD and CV text together, uses none of the F12 features and learns nothing here. About 5-7 s per pair.
  In the reported system it only reorders L1's top 50, by min-max L1 + 0.25 × min-max C2.

Chosen on `chat-3` by 5-fold CV from nine options (k ∈ {20, 30, 50} × w ∈ {0.25, 0.5, 1.0}): +0.019 [+0.003, +0.034] there (best of nine, so optimistic).
The registered bar on the test was Δ ≥ +0.02 with the CI above 0. **It was missed**; C2 was adopted anyway by the owner (D64) on the grounds that Recall@10 and nDCG@10
are both higher. Human check: no change (AUC −0.003, pairwise −0.006). Cost: about 5-7 s per pair, ~5 min per JD for the top 50.

### 1.3 What the numbers mean

- The LLM scores are the target *and* the reference; human grades are the independent check.
- Training on chat scores buys about +0.03-0.04 over the chat-free order.
- C2 is the only extra sophistication adopted, and only post hoc: it helps the order inside the top 10 (nDCG +0.012) more than it brings more of the chat's top 10 in (Recall@10 +0.005, CI across 0).
- Learned weights, richer concepts and the other zero-shot variants added nothing measurable (§5).

¹ The reference is the owner's claude.ai LLM scores (`chat-1`: X2's top 100 for the 60 final-set JDs; `chat-2`: extra CVs that L1 and C2 pull into the top 10, so every compared top 10 is fully scored; latest score per pair). Since D62 it is test-only; models are trained on `chat-3`, a separate set of 48 train-region JDs. Earlier versions of the stage-2
work trained and chose models on it (L2/L3, steps 3-5d of the classifier-then-rerank plan); those results are superseded and not reported here (see
`plan/2026-10-18-classifier-then-rerank/` and `plan/2026-10-21-chat3-holdout/`).

---

## 2. Solution architecture and design 🔎

### 2.1 Design as built and tested: retrieve, then rerank

```
 JD text + JD title ─┐
                     ├─► EXTRACTION (§4) ─► features per (JD, CV) pair ─┐
 CV text (text_v2) ──┘   role family · experience · skills/concepts      │
                                                                         ▼
 500-CV pool ─► STAGE 1: X2 = 2·role_match + experience_compatible + 0.999·TF-IDF ─► top 100
                                                                         ▼
                        STAGE 2: L1 (ridge) on F12, trained on chat-3 LLM scores ─► [top 50 reordered with C2, D64] ─► top 10 ─► evidence
```

Why two stages: X2 is cheap (about 0.6 s per 500-CV role) and fixed, and the LLM scoring that supplies the training signal
is expensive, so it was collected only for X2's top 100 per JD (K = 100, owner's cost choice; the registered rule gave K = 200).
That shortlist also made the test clean: `chat-3` (48 train-region JDs) trains stage 2, `chat-1` (the 60 final-set JDs) only tests it.

### 2.2 The stages collapse to one scorer (D63, 22 Oct)

Once stage 2 was a linear model on cheap features, it could rank the whole pool itself.

| Check (60 JDs, reference `chat-1`) | Result |
|---|---|
| L1 over all 500 CVs vs X2 top 100 → L1: same top 10 | **59 of 60 JDs** (one lower-ranked CV differs) |
| Recall@10 | 0.484413 for both (identical to six digits) |
| Share of the chat's top 10 inside the top k, k = 10 / 30 / 100 | L1 over pool 0.475 / 0.774 / 0.960; two-stage 0.475 / 0.770 / 0.964; X2 alone 0.417 / 0.744 / 0.964 |

So the one-stage system L1 over the pool is equivalent to the tested two-stage pipeline (equivalence, not improvement), and X2 is reported as the baseline.
The two-box figure above stays as the *evaluation pipeline* (how the data was collected and the test run).

### 2.3 Adding a zero-shot reranker, C2 (D64, 22 Oct)

**What C2 is.** Qwen3-Reranker-0.6B, used zero-shot with a "fit" instruction, no training. It reads a JD and a CV together and returns a relevance score.
It costs about 5-7 s of CPU per pair (about 5 min per JD for 50 CVs), so it can only run on a short list, not on all 500 CVs.

**Design.** L1 ranks the pool. Within L1's top k, the order is min-max(L1) + w × min-max(C2); CVs below the top k keep L1's order.

**History: C2 had been tried several times without a clean answer.**
- Alone, C2 was worse than X2's order (Recall@10 0.379 vs 0.419) and 0.106 below stage 2 (`chat-3`).
- Fused with a stage 2 trained on `chat-1`, it gave +0.013 to +0.025 (steps 5b-5d). Those runs chose their settings after seeing the test JDs.
- L1 top 30 → C2 + L1 gave −0.002 [−0.026, +0.024].
- C2 had never been run on `chat-3`'s JDs, so no setting had been chosen on data other than the test JDs.

**Validation (plan `2026-10-22-c2-validation`, registered before any scoring).**
1. C2 was run on `chat-3`'s 48 JDs: L1's out-of-fold top 50, 2,400 pairs, 3 h 50 min.
2. Nine options (k ∈ {20, 30, 50} × w ∈ {0.25, 0.5, 1.0}) were compared by 5-fold CV by JD on `chat-3` only. The winner was **k = 50, w = 0.25**,
   Recall@10 0.532 vs 0.513 for L1 alone: +0.019 [+0.003, +0.034] (the best of nine, so optimistic). The gate (Δ > 0 on `chat-3`) passed.
   Pattern across the grid: w = 0.25 helps at every k (+0.013 to +0.019); w ≥ 0.5 does not help at any k. Every option cut chat No Fits.
3. One test on the 60 final-set JDs (reuse approved by the owner, exploratory), reference `chat-1` + `chat-2`:

| | Recall@10 | nDCG@10 | Chat No Fits in top 10 |
|---|---|---|---|
| L1 alone | 0.475 | 0.820 | 1.38 |
| L1 top 50 → L1 + 0.25 C2 | 0.480 | 0.832 | 1.28 |
| Δ [CI] | **+0.005 [−0.010, +0.021]** (generic +0.003, targeted +0.011) | **+0.012 [+0.005, +0.019]** | −0.10 |

**The registered bar was not met.** It required Recall@10 Δ ≥ +0.02 with the CI above 0; the result is a quarter of that, CI across 0.
Under the plan's own rule, C2 would have been retired. Human check (422 judged CVs): no change (per-JD AUC −0.003, pairwise −0.006, CIs across 0).

**Why C2 was adopted anyway (owner, D64, post hoc).** After seeing the test, the owner decided to use C2 because **both Recall@10 and nDCG@10 are higher**,
and the nDCG gain has a CI above 0. In other words, C2 does not bring more of the chat's top 10 in, but it orders the top 10 a little better and puts
fewer chat "No Fit" CVs there. This is a judgement call, not a result that met its bar:
- the decision was made after the test, so it is post hoc;
- the Recall@10 improvement is within noise, and the human check shows no change;
- the same pattern appeared in the earlier C2 runs (adds 0 to +0.02);
- the cost is about 5 min of CPU per JD.
Every report must therefore state the bar and that it was missed. Nothing was retuned: k = 50 and w = 0.25 stay as chosen on `chat-3`, although top 30 was
within 0.001 there and needs about 40% fewer C2 calls.

**Reported system (D64):** L1 on F12 ranks all 500 CVs; its top 50 are reordered by min-max L1 + 0.25 × min-max C2; the top 10 are returned.
Without C2 it is the D63 system. X2 stays the baseline.

### 2.4 Caveats

- L1 was trained only on X2's top-100 CVs, the only ones `chat-3` scored. Outside that range it extrapolates.
- Beyond k = 50, 22% of L1's picks have no chat score and count as misses, so deeper figures are lower bounds.
- The choice to report the one-stage system (D63) and to add C2 (D64) were both made after the tests, as reporting decisions, not pre-registered rules.
- **Reference coverage gap: closed for this comparison.** The pooled top 10s of the winner, L1 alone and X2 were all already scored (`chat-1` or `chat-2`),
  so the extra labelling batch (chat-4) came out empty and was never run or logged. CVs deeper than the top 10 (L1's ranks 31-50) are still partly unscored.
- The reference (`chat-1` plus `chat-2`) otherwise covers X2's top 100 plus a partial extension, so a strong CV outside that range counts as a miss
  for any method that ranks it high. For the compared top 10s this did not occur.

### 2.5 F12 (the feature set), by extraction component

| Feature(s) | Produced by |
|---|---|
| `role_match` | role-family classifier (§4.3) |
| `experience_compatible`, `exp_gap`, `exp_known` | experience rules (§4.4) |
| `coverage`, `coverage_unweighted`, `log_matched`, `log_jd_concepts`, `log_cv_concepts`, `skill_overlap` | vocabulary matcher over the ESCO-based vocabulary (§4.1) |
| `tfidf_scaled` | TF-IDF within JD (not extraction) |
| `sbert` | SBERT weak cosine (fine-tuned bi-encoder, §4.7) |

F9 drops `coverage_unweighted`, `skill_overlap`, `log_matched` (near-copies of `coverage`).

**Design rules** (`AGENTS.md`): one frozen reference (D41); no metadata at inference except the JD title (D53/D54);
nothing fitted on the final set (D46); generic + targeted + total with CIs over JDs; bars registered before scoring.


### 2.6 Calling the system: the inference API (plan `2026-10-23-inference-api`)

For the teammate's pipeline and chatbot (Streamlit, local, on a machine with `data/`). Nothing is fitted at run time:
a new JD or CV is inference only.

```python
from candidate_screener import inference
s = inference.Screener.load()                 # ~4 s; once, under st.cache_resource
res = s.rank(jd_text, cvs, jd_title="QA Engineer")   # D63; cvs = cv_id + cv_text only (D53)
res = s.rank(jd_text, cvs, rerank=True)        # D64: C2 on L1's top 50
res.top(10); s.explain(res, cv_id)             # explain: for the recruiter
```

| Owner's decision (23 Oct 2026) | Built as |
|---|---|
| Local, with `data/`; portability later | `Screener.load()` reads saved parts; the HF Hub bundle stays in the on-hold bundle plan |
| D63 by default, C2 optional | `rank(..., rerank=True)` = D64; `system="X2"` for the baseline |
| CVs are uploads only | no CV store or search; per-text in-memory caches |
| **Pool scaling B** | uploads padded to 500 with a fixed sample of A2 train-region CVs (scored, never returned), as L1 was trained and tested; `scaling="pool"` (A) stays an option |
| `present` in CV date ranges | `reference_date`: today for uploads, 2023-09-01 for A2 documents |
| `explain` is for the recruiter | role, matched / missing JD skills, years required vs stated, plain `reasons`; L1 feature shares last |
| Freezing L1 is an artifact, not training | `output/rerank/l1-chat3-model.json` (coefficients of the existing chat-3 fit) |

**What was added.** The frozen L1 (role_match +13.1, sbert +5.5, tfidf_scaled +3.4 standardised; coverage, coverage_unweighted
and skill_overlap at the non-negative bound 0); the fitted concept frequencies and TF-IDF space saved as parquet/npy/json, no pickle
(start-up no longer rebuilds the dev pools); `inference/` (`Screener`, dataclass results); `--check` (registered) and `--demo`.

**Why B (pool scaling).** `tfidf_scaled` is min-max scaled over the CVs scored together; L1 was fitted on 500-CV pools.
Measured on 50 A2 train-region dev JDs (exploratory, `output/inference/pool-scaling.json`), scaling over the uploads alone (A)
vs over 500 (B), same CVs: the top candidate agrees 0.80-0.82 at 5 CVs, 0.86 at 10, 0.89-0.90 at 20, 0.92-0.94 at 50
(Kendall tau 0.74-0.77 → 0.95). B costs 18 s on the first call (background CVs embedded once) and ~1.2 s after.

| Cost (measured 23 Oct 2026) | Wall |
|---|---|
| `Screener.load()` | 4.4 s |
| D63, 500-CV pool | 4.0 s cold, 1.25 s warm |
| D63, 10 uploads (B) | 18 s first call, 1.2 s after |
| D64 | + ~2.5 s per uncached C2 pair (~2 min for a top 50); 1.5 s cached |

**Verification.** `python -m candidate_screener.inference --check` (14 items, ~9 min, C2 from cache only): the API reproduces
`a2-features.csv`, both extraction tables and `a2-x2-scores.csv` exactly; the frozen L1 equals a refit to 1e-9; over the 60
final-set pools, D63's top 10s equal the evaluation path's on all 60 JDs and Recall@10 is 0.484413 (`chat3-l1-pool.json`);
D64 reproduces `c2-validation-test.json` (Recall@10 0.48047, nDCG@10 0.831809).

**Caveats.** Uploaded PDFs differ from Djinni text (`text_v2`) and nothing measures accuracy on them; a non-IT JD is still
forced into one of the 60 IT role families (low `role_prob` is the tell); skills outside the vocabulary are not seen.

---

## 3. Data sources, in the order the design uses them

```
Reference ─ A2 final set (60 JDs × 10 CVs)             humans yc, van + LLM tie-break (D41)
Stage 1 ──► A2 CV/JD text (text_v2)                    inputs to every extractor
            ESCO ICT + data_jobs                       skill vocabulary (§4.1)
            SkillSpan, DataTurks                       train/evaluate skill tagger (§4.2)
            A2 train region + Primary Keyword/Position role-family labels (§4.3)
            weak pairs (7,548)                         SBERT fine-tune (§4.7)
            A2 dev pools (400 JDs × 500)               choose stage-1 variants without the final set
Stage 2 ──► D50 batch (1,000 pairs)                    S1 classifier; R1-R3 rerankers; learned-weights notebooks
            chat-3 (48 JDs × 100)                      TRAINS stage 2 (D62)
            chat-1 (60 JDs × 100)                      stage-2 TEST reference only
Context ──► A1 (8,000 pairs; labels do not reproduce)  secondary benchmark; learned-weights training
            livecareer PDFs/HTML, French clip          demo only (deprecated path)
```

| Source | What | Used for | Caveat |
|---|---|---|---|
| **A2** Djinni (MIT) | 141,897 JDs, 210,250 CVs, English | Everything. Partitioned at document level into `train` / `eval` before any sampling (D19) | Ukrainian/EE IT market. No fit labels in the source |
| **A2 final set** | 600 pairs; 219 humans agree, 107 split (LLM breaks tie), 274 one human | Scoring only | `llm_free` = pairs without LLM tie-breaks. Annotators agree 0.672, kappa 0.353 |
| **ESCO v1.2.1** | Skills ontology; ICT subset used | Vocabulary, normalisation (§4.1, §4.6, §6.Q) | |
| **`data_jobs`** | 252 posting skills | Vocabulary (tool names ESCO lacks) | |
| **SkillSpan** | JD skill/knowledge spans | Tagger training + evaluation; vocabulary filter selection | Validation used to select; **test never used for selection** |
| **DataTurks** | 220 resumes, Skills spans | Evaluation only (neither system fitted on it) | PII classes dropped; 222 spans repaired (HTML-escape offsets) |
| **D50 batch** | 100 A2-train JDs × 10 CVs; LLM 0-100 + Fit/No Fit (haiku, run 11) | S1 classifier, R1-R3 | None from final set |
| **`chat-3`** | 4,800 pairs scored by the owner in claude.ai (`claude-haiku-5-5`) | Stage-2 training and design choice | Same instrument as `chat-1` |
| **`chat-1`** | 6,000 pairs, X2's top 100 for the 60 final JDs | Stage-2 test reference | Never trained on since D62 |
| **A1** | `cnamuangtoun` fit pairs | Learned weights | A13 fails: kappa(A1, human) 0.08 |
| Rejected: A3-A4, B3-B4, C3-C4, D3-D4 | | | catalog Tier 3 |

---

## 4. Information / entity extraction framework 🔎

Extraction is split by *what the downstream stage needs*, not by a single model. Seven components:

```
Extraction
├── 4.1 Vocabulary + rules matcher     (ESCO ICT + data_jobs)           no training; filters tuned on SkillSpan val
├── 4.2 Skill taggers                  MiniLM token tagger (trained)  ·  GLiNER (zero-shot)
├── 4.3 Role family                    text → 60 role classes (trained logistic regression) + title spans
├── 4.4 Experience & seniority         rules (no training)
├── 4.5 JD title                       lookup + classifier fallback (D54)
├── 4.6 Concept normalisation          SBERT → ESCO (zero-shot)
└── 4.7 Related text models            SBERT weak (fine-tuned) — a text feature, not an extractor
```

### 4.0 Where each component is used (cross-reference for the methodology sections)

| Component | Ladder rungs / stage 1 | Stage 2 features | Tested and rejected as an enhancement |
|---|---|---|---|
| 4.1 vocabulary matcher | rung 1b; X3 | `coverage*`, `log_*`, `skill_overlap` | coverage weighting W1-W3; soft matching (0.60) |
| 4.2 MiniLM tagger | – | – | R+N concepts (gliner-concepts plan; step 4b arm A6) |
| 4.2 GLiNER | – | – | concept overlap gate failed; EDA only |
| 4.3 role family | X1, X2, X3, X4 (`role_match`) | `role_match` | JD-title role (N1/Y rungs); hard filters; role-set |
| 4.4 experience | X1 (`experience_compatible`) | `experience_compatible`, `exp_gap`, `exp_known` | no-experience flag (F2/N3); unknown = 0 (F2b) |
| 4.5 JD title | – | – | shortlist-recall Y/Z rungs; arm A2 (N1) |
| 4.6 normalisation | – | – | measured only |
| 4.7 SBERT weak | X4 | `sbert` | zero-shot SBERT as extra feature (A5) |

### 4.1 Vocabulary and rules matcher (no training)

- **Sources:** ESCO v1.2.1 **ICT scope** (1,489 concepts: digital-skills collection, or below skill group S5, or knowledge field 06) +
  `data_jobs` (252 posting skills) + the 8 baseline regex families (45 literals).
  Result: **9,124 aliases, 1,691 canonicals** (8,927 carry ESCO, 251 `data_jobs`, 44 regex), `data/vocab/skills.csv`.
- **Merge policy:** canonical from the highest-priority source (ESCO > data_jobs > regex). A `data_jobs` term joins an ESCO
  canonical only via that concept's *own* label, never an alternative label, because ESCO lists narrow tools (`tensorflow`,
  `git`) under broad competences. ESCO's `SPARK` (the Java framework) is withheld so Apache Spark wins.
- **Matcher:** leftmost-longest, token-bounded, case-insensitive (`C#`, `.NET`, `Node.js` match; `java` never inside `javascript`).
  Line breaks end phrases. Context filter marks mentions as unqualified / qualified (learning, negated).
- **Filters tuned, not trained:** 390 configurations of min-chars × max-tokens × stopwords × blacklist rules, selected on SkillSpan
  *validation* by strict F1. Chosen: English stopword filter on, blacklist {`email`, `it`}. Only the stopword filter matters
  (strict precision 0.404 → 0.480 by removing `it` → "computer technology").
- **Result:** SkillSpan test strict F1 **0.237**, partial **0.305** (strict recall 0.15: it finds tools, not soft-skill phrases).
  DataTurks (resumes), `all` 220 docs, partial F1 **0.380**.

### 4.2 Skill taggers

| | MiniLM token tagger | GLiNER |
|---|---|---|
| Model | `all-MiniLM-L6-v2` + token-classification head (3 labels O/B/I, one SKILL class = skill ∪ knowledge) | `gliner_small-v2.1` (166M, Apache-2.0), pinned `gliner==0.2.21` |
| Training | **Fine-tuned** on SkillSpan train. AdamW, lr 1e-4, weight decay 0.01, 10% warmup, batch 32, ≤ 6 epochs, early stop patience 2 on validation strict F1, seed 0, deterministic. 792 s on CPU. Long sentences split into chunks, never truncated. No hyperparameter search | **None** (zero-shot) with 5 prompted labels: KNOWLEDGE, SKILL, TITLE, DEGREE, EXPERIENCE_YEARS. Threshold 0.5, fixed |
| Convergence | Best epoch was the last (val strict F1 0.506, still rising): **budget-limited, a lower bound** | n/a |
| Result | SkillSpan test strict F1 **0.523**, partial **0.837** (vs rules 0.237 / 0.305; Δ +0.286 strict). DataTurks partial **0.224** (vs rules 0.380; Δ −0.156) | No accuracy figure (no validated gold adapter). Counts only |
| Failure mode | On resumes it tags requirement-like phrases DataTurks does not annotate (3.4× more spans) | Long phrases rarely match across JD and CV; adds overlap to 2 of 320 no-overlap pairs vs MiniLM's 35 |
| **D49** | Tagger adds free-text JD requirements; **rules** do CV skills and all exact/alias matches | EDA / exploration only |

Gate for using tagger concepts in a score: ≥ 5 pp fewer no-overlap pairs. Only R+N (rules ∪ MiniLM) passed, and no learner
beat X2 with it (§5.F).

### 4.3 Role family (E2) — the highest-value extractor, and a trained model

- **Why:** `role_match` replaces the metadata `Primary Keyword`.
- **Labels (training and oracle only, never an input):** `role_family(Primary Keyword, Position)` over the A2 train region. 38 keyword classes
  with ≥ 20 train JDs *and* ≥ 20 train CVs; 4 JD-only singletons folded into `Other`; the generic keywords (`Data Science`, `Lead`, `Other`)
  split by normalised `Position` into families (m = 20); a merge step removes near-duplicates (`Data Science/data engineer` → `Data Engineer`).
  **60 classes**, same space on JD and CV side. Sizes 24-25,474 documents per class.
- **Inputs:** JD `Long Description`; CV `CV`, `Highlights`, `Moreinfo`, `Looking For` (`text_v2`, no `Position`).
- **Title spans:** an alias table of 518 normalised titles (≥ 20 documents, modal purity ≥ 0.5; one-token titles ≥ 0.8) built on the fit part only;
  leftmost-longest matching up to 6-grams. Becomes 120 binary features (family named anywhere / first span is family).
  Alias family = label for 89.6% (JD) / 93.9% (CV) of covered titles, but the *first* span is right only ~50% of the time, so spans are features, not the label source.
- **Two candidates, fixed in advance:**
  (a) **TF-IDF** (hashing word 1-2-grams, 2^18 features, sublinear, IDF on the fit part) + span indicators → logistic regression (saga, C = 10, 30 epochs);
  (b) **frozen MiniLM chunk-mean** embeddings + span indicators → logistic regression (lbfgs, C = 10), on a 6,000/side subsample (CPU limit).
  Split: fit / validation by document with duplicate texts grouped; selection draw n = 2,500 and calibration draw n = 1,000 per side.
  Selection rule: higher mean JD/CV macro-F1. **(a) selected** (0.530 vs 0.472). Temperature scaling per side (T ≈ 0.91-0.93): ECE barely changes.
- **Quality (validation, A2 train docs):** accuracy 0.784 JD / 0.785 CV; macro-F1 0.54 / 0.52; top-3 0.92 / 0.93.
- **On the final-set documents (measurement only):** JD accuracy 0.683 (generic 0.775, **targeted 0.500**), CV 0.700.
  The pair score is soft: `role_match = Σ_k p_jd(k)·p_cv(k)`. It separates oracle-same-family pairs with **AUC 0.901** (generic 0.960, targeted 0.761).
  Hard argmax matching under-calls sameness (42% vs 74% oracle), which is why the soft form is used.
- **Cost:** ~52 min cold to train, ~13 min warm. Retrained byte-identically three times.
- **Caveat:** validation is against metadata-derived labels, not hand labels; span precision/recall on a hand-checked sample was never done (Q-B2 open).

### 4.4 Experience and seniority (E3, E4) — rules, no training

- **JD requirement → (min, max):** shapes `2-5 years`, `at least N`, `5+`, `up to N`, plain `N years`; months → fractions; experience-context cue required;
  company-context mentions dropped ("we have 10 years in the market") unless the line addresses the candidate; aggregation = largest minimum among
  required mentions (nice-to-have mentions only if nothing else). Dates (`2019-2022`) and numbers ≥ 21 are never a requirement.
- **CV years:** largest explicit statement wins; else merge overlapping date ranges and sum; `present` = **2023-09-01** (latest A2 `Published`) as a parameter, never today.
- **Bands:** snapped to Djinni's `Exp Years` scale (no 4-year option, so 4 → `3y`). `experience_compatible` reuses the rung-1a rule.
- **Variants selected on A2 train only** (JD aggregation `max` vs `first`: 0.554 vs 0.552). One rule change after the first train run (narrowing the company filter lifted JD coverage 0.640 → 0.659), before final-set documents were scored.
- **Quality:** coverage JD 0.659, CV 0.444 (train). Band accuracy where extracted: JD 0.841, CV 0.678; final set 0.694 / 0.718.
  Most CVs simply do not restate their years (CV coverage bounded ~0.45).
- **The key decision (registered amendment A1, 7 Oct, before X1 was scored):** the 1a rule scores an unknown band as 0, which made the extracted check
  disagree with the true one on 56% of pairs (agreement 0.435; 0.863 where both sides state years). Owner decided: **unknown on either side counts as compatible** in X1/X2.
  This is also why a CV with no professional experience still passes (a defect measured later, §5.F).
- **Seniority (E4):** title-token heading or years fallback; accuracy 0.47 JD / 0.35 CV overall, weak; used only as a source-tagged signal, not a filter.
- **Invariant tests**, each shown to fail against a naive implementation: overlapping ranges not double counted; `5+` is a minimum; `present` is a parameter; date range is not a requirement; bands match metadata; unknown = 0 under 1a.

### 4.5 JD title (D54)

The JD `Position` is an allowed inference input (a recruiter types it). Title → family: lookup of the alias table (62.5% of dev JDs, 90% accurate), classifier fallback
(55%). Overall 0.768 on dev, **no better than the JD-text classifier (0.778)**, so it was tested as a role signal (shortlist-recall Y/Z rungs, step-4b arm N1) and not adopted.
CV-side `Position` stays a label. CV *role sets* (the top-k title spans) were also tried: recall of the oracle family 0.86 at k = 1 (context off), set size 2.8.

### 4.6 Concept normalisation (zero-shot)

SBERT (`all-MiniLM-L6-v2`, cosine) maps a phrase to an ESCO concept. On 1,000 held-out ESCO alternative labels, ICT scope, preferred-label index:
**top-1 0.766** [0.740, 0.792] vs string cascade 0.602 (+0.164). This measures recovery of ESCO's own paraphrases, not mapping of free-text tagger spans.
Used in the soft-matching experiment (threshold 0.60) only.

### 4.7 SBERT weak (the text feature that sits beside extraction)

Bi-encoder `all-MiniLM-L6-v2` fine-tuned on **7,548 weak A2 train pairs** (+840 validation) labelled from metadata (keyword + experience), with hard negatives.
Two-point grid on its own validation split: (lr 2e-5, 1 epoch) and (lr 5e-5, 3 epochs); batch size 16 (32 ran out of memory); 2,850 s, 6.7 GB peak.

### 4.8 Other extraction work that did not enter the system

- Years-of-experience as a GLiNER label (`EXPERIENCE_YEARS`): kept as exploration; the rule extractor is what scores.
- Seed lexicon `seed-v1` (tiny, hand-written, explicitly **not ESCO**): used only for the first extraction notebooks (10-11) before the vocabulary existed.
- DataTurks repair and PII masking; entity-distribution EDA (notebooks 14-15); evidence-record schema (`doc_id`, offsets, `canonical_id`, `method`, `context_status`).
- Multimodal (French ASR → MarianMT → extraction) prototype, notebook 16.

---

## 5. Methodology (git-tree view; ✅ kept · ⚪ neutral · ❌ rejected · 🧪 exploratory)

```
Methodology
├── A. Foundations                                       ✅
│   A1 recheck, judge calibration, final set (D41), success measures, check tiers (D55)
├── B. Text-only ladder (random, BM25, TF-IDF, SBERT)    in the §1.1 table
├── C. Stage 1 (extraction used: 4.1, 4.3, 4.4, 4.5, 4.7)
│   ├── C.a X1-X4 registered rungs                        ✅ X2 is stage 1
│   ├── C.b Shortlist recall (4.3, 4.5)                   ❌ hard filters; soft weight ties X2
│   └── C.c Learned stage-1 classifier S1 + threshold + features (4.1, 4.3, 4.4, 4.7)   ⚪
├── D. Stage 2 (features from 4.1, 4.3, 4.4, 4.7)
│   ├── D.a R1-R3 trained on D50                          ❌
│   ├── D.b LLM as reranker (D59/D60) → superseded        ⚪
│   ├── D.c Zero-shot C2 on top of L1 (validated on chat-3)  ⚪ adopted post hoc (D64), registered bar missed
│   └── D.d chat-3 hold-out (D62) → L1 tested once        ✅
├── E. Exploratory side branches (D56; extraction used: 4.1, 4.2, 4.3, 4.4, 4.6)
│   learned weights · concepts · coverage weights · soft matching · error analysis · score defects · ceiling checks
└── F. Peripheral: annotation campaign, evidence layer (deprecated), multimodal prototype, ownership audit
```

### A. Foundations (summary)

A1's labels do not reproduce (A13 fails), so A1 is a secondary benchmark (D40). The LLM judge was calibrated, then collection closed (D42).
One frozen final set (D41). Bars missed and reported plainly (accuracy 0.768 vs 0.85; extraction F1 0.523 vs 0.85; annotators themselves agree 0.672).

### C. Stage 1 (expand: 🔎)

| Step | Extraction used | What | Result |
|---|---|---|---|
| `text_v2` | – | CV text without `Position` | 22.7% of CVs still mention the removed title (reported beside figures) |
| X1-X4 | 4.3, 4.4 (+4.1 for X3, 4.7 for X4) | Registered rungs | §1.1. K1 justified; K2-K5 not resolved |
| Shortlist recall | 4.3, 4.5 | Hard role filters Y1/Y2/Y2t/Y4 on a 400-JD dev set; then soft role weight Z1/Z1L | ❌ hard filters lost to X2 (dev R@50 0.664-0.678 vs 0.727: early stop). Soft weight beat X2 on dev but tied on final (0.781 vs 0.779); S2 still missed, nothing adopted |
| S1 classifier (D61) | all of F12 | Logistic regression on D50 Fit/No Fit labels | AUC +0.009 over X2 (CI crosses 0). At 0.5 it passes ~23 CVs/JD and keeps 240/367 judged fits (X2 top 100 keeps 332) |
| L1 over the whole pool | – | Drop stage 1 | Same top 10 in 59/60 JDs → simplification (D63) |

### D. Stage 2 (expand: 🔎)

| Step | What | Result |
|---|---|---|
| R1/R2 cross-encoders, R3 bi-encoder on D50 | MiniLM-based, pointwise/pairwise | ❌ R@10 0.19 / 0.23 / 0.41 vs X2 0.419 |
| Zero-shot C2 (Qwen3-Reranker-0.6B) | Alone, fused with S1, fused with L1; top-30 cascade; then the c2-validation menu (9 options on `chat-3`) | Alone −0.106 vs stage 2 (`chat-3`); top-30 cascade −0.002. Validation: winner k = 50, w = 0.25, +0.019 [+0.003, +0.034] on `chat-3`; test +0.005 [−0.010, +0.021], bar (+0.02) missed; adopted post hoc (D64) for nDCG +0.012 |
| **`chat-3` hold-out (D62)** | Choose among L1/L2 × F12/F9 by 5-fold CV on 48 new JDs, fit once, test once on the 60 | L1 F12 chosen (0.513 OOF; menu within 0.011). Test: §1.2. Replaces the `chat-1`-trained L2, which was chosen and tested on the same 60 JDs |

**First-pass figures on `chat-1` alone** (X2's top 100 only, 60 JDs; X2 Recall@10 is 0.419 on this reference). Compare rows only within this block.

| Pipeline | Recall@10 | Δ vs X2 [CI] | nDCG@10 | Trained on |
|---|---|---|---|---|
| X2's order | 0.419 | – | 0.776 | nothing |
| → S1 probability (D50 classifier: logistic regression on the F12 features, Fit / No Fit labels) | 0.454 | +0.034 [+0.005, +0.063] | 0.801 | D50 |
| → **L1 on F12** | **0.484** | **+0.065 [+0.036, +0.096]** | 0.822 | `chat-3` |
| → C2 zero-shot reranker alone | 0.379 | −0.040 | 0.751 | nothing |
| L1 top 50 → L1 + 0.25 C2 (D64) | 0.490 | +0.071 (+0.006 vs L1) | 0.832 (on `chat-1` + `chat-2`) | `chat-3` (L1); C2 none |

Pre-registered rules for L1 (on `chat-1`) all pass: Δ ≥ +0.05; beats the LLM-free S1 order (+0.031 [+0.007, +0.056]); human per-JD AUC vs X2 +0.068 [+0.002, +0.143]
(a CI edge). The `llm_free` human AUC gain is +0.040 [−0.046, +0.135]. Pairwise on `yc`'s grades +0.104 [+0.049, +0.165].
The §1.2 table uses `chat-1` + `chat-2` instead, because L1 ranks all 500 CVs and `chat-2` scores the CVs outside X2's top 100 that it pulls in.

### E. Exploratory side branches (compress to one row each in the report)

| Branch | Extraction tested | Result |
|---|---|---|
| Learned weights, learning curves (nb 18-19) | 4.1, 4.3, 4.4 | ❌ X2 not beaten (0.850). Learners fit training JDs no better than held-out ones (gap 0.01): features, not labels, are the limit |
| GLiNER / MiniLM concepts | 4.2 | Gate passed for R+N only; no learner beat X2 |
| Coverage weighting W0-W3 | 4.1, 4.2 | ❌ None above W0 (84% of MiniLM concepts unknown to the vocabulary, weight 1) |
| Soft matching, threshold 0.60 | 4.6 | Coverage-only +0.07 AUC, but ❌ no gain in the full score |
| Error analysis + manual review (45 of 111 cases) | 4.3, 4.4 | Role oracle shows no headroom. **23 of 45 flagged "label looks wrong"**, mostly LLM-decided Fits |
| Score defects (nb 22) | 4.1, 4.4 | ❌ Negation drop −0.003; no-experience flag +0.002; unknown = 0 −0.020 |
| Ladder on clean labels (nb 23) | – | Dropping the 107 split pairs lifts X2 0.850 → 0.900; sophisticated rungs do not gain more |
| Ceiling checks (experience oracle, must-have signal) | 4.4 | Registered; I found no recorded result |

---

## 6. Answers to specific questions

**Q. Did we use the ESCO skill ontology?** Yes, partly.
- **Used:** the ESCO v1.2.1 *skills pillar, ICT scope* (1,489 concepts) is the backbone of the vocabulary: preferred + alternative labels (and labels with a trailing
  parenthetical removed) become aliases, preferred label = canonical, and the ESCO URI is kept per row. It drives the rules matcher (4.1), all concept features, and
  the normalisation test (4.6). Not used: hidden labels (misspellings, version strings).
- **Restricted/modified:** ICT only, decided before measurement; `data_jobs` and regex terms attach only through a concept's own label; `SPARK` withheld.
- **Not used:** the ontology's hierarchy or relations (broader/narrower, essential vs optional skills for occupations) for scoring; ESCO *occupations* were
  measured for title coverage but map to no class of the 60-class role space, so they are not in the title alias table (that table is built from A2 `Position`).
- **Practical limit:** ESCO covers few recruiter-relevant tool names, so `data_jobs` fills the gap; 84% of MiniLM tagger concepts are not in the vocabulary at all.

---

## 7. Still open

| Item | Why it matters |
|---|---|
| **Report wording for D64** | C2 was adopted over a missed registered bar: say so beside every figure |
| Final method in the report | D64 is the reported system; confirm it is the final conclusion |
| Human evidence for stage 2 is thin | AUC CI edge +0.002; `llm_free` gain not above 0 |
| Role extraction on targeted JDs is weak | JD accuracy 0.50, soft AUC 0.76; role is X1's main contribution |
| Span precision/recall on a hand-checked sample (Q-B2) | Never done; role-family validation uses metadata labels |
| Ceiling-check results | Not found in records |
| Reference covers X2's top 100 (+ partial `chat-2`) | Fine for compared top 10s (chat-4 empty); a wider comparison would need more scoring |
| Hosting / packaging the system | The local entry point exists (§2.6); portability (no-pickle role model, HF Hub bundle, PII filter on vocabularies) is still the on-hold plan `2026-10-22-model-bundle-and-hosting`. The old recruiter app (`app/`) still runs its own TF-IDF draft ranker |
| Inference API: warm checks of the shared code | `learned.py`, `extracted.py`, `a1_weights.py`, `cross.py` gained default-preserving arguments; the `inference` check passes, the other affected checks still need a warm run (owner to schedule) |
| `entity_weighted` record | A teammate's `blocked` placeholder was regenerated (pilot_complete, not adopted); the author may want to confirm |
| Cross-encoder fine-tuning from a reranker checkpoint | Literature suggests it; not tried |
| Replicate of the D50 batch (Q-A1b) | Score noise unknown |

---

## 8. Where to look

| For | Open |
|---|---|
| Every decision | `docs/decisions.md` |
| Extraction | `plan/2026-09-26-top-down-reset/08-extraction.md`, `plan/2026-10-06-extraction-and-rerank/{11-p2,12-p3,13-p4}*.md`, `plan/2026-10-07-shortlist-recall/` |
| Stage 1 | `plan/2026-10-06-extraction-and-rerank/16-p7-stage1.md` |
| Stage 2 | `plan/2026-10-18-classifier-then-rerank/README.md`, `plan/2026-10-21-chat3-holdout/README.md`, `plan/2026-10-22-c2-validation/README.md`, notebook `26-chat3-holdout-and-c2.ipynb` |
| Records | `output/extraction/`, `output/matching/`, `output/rerank/`, `output/inference/` |
| Calling the system from an app | `plan/2026-10-23-inference-api/README.md`, `src/candidate_screener/inference/`, `README.md` ("Using the system from an app") |
