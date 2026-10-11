# 02 — Design decisions

*Written 11 Oct 2026 against `5fc2856`. This document covers the **research** system: the
ladder, the stage-1 X rungs and stage 2. Where the app differs, it says so.
"Origin" means where the value came from. `D##` is the decision log; "unknown" means
I found no rationale in code, decisions or plans.*

## 1. Decisions, one by one

### 1.1 Input text: what a system reads

| | |
|---|---|
| **Chosen** | JD = Djinni `Long Description` only. CV = `CV` + `Highlights` + `Moreinfo` + `Looking For`, joined by blank lines (`text_v2`, D53). Since D54, the JD title is also an input for the Y/Z rungs only. English only, filtered by Djinni's own `*_lang` column |
| **Where** | `matching/corpus.py:36-39` (fields), `:54-65` (`build_cv_text`), `:82-86` (language filter) |
| **Alternatives** | Keep `Position` (v1, what annotators saw); include metadata fields (keyword, experience) as features; detect language yourself; translate non-English |
| **Trade-off** | v2 is honest about open-world input, because a recruiter's CV has no structured title field. But the **labels were made reading v1**, so v2 systems are judged against labels that used information they cannot see. Residual title leak in v2 bodies is 22.7% of CVs (`text_leak.py`). The English filter trusts the publisher's language tag, and how accurate that tag is has not been measured |

### 1.2 Parsing

| | |
|---|---|
| **Chosen** | No resume parser in the research path: the Djinni fields are already text. The app uses `pypdf` page text and regex HTML stripping (`app/files.py:9-20`). The only structure extracted is: skill mentions (§1.4), role family (§1.6), experience years and bands (§1.7) |
| **Alternatives** | A layout-aware PDF parser (e.g. pdfplumber, unstructured, docTR+OCR); a section segmenter; a commercial CV parser; an LLM extractor |
| **Trade-off** | Simple and deterministic, and Djinni does not need it. **Nothing about parsing is evaluated**: the livecareer PDFs (C1) were acquired "for the ingestion demo only" (D10) |

### 1.3 Text cleaning / normalisation

| Component | What is done | Where |
|---|---|---|
| TF-IDF | sklearn defaults + `lowercase`, English stop words, 1–2-grams, `min_df=2`, `max_df=0.8`, `max_features=5000`. One vectoriser for both sides (a past bug: two vectorisers produced near-random cosines) | `baselines/tfidf.py:27-34,37-53` |
| BM25 | Same token pattern as TF-IDF (shared tokeniser, another past bug), `BM25Okapi` defaults `k1=1.5`, `b=0.75` | `baselines/lexical.py`, `baselines/bm25.py:34,64-65` |
| SBERT | Whitespace collapsed, then **800-character chunks**, each embedded, mean-pooled, L2-normalised | `matching/sbert.py:28,35-37,70-72` |
| Skill matcher | Case-folded tokens (letter/digit runs; every other char is its own token); the gap pattern (touching / space / line break) must match the alias | `extraction/rules.py:27-57` |
| No stemming, lemmatisation, spelling correction, or de-duplication of boilerplate | — | — |

**Alternatives:** sub-word or char n-grams for TF-IDF (would help `ReactJS`/`React`),
lemmatisation, boilerplate removal ("About us…" in JDs), section weighting.
**Trade-off:** fixed and reproducible. Every config is digested into the record, so a change
fails the check. **None of the TF-IDF settings were tuned**; they are the Phase-4
baseline's, frozen (D19 baseline).

### 1.4 Skill vocabulary and matching

| | |
|---|---|
| **Chosen** | `data/vocab/skills.csv`: 9,124 aliases. ~97% come from **ESCO v1.2.1 ICT scope** (skill group `S5` *working with computers* or `06`), plus 191 rows from a data-jobs postings corpus and a handful of regex synonyms (`nodejs`, `postgres`). Matching is leftmost-longest, non-overlapping, exact-alias trie |
| **Where** | `extraction/vocab.py:41-78`; `extraction/rules.py:60-104` |
| **Alternatives** | A curated IT tech taxonomy (e.g. Stack Overflow tags, LinkedIn skills, O*NET Hot Technologies); embedding nearest-label normalisation (`extraction/normalise.py` measured this); an LLM extractor; fuzzy matching |
| **Trade-off** | ESCO gives free, licensed, URI-linked canonicals. But it is an EU occupational taxonomy, **not** a developer vocabulary: it lacks `JS`, `k8s`, `LLM`, `Figma`, `QA`, and maps `ML`→the ML language and `.NET`→Visual Basic (probes in `03` §3). The filter grid (`vocab.py:297-307`) was tuned for **strict span F1 on SkillSpan**, which is job postings, not CVs |
| **Magic numbers** | `BLACKLIST = ("email", "it")`, `BLACKLIST_RULE = ("partial", 10, 0.25)`, `ESCO_HOMONYMS = ("spark",)`, `REGEX_SYNONYMS` (2 entries). **Origin:** the grid was tuned on SkillSpan validation (`extraction/evaluate.py` checks that the committed constants equal a re-tune); the blacklist and homonyms were added by hand (unknown which errors motivated them) |

### 1.5 Learned skill tagger (NER)

| | |
|---|---|
| **Chosen** | MiniLM token classifier, BIO over SKILL = skill ∪ knowledge, trained on SkillSpan train, early-stopped on validation. Used only to add free-text JD requirements of ≤ 6 words (D49) |
| **Where** | `extraction/ner.py`; `evidence/requirements.py:18,38-53` |
| **Alternatives** | JobBERT / ESCOXLM-R (domain-pretrained, not cached); spaCy; no tagger |
| **Trade-off** | Strict F1 0.523 vs rules 0.237 on SkillSpan (README). Strict precision is ~0.50, hence the 6-word cap. It does not affect ranking, only evidence |
| **Magic** | `MAX_FREE_TEXT_WORDS = 6`. **Origin:** judgement, justified only as "a long span makes a vague requirement" (`requirements.py:7-8`) |

### 1.6 Role family (the strongest signal)

| | |
|---|---|
| **Chosen** | 60 role families derived from Djinni `Primary Keyword`. Generic keywords (`Data Science`, `Lead`, `Other`) are split by normalised `Position` title (Q-B1, "T-A split generic"). Families with fewer than `MIN_SUPPORT=20` train JDs **and** CVs are folded into `Other`. Each side gets a classifier: TF-IDF-hash or MiniLM features + title-span features → L2 logistic regression (`C=10`) → temperature-calibrated softmax. **Role match = Σₖ p_jd(k)·p_cv(k)**, the probability that two independent draws agree |
| **Where** | `extraction/role_family.py:64-86,404-416,551-570,833-846` |
| **Alternatives** | Hard argmax match (the oracle 1a is the hard version); top-k set overlap (Y rungs, `role_set.py`); title-only lookup (C1, `jd_title.py`); cosine of role embeddings |
| **Trade-off** | Soft agreement degrades gracefully: the hard filter (Y2) lost badly when the JD family was wrong (Recall@50 0.305 vs 0.546, `shortlist-recall/05-findings.md`). But a soft term **lets text overtake role**: at weight 2, text in [0, 0.999] can reorder CVs whose role agreement differs by < 0.5 (`extracted.py:12-15`). That is the main reason X2 retrieval (0.779) trails the oracle (0.924) |
| **Accuracy** | JD 0.683 / CV 0.700 top-1, 0.850 / 0.877 top-3 on final-set documents (`16-p7-stage1.md` §6a) |
| **Magic numbers** | `MIN_SUPPORT=20`, `MERGE_PURITY=0.5`, `LR_C=10.0`, `LR_MAX_ITER=30`, `HASH_FEATURES=2**18`, `MINILM_TRAIN_N=6000`, `MINILM_CLASS_FLOOR=15`, `N_SELECT=2500`, `N_CALIB=1000`, `VAL_SHARE=0.10`, `TOP_K=3`, `ECE_BINS=15`. **Origin:** recorded in `CONFIG` and the P3 record (`12-p3-role-family.md`). `LR_MAX_ITER=30` with `saga` looks like a runtime cap; whether it converges is **unknown** (no warning check found). The rest read as reasonable defaults, not tuned values |

### 1.7 Experience

| | |
|---|---|
| **Chosen** | Bands `0-1 / 2-3 / 4-6 / 7+`. JD years = regex mentions in candidate context (not "our company has 10 years"), optional lines dropped, **max** aggregate. CV years = **merged date ranges** up to `REFERENCE_DATE = 2023-09-01`, else stated "N years". Compatible = CV band ≥ JD band; **unknown on either side → compatible** (amendment A1) |
| **Where** | `extraction/experience.py:81-135,239-402`; `data/a2_finetune.py:53-61`; `matching/corpus.py:132-135` |
| **Alternatives** | Continuous years with a tolerance; penalise over-qualification; seniority words (E4 exists, not used in X1) |
| **Trade-off** | Experience is unknown for 71% of pairs, so under A1 it equals 1 for **95%** of pairs (`16-p7-stage1.md` §6). **In practice X1 is a role-only rule.** The app's `years_stated` (max "N years" anywhere) is a different, cruder extractor from this one |
| **Magic** | Band edges: mirror Djinni's `Exp Years` options (`JD_EXP_TO_BAND`; `5y` → `4-6`, **so a "5+ years" JD and a 6-year CV are compatible, but nothing maps to `7+` on the JD side**). `REFERENCE_DATE` ≈ the Djinni dump date (inferred). `JD_MAX_YEARS=20`, `CV_MAX_YEARS=45`, `MIN_RANGE_YEAR=1970`: sanity caps, origin unknown |

### 1.8 Scoring: how the signals combine

| | |
|---|---|
| **Chosen** | **`score = 2 × role + experience_compatible + 0.999 × minmax_perJD(text)`**. The 1a rule with the true fields (oracle), or extracted (X1–X4). Text = TF-IDF (X2), skill overlap (X3) or SBERT weak (X4) |
| **Where** | `matching/rules.py:36-43`; `matching/lexical.py:83-115`; `matching/extracted.py` |
| **Alternatives** | A learned combination (rung 5, `hybrid.py`: L2 logistic on A1, **not resolved** vs metadata + TF-IDF); a weighted sum tuned on dev; learning-to-rank (LambdaMART) on the weak or LLM batch; the Z rungs (`64 × role`, chosen on dev) |
| **Trade-off** | Fixed **before** scoring, so it cannot overfit the final set. But it is lexicographic by design: role ≫ experience ≫ text. **Skills and education have no explicit weight.** Skills enter only via TF-IDF/SBERT text similarity (X3's skill-overlap variant added +0.002, K5). Education is not modelled at all |
| **Magic numbers** | `2`, `1`, `0.999`. **Origin:** "keyword first, experience breaks ties, text orders within" (`rules.py:37-38`). `0.999` keeps the text term strictly below one experience step. The ×2 makes keyword dominate both. Judgement, chosen before any held-out rung was scored (D48 registration) |

### 1.9 Embedding model

| | |
|---|---|
| **Chosen** | `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-d, 256-token window), everywhere: SBERT, role classifier, NER, cross-encoders, evidence |
| **Where** | `matching/neural.py:19` |
| **Alternatives** | `bge-small/base`, `e5-base`, `gte`, `nomic-embed`; multilingual models (`paraphrase-multilingual-MiniLM`) for Djinni's UA/RU text; an MS MARCO cross-encoder checkpoint |
| **Trade-off** | CPU-cheap and already cached. The top-down reset fixed "the cached MiniLM unless more are approved" (`2026-09-26-top-down-reset/02-design-spec.md:375`). **No other model was ever tried**, so "SBERT vs TF-IDF" really means "*this* MiniLM vs TF-IDF". P8 notes the cross-encoders started from a bi-encoder checkpoint with an untrained head, a likely cause of their weakness (`17-p8-stage2.md` §6) |

### 1.10 SBERT fine-tuning (weak supervision)

| | |
|---|---|
| **Chosen** | `MultipleNegativesRankingLoss` on (JD, positive CV, hard-or-easy negative) triplets from the **A2 train region**. Positive = same `Primary Keyword` and CV band ≥ JD band; hard negative = same keyword, lower band; easy = different keyword. 3,000 JDs, 10% JD/CV-disjoint validation. 1 epoch, lr 2e-5, batch 16, warmup 0.1, max_seq 256, seed 0 |
| **Where** | `data/weak_pairs.py:1-13,31-33,49-70`; `matching/sbert.py:31-32,118-183` |
| **Alternatives** | Train on the LLM 0–100 batch (R3: tied, O3 not resolved); contrastive on title only; TSDAE domain adaptation; no fine-tuning (zero-shot 0.737 v2) |
| **Trade-off** | Cheap labels at scale, and the biggest single gain (+0.08–0.10 AUC over zero-shot). But **it teaches the model to predict the keyword**, the same thing the final-set sampling enriched for. It is the most "metadata-shaped" text model. The docstring says so (`weak_pairs.py:11-13`) |
| **Magic** | `N_JDS=3000`, `VAL_FRACTION=0.10`, `FT_CONFIG`. **Origin:** "fixed in advance: nothing is tuned (D46)" (`sbert.py:6-7`). They are library defaults, not searched |

### 1.11 Similarity metric

Cosine everywhere (TF-IDF L2-normalised vectors; SBERT unit vectors via dot product).
BM25 is asymmetric, query = JD. No calibration to a probability anywhere; scores are
comparable **within one JD only** (`rerank/base.py` docstring). The app's "confidence" is
the within-pool percentile (`pipeline.py:86`).

### 1.12 Thresholds

| Threshold | Value | Origin | Where |
|---|---|---|---|
| Evidence "supported" τ | **0.28** | Max F1 on SkillSpan validation: phrase vs its own sentence (positive) or a **random other** sentence (negative). F1 0.914. The negatives are easy, so τ is too low for in-domain CV sentences: in the walkthrough notebook §4, 60% of unmatched requirement cells are 'supported', e.g. "DevOps practice is a hobby for me" supports both *Agile development* and *Jenkins* | `evidence/evidence.py:110-159`; `output/evidence/tau.json` |
| Sentence minimum | 12 chars | unknown | `evidence.py:37` |
| Fit / No Fit | None in research: systems rank. The cross-encoders had a validation-split threshold (0.636 / 0.530) and were rejected | `indomain-metrics.json` `classifiers` |
| S2 retrieval bars | Recall@50 ≥ **0.80** classical / **0.90** advanced | The proposal's bars were **Recall@10 > 80% / > 90%** (`proposal v3:310`). The reset **kept the numbers but moved them to Recall@50** (`04-success-measures.md:30,58`). Defensible (Recall@10 over 10 judged CVs is near-impossible), but it is a changed metric with the old numbers | `evaluation/stage1.py:60` |
| Verdict rule | "Justified" = paired-Δ CI lower > 0 **and** both strata Δ > 0 **and** `llm_free` Δ > 0 | Registered 26 Sep (`06-registration.md`) | `matching/selection.py:38-52` |
| Bootstrap | 10,000 resamples, α 0.05, over JDs | Standard | `evaluation/resample.py:14-15` |
| Pool | 500 candidates = 10 judged + 490 random eval-region CVs; shortlist 50 | "pilot scale" (02 §5.2); 500 stands in for a role's applicant volume (not sourced) | `retrieval_pilot.py:27-29` |

### 1.13 Rules and filters

| Rule / filter | Effect | Where |
|---|---|---|
| English-only | Non-English documents never enter evaluation or training | `corpus.py:82-86` |
| Document-level train/eval partition | Eval-region documents never train anything (D19). Verified by id, `03` §1.1 | `data/a2_finetune.py` |
| No metadata at inference | `extracted._text_only` hands extractors only `jd_text`/`cv_text` (D53) | `extracted.py` |
| Nothing fitted on the final set | AST tests enforce that training modules don't read labels (D46) | `tests/` |
| LLM dispatch closed | `DispatchClosed`, except the D50 batch (D42) | `annotation/llm_recheck.py` |
| Contamination | 44 livecareer resumes overlapping A1 are excluded from NER/distractors | AGENTS.md data rule 5 |
| **Not present** | No hard filter on role, years, location, salary, language level or education. No protected-attribute masking | — |

### 1.14 Evaluation design (also a design decision)

| Decision | Alternative | Trade-off |
|---|---|---|
| 10 CVs per JD, 8 from its keyword (D31) | Random CVs, or the top-k from several systems (pooling, TREC-style) | Random gave 28/30 No Fit in a pilot (wasteful). But enrichment by keyword makes "same role" the dominant signal (`03` §2.2 T1). **TREC-style pooling of system outputs was not considered in the records I read**, and it would test discrimination where systems actually disagree |
| Fit = Good ∪ Potential (D47) | 3-class; Good vs rest | Raises agreement and sample size; loses the "strong candidate" distinction (partly recovered by D52 ordering) |
| LLM breaks human ties (D41) | A third human; drop ties; keep three-class mean | Uses all 600 pairs; injects one model's judgement into 18% of labels |
| Per-JD AUC headline | nDCG@k, P@k, MRR | Threshold-free and matches "rank within a job". But it is not what a recruiter sees ("are my top 5 good?") |
| Cheapest-unbeaten selection rule (D48) | Pick the best point estimate | Guards against chasing noise; with CIs this wide it almost always picks the cheaper rung |

## 2. Hard-coded values: consolidated index

| Value | Where | Origin |
|---|---|---|
| `max_features=5000, min_df=2, max_df=0.8, ngram=(1,2)` | `baselines/tfidf.py:27-34` | Phase-4 baseline, untuned |
| IDF sample 20,000 JDs + 20,000 CVs, seed 0 | `matching/corpus.py:139` | unknown (size choice); seed fixed |
| `k1=1.5, b=0.75` | `rank_bm25` defaults | library default |
| `CHUNK_CHARS=800` | `matching/sbert.py:28` | ≈ 256 tokens of English (inferred) |
| `FT_CONFIG` (1 ep, 2e-5, bs 16, warmup 0.1, seq 256) | `matching/sbert.py:31-32` | library defaults, "nothing tuned" |
| `CE_CONFIG` (BCE, 1 ep, 2e-5, bs 16, max_len 384) + `CE_GRID` {defaults, 5e-5/3 ep} | `matching/cross_encoder.py:37-40` | registered grid |
| `DELTAS=(10,20)`, `MICRO_BATCH=8`, `COSINE_SCALE=20.0` | `rerank/models.py:43-50` | registered (A3); 8 forced by RAM (`17-p8` §7); 20 = sentence-transformers' usual scale |
| Score formula `2·role + exp + 0.999·text` | `rules.py`, `lexical.py:106` | judgement, registered before held-out scoring |
| Z1 weight 64, Z1L 32 | `evaluation/shortlist_final.py:32` | chosen on the dev set, grid {2,…,64, 1e6} (amendment Z′) |
| `MIN_SUPPORT=20`, `LR_C=10`, `TOP_K=3`, … | `extraction/role_family.py:70-414` | P3 config; partly judgement |
| `REFERENCE_DATE=2023-09-01`, caps 20 / 45 / 1970 | `extraction/experience.py:81-85` | dump date (inferred); caps unknown |
| Band edges 0-1/2-3/4-6/7+ | `data/a2_finetune.py:53-61` | Djinni's own options |
| `tau=0.28`, grid 0.10–0.80 step 0.01, n=1500 | `evidence/evidence.py:110-159` | calibrated on SkillSpan |
| `MAX_FREE_TEXT_WORDS=6`, sentence ≥ 12 chars | `evidence/requirements.py:18`; `evidence.py:37` | judgement |
| `POOL_SIZE=500`, `SHORTLIST=50`, `KS=(10,50,100)` | `evaluation/retrieval_pilot.py:27-29` | plan 02 §5.2 |
| S2 bars 0.80 / 0.90 | `evaluation/stage1.py:60` | proposal Recall@10 bars moved to Recall@50 |
| 10 CVs/JD, 8 same keyword; 40 + 20 JDs | annotation campaign (D31, D28) | annotation budget (~250 → 1,034 labels) |
| `N_BOOT=10_000`, `ALPHA=0.05` | `evaluation/resample.py:14-15` | standard |
| Redaction: phone `\d{2,4}([\s.-]\d{2,4}){1,3}`, digits ≥ 7 | `annotation/redact.py:33-35` | tuned by eye to the corpora's phone shapes (docstring); over-matches date ranges |
| App: `[:12]` id truncation, `head(50)`, `[:5000]` resume chars | `app/streamlit_app.py:122,130,137` | UI choices |

## 3. Decisions that look fragile to me

1. **The ×2 role weight with a soft role term.** The formula was designed for a 0/1
   keyword. With a probability in [0, 1] it no longer acts as a filter. The dev set says a
   much larger weight (16–64) recovers recall, which the registered X rungs could not use.
2. **One embedding model for everything.** Every "advanced" conclusion is conditional on
   MiniLM-L6. The research question ("does sophistication earn its complexity?") is
   answered for one small model only.
3. **Weak labels = the keyword.** SBERT weak's win and the evaluation's keyword enrichment
   are the same signal. A model trained on genuinely different supervision (R3, LLM batch)
   ties it. That is consistent with "nothing beyond role is being learned or measured".
4. **The ESCO-based vocabulary for an IT corpus.** It drives evidence, skill overlap and
   requirement extraction; its gaps are systematic, not random.
5. **"Unknown experience = compatible."** It makes the experience term almost a constant
   (95% ones), so any claim that the system "weighs experience" is not supported.
