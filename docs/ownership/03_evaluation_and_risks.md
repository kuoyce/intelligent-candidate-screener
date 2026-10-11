# 03 — Evaluation and risks

*Written 11 Oct 2026 against `5fc2856`. Every figure is read from a committed record or
measured by me on this machine. Probes I ran are marked **(probe)**. They are ad-hoc
scripts, not committed, and the notebook (deliverable 4) will make them reproducible.
Side effect of the probes: they added embeddings to the git-ignored cache
`data/processed/embeddings/sbert-weak-*.npz`. No code or committed file changed.*

---

## 1. What "good" means here

### 1.1 Ground truth: the "final set"

| Fact | Value | Source |
|---|---|---|
| Corpus | A2 = Djinni (Ukrainian / Eastern-European IT job board), English documents only | `matching/corpus.py:82-84` (filters `_lang == "en"`) |
| Size | **600 pairs = 60 JDs × 10 CVs**. `generic` stratum 40 JDs (random draw); `targeted` 20 JDs (chosen job families) | `docs/data/manifests/a2-final-labels.csv` |
| How the 10 CVs were chosen | **8 of 10 share the JD's `Primary Keyword`** (D31), so the set is enriched on purpose | README §annotation; D31 |
| Labels collected | Good Fit / Potential Fit / No Fit, from two humans (`yc`, `van`) and one LLM (claude-haiku-4-5, instrument `5d986c82`) | `evaluation/final_set.py:39-46` |
| Label used | **Fit** = Good ∪ Potential, otherwise **No Fit** (D47) | `final_set.py:68-74` |
| Combination rule | humans agree → theirs · humans split → the LLM decides · one human → that human · none → LLM | `final_set.py:81-102` |
| Tier counts (total) | humans agree **219** · LLM broke a tie **107 (18%)** · single human **274 (46%), all `yc`, all in `generic`** · LLM only 0 | `output/annotation/a2-final-report.json` |
| Fit rate | **61%** overall (generic 62.5%, targeted 58.5%) | same |
| Human–human agreement | Cohen's κ **0.35** [0.25, 0.46] total; **0.29** [0.15, 0.43] targeted; raw agreement 67% | same |
| Annotator leniency | yc calls 54% Fit, van 71%, LLM 62% | same |
| Train/eval separation | A2 is partitioned by **document** into train/eval regions (D19). I checked by id: **0** final-set documents appear in the weak pairs or in the LLM training batch **(probe)**. Near-duplicate CVs re-posted under new ids were **not** checked | `docs/data/manifests/a2-partition.csv` |

**Plain-English reading:** the 600 labels are the ground truth. Two people looked at the
same pair and agreed on Fit / No Fit only two times in three (κ 0.35, "fair" agreement).
Almost half the pairs were labelled by one person, and the LLM settled 107 disagreements.
So the reference is noisy, and in places it reflects one person's or the LLM's view.

### 1.2 Metrics actually computed

| Metric | Plain meaning | How it is computed | Where |
|---|---|---|---|
| **Per-JD AUC** *(headline)* | Pick a random Fit CV and a random No-Fit CV for the same job. AUC is the chance the system scores the Fit one higher. 0.5 = coin flip, 1.0 = perfect | `roc_auc_score` within each JD that has both classes (**59 of 60**), then the mean over JDs. 95% CI = bootstrap of the **JD** means (10,000 resamples) | `evaluation/indomain.py:92-96`, `resample.py:43-49` |
| **Within-family AUC** | The same, but only among CVs sharing the JD's `Primary Keyword`. Asks: can the system tell good from bad **inside the right role**? | Same function on the keyword-matched subset (**48** JDs scoreable) | `indomain.py:137-144` |
| **Paired Δ AUC** | Is system B better than A **on the same jobs**? | Mean of per-JD (AUC_B − AUC_A) with a JD-bootstrap CI. A registered rule turns it into a verdict: *justified* (CI > 0, both strata positive, holds without the LLM labels) / *harmful* (CI < 0) / *not resolved* | `indomain.py:171-186`; `matching/selection.py:38-52` |
| Accuracy, macro-F1, F1(Fit), κ | For the Fit / No-Fit **call** only | Counts, JD-block bootstrap | `indomain.py:191-221` |
| Always-Fit floor | What a classifier gets by saying "Fit" for everyone: **0.612** accuracy | `indomain.py:225-228` | |
| **Recall@10 / Recall@50** (500-pool) | Hide the 10 judged CVs among 490 random others. What share of the Fits lands in the top 10 / top 50? | Per-JD recall, mean over JDs. Distractors are **assumed** No Fit | `evaluation/retrieval_pilot.py:36-81`, `stage1.py` |
| Pairwise ordering / nDCG (D52) | Among CVs with *different* human grades (Good=2 > Potential=1 > No=0), the share ordered correctly | The verdict cell is **`yc`'s grades only** (amendment A3) | `evaluation/ordering.py`; `17-p8-stage2.md` |

**Not computed on A2:** precision@k and MRR. Precision@k exists only for the superseded A1
pools (`evaluation/metrics.py`). For a recruiter, "how many of my top 5 are Fits" is the
most intuitive number, and it is not reported for the primary benchmark.

---

## 2. Current results, and how far to trust them

### 2.1 The numbers

Per-JD AUC, total [95% CI], n = 59 JDs. `v1` = CV text includes the `Position` (job title) field; `v2` = removed (D53).

| System | Uses metadata? | AUC (v1) | AUC (v2) | Within-family (v2) | Status |
|---|---|---|---|---|---|
| random | — | 0.556 | 0.556 | 0.580 | floor |
| keyword match | **yes** | 0.766 | — | 0.500 (by construction) | reference |
| metadata rules (keyword + exp) | **yes** | 0.778 | — | 0.540 | reference |
| TF-IDF | no | 0.797 | 0.789 [0.74, 0.84] | 0.714 | exploratory |
| BM25 | no | 0.636 | 0.632 | 0.616 | exploratory |
| SBERT zero-shot | no | 0.774 | 0.737 | 0.731 | exploratory |
| **SBERT weak** (app, uploads) | no* | **0.854** [0.82, 0.89] | 0.834 [0.80, 0.87] | 0.752 | held-out (v1) |
| **metadata + TF-IDF** (app, demo) | **yes** | **0.849** [0.80, 0.89] | — | 0.708 (v1) | selected by D48, now **forbidden** by D53 |
| X2 = extracted role + exp + TF-IDF | no | — | 0.850 [0.81, 0.89] | 0.745 | held-out, **not adopted** (missed recall bar) |
| X4 = extracted role + exp + SBERT weak | no | — | 0.855 [0.82, 0.89] | 0.735 | held-out, not adopted |
| cross-encoder (A1 / weak) | no | 0.488 / 0.661 | — | — | **rejected** |
| Stage-2 rerankers R1 / R2 / R3 (pairwise ordering, yc) | no | 0.620 / 0.616 / 0.810 vs SBERT weak 0.809 | | | **rejected** |

\* *SBERT weak uses no metadata at inference, but it was trained on pairs labelled by
metadata (`data/weak_pairs.py:1-13`). It learned to predict the keyword.*

Fit/No-Fit call (keyword equality): accuracy **0.768**, κ 0.465, against the always-Fit floor of 0.612.
Retrieval (500-pool, v2): X2 Recall@50 **0.779**, X4 0.761, vs the registered bars 0.80 / 0.90 (both **missed**). The oracle keyword rule gets 0.924.

### 2.2 Trustworthiness: candid assessment

| # | Issue | Severity | Evidence |
|---|---|---|---|
| T1 | **The task is mostly "same role?", and the sampling made it so.** 80% of pairs share the JD's keyword by design. **96.5% of all Fits share it**; fit rate is 74% with a keyword match vs **11%** without **(probe)**. Any system that recovers the job title gets most of the AUC for free. The question a recruiter cares about, *which of the right-role candidates is better*, is the within-family AUC: **0.54–0.75 for every system**, ~0.70–0.75 for the best. That is the honest headline, and it sits well below 0.85 | **High** | `a2-final-labels.csv` joined with `corpus.keyword_match` |
| T2 | **Title leakage into labels and models.** Annotators and the LLM read CV text *with* `Position` (v1). SBERT weak was trained to predict the keyword from v1 text. Removing `Position` (v2) still leaves the title spelled out in the CV body for **22.7%** of CVs (`indomain-metrics-v2.json` `residual_leak`) | High | D53; `evaluation/text_leak.py` |
| T3 | **Label noise is about the size of the effects being compared.** Human κ 0.35 (0.29 on targeted). Most rung-vs-rung deltas are 0.005–0.03 with CI half-widths of ~0.03–0.05, so they come out "not resolved". No ranking among the top four systems (≈0.85) is supported by the data | High | `a2-final-report.json`; `indomain-metrics.json` `deltas` |
| T4 | **Single-annotator and LLM dependence.** 274 pairs (46%) carry only `yc`'s label, and that annotator is the stricter one (54% Fit vs van's 71%). 107 pairs were decided by an LLM that also read `Position`. Each figure has an `llm_free` sibling (AUCs go *up* there, e.g. X2 0.900), but no `van`-only or "both humans" sensitivity is reported. Stage-2 ordering verdicts use **`yc` alone**, which sits uneasily with AGENTS.md evaluation rule 1 ("Do not score systems against one annotator") | Medium-High | `final_set.py:48-49`; `17-p8-stage2.md` §6 |
| T5 | **Small n.** 59 scoreable JDs (48 within family). Targeted CIs span ±0.07. Per-group gaps (`group-gaps.json`) have ~20 JDs per band | Medium | records |
| T6 | **Test-set reuse / adaptive overfitting.** The same 600 pairs have been scored by at least 5 record sets (indomain v1, v2, stage1, stage2, shortlist_final), with ~30 systems. A post-hoc diagnosis that **read the final labels** (`16-p7-stage1.md` §6a) motivated the shortlist-recall Z rungs. The project's controls (registration before scoring, `exploratory` labels, tuning on dev) are better than most. But the D48 winner, `metadata + TF-IDF`, was itself an **`exploratory`** rung (`matching/ladder.py:57`). Expect some winner's curse in the 0.85s | Medium | `matching/ladder.py:3-6`; plan records |
| T7 | **Retrieval recall is partly an artefact of the pools.** Each pool = 10 judged CVs (8 same-keyword) + 490 random CVs from **all** roles (`retrieval_pilot.py:52-58`). A role-aware ranker floats the judged 8 to the top without assessing fit. The oracle keyword rule's 0.924 Recall@50 mostly measures "find the CVs we sampled by keyword". Unjudged distractors are assumed No Fit (472–516 of the 600 top-10 slots are unjudged, `17-p8-stage2.md` §6) | Medium | code |
| T8 | **The enriched 61% Fit rate is not production.** In a real applicant pool the base rate and the role mix differ. Precision and the Fit call's accuracy will not transfer. README says so; the UI does not | Medium | README; `streamlit_app.py:120-121` |
| T9 | **A1 is unreliable by the project's own measurement.** Its labels do not reproduce (κ(A1, human) = 0.010, `annotation/llm_recheck.py` docstring). The A1-trained cross-encoder (0.488) is uninformative as a model test | Known | D33, D40 |
| T10 | **No end-to-end evaluation of the app as shipped.** No record scores `pipeline.screen` on uploaded/PDF text, the evidence layer's correctness (are "matched"/"supported" right?), or the Fit call on text-only input. `latency.json` excludes the evidence encoder | Medium | `evaluation/latency.py` |

**What is solid:** the engineering of the evaluation. JD-level bootstrap, paired deltas,
strata always reported, no fitting on the final set (D46), document-level train/eval split
(id-level verified), deterministic `--check` diffs, and a real effort to separate
exploratory from held-out results. The weakness is the **construct**, not the arithmetic:
what the labels measure and how the candidates were sampled.

---

## 3. Failure modes

| Area | What happens | Evidence | Risk |
|---|---|---|---|
| **PDF / layout parsing** | `files.to_text` = `pypdf.extract_text` per page, joined (`app/files.py:9-20`). No OCR (scanned PDF → `""`), no column or table handling, no `.docx`. There is no parse-success check: an empty CV is still scored and shown. An empty CV scores TF-IDF 0 (ranks last) but SBERT **0.25** cosine to the JD **(probe)**. It still ranks last on the demo data, though an empty or garbled CV gets no warning. The PDF corpus (C1, livecareer) is never used in any evaluation (D10: "PDFs feed the ingestion demo only") | code; probe | **High** for real use |
| **No structured CV parsing** | There are no sections, employers, dates or education fields. "Parsing" is a regex for `N years` and a skill matcher. `years_stated` takes the **max** "N years" in the CV: "a company with 25 years of history" → **25** **(probe)**. `years_required` takes the **min**: "1-3 years, ideally 5+" → 1 | `evidence/requirements.py:56-65` | Medium (shown to recruiters, used by `at least N years` filter) |
| **Keyword stuffing** | Appending the JD's own vocabulary skills to a CV lifts TF-IDF by +0.046 and SBERT weak by +0.046 on average. The stuffed CV then beats the JD's **best original CV in 30% (TF-IDF) / 24% (SBERT) of cases** (20 JDs × 10 CVs, **probe**). Evidence "coverage" rises to 100% by construction | probe | **High** (trivially gameable) |
| **Vague "supported" evidence** | τ = 0.28 accepts topically close but empty sentences. For one JD's top 8 candidates, 60% (25/42) of unmatched requirement cells were marked `supported`; "DevOps practice is a hobby for me" was evidence for both Agile and Jenkins (walkthrough notebook §4) | `evidence/evidence.py:57-93`; `output/evidence/tau.json` | **High** for an explainability claim |
| **Negation / context** | "I have **no** experience with Kubernetes, Python or AWS" → all three **matched**, with quotes **(probe)**. The evidence layer is a string match; the UI shows ✔ | `evidence/evidence.py:57-93` | **High** for an explainability claim |
| **Synonyms / abbreviations** | Vocabulary = 9,124 aliases, **97% ESCO** (with entries like "design the digital interface of gambling games"). **Probes:** `ML` → *"ML (computer programming)"*, i.e. the ML functional programming language, **not** machine learning; `.NET` and `Visual Studio` → **Visual Basic**; no match at all for `JS`, `ReactJS`, `k8s`, `Sklearn`, `Torch`, `TF`, `LLM`, `GenAI`, `AI`, `artificial intelligence`, `QA`, `manual testing`, `Figma`, `Go` (only `Golang`), `communication`, `teamwork`. TF-IDF has no synonym handling. SBERT partly covers this, but only at the document level | `data/vocab/skills.csv`; `extraction/rules.py` | **High** for evidence and skill-overlap; Medium for ranking |
| **Long vs short CVs** | TF-IDF cosine is length-normalised. SBERT mean-pools 800-char chunks, so a long CV dilutes its relevant chunk. Adding one name line moves short CVs more than long ones: corr(|Δ|, length) = −0.31 **(probe)**. Length bias in the ranking itself is **not measured** | `matching/sbert.py:35-37,70-72` | Medium, unmeasured |
| **Non-English** | Every evaluation document is filtered to English. The app does no language detection. MiniLM and the English stop-word TF-IDF will silently mis-score Ukrainian/Russian or mixed text, which is common on Djinni | `corpus.py:82-84`; `baselines/tfidf.py:27-34` | Medium |
| **Redaction damages content** | The phone regex masks **date ranges and salary ranges**: "Worked 2019-2021…, salary 1500-2000" → "Worked [PHONE]…, salary [PHONE]" **(probe)**. That hurts every displayed quote and resume about experience. It does **not** remove names | `annotation/redact.py:33-34` | Medium (UX), High (privacy claim) |
| **Silent degradation** | No MiniLM in the HF cache → uploads fall back to TF-IDF, no tagger, no "supported" status. Nothing tells the user | `pipeline.py:52-56`; `streamlit_app.py:28-46` | Medium |
| **Embedding cache write cost** | Every `Encoder.encode` call with a cache miss rewrites the **whole** `.npz` (`sbert.py:76-81`). The cache is 152 MB here, and the evidence step calls `encode` twice per CV. Screening 500 uploaded CVs with the encoder on could mean ~1,000 full rewrites *(inferred, not timed)* | code | Medium (performance) |

---

## 4. Fairness and bias

**What exists:** `evaluation/group_gaps.py` reports AUC by the JD's experience band and by
stratum (no target, no `--check`), for the **pre-D53** pipeline. Djinni is pre-anonymised,
so the evaluation data has **no names, gender, age, nationality or school fields to audit
against**. That is the stated reason no demographic audit exists (`group_gaps.py:3-6`).

**What does not exist:** any name-swap, counterfactual or proxy test. There is no masking
of names, age, gender or nationality, although proposal v3 §(privacy) promises it
(`docs/proposal/…-v3.md:227`). `redact.py:8-17` explicitly disclaims removing names. No
filter or rule refers to a protected attribute.

| Attribute | Direct path into the score? | Proxy path? |
|---|---|---|
| **Name** | **Yes, for SBERT** (uploads). **Probe** (all 600 pairs, SBERT weak v1): prepending one name line ("John Smith", "Olena Shevchenko", "Mohammed Al-Rashid", "Li Xiaomei") changes the score by up to 0.13 (p95 ≈ 0.04, against a within-JD SD of 0.09). It **moves the CV's rank among its 10 peers for 173–198 of 600 CVs**. Mean shifts are tiny (−0.001 to +0.004) and show no consistent direction by name origin in this probe. So the measured effect is *arbitrary sensitivity*, not demonstrated group bias, but it is real. Caveat: Djinni CVs have no names, so any name is out of distribution. TF-IDF: **zero** change (names fall outside the 5,000-term vocabulary) | Upload filenames (often names) are shown unredacted as `cv_id` |
| **Gender** | No field. Pronouns and gendered words can enter SBERT the same way names do *(inferred)* | The LLM annotator (haiku) labelled 107 tie-breaks, and its 0–100 scores trained the (rejected) rerankers. Any LLM bias enters the reference through that route |
| **Age** | `years_stated` uses the max "N years" in the CV. `experience_compatible` rewards CV band ≥ JD band. There is no over-qualification penalty, which favours longer careers | Graduation years, "20+ years" statements, seniority words (`extraction/experience.py`) |
| **School** | No education feature | University names can reach SBERT. They are mostly absent from the TF-IDF vocabulary *(inferred)* |
| **Nationality / language** | The English-only filter. The Ukrainian/EE market (D11) | Non-native English phrasing changes embeddings and TF-IDF overlap. Unmeasured |
| **Career gaps** | Not modelled (no dates are parsed) | Gaps lower stated years, so a lower experience band |

**Bottom line:** fairness is **untested** for the product, not shown to be fair. The data
cannot answer the question, and no synthetic counterfactual test exists.

---

## 5. Test coverage

`uv run pytest`: **639 passed, 0 skipped, 54.7 s** (verified 11 Oct, with `data/` present;
`logs/pytest-ownership.log`). There is no CI.

| Well tested (named invariant tests, many shown to fail against a mutant) | Thin or untested |
|---|---|
| Evaluation arithmetic: per-JD AUC, bootstrap over JDs, "total needs strata", final-set rule v1 and its provenance (`test_final_set.py`, `test_indomain.py`, `test_metrics.py`) | **`pipeline.screen`**: 2 tests with a stub ranker and a stub matcher (`test_evidence.py:106-120`). The default ranker path is never exercised in a test |
| TF-IDF shared vocabulary, BM25 scoring and tokeniser parity (past real bugs) | **Streamlit app**: one test, "renders its empty state" (`test_app.py:15`) |
| Skill matcher tokenisation (`C++`, `java`/`javascript`, line breaks) | **`files.to_text`**: HTML and TXT only. **No PDF test** (`test_app.py:9`) |
| Evidence quotes are verbatim; fabricated quotes are refused | Evidence **correctness** (negation, wrong canonical, e.g. `.NET` → Visual Basic). The vocabulary content has no regression tests for common IT aliases |
| No training module reads the final set (AST tests); dispatch to the LLM is closed | `years_stated` / `years_required` edge cases; `redact` false positives (date ranges) |
| Notebooks contain no model code (`test_notebook_is_thin.py`) | Robustness (stuffing, length, empty text), fairness (name swap), non-English input: **none** |
| The `checks` registry covers every `--check` | `group_gaps`, `latency`: no `--check`, and both measure the superseded pipeline |
| Synthetic-only tests run on a fresh clone | Neural tests `importorskip`, so the base `uv sync` suite never runs them |

The suite protects **the research numbers** well and **the product** barely at all.

---

## 6. Top 10 things to understand or fix first (ranked by risk)

| # | What | Why it matters | First action |
|---|---|---|---|
| 1 | **The app ranks and labels "Fit" by a metadata field (`Primary Keyword`) that real input will not have, and that D53 forbids** | Demo numbers do not describe uploads. The "Fit call" shown to recruiters is string equality of a Djinni dropdown | Decide: finish P9 (wire in X2/X4 or `SbertWeak("v2")`), or label the demo mode as an oracle demo |
| 2 | **The headline AUC ≈ 0.85 is mostly "same role" detection** (96.5% of Fits share the keyword). Within-role quality is 0.70–0.75 | This is the number you will be asked to defend. Overclaiming here is the biggest reputational risk | Lead with within-family AUC and say how the candidates were sampled |
| 3 | **Evidence ✔ marks are string matches**: negation passes, `ML`→the ML programming language, `.NET`→Visual Basic, `JS`/`k8s`/`LLM` unknown | The UI's explainability is what recruiters will trust most, and it is the least validated part | Audit the top ~200 IT aliases; add a negation guard; build a small hand-checked evidence test set |
| 4 | **Trivially gameable**: pasting the JD's skills beats the best real CV 24–30% of the time | Any deployed screener gets gamed | Down-weight skill lists, cap repetition, flag CVs with high skill coverage but low semantic similarity |
| 5 | **Label reliability**: κ 0.35, 46% single-annotator, 18% LLM-decided, ordering verdicts on `yc` alone | Every comparison inherits this noise, and most deltas are "not resolved" | Report a "both humans agree" subset beside every headline; stop ranking systems inside overlapping CIs |
| 6 | **No PDF / parse-quality guard**, no OCR, no docx, no language check | The first real upload batch will contain scanned or two-column PDFs, and they will rank silently | Add text-length/garbage checks and surface "could not read" to the user; test on the C1 PDFs |
| 7 | **Privacy and fairness claims exceed the code**: names not masked, filenames shown, no name-swap test; SBERT rank moves for ~30% of CVs when a name line is added | Proposal promises masking; a reviewer will try a name swap | Strip a header/name block before scoring; add a counterfactual test to the suite |
| 8 | **Undocumented model prerequisite and silent fallbacks** (MiniLM must already be in the HF cache) | A fresh machine runs a different system without saying so | Document the download; make the UI show which ranker / tagger / encoder actually ran |
| 9 | **Test-set reuse**: the same 600 pairs have been scored by ~30 systems over 5 rounds, including a post-hoc diagnosis on final labels | The next "improvement" measured on this set is increasingly optimistic | Treat further final-set scoring as spending a budget; keep the dev set (A2 train) for iteration |
| 10 | **Docs drift**: README results pre-date D53; plan says the cold stage-2 check is owed but a local log shows PASS; `pipeline.py` docstring describes a fitted threshold that does not exist | You will quote the wrong number to someone | Refresh README "Results" and the P8 record. Commit check evidence, or note where it lives |

Open questions for the owner (the full list goes in deliverable 5):
(a) Is the Djinni demo meant as the product or as an oracle illustration?
(b) Is the target use ranking within one role family, or across roles? The answer changes which AUC is the headline.
(c) Who decided the `yc`-only ordering cell, and was the alternative (both humans) considered?
