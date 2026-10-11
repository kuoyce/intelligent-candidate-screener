# 12 — P3 record: E2, job title → role family

*6–7 Oct 2026. Implements 03-work-plan P3 under the user's Q-B1 decision (6 Oct): **"T-A,
split generic"**, and **a text → family classifier, with title spans as features and as
UI evidence**.*

Reproduce:

```bash
uv run python -m candidate_screener.extraction.role_family --train      # fit both candidates, write the record (~52 min CPU)
uv run python -m candidate_screener.extraction.role_family --evaluate   # rebuild the record from the cached model
uv run python -m candidate_screener.extraction.role_family --check      # refit and diff the committed record
uv run python -m candidate_screener.extraction.role_family --derive     # print the label space; says whether it matches the committed constants
uv run pytest tests/test_role_family.py
```

- Code: `src/candidate_screener/extraction/role_family.py`
- Record: `output/extraction/role-family-metrics.json` (deterministic, no timestamps; fingerprint `44b7fff54614`)
- Model cache: `data/processed/extraction/role-family/role-model-<fingerprint>.pkl` + `LATEST`.
  The fingerprint covers the config, the label space and the partition manifest digest. Nothing is committed (D23)
- Tests: `tests/test_role_family.py` (synthetic data only, 25 tests)

Every figure below is read from the record.

## 1. The label: `role_family(keyword, position)` (task 3.1)

It reads metadata, so it is **only** for training labels and the oracle. It is never used at inference (D53).

1. **Keyword classes.** Each `Primary Keyword` with at least *m* train JDs **and** *m* train
   CVs (fit part) is a class: **38** keywords.
2. **Fold.** The other keywords are the four JD-only singletons (`Block-chain`, `Product Owner`,
   `React`, `SAP`; P2 §3.2). They are folded into `Other` before the split. **This is how
   the 45 JD / 41 CV keyword sets become one shared label space.** On the CV side nothing is
   folded.
3. **Split.** `Data Science`, `Lead` and `Other` are split by the document's normalised
   `Position` (`title_profile.normalise_title`, T-B1, P2 §1). Families are the titles with at
   least *m* train JDs **and** *m* train CVs *under that keyword*. A document whose title is
   in no family gets the residual `<keyword>/other`. That includes an empty normalised
   title, such as `Team Lead`.
4. **Merge (my addition, flagged in §7).** A split family whose title is a specific
   keyword's **own** title is merged into that keyword. "Own title" needs two things: the
   title is that keyword's modal normalised title (JD and CV pooled), and the keyword holds
   ≥ 0.5 of the train documents that carry the title under a non-generic keyword. Merged:
   `Data Science/data analyst` and `Other/data analyst` → `Data Analyst`;
   `Data Science/data engineer` → `Data Engineer`; `Other/frontend developer` → `JavaScript`;
   `Other/technical writer` → `Technical Writing`. Without the "own title" condition, a
   plain modal-keyword rule moved `data scientist` and `ml engineer` into `Python` and
   `computer vision engineer` into `C++`. That would undo the split, so the condition stays.
   It is regression-tested (`test_merge_only_into_a_keywords_own_title`).

**m = 20.** This is P2's headline m (11 §1, §3.2). Within the generic keywords (train
region, JD/CV coverage of the keyword's documents):

| m | Data Science families | Lead | Other | DS JD/CV covered | Lead | Other |
|---|---|---|---|---|---|---|
| 10 | 7 | 12 | 33 | .54/.77 | .29/.39 | .20/.30 |
| **20** | **5** | **5** | **15** | .53/.76 | .18/.28 | .12/.16 |
| 30 | 3 | 5 | 10 | .37/.73 | .18/.28 | .09/.13 |
| 50 | 3 | 2 | 2 | .37/.73 | .10/.18 | .02/.03 |

m = 20 keeps the four Data-Science roles P2 named (scientist, data engineer, ML engineer,
analyst), all five Lead families (architects, CTO, engineering manager), and the Other
families with ≥ 20 documents per side. At m = 10, several classes have 10–15 documents
per side, which is too few to train on or to validate. Word-order merging (a token-set key)
was tried and changed nothing at m = 20, so it is not used.

**The label space: 60 classes**, shared by both sides. They are 38 keyword classes, 3
`Data Science/*` families, 5 `Lead/*` families, 11 `Other/*` families and 3 residuals. The
space is committed as constants (`KEYWORD_CLASSES`, `SPLIT_FAMILIES`). It was derived from
the **fit part** of the train region, and the record states `committed_constants_match:
true`.

**Class profile (whole train region, record `class_profile_train_region`).**

- JDs: 106,423 documents, all labelled, 60 classes, sizes 24 – 13,423 (median 680.5).
- CVs: 157,688 documents, 157,658 labelled (30 have a null keyword and are excluded), 60
  classes, sizes 27 – 25,474 (median 921.5).
- Residuals: `Other/other` 4,987 JD / 3,364 CV; `Lead/other` 1,137 / 1,299;
  `Data Science/other` 814 / 478.
- Smallest classes: `Data Science/computer vision engineer` 24 / 27,
  `Other/operations manager` 28 / 37, `Other/software developer` 29 / 28.

## 2. Title spans (task 3.2)

`title_spans(text) -> list[TitleSpan(start, end, text, title, family)]`

- **Matching.** It uses `title_profile`'s rule unchanged: `canonical_tokens`, n-grams up to
  6, token-bounded. The search is leftmost-longest and spans do not overlap.
- **Offsets.** They index the **original** text. The `.net` → ` dotnet` rewrite and
  multi-character lower-casing are mapped back, and `text[start:end]` is the surface string
  (regression test).
- **The alias table.** It is built from normalised train `Position` strings (**fit part
  only**), each mapped to its modal role family. An entry needs ≥ 20 documents and modal
  purity ≥ 0.5. A one-token title needs purity ≥ 0.8, so `developer` is out and `cto` is in.
  Result: **518 entries** (460 multi-token).
- **Coverage on the validation part's normalised `Position`:** 61.1% of JDs and 79.1% of
  CVs. Where covered, the alias's family equals the label 89.6% (JD) and 93.9% (CV) of the
  time (plan 3.1 acceptance).
- **Spans in text (selection draw, n = 2,500 per side, `text_v2`).** At least one span:
  97.6% of JDs, 93.0% of CVs. The label's family is named somewhere: 78.4% / 73.1%. The
  **first** span is the label: 47.7% / 51.6%. This matches P2 §4 (first title right about
  half the time), so spans are features, not the label source.
- **ESCO and `data_jobs`** are not in the span table. They recognise titles but map to no
  class of this label space (P2 §2), so they would add unclassed spans only.
- **Features.** For each class *k*, two indicators: family *k* is named anywhere, and the
  first span is family *k*. That gives 120 binary columns.
- **Not done:** precision and recall on a hand-checked sample (plan 3.2; Q-B2, which is
  still open: no hand labels exist).

## 3. Classifier and selection (task 3.3)

**Inputs.**

- JD = `Long Description`.
- CV = `CV`, `Highlights`, `Moreinfo`, `Looking For` (`text_v2`; `Position` removed).
- No label field is an input.
- `cv_text_v2` is built in this module, mirroring `annotation/sample.py` minus `Position`,
  with the same join and newline collapsing. It equals P1's
  `matching.corpus.CV_FIELDS_BY_VARIANT["v2"]`, so it can be unified later.

**Split.** The train region is split into a fit part and a validation part.

- 10% of the validation part, seeded, is disjoint **by document**.
- Documents whose texts are identical after whitespace and case folding go to the same
  side, so a re-posted JD cannot straddle the split.
- Sizes: JD fit 95,781 / validation 10,642; CV fit 141,894 / validation 15,764.
- From the validation part, two disjoint seeded draws per side: **selection** (n = 2,500)
  and **calibration** (n = 1,000).

**Candidates.** Both were fixed in advance, with no hyperparameter search.

- **(a) tfidf:** HashingVectorizer, word 1–2-grams, 2^18 features → sublinear TF-IDF (IDF
  from the fit part) + span indicators → logistic regression (saga, C = 10, 30 epochs, seed
  0). No char n-grams: memory on this shared machine.
- **(b) minilm:** frozen `all-MiniLM-L6-v2`, chunk-mean exactly as `sbert.Encoder` (own
  cache key `minilm-role-family`) + span indicators → logistic regression (lbfgs, C = 10).
  It is trained on a **seeded subsample of 6,000 per side**, topped up to ≥ 15 per class
  (6,249 JD / 6,241 CV), because CPU embedding was the limit (~20 chunks/s with other
  agents running).
- Both use one model per side over the full 60-class space.

**Validation (selection draw, n = 2,500 per side, 95% CI by bootstrap over documents, 1,000 resamples):**

| Candidate | Side | n train | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|---|---|
| **(a) tfidf** | JD | 95,781 | **0.784** [0.768, 0.800] | **0.540** [0.514, 0.602] | **0.924** [0.913, 0.934] |
| **(a) tfidf** | CV | 141,894 | **0.785** [0.769, 0.802] | **0.520** [0.507, 0.597] | **0.931** [0.921, 0.941] |
| (b) minilm | JD | 6,249 | 0.694 [0.676, 0.712] | 0.467 [0.441, 0.521] | 0.876 [0.863, 0.889] |
| (b) minilm | CV | 6,241 | 0.719 [0.701, 0.738] | 0.477 [0.458, 0.546] | 0.878 [0.865, 0.891] |

**Selection rule (fixed beforehand): the higher mean of JD and CV macro-F1.** tfidf 0.530
vs minilm 0.472, so **(a) tfidf is selected**.

Caveat: (b) saw ~6% of (a)'s training documents. The comparison is between the two
*deployable* configurations, not the two feature sets at equal n.

**Calibration.** Temperature scaling per side, fitted by NLL on the calibration draw:

| Side | T | ECE on the selection draw, before | after |
|---|---|---|---|
| JD | 0.914 | 0.032 | 0.033 |
| CV | 0.933 | 0.038 | 0.031 |

Logistic regression was already close to calibrated. Argmax, and therefore accuracy, is
unchanged.

## 4. Final-set measurement (task 3.4): measurement only, D46

- **Documents:** the 60 final-set JDs and the CVs of the 600 pairs (600 unique CVs), on
  `text_v2`.
- **Oracle:** `role_family(Primary Keyword, Position)` of each document. Every document has
  a label. 26 oracle classes among the JDs, 37 among the CVs.
- **CIs:** bootstrapped over **JDs** (2,000 resamples for pairs, 1,000 for documents). CV
  figures are over the pair rows, clustered by JD.
- Nothing was tuned on these figures. The module never reads `a2-final-labels.csv` or
  `judgements.csv` (test).

**Document level: predicted family (argmax) vs oracle.**

| Scope | Side | n | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|---|---|
| generic | JD | 40 | 0.775 [0.649, 0.900] | 0.762 [0.629, 0.875] | 0.950 [0.875, 1.000] |
| targeted | JD | 20 | 0.500 [0.300, 0.750] | 0.609 [0.292, 0.758] | 0.650 [0.450, 0.850] |
| **total** | JD | 60 | **0.683** [0.567, 0.800] | 0.705 [0.567, 0.800] | **0.850** [0.750, 0.933] |
| generic | CV | 400 (40 JDs) | 0.740 [0.673, 0.803] | 0.617 [0.567, 0.700] | 0.893 [0.850, 0.933] |
| targeted | CV | 200 (20 JDs) | 0.620 [0.530, 0.715] | 0.536 [0.469, 0.690] | 0.845 [0.790, 0.895] |
| **total** | CV | 600 (60 JDs) | **0.700** [0.647, 0.753] | 0.565 [0.513, 0.637] | **0.877** [0.842, 0.908] |

**Pair level: same family? (oracle same = both oracle labels equal).**

| Scope | Pairs (JDs) | Oracle same | Predicted same (argmax) | Agreement | κ | Soft score AUC vs oracle same | Mean soft, oracle same / differ |
|---|---|---|---|---|---|---|---|
| generic | 400 (40) | 0.798 | 0.485 | 0.683 [0.598, 0.763] | 0.376 [0.278, 0.485] | **0.960** [0.930, 0.982] | 0.482 / 0.016 |
| targeted | 200 (20) | 0.630 | 0.295 | 0.625 [0.490, 0.750] | 0.322 [0.140, 0.517] | **0.761** [0.630, 0.882] | 0.304 / 0.054 |
| **total** | 600 (60) | 0.742 | 0.422 | **0.663** [0.592, 0.732] | 0.374 [0.285, 0.475] | **0.901** [0.852, 0.943] | 0.432 / 0.034 |

**Reading.**

- The **soft score** `sum_k p_jd(k) p_cv(k)` separates oracle-same from oracle-different
  pairs well: AUC 0.90 total.
- The **hard** argmax match under-calls sameness: predicted 42% vs oracle 74%. It needs
  both sides to pick the same one of 60 classes. This is why 02 §3.4 specifies the soft
  form for `role_match`.
- `targeted` is clearly harder (JD accuracy 0.50, AUC 0.76). Its JDs were chosen to be
  hard, and its CVs are less often in the JD's family.
- The P7 X rungs are not scored here.

## 5. Public API (for P7)

```python
from candidate_screener.extraction import role_family as rf
model = rf.load_role_model()                  # cached, calibrated (run --train first)
rf.predict_proba(texts, side="jd"|"cv")       # (n, 60), columns = model.labels; text only
rf.role_match(jd_texts, cv_texts)             # aligned pairs -> soft agreement in [0, 1]
rf.title_spans(text)                          # [TitleSpan(start, end, text, title, family)]
rf.role_family(keyword, position)             # LABEL ONLY: training / oracle, never inference
rf.label_space()                              # the 60 labels
```

`predict_proba` scores duplicate texts once. A class the side model never saw gets
probability 0, and with two classes the single binary logit is expanded (regression test).

## 6. Tests (each shown to fail against the code it guards)

Each test below was checked by mutating the guarded code, running the test (it failed),
and restoring the code:

| Test | Mutation that makes it fail |
|---|---|
| `test_merge_only_into_a_keywords_own_title` | drop the "own title" condition |
| `test_span_offsets_index_the_original_text` | use normalised-text offsets |
| `test_role_match_is_soft_agreement_in_unit_interval` | remove the binary-logit expansion. Found a real bug: sklearn returns a 1-D decision for two classes |
| `test_jd_only_singleton_keyword_folds_into_other` | remove the fold |
| `test_identical_texts_share_a_side` | group by raw text |
| `test_cv_text_v2_never_contains_position` | add `Position` to the fields |
| `test_save_model_refuses_classes_defined_in_main` | drop the save guard. Found a real bug: the first `--train` (run as `python -m`) pickled `__main__.SideModel`, which no importer could load. `main` now re-imports the module, and `save_model` refuses `__main__` classes. The model was retrained after the fix |

Others cover:

- label-space closure, the support threshold on both sides, alias-table purity and support;
- macro-F1 = sklearn's, temperature keeps argmax and lowers NLL;
- the inference API takes text only;
- the module never reads the final labels or judgements (AST check).

## 7. Deviations, assumptions, open items

- **Assumption to confirm: the merge step (§1.4).** Not in the user's wording. It removes
  four near-duplicate classes (e.g. `Data Science/data engineer` vs `Data Engineer`), which
  would otherwise make `role_match` and the oracle call two data engineers different roles.
  Without it the space would have 64 classes.
- **Plan 3.1 (alias table from ESCO + `data_jobs` + A2).** Only the A2-train part is used
  for spans (§2). Coverage on A2-train `Position` is reported for the A2 table. P2 already
  reports the external tables' coverage.
- **Plan 3.2 (hand-checked span P/R)** is not done: Q-B2 is open.
- **(b) minilm trained on a subsample** (6k per side), documented above. **No char n-grams**
  in (a).
- **Validation is on A2-train documents, not hand labels.** The labels are the metadata
  themselves.
- **Normaliser reconciliation (for P1/P9).** This module uses
  `title_profile.normalise_title` (cuts at separators, drops seniority, phrase and synonym
  merges). P1's `evaluation.text_leak.normalise_title` strips seniority words only. They
  differ, and should be unified to one rule (not edited here: P1's file).
- **CV `text_v2` builder duplicated** (`role_family.cv_text_v2` vs `matching.corpus.build_cv_text("v2")`).
  Same fields and join; unify later.
- **Cost and reproducibility.** `--train` and `--check` each refit: ~52 min on CPU on a
  cold MiniLM cache, ~13 min warm. `--evaluate` rebuilds the record from the cached model.
  Three refits (6 Oct 22:46, 7 Oct 07:15, and `--check` at 07:28) gave byte-identical
  records: `--check` prints **OK**.
