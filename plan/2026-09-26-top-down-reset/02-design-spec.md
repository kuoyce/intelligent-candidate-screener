# 02 — Design specification

## 1. Direction in one paragraph

First, finish what the annotation campaign was for. Reconcile `yc`, `van` and the LLM
(as a **secondary annotator**) into **one final Fit / No Fit annotation set** for the 600
A2 pairs, using the user-approved rule:
- humans agree → theirs;
- humans disagree → the LLM breaks the tie;
- one human → that human;
- no human → the LLM.

Freeze the result as a committed manifest. Three classes are dropped everywhere (D47).

Then build the ladder the proposal promised: rules → lexical → semantic → cross-encoder.
Score every rung against that one set, with a decision rule registered before the held-out
rungs run. After that, build the extraction component, the evidence layer and the
conversational UI on top of the rung the rule selects.

A1 becomes a training resource plus a secondary benchmark. Annotation collection and
judge calibration are closed.

## 2. Decisions this plan asks to record

| # | Decision | Closes / supersedes |
|---|---|---|
| **D40** | **A2 is the primary benchmark.** A1 is a training resource and a *secondary* benchmark, always quoted with the A13 caveat. D26's attainable ceilings are **retired**: superseded in place in the catalog, the A1 card and the README. They are not restated | Q48; supersedes D26's headline use |
| **D41** | **One final annotation set for A2** (`a2-final-labels.csv`), built by the user-approved rule of §3.2 (26 Sep 2026): humans agree → theirs; humans disagree → the LLM breaks the tie; one human → that human; no human → the LLM. **Every system is scored against it, and only against it**, reported for `generic`, `targeted` and `total` (Q56, amending D29). It is frozen once committed: a changed rule is a new `rule_version` and a new file, never an edit | Q46, Q47, Q49, Q57. Supersedes D37's "consensus licenses measurement, not substitution" for A2, the reference design's §2 closure of option G, and `plan/2026-09-10-a2-reference-design/`. **D36 stays unmade in its own sense**: the LLM never outvotes humans who agree, and it labels alone only where no human looked (0 of today's 600 pairs) |
| **D42** | **Annotation collection and judge calibration are closed.** No new labels, no LLM dispatch (user constraint, 26 Sep 2026). The existing `annotation/` modules are frozen. `llm_recheck --dispatch/--judge` and `prompt_calibration` refuse, in code, with a pointer to D42 | Q34 stays deferred indefinitely |
| **D43** | **The LLM is a secondary annotator, not a system.** Only one LLM series enters the final set: the committed instrument `5d986c82` under `claude-haiku-4-5` (runs 9/10), which covers all 600 pairs. No system may be trained on LLM labels. If the LLM is ever scored as a system, it is scored on the `llm_free` subset (§3.4) | Revises D37 §2 |
| **D44** | **Success measures, partly decided** (Q53, 26 Sep 2026). Three-class targets are dropped. Inter-annotator agreement is **reported, with no target**. The remaining targets are open and listed in [04-success-measures.md](04-success-measures.md) §3. They must be fixed **before** P1.5 registration, so targets are not set after results are seen | Proposal §7 |
| **D45** | **`AGENTS.md` holds conventions and live rules only** (≤ 1,800 words). Decision history moves to `docs/decisions.md` (one line per decision, linked to its plan record) | — |
| **D46** | **The final set is never used to fit or tune anything**, hyperparameters, combination weights and thresholds included (proposal §5.2). Tuning uses A1's doubly-disjoint CV folds (`fit_split.cv_folds`) or a validation split of the weak-supervision set. The final set's construction reads annotator labels only, never a system's output | Refuses A2 cross-fitting in the main line (Q54) |
| **D47** | **Fit / No Fit is the only label scheme**, everywhere: the final set, every metric, every trained head, the evidence layer and the success measures (user, 26 Sep 2026). `Fit` = `Good Fit` ∪ `Potential Fit`. The annotators' raw three-class labels stay in the manifests for audit only, and no figure is computed on three classes | Retires the `Good`/`Potential` boundary work (Q38, move 3) from the evaluation. The guide itself is unchanged, since the labels were collected under it |

## 3. The final annotation set

### 3.1 Inputs

| Annotator | Role | Coverage |
|---|---|---|
| `yc` | human, primary | 600 / 600 |
| `van` | human, primary | 326 / 600 (126 generic, 200 targeted) |
| `llm` | secondary: `5d986c82`, `haiku`, runs 9/10, `parsed_ok` rows only | 600 / 600 |

Every raw label is collapsed to `Fit` (`Good Fit` or `Potential Fit`) or `No Fit` before
the rule runs (D47).

**Why exactly one LLM series.** Several LLM runs would be correlated votes, one judge
counted several times. The committed instrument is the one D39 kept. `sonnet` (runs 7/8)
was shown to track `yc` specifically (kappa 0.45 vs `yc`, 0.14 vs `van` on targeted), so
using it would double-count `yc` whenever it breaks a tie.

### 3.2 Rule (`rule_version = 1`, approved by the user 26 Sep 2026)

| # | Case | Final label | Tier | Pairs today (generic / targeted / total) |
|---|---|---|---|---|
| 1 | Both humans, and they agree | the humans' label (the LLM is ignored) | `humans_agree` | 93 / 126 / **219** |
| 2 | Both humans, and they disagree | **the LLM's label** (tie-breaker) | `llm_breaks_tie` | 33 / 74 / **107** |
| 3 | One human | **that human's label** (the LLM is ignored) | `single_human` | 274 / 0 / **274** |
| 4 | No human | **the LLM's label** | `llm_only` | 0 / 0 / **0** |

- **Rule 4 is implemented and tested but applies to no pair today.** Every A2 pair has at
  least `yc`. It exists so that any future pair without a human label still gets a label,
  under a named tier that a report can count and exclude.
- **Rules 1–3 give exactly the same Fit / No Fit labels** as the three-class
  median-of-three rule in [05](05-annotator-vs-system.md), collapsed to two classes. This
  was checked on all 600 pairs. The median of three is ≥ `Potential` exactly when two of
  three are ≥ `Potential`. So every system figure shown in 01 §5.4 and 05 carries over
  unchanged.
- **The fit rate is 61.2%.** `yc` gives 53.8%, `van` 70.6% and the LLM 61.5%.
- The fitted alternative (Dawid–Skene) was measured and rejected: it makes the LLM the
  dominant rater (01 §5.3).

### 3.3 Artefact

- **`docs/data/manifests/a2-final-labels.csv`** (committed; ids and labels only, no
  text), with columns:
  - `pair_id, stratum, batch, jd_id, cv_id`
  - `yc_raw, van_raw, llm_raw`: the three-class labels as recorded, for audit
  - `yc, van, llm`: collapsed to Fit / No Fit
  - `final_label` (`Fit` / `No Fit`), `tier`, `n_humans`
  - `rule_version, llm_agent_sha256, llm_model, llm_run`
- **`output/annotation/a2-final-report.json`**, for `generic`, `targeted` and `total`:
  - tier counts;
  - fit rate of the final set beside each annotator's;
  - **inter-annotator agreement** (§3.6);
  - agreement of the final set with each annotator. This is descriptive only: each
    contributed, so the figure is inflated by construction and the report says so.
- **Builder:** `evaluation/final_set.py`, registered as `data.build --task a2-final`, with
  a `--check`.

### 3.4 Invariants (tests and `verify --derived`)

1. Rule 1: where the humans agree, `final_label` equals their label, whatever the LLM
   said.
2. Rule 2: where the humans disagree, `final_label` equals the LLM's label.
3. Rule 3: on single-human pairs, `final_label` equals the human's label.
4. Rule 4: on pairs with no human, `final_label` equals the LLM's label and
   `tier = llm_only`.
5. The LLM column comes from exactly one `(agent_sha256, model)`, and every value matches
   its source row in `llm-a2-*.csv`. Every human value matches `judgements.csv`.
6. 600 rows, no missing `final_label`. The rebuild is byte-identical.

**`llm_free` subset.** The pairs whose label needed no LLM input (tiers 1 and 3: 493
pairs). It is never a headline. It exists for two jobs only:
- the sensitivity check in §5.4;
- scoring the LLM, if it is ever scored as a system (D43).

### 3.5 Stated limitations (they travel with every figure)

- On `generic`, 274 of 400 pairs have one human, so the final set there is **mostly
  `yc`'s labels**.
- The LLM decides **107 of the 326 double-labelled pairs** (33 generic, 74 targeted):
  the ones where the humans split.
- The LLM's semantic reading could favour semantic systems in exactly those tie-breaks.
  §5.4's check exists to catch that.
- There is no ground truth. A system's figure is its **agreement with a documented panel
  rule**, not correctness.

### 3.6 Inter-annotator agreement: reported, no target (D44)

Fit / No Fit, on the pairs both raters labelled:

| Pair | generic | targeted | total |
|---|---|---|---|
| `yc` vs `van`: agreement / kappa | 0.738 / 0.419 (n=126) | 0.630 / 0.292 (n=200) | 0.672 / 0.353 (n=326) |
| `yc` vs LLM | 0.800 / 0.581 (n=400) | 0.670 / 0.362 (n=200) | 0.757 / 0.505 (n=600) |
| `van` vs LLM | 0.746 / 0.431 (n=126) | 0.690 / 0.339 (n=200) | 0.712 / 0.371 (n=326) |

These go in the report as context for every system figure. None is a target.

## 4. The ladder

Every rung implements one interface (`matching.base.System`):

```python
class System(Protocol):
    name: str
    def fit(self, train: pd.DataFrame | None) -> None: ...          # no-op for unsupervised rungs
    def score(self, pairs: pd.DataFrame) -> pd.Series: ...          # index = pair_id, higher = better fit
    def card(self) -> dict: ...   # params, training data, fit seconds, ms/pair (CPU), can_explain
```

`pairs` carries `jd_text, cv_text` plus the metadata both corpora share where it exists
(`jd_keyword, cv_keyword, jd_exp_band, cv_exp_band`). Metadata features are **reported as
their own rung**, so text models are never credited with what a category field does for
free.

| Rung | System | Trained on | Proposal stage |
|---|---|---|---|
| 0 | random | — | floor |
| 1a | **Metadata rules**: keyword match + experience-band compatibility, fixed formula | — | Stage 1 |
| 1b | **Skill overlap**: JD requirement coverage by CV, over the `data/vocab/` phrase matcher | — | Stage 1 (as specified) |
| 2 | TF-IDF and BM25 (existing modules), IDF from the A2 **train** region (D19). Hybrid 1+2 as a fixed rank-normalised sum | — | Stage 2 |
| 3a | SBERT `all-MiniLM-L6-v2`, zero-shot, chunked mean pooling | — | Stage 3 |
| 3b | SBERT fine-tuned (MultipleNegativesRankingLoss) on **weak pairs** (§4.1) | A2 train region | Stage 3 |
| 4a | Cross-encoder from MiniLM, **Fit / No Fit head**, on **A1 train** (3,990 pairs, collapsed) | A1 | Stage 4 (as proposed, binary per D47) |
| 4b | Same architecture on **weak pairs** | A2 train region | Stage 4 |
| 4c | Retrieve-then-rank: best of 2/3 → top-K → best of 4 | — | Stage 3 + 4 |
| 5 | Hybrid: logistic combination of rung scores | A1 train or weak validation (D46) | — |

The LLM is **not** a rung (D43). The proposal marks the LLM comparator as optional. If it
is added later, it is scored on the `llm_free` subset.

Every rung is reported with four things:
- **quality** (§5);
- **cost**: fit time, CPU ms per pair, and ms per 500-candidate role (median of 5);
- **complexity**: parameters, training data needed, lines of code;
- **explainability**: whether it can emit evidence spans, and whether it is deterministic.

4a against 4b is a finding the report needs in its own right. **Does supervision from a
public benchmark whose labels do not reproduce transfer to in-domain judgement better or
worse than free weak supervision from the target domain?**

### 4.1 Weak supervision (A2 train region only)

Built by `data.build --task weak-pairs --seed 0`. The committed manifest holds ids only.

- **Positive:** CV and JD share `Primary Keyword` and the CV's experience band satisfies
  the JD's `Exp Years` band.
- **Hard negative:** same keyword with an incompatible experience band, or same keyword
  with the lowest TF-IDF tercile of that JD's candidates.
- **Easy negative:** different keyword.
- **Size:** ~20k pairs over ~2k JDs. A 10% validation split is disjoint by JD **and** CV,
  and it is the only thing 3b, 4b and 5 are tuned on.

Known risk: the weak label is largely "predict `Primary Keyword`", which rung 1a gets for
free. A model trained on it earns its place only if it beats **keyword + rung 3a**. The
within-family metric (§5) is where that shows.

## 5. Evaluation protocol

### 5.1 Primary benchmark: A2, within-JD re-ranking, against the final set

- **Unit:** the JD, with its 10 candidates. There are 40 `generic` and 20 `targeted` JDs.
  **Every figure is reported three ways: `generic`, `targeted` and `total`** (Q56). The
  total is all 60 JDs with each JD weighted equally, so generic counts 2:1. It is labelled
  as such, and the strata always sit beside it. This amends D29, which forbade the pooled
  figure. `stratum` stays stamped on every pair and label, so the split can always be
  re-derived.
- **Labels are Fit / No Fit only (D47).** No metric uses three classes.
- **Primary metric (rankers):** **per-JD AUC, `Fit` vs `No Fit`**, averaged over JDs.
  It is rank-based, so no system needs a calibrated threshold, and it answers "is a fit
  candidate ranked above a non-fit one for the same job?".
- **Secondary (rankers): within-family AUC**, over only the 8 keyword-matched candidates
  per JD. This is where the metadata cannot help, and where richer models have room to
  earn their place.
- **Classifiers:** for systems that emit Fit / No Fit against `final_label`:
  - accuracy (beside the always-`Fit` floor of 0.612);
  - binary macro-F1;
  - F1 of `Fit`;
  - kappa.
- **Considered and not used:** per-JD average precision. With 61% of candidates labelled
  `Fit`, a random ranker already scores 0.71, so it separates little.
- **CIs:** a percentile bootstrap over JDs (10,000 resamples, seed 0). Rung-to-rung
  deltas are **paired**.
- **The code enforces this:** a figure cannot be built without its `scope` (`generic`,
  `targeted` or `total`), `n_jd`, `ci` and the final set's `rule_version`. A report that
  emits a `total` must emit both strata beside it. This mirrors `metrics.Figure`.

### 5.2 Pilot-scale retrieval

For each labelled A2 JD, a 500-candidate pool: its 10 labelled CVs plus 490 CVs drawn at
random (seed 0) from `indomain-holdout.csv`.
- **Metric:** Recall@50 and Recall@100 of the JD's final-set fits. This asks whether cheap
  retrieval keeps them in the shortlist the re-ranker sees.
- **Latency** per 500-candidate role is measured here.
- **Stated limitation:** the 490 are unjudged and assumed not to be fits. This is the same
  exposure Q18 recorded for the A1 pools.

### 5.3 Secondary benchmark: A1 (D40)

- Every rung is scored on A1 test, Fit / No Fit throughout:
  - binary macro-F1 against the majority floor;
  - pools Recall@10 over N100, with `n` and CI, using `metrics.score`. It uses the `graded`
    definition, which is exactly `Fit` = `Good` ∪ `Potential` (n=64 queries). The `strict`
    (`Good`-only) definition is a three-class artefact and is dropped.
- The committed Phase 4 record (`baseline-metrics.json`, three-class) stays as it is,
  since `run --check` guards it. The binary figures are new records beside it.
- No "% of attainable" is quoted, and every table carries the A13 line.
- Rung 4a is trained on A1 train and is therefore in-distribution there. The table says
  so.

### 5.4 Decision rule, registered before any held-out rung is scored

For rung B against the cheaper rung A it would replace, on the primary metric. The
**total** carries the decision (60 JDs, the most power). The strata are a consistency
check:

| Verdict | Condition |
|---|---|
| **Justified** | the paired Δ(B − A) 95% CI on the **total** excludes 0 (above), **and** the Δ point estimate is > 0 in both strata |
| **Justified, generic only / targeted only** | the total CI excludes 0 but one stratum's point estimate is ≤ 0. Reported as such, and it does not count as justified for selection |
| **Justified, LLM-dependent** | justified, but the total Δ changes sign on the `llm_free` subset (the 493 pairs no tie-break touched). Reported as such, and it does not count as justified for selection |
| **Harmful** | the total CI lies entirely below 0 |
| **Not resolved** | anything else. The report quotes the minimal detectable effect (the total CI half-width) |

**Selection:** the pipeline adopts the **cheapest** rung that is not beaten by a
justified, more expensive rung. It must also meet the quality floor Q53 chooses (if none
is chosen, no floor). This is proposal §5.3.2's "simplest approach that achieves
acceptable quality", made mechanical.

**Already seen:** rungs 0, 1a, 2, 3a and the keyword+X sums were looked at in the audit
probe. They are marked `exploratory` in the record. Rungs 1b, 3b, 4a–c and 5 are the
held-out tests of the rule.

## 6. Extraction component (a separate experiment, proposal §4.3)

- **Vocabulary** (the never-done task 3.5): `data/vocab/skills.csv`, built from:
  - ESCO skills (preferred and alternative labels; digital collection and ICT skills
    groups);
  - `data_jobs` skill frequencies;
  - the existing 8 regex families.
  It is deterministic and committed with a digest (D15 permits derived extracts).
- **Rules:** a longest-match phrase matcher with alias normalisation, in pure Python.
  spaCy can now be installed (Q51), so the matcher may be built on a spaCy
  `PhraseMatcher` over a blank English pipeline. No trained spaCy model is needed, so no
  model download. Either implementation is fine. The comparison is rules vs learned, not
  the library.
- **Learned:** a MiniLM token classifier (BIO over skill and knowledge), fine-tuned on
  SkillSpan train and early-stopped on SkillSpan validation.
- **Evaluation:**
  - strict and partial span F1 on SkillSpan test (the job-posting domain);
  - the same on DataTurks test `Skills` (the resume domain), after the task 3.1 offset
    repair, with `Name` and `Email` spans dropped;
  - both reported with *n* documents and sentences. 20 DataTurks documents decide
    nothing, and the table says so.
- **Normalisation** (proposal table row 2): string match vs SBERT nearest-label, on a free
  gold set. It holds out ESCO alternative labels and maps each back to its preferred
  label. Top-1 accuracy is the metric.

## 7. Evidence layer and pipeline

`pipeline.screen(jd_text, cvs) -> list[Result]`, where each result holds a rank, a score,
a Fit / No Fit call, a confidence and evidence:

- **Requirements:** skills extracted from the JD (the selected extractor), plus the
  experience field when present.
- **Per requirement:**
  - `matched`: an exact or alias hit, with the CV span;
  - `supported`: the best CV sentence with SBERT similarity ≥ τ, quoted verbatim. τ is
    fixed on SkillSpan validation;
  - `missing`.
- **Fit / No Fit call:** from the selected rung's score, with one threshold fitted on A1
  train or the weak validation split (D46). It is labelled as a screening aid, not a
  verdict.
- **Faithfulness invariant (tested):** every quoted span is a verbatim substring of the
  source CV, and the pipeline refuses to emit otherwise. This is the proposal's "evidence
  faithfulness", met by construction rather than estimated.
- **Completeness proxy:** on A2, the share of JD requirements marked `matched` or
  `supported`, compared between final-set `Fit` and `No Fit`.

## 8. Conversational UI

- **Transport:** **Streamlit**, as the proposal names, installed through uv (Q51). It uses
  `st.chat_input` / `st.chat_message`. The rules live in plain modules, as `session.py`
  does for the annotation UI, so they are tested without a browser. Streamlit is thin
  transport, smoke-tested with `streamlit.testing.v1.AppTest`.
- **Input:** paste or upload a JD and CVs (`.txt`, `.pdf` via `pypdf`, `.html`), or pick
  an A2 JD from the eval region for the demo.
- **Scripted intents**, parsed by rules and answered from templates, with no generative
  model (proposal §4.6; no LLM at run time):
  - `top k`
  - `filter by skill / years`
  - `compare A and B`
  - `why is A above B`
  - `what is X missing`
  - `explain candidate A`
- **Display** goes through `annotation.redact`.
- **Tests:** one scenario test per intent, asserting the answer cites only evidence present
  in the pipeline output.

## 9. Codebase changes

```
src/candidate_screener/
  evaluation/      final_set.py (the D41 set), indomain.py (the protocol), retrieval_pilot.py;
                   metrics.py extended
  matching/        base.py, rules.py, lexical.py (wraps baselines/), sbert.py, cross_encoder.py,
                   hybrid.py
  extraction/      vocab.py, rules.py, ner.py, normalise.py
  evidence/        requirements.py, evidence.py
  pipeline.py      screen()
  app/             intents.py (rules, tested), streamlit_app.py (transport)
  data/            weak_pairs.py, dataturks.py (task 3.1); build.py registry extended
  annotation/      existing modules frozen (D42); dispatch refuses
docs/data/manifests/a2-final-labels.csv, weak-pairs.csv
output/annotation/a2-final-report.json
output/matching/   indomain-metrics.json, a1-metrics.json, retrieval-pilot.json (+ --check)
output/extraction/ extraction-metrics.json
docs/decisions.md  D1–D47, one line each
```

- **Packages, through uv only** (Q51). Commands:
  - `uv add --group ml torch sentence-transformers transformers`
  - `uv add --group app streamlit`
  - `uv add spacy` (optional)

  Torch comes from the CPU index, so the lock does not pull CUDA wheels:

  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cpu" }]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```

  - Python stays at 3.12. `uv sync --all-groups` for development.
  - The base `uv run pytest` must still pass **without** the `ml` group: tests needing a
    model use a stub encoder, and real-model tests `importorskip`.
  - Model weights: the cached `all-MiniLM-L6-v2` unless more are approved.
  - Caches (embeddings, fine-tuned weights) live in git-ignored `data/processed/`.
    Records hold config, digests and figures, never bytes (D23).
- **Determinism:** seeds fixed, `torch.use_deterministic_algorithms(True)`, one thread for
  `--check`. The float tolerance for torch-derived figures is measured in P3 (as Q23 was
  for the baseline), not assumed.
- **Style for new code:** docstrings state what and why in ≤ 10 lines, and history lives
  in `plan/` and `docs/decisions.md`. Guards stay where a silent failure is plausible:
  pooled strata, a figure without its `n` or `rule_version`, an LLM value overriding
  agreeing humans.

## 10. What this design does not do

- It collects no labels and dispatches no LLM (D42).
- It downloads no datasets. Packages come through uv.
- It never trains or tunes on the final set (D46).
- It makes no claim that any system is "correct". Every quality figure is agreement with a
  documented panel rule.
- It makes no claim about production deployment (proposal §5.4).
