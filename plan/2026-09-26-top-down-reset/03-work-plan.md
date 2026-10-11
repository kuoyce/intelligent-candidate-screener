# 03 — Work plan

Nine phases in two tracks that can run in parallel once P1 lands. The effort figures are
team-days and are estimates. About 30 days remain (Q50), and the full plan is ~21.5
team-days, so it fits with slack for review and rework.

Every phase ends with the workflow's step 4 (`uv run pytest`, every `--check`,
`verify --derived`) and a record in this directory (`NN-*.md`).

```
Track A (matching)   P0 ─► P1 ─► P2 ─► P3 ─► P4 ─┐
Track B (system)            └──► P5 ─► P6 ─► P7 ──┴─► P8 (report assets)
```

P5 needs only P0. P6 needs P2's vocabulary and P4's selected rung (it can start on rung 2
and swap later). P7 needs P6.

## P0 — Reset (0.5 d)

| # | Task | Acceptance |
|---|---|---|
| 0.1 | **Merge `feat/a2-judge-calibration` into `main`** (Q52: `--no-ff`, following the repo's merge-commit convention). Then commit this plan on `feat/top-down-reset`, which is already based on that branch's head `b39fce0`. Nothing is pushed unless asked | `main` contains `b39fce0`. The plan commit holds nothing else |
| 0.2 | `docs/decisions.md`: D1–D47 one line each, with a link to the plan record | Every `D\d+` cited anywhere in the repo resolves there (a test greps for it) |
| 0.3 | Rewrite `AGENTS.md` to live rules only (D45). Update the README's "current phase" | ≤ 1,800 words. Every rule removed is either dead or reachable through `docs/decisions.md` |
| 0.4 | Freeze `annotation/` (D42): `--dispatch`, `--judge` and `prompt_calibration` raise `DispatchClosed` naming D42 | A test that each entry point refuses, and that the report readers still run |
| 0.5 | Retire D26 (D40): supersede in place, dated, in `data-catalog.md`, the A1 card, the README and `output/baselines/README.md` | Supersession notes present. No figure is deleted |
| 0.6 | Packages through uv (Q51): the `ml` and `app` groups and the PyTorch CPU index (02 §9) | `uv sync --all-groups` works on 3.12. Base `uv run pytest` passes without the groups |

## P1 — The final annotation set and the evaluation protocol (3 d)

| # | Task | Acceptance |
|---|---|---|
| 1.1 | `evaluation/final_set.py`: the user-approved rule v1 (02 §3.2: humans agree → theirs; split → LLM; one human → human; no human → LLM), Fit / No Fit, plus the `llm_free` subset. `data.build --task a2-final`. Writes `a2-final-labels.csv` and `a2-final-report.json`, including inter-annotator agreement (02 §3.6) | The 02 §3.4 invariants hold as tests (synthetic, including a synthetic no-human pair for rule 4) and as `verify --derived` checks (real). Byte-identical rebuild. Tier counts 219 / 107 / 274 / 0 and fit rate 0.612 reproduce |
| 1.2 | `matching/base.py`: the `System` protocol and `card()` with timing helpers | A stub system round-trips through 1.3 |
| 1.3 | `evaluation/indomain.py`, Fit / No Fit only: per-JD AUC (primary) and within-family AUC for rankers; accuracy, binary macro-F1, F1(`Fit`) and kappa for classifiers. Paired bootstrap over JDs, for `generic`, `targeted` and `total` | `IndomainFigure` refuses a missing `scope`, `n_jd`, CI or `rule_version`. A `total` is never emitted without both strata beside it. Tested on synthetic frames |
| 1.4 | First record, `output/matching/indomain-metrics.json`: rungs 0, 1a and 2, flagged `exploratory`. CLI: `python -m candidate_screener.evaluation.indomain --seed 0 [--check]` | `--check` reproduces byte-for-byte. The 01 §5.4 figures are reproduced, or the difference is explained (the IDF corpus differs by design) |
| 1.5 | **The open success items answered first** (04 §3: S0–S4). Then `06-registration.md`: the §5.4 rule, the absolute bars chosen under S1–S4 if any, and the held-out rung list, committed **before** any held-out rung is scored | The commit precedes the first 1b/3b/4x figure in git history |

## P2 — Stage 1 rules, vocabulary, Stage 2 on A2 (2 d)

| # | Task | Acceptance |
|---|---|---|
| 2.1 | `extraction/vocab.py` (task 3.5): build `data/vocab/skills.csv` from ESCO, `data_jobs` and the regex families, with a recorded digest. `verify --derived` checks the digest | Deterministic rebuild. The digest matches |
| 2.2 | `extraction/rules.py`: phrase matcher with alias normalisation (pure Python or a spaCy `PhraseMatcher`) | Unit tests: overlapping aliases, case, punctuation (`C++`, `.NET`, `Node.js`) |
| 2.3 | Rung 1b (skill coverage) and rung 1a (metadata rules), both fixed formulas | In the record, with `card()` |
| 2.4 | Rung 2: TF-IDF and BM25 with IDF from the A2 train region, via the existing `baselines/` functions. Plus the hybrid 1+2 | In the record. `baselines/run --check` still green, untouched |

## P3 — Stage 3 semantic, and pilot-scale retrieval (3 d)

| # | Task | Acceptance |
|---|---|---|
| 3.1 | `matching/sbert.py` with chunked pooling, and an embedding cache keyed by (model digest, text digest) under `data/processed/embeddings/` | Real-model tests `importorskip` |
| 3.2 | Rung 3a (zero-shot) on A2 and A1 | In both records |
| 3.3 | `data/weak_pairs.py`, `build --task weak-pairs`: the manifest (ids only) and a JD+CV-disjoint validation split | `verify --derived`: no eval-region document, no train/validation overlap |
| 3.4 | Rung 3b: fine-tuned bi-encoder, selected on weak validation only | Config and digests in the record. Weights cached, not committed |
| 3.5 | Measure torch determinism and set the `--check` tolerance (as Q23 did) | Tolerance recorded, along with how it was measured |
| 3.6 | `evaluation/retrieval_pilot.py`: 500-candidate pools, Recall@50/100 of final-set fits, latency per role | `output/matching/retrieval-pilot.json` with `--check` |

## P4 — Stage 4, hybrid, selection (3 d)

| # | Task | Acceptance |
|---|---|---|
| 4.1 | Rung 4a: cross-encoder with a Fit / No Fit head on A1 train (labels collapsed, D47), hyperparameters chosen on `fit_split.cv_folds` | Fold figures recorded. Test pairs are never touched during selection |
| 4.2 | Rung 4b: cross-encoder on weak pairs | Weak validation only |
| 4.3 | Rung 4c: retrieve-then-rank over the P3.6 pools | Recall and re-ranking quality both reported |
| 4.4 | Rung 5: hybrid combiner (D46-compliant training data) | Record |
| 4.5 | A1 secondary benchmark for every rung, Fit / No Fit: binary macro-F1, and pools Recall@10 under the `graded` definition (= `Fit`), with the A13 line | `output/matching/a1-metrics.json` with `--check` |
| 4.6 | `07-selection.md`: apply the registered rule, including the `llm_free` check, and record the **selected pipeline** as D48 | The ladder table is complete: quality × cost × complexity × explainability, for generic, targeted and total |

## P5 — Extraction component (3 d, track B)

| # | Task | Acceptance |
|---|---|---|
| 5.1 | `data/dataturks.py` (task 3.1): offset repair of the 222 broken spans, converted to BIO, dropping `Name`/`Email` spans | Every surviving span's text equals its slice. A test on synthetic broken offsets |
| 5.2 | Rules extractor evaluated on SkillSpan test and DataTurks test | Strict and partial span F1, with *n* |
| 5.3 | `extraction/ner.py`: MiniLM token classifier on SkillSpan train, early-stopped on validation | Same metrics. Weights cached |
| 5.4 | `extraction/normalise.py`: string vs SBERT on held-out ESCO alternative labels | Top-1 accuracy with CI |
| 5.5 | `output/extraction/extraction-metrics.json` with `--check`. `08-extraction.md` selects the extractor (D49) | Record committed |

## P6 — Evidence layer and pipeline (2 d)

| # | Task | Acceptance |
|---|---|---|
| 6.1 | `evidence/`: requirements, matched / supported / missing, experience check | Unit tests |
| 6.2 | `pipeline.screen()` using the selected extractor and rung | Deterministic output for a fixed input |
| 6.3 | Faithfulness invariant | A test that a fabricated span is refused |
| 6.4 | Completeness proxy on A2, final-set `Fit` vs `No Fit` | Figure in `indomain-metrics.json` |

## P7 — Conversational UI (3 d)

| # | Task | Acceptance |
|---|---|---|
| 7.1 | `app/streamlit_app.py`: upload or paste; ranked shortlist; evidence table; chat panel; redacted display | Launches with `uv run streamlit run src/candidate_screener/app/streamlit_app.py`. An `AppTest` smoke test |
| 7.2 | `app/intents.py`: the six scripted intents, rule-parsed and answered from templates | One scenario test per intent. Answers cite only pipeline evidence |
| 7.3 | Latency at 500 candidates per role, end to end | Recorded next to the P3.6 figure |

## P8 — Report assets (2 d)

| # | Task | Acceptance |
|---|---|---|
| 8.1 | Notebooks `08-final-annotation-set`, `09-indomain-ladder`, `10-extraction`, `11-end-to-end`: thin, reading the committed records | `test_notebook_is_thin` passes. Outputs are capped (D15) |
| 8.2 | `09-findings.md`: the answer to the research question, the success-measure table (04 §2, with pass/miss where S1–S4 set a bar), inter-annotator agreement as context, and the limitations: the final set's construction (01 §5.3, 02 §3.5), n = 60 JDs total (40 generic, 20 targeted), the Djinni market, unjudged distractors, A13 | Every number traces to a committed record |
| 8.3 | Demo script: the UI walkthrough over two A2 JDs | Rehearsed once |

## Effort

| Phase | Days | Track |
|---|---|---|
| P0 | 0.5 | A |
| P1 | 3 | A |
| P2 | 2 | A |
| P3 | 3 | A |
| P4 | 3 | A |
| P5 | 3 | B |
| P6 | 2 | B |
| P7 | 3 | B |
| P8 | 2 | both |
| **Total** | **21.5** | ~12 elapsed days with two people, against ~30 remaining |

**Contingency cut (~15 d)**, only if P3/P4 overrun. It drops:
- 3.4 (fine-tuned bi-encoder)
- 4.2 (weak-pair cross-encoder), which means 3.3 goes too, unless 3.4 stays
- 4.4 (hybrid)
- 5.4 (normalisation)

It keeps every proposal stage (1b, 2, 3a, 4a), the final set, the protocol, extraction,
evidence, the UI and the report.

## Sub-agent use (workflow step 3)

One sub-agent per task row, briefed with the design section it implements and the
acceptance line it must meet. The orchestrator runs step 4 after each phase.

Sub-agents never run `llm_recheck` or `prompt_calibration` (the user's constraint, enforced
by 0.4). A sub-agent building a rung never reads `a2-final-labels.csv` or
`judgements.csv`. Only `evaluation/final_set.py` and `evaluation/indomain.py` read them,
which keeps the labels out of any code path that fits a model.

## Risks

| Risk | Mitigation |
|---|---|
| No rung-to-rung delta resolves at 60 JDs (CI half-width ≈ 0.04 on the total) | Expected, and pre-registered as a legitimate outcome ("not resolved", with the minimal detectable effect). The proposal already says a simpler method winning is not a failure |
| The LLM's tie-breaks favour semantic rungs | The `llm_free` check downgrades any verdict that depends on them (02 §5.4) |
| Generic's final set is mostly one annotator | Stated with every figure (02 §3.5). Generic, targeted and total are always reported together (Q56) |
| Weak labels teach keyword prediction only | Rung 1a is its own row, and the within-family metric exposes it |
| Torch nondeterminism breaks `--check` | A measured tolerance (3.5) and a single-thread check |
| Time | Tracks run in parallel. Contingency cut |
