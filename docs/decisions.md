# Decision log

One line per decision, newest last, each linked to the plan record that argues it. This is
where history lives. `AGENTS.md` carries only the rules that are still live (D45).

**Status:** `live`, `superseded`, `amended`, `closed` (it did its job), or `unmade`
(proposed and never taken).

**Number collisions.** D19–D23 were minted twice, by `derived-artefacts` /
`unified-judging-wave` and by `baseline-repeatability`, on the same day. Both meanings
are listed, qualified by plan. Cite them as e.g. "D23 (baseline)".

| # | Decision | Status | Record |
|---|---|---|---|
| D1 | Retrieval is evaluated on synthetic candidate pools built from the fit dataset | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D2 | Success targets rebased on the measured Stage-1 baseline as relative deltas | superseded by D44 | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D3 | In-domain data from public JD dumps and resume corpora (no scraping) | amended (Djinni, see D4/D11) | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D4 | In-domain fit labels are team-adjudicated, with inter-annotator agreement reported | live (agreement reported with no target, D44) | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D5 | DataTurks kept as the resume-native NER gold seed and supplemented | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D6 | A real PDF/HTML resume corpus is sourced for parsing robustness | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D7 | A1's undeclared licence accepted; recorded in the report | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D8 | Re-split A1 at candidate level (doubly disjoint); not comparable to published numbers | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D9 | Pool size and retrieval metrics decided from the data | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D10 | Kaggle access available; PDFs feed the ingestion demo only | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D11 | Djinni's Ukrainian/EE IT market accepted as in-domain, stated as a limitation | live | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D12 | 200 in-domain pairs hand-labelled by the team | superseded by D25/D28 | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D13 | ESCO v1.2.1 acquired | closed | [data-strategy](../plan/2026-08-19-data-strategy/README.md) |
| D14 | The first in-domain draw is not stratified by role family | live (it is the `generic` stratum) | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D15 | Derived ESCO extracts may be committed; raw data may not; notebook output capped | live | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D16 | A1's shipped train and test are pooled, then re-split | live | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D17 | The in-domain set uses A1's three-class scheme | amended by D47 (collected in three classes, evaluated as Fit / No Fit) | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D18 | A2 may feed unlabelled pretraining only; eval documents excluded | amended by D19 | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D19 (derived-artefacts) | A2 is partitioned at document level into `train` / `eval` regions before any sampling; `train` may supervise, `eval` never | live | [derived-artefacts](../plan/2026-08-23-derived-artefacts/README.md) |
| D19 (baseline) | The baseline's logic lives in `src/candidate_screener/baselines/` | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D20 (judging wave) | Our judgements are a separate committed layer, never merged into `pools.csv` | live | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D20 (baseline) | The baseline's committed record is `output/baselines/baseline-metrics.json` | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D21 (judging wave) | A judging wave would cover the 64 graded-scoreable queries | superseded in scope by D25 | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D21 (baseline) | Reproducibility is asserted by `run --check` against the committed JSON | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D22 (judging wave) | One judging wave, run once after the last stage exists | superseded in scope by D25 | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D22 (baseline) | A pytest suite, with each shipped bug as a named regression test | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D23 (judging wave) | The wave would cover all 100 test queries | superseded in scope by D25 | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D23 (baseline) | Fitted models (pickles, weights) are a cache, never an artefact | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D24 | BM25 is scored from `idf`/`avgdl` against a document's own tokens | live | [baseline-repeatability](../plan/2026-08-29-baseline-repeatability/README.md) |
| D25 | Manual annotation capped at one session (~250 judgements) | closed (exceeded: 1,034 labels; collection closed by D42) | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D26 | A1 pool precision is reported beside its attainable ceiling (0.7806 / 0.7188); Recall@10 headline | superseded by D40 (ceilings retired; built on labels that do not reproduce) | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D27 | The retrieval scoring pass over the A1 pools runs | closed | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D28 | The in-domain set is a frozen, append-only batch campaign | live (frozen; no batches will be added, D42) | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D29 | Reporting by stratum (`generic`, `targeted`), never averaged | amended by D41/Q56: a total is reported beside both strata | [unified-judging-wave](../plan/2026-08-29-unified-judging-wave/README.md) |
| D30 | Labelling happens in a local UI | closed | [annotation-ui](../plan/2026-08-29-annotation-ui/README.md) |
| D31 | 10 candidates per JD, 8 sharing its `Primary Keyword` | live (describes the set) | [annotation-ui](../plan/2026-08-29-annotation-ui/README.md) |
| D32 | The A1 recheck is labelled in the same UI | closed | [annotation-ui](../plan/2026-08-29-annotation-ui/README.md) |
| D33 | A tool-free LLM judge rechecks A1; A13 fails (A1's labels do not reproduce) | live (the finding); judge frozen by D42 | [llm-recheck](../plan/2026-08-30-llm-recheck/01-findings.md) |
| D34 | The annotation guide defines `No Fit` positively (an ordered decision) | live (guide) | [prompt-calibration](../plan/2026-09-05-prompt-calibration/02-calibration-run.md) |
| D35 | One draw per run, one file per draw | live (frozen files) | [recheck-file-split](../plan/2026-09-06-recheck-file-split/README.md) |
| D36 | LLM labels substitute for human labels | unmade; stays unmade (D41: the LLM only breaks human ties or labels where no human looked) | [a2-llm-prelabel](../plan/2026-09-06-a2-llm-prelabel/06-phase2-findings.md) |
| D37 | Consensus is a reference; the judge-acceptance gate asks interchangeability | superseded for A2 evaluation by D41 | [judge-calibration](../plan/2026-09-08-judge-calibration/03-consensus-findings.md) |
| D38 | Model is an axis: judge series keyed by (instrument, model) | live (frozen code) | [judge-calibration](../plan/2026-09-08-judge-calibration/README.md) |
| D39 | Keep the graded instrument `5d986c82` committed | live (it is the LLM annotator in D43) | [judge-calibration](../plan/2026-09-08-judge-calibration/08-q45-decision.md) |
| D40 | A2 is the primary benchmark; A1 is a training resource and secondary benchmark; D26's ceilings retired | live | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D41 | One final Fit / No Fit set: humans agree → theirs; split → LLM; one human → human; no human → LLM. Reported for generic, targeted and total | live | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D42 | Annotation collection and judge calibration closed; dispatch refuses in code | live (amended by D50: one scoped scoring batch) | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D43 | The LLM is a secondary annotator (one series), not a system; no system trains on LLM labels | live (amended by D51: the D50 batch may train a reranker) | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D44 | Success measures: three-class dropped; inter-annotator agreement reported with no target; remaining bars registered before held-out results | live | [top-down-reset](../plan/2026-09-26-top-down-reset/04-success-measures.md) |
| D45 | `AGENTS.md` holds live rules only; history lives here | live | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D46 | Nothing is fitted or tuned on the final set | live | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D47 | Fit / No Fit is the only label scheme, everywhere | live (amended by D52: ordering figures on the human grades) | [top-down-reset](../plan/2026-09-26-top-down-reset/02-design-spec.md) |
| D48 | Selected pipeline by the registered rule: metadata + TF-IDF where a `Primary Keyword` exists on both sides; the weakly fine-tuned SBERT for text-only input. No cross-encoder or learned hybrid earned its place | live | [top-down-reset](../plan/2026-09-26-top-down-reset/07-selection.md) |
| D49 | Extraction split by role: the learned tagger adds free-text JD requirements; the rules matcher does CV skills and every exact/alias match; rung 1b stays on rules | live | [top-down-reset](../plan/2026-09-26-top-down-reset/08-extraction.md) |
| D50 | D42 reopened once: one LLM scoring batch (instrument `5d986c82`, haiku), 100 A2-train JDs × 10 CVs = 1,000 pairs, none from the final set or the eval region; scoped in code, every other dispatch still refuses | live | [extraction-and-rerank](../plan/2026-10-06-extraction-and-rerank/README.md) |
| D51 | A reranker may be trained on the D50 batch's 0-100 scores, and on nothing else from the LLM; every ordering figure is reported beside the `llm_free` subset | live | [extraction-and-rerank](../plan/2026-10-06-extraction-and-rerank/README.md) |
| D52 | Ordering figures (per-JD nDCG, pairwise agreement) on the human grades Good=2 / Potential=1 / No Fit=0 are allowed as a secondary metric; classification stays Fit / No Fit | live | [extraction-and-rerank](../plan/2026-10-06-extraction-and-rerank/README.md) |
| D53 | No metadata at inference. The evaluation text is `text_v2`: A2 CV text without `Position`; `Primary Keyword`, experience and `Position` fields are training labels or oracles only | live (amended by D54: the JD title is an input) | [extraction-and-rerank](../plan/2026-10-06-extraction-and-rerank/README.md) |
| D54 | The JD title (A2 JD `Position`, as given) is an inference input, `jd_title`: a recruiter always types it. Amends D53 for the JD side only; CV `Position`, `Primary Keyword` and the experience fields stay labels or oracles | live | [shortlist-recall](../plan/2026-10-07-shortlist-recall/README.md) |
| D55 | Two check tiers. **Cold** (`--check`) rebuilds from scratch, retraining included, and is the reproducibility proof. It is mandatory before a commit that touches model or training code, before a plan is marked complete, and before a final-set score. **Warm** (`--check --warm`, or `checks --warm`) reuses cached model outputs (NER weights, role-family model, cross-encoder call cache) and serves the everyday loop. Superseded records (indomain v1, `retrieval`, `retrieval_pilot`, `a1_secondary`) run cold only, or when a change affects them | live | [test-and-check-runtime](../plan/2026-10-08-test-and-check-runtime/README.md) |
| D56 | Exploratory exception to D43/D51, authorised by the project owner on 12-13 Oct 2026: models may be fitted on LLM labels (the A1 recheck, `claude-sonnet-5` run 2, and the D50 batch) to measure what labels could add. Fitted only in notebooks 18-20 and `evaluation/a1_weights.py`; the A2 final set is scored on, never fitted on (D46); no result selects a rung. Findings: learned weights and richer features do not beat the hand-set X2 on A2 (0.850); the D50 curve is no steeper than A1's. Amended 13 Oct 2026: the same exception covers GLiNER and MiniLM-tagger concept features, registered in [gliner-concepts](../plan/2026-10-13-gliner-concepts-in-learned-weights/README.md) (gate: >= 5 pp fewer no-overlap pairs; bar as notebook 18; result: only R+N passed, no learner beat X2) and coverage weighting variants in [coverage-weighting-variants](../plan/2026-10-13-coverage-weighting-variants/README.md) | exploratory | [learned weights](../notebooks/18-a1-learned-weights.ipynb), [learning curves](../notebooks/19-learning-curves.ipynb), [error analysis](../notebooks/20-error-analysis.ipynb) |
| D57 | Manual review of scoring mistakes (error-analysis-2): the owner tags why a sampled mistake happened (role mismatch with transferable skills, seniority, missing core tool, label looks wrong, other). These are **error-reason tags, not relevance labels**: stored in `output/extraction/error-review-reasons.csv`, never in `judgements.csv`, the final set or any training set, and no document text is stored. The frozen annotation instrument (D42) is not reused. Amended 17 Oct 2026 (owner): the reviewer may also set an **override flag** (`fit_side`, `nofit_side`, `both`) when very sure a final label is wrong; it needs reason D and a note, is a flag on a disputed label, not a replacement, and changes no label anywhere (D41, D42, D46). Redacted case texts are cached in git-ignored `data/interim/` only. Review stopped at 45 of 111 cases (owner, 17 Oct 2026): 23 flagged, mostly Fit sides from the D50 LLM-only batch (11/21) and LLM tie-breaks (7/10); counts in `output/extraction/error-review-summary.json` | exploratory | [error-analysis-2](../plan/2026-10-13-error-analysis-2/README.md) |
| D58 | `uv run pytest` alone is the default verification; record checks run only when a change touches underlying code (models, extractors, features, scorers, data layer). Before `--all`, `--cold` or any long check run, the agent names the checks and the estimate and asks the owner. Owner, 17 Oct 2026, after an affected-only warm run took over an hour for a standalone review tool. D55's cold triggers still apply when the run happens | live | [AGENTS.md](../AGENTS.md) Testing |
