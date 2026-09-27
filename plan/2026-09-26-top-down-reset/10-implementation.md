# 10 — Implementation record

What was built against the approved plan (v4), phase by phase, and every place it
deviated. Figures live in the committed records under `output/`; this file points to
them rather than restating them, except where a number explains a decision.

## P0 — Reset

- `feat/a2-judge-calibration` merged into `main` locally (`--no-ff`, Q52). Nothing
  pushed.
- `docs/decisions.md` holds D1–D49. **D19–D23 were minted twice** (`derived-artefacts` /
  `unified-judging-wave` and `baseline-repeatability`, both on 29 Aug); both meanings are
  listed, qualified by plan, rather than renumbered. `tests/test_decisions.py` fails on
  any cited decision the log does not hold.
- `AGENTS.md` went from 8,460 to 1,800 words of live rules (D45).
- **D42 in code:** `annotation.llm_recheck.DISPATCH_CLOSED` guards every dispatch and
  spawn path (`dispatch*`, `judge`, `judge_one`, `prompt_calibration`), raising
  `DispatchClosed`. The frozen module's own regression tests reopen the gate in-process,
  with the subprocess mocked. `tests/test_dispatch_closed.py` pins the closed default.
- D26's ceilings are superseded in place: catalog, A1 card, README and baselines README.
- Packages through uv: groups `ml` (torch from the PyTorch CPU index,
  sentence-transformers, transformers, **accelerate**, added in P3 because `Trainer`
  requires it) and `app` (streamlit). Python stays 3.12.

## P1 — The final set and the protocol

- `evaluation/final_set.py` builds `a2-final-labels.csv` by rule v1.
  - Tiers: 219 humans-agree, 107 LLM tie-breaks, 274 single-human, 0 LLM-only.
  - The fit rate is 0.612.
- **Bug caught by the invariants before anything was written.** A missing `van` label
  arrives as NaN, not `None`. `resolve()` counted it as a second, disagreeing human, and
  55 single-human pairs silently took the LLM's label. It was fixed, and
  `test_a_missing_human_as_nan_is_not_a_second_human` was shown to fail against the old
  code.
- **Second silent bug, caught by the NaN guard in `matching.base`.**
  `pd.Series(series, index=pair_ids)` re-aligned on the old RangeIndex and returned
  all-NaN scores. `base.score` now takes values. It has a regression test.
- The bootstrap was vectorised: `resample.pair_stat_ci` passes numpy arrays. The first
  version took minutes per figure.
- **A scope with no JD holding both classes** skips its whole (metric, subset) group,
  and the group is listed under `unscoreable`. It is never emitted as a total without
  strata.
- **Rung definitions fixed before any held-out rung was scored:**
  - metadata = `2 × keyword + experience-compatible`;
  - hybrids = metadata + the text score min-max scaled per JD × 0.999, so metadata
    decides the level and text orders within it;
  - TF-IDF/BM25 use the frozen baseline's config, **without** the skill-boost augment,
    with IDF from 20k + 20k A2 train-region documents.
- `06-registration.md` was committed (`c29445a`) before the first held-out figure.

## P3 — Semantic rungs, weak pairs, pilot retrieval

- **Weak pairs:** 8,388 (7,548 train / 840 validation). There are 3,000 JDs targeted, and
  hard negatives are missing where a keyword has no under-experienced CV. They pass three
  checks: no eval-region document, no JD or CV shared across splits, and a deterministic
  rebuild.
- **3b batch size 16, not 32.** The first run was killed for running out of memory (exit
  137, while the extraction agent's NER training shared the machine). The config was
  changed before any weights existed. It trained in 2,850 s, with peak RSS 6.7 GB.
- **Pilot pools:** 60 JDs × 500 = 30,000 pairs, 12,592 unique CVs, 367 relevant fits.

## P4 — Cross-encoders, hybrid

- **4a hyperparameters are fixed in advance, not chosen on `fit_split.cv_folds`**
  (a deviation from 03 row 4.1). With the final set out of bounds and one epoch
  affordable on CPU, a search would have tuned noise. Fold 0 of the doubly-disjoint CV
  (135 pairs) is held back and used only for the class threshold. A `fold_masks` helper
  was added to `fit_split`, and `cv_folds` now uses it (its counts are unchanged).
- **The learned hybrid (rung 5) trains on A1 fold 0, over text scores only.** The weak
  labels are a function of keyword and experience, so a combiner fitted there would
  relearn the metadata rule. It emits no class calls: no split is left for a threshold.
  This is a deviation from registration §5's S1 list.

## P6 / P7 — Evidence, pipeline, UI

- τ = 0.28, fixed on SkillSpan validation (phrase vs its own sentence / a random other
  sentence; F1 0.91). **Limitation:** those positives contain the phrase literally, so τ
  may be lenient for purely semantic support.
- The pipeline makes a Fit / No Fit call only when both sides carry a `Primary Keyword`.
  Uploaded CVs have none, so the UI shows rank and evidence but no call. "Confidence" is
  the within-pool percentile, labelled as relative.
- The UI is Streamlit (Q51), with the rules in `app/intents.py`. Six intents, each with a
  scenario test, plus an `AppTest` smoke test.

## Open at the time of writing

Results are filled in from the records once they are run: `07-selection.md`,
`08-extraction.md`, `09-findings.md`.
