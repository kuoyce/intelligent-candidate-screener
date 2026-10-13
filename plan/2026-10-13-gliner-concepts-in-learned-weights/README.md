# GLiNER concepts in the learned-weights test

Status: **draft v2, owner answers folded in (13 Oct 2026); awaiting final approval.** Nothing
below has been run except the GLiNER reproducibility check (Evidence). Exploratory under D56; nothing here is adopted.

## Question

Notebooks 18-20 found that learned weights (A1 or D50 labels, 4 or 11 features) do not beat the
hand-set X2 on the A2 final set (AUC 0.850), and that the 11-feature learners fit their own
training JDs no better than held-out ones (gap 0.01), so the features, not the labels, look like
the limit. Notebook 20 (Lead 1) traced part of that to the seed-rule concept vocabulary: **53%
of A2 pairs have no JD-CV concept overlap**, 33% of JDs yield <= 3 concepts, and only 38% of
the tool terms in the LLM's notes are in the vocabulary.

**Does swapping the seed-rule concepts for learned taggers (GLiNER, and our own MiniLM skill
tagger) (a) raise concept overlap and (b) let a learned model beat hand-set X2?** The answer decides whether more LLM labels for A1 are worth
paying for: more labels can only help if a model that can use them exists.

## Evidence

- GLiNER (`gliner_small-v2.1`, rev `4e09141`) runs here: 200 documents in 64 s, 3.4 GB peak,
  0 quarantined predictions (0.32 s/document). Counts match the saved 100 CV + 100 JD run to
  within ~1% (CV 1,082 vs 1,077; JD 747 vs 755, excluding the later `EXPERIENCE_YEARS` label).
  The drift is expected from library versions; raw scores are not claimed to replay.
- The `ml` group now pins `gliner==0.2.21`. 0.2.29 breaks `load_model`
  (`encoder_from_pretrained`), so the pin is required. This machine runs transformers 5.16,
  the saved run used 4.48.3 in the isolated runtime.
- **Our own tagger already exists:** `extraction/ner.py`, a MiniLM token classifier (BIO over
  SkillSpan SKILL = skill ∪ knowledge), weights present at `data/processed/extraction/ner/`.
  It loads and runs here (smoke: `AWS`, `Kafka`, `SQL` found in a test sentence; it also
  tagged `BSc`, `Computer Science`). It returns `(start, end)` spans only, no label or score,
  and is used today by `evidence/requirements.py`. It is the like-for-like comparison to
  GLiNER KNOWLEDGE + SKILL.
- `ConceptExtractor(entities=...)` already accepts supplied GLiNER entities
  (`matching/entity_weighted.py`). `a1_weights.build_scorer` currently builds it with none.

## Design

**Documents to extract** (GLiNER, threshold 0.5, the five registered labels; threshold is
fixed and never tuned on the final set, D46): the JDs and CVs of the A1 LLM-labelled 639 pairs
(100 JDs), the D50 batch (1,000 pairs, 100 A2-train JDs), and the A2 final set (60 JDs, 600
CVs). About 2,500 documents, ~13 min at the measured rate. Cached under `data/processed/`
(git-ignored), keyed by document id and text hash. Extraction reads text only; no label file.

**Features.** Rebuild the 11-feature set of notebook 18 with `ConceptExtractor` fed
an alternative concept source. Five sources, nothing else changes:

| Arm | Concepts |
|---|---|
| R (baseline) | seed rules, as notebooks 18-20 |
| G | GLiNER KNOWLEDGE + SKILL only |
| N | our MiniLM tagger only (`extraction/ner.py`) |
| R+G | union of rules and GLiNER, deduplicated by normalised phrase |
| R+N | union of rules and the MiniLM tagger |

N needs a thin adapter: its spans become entity dicts (`label` SKILL, `surface`, and
`context_status` from the same `rules._context` heuristic GLiNER uses), because
`ConceptExtractor` keeps only `unqualified` SKILL/KNOWLEDGE rows. The adapter is tested.
The G-vs-N comparison is also reported on its own: concepts per document and no-overlap share
for each, plus the overlap of their concept sets (Jaccard on normalised phrases).

IDF population, role model, TF-IDF and experience features are unchanged. Skill-overlap
(`matcher.canonicals`) stays rule-based in every arm (it is a vocabulary-matcher feature).

**Stage 1, label-free (reads no label):** on the A2 final pairs and on D50, report per arm the
share of pairs with no JD-CV concept overlap, the share of JDs with <= 3 concepts, and the share
of CVs with 0 concepts. Computed from text alone, so it is safe to look at before Stage 2.

**Gate (registered now, owner-set 13 Oct):** proceed to Stage 2 only if some arm other than R
cuts the no-overlap share by **>= 5 percentage points** (53% -> <= 48%) on the A2 final pairs.
Only arms that pass go to Stage 2. If none passes, stop: neither tagger repairs the gap, and
the answer is "more A1 labels will not help via this route".

**Stage 2** (only if the gate passes). Repeat the registered protocols with each arm that
passed the gate, against the same hand-set X2 and the same baseline R results:

1. Notebook 18 Part 2: fits on A (3,990 original), B (639 original), C (639 LLM).
2. Notebook 19 Parts 3-4: learning curves (10/25/50/70 JDs, 30 held-out, 20 repetitions) for
   logistic-11 and boosting-11, A1-LLM labels and D50 labels.

**Bar (written before any Stage 2 score, rule 8).** A learner beats hand-set X2 only if its
paired total dAUC has a JD-bootstrap CI above 0 **and** `generic` and `targeted` both have a
positive delta (the notebook 18 bar). Stage 2 has up to 4 feature arms x (3 label fits + 2 curve
sources x 2 learners) comparisons, so **a single nominal pass is not evidence**: it counts only
if it passes for the same feature arm on both an A1-LLM fit and a D50 fit.

**Decision rule for "are more A1 LLM labels worth it"** (registered now):

- **Yes, worth scoping** only if the bar is met as above **and** the 25 -> 70 JD rise on A1-LLM
  labels for that learner is > 0.01 (curve still rising, notebook 19's own criterion).
- **No** if the gate fails, or no arm meets the bar, or the curve is flat.
- Anything between (e.g. bar met only on D50) is reported as "inconclusive" with the paired CIs;
  no label spend is recommended on that basis.

Reported as always: `generic`, `targeted` and `total` together, n and a JD-bootstrap CI,
within-family AUC beside the headline, `llm_free` subset for any LLM-involved figure.

## Constraints and compliance

- D56 covers fitting on the A1 recheck and D50 labels in notebooks 18-20 and `a1_weights.py`.
  This plan extends its use to GLiNER features; it needs a one-line amendment to D56 on approval.
- No new labels, no model dispatch (D42). GLiNER is local inference on text, not an annotation.
- Code that fits a model never reads `a2-final-labels.csv` / `judgements.csv`; the scorer does.
- GLiNER cache, extraction code and the arm switch go in the package (`extraction/`,
  `evaluation/a1_weights.py`); notebooks only report (`test_notebook_is_thin`). No scripts
  under `plan/`.
- Everything is `exploratory`; the A2 final set is scored on, never fitted on.

## Cost

Extraction ~13 min (one background job, `/usr/bin/time -v`, exit marker, then end the turn).
Stage 1 is seconds on cached spans. Stage 2 refits are logistic/boosting on <= 1,000 rows and
were minutes each in notebooks 18-19. Time one fit before the grid; if the total exceeds about
an hour, say so before starting.

## Decisions (owner, 13 Oct 2026) and remaining questions

Resolved:

1. **Gate:** >= 5 percentage points (above).
2. **Version drift:** accepted (transformers 5.16, `gliner==0.2.21`; counts agree to ~1%). The
   tokenizer's Mistral-regex warning is noted, not chased.
3. **`transformer_run --check` / registry:** ignored for now; not part of this plan.
4. **`pyproject.toml` / `uv.lock`** (`gliner==0.2.21` in `ml`) go in the same commit as the
   implementation.
5. **Branch:** stay on `exp/learned-weights-and-error-analysis`.

Open:

- **MiniLM tagger cost and determinism.** Not yet timed on ~2,500 documents (the smoke test
  included a 3.4 s load). Time a 50-document batch before the full run; weights are a cache
  (D23), so a retrain is out of scope and the existing weights are used as they are.
- **Entity cache location** for the GLiNER and MiniLM spans: `data/processed/extraction/`
  (git-ignored), keyed by document id and text hash, unless you prefer otherwise.
