# 01 — Top-down audit

Starts from what the project promised (proposal v3), checks each deliverable against the
repository as of `b39fce0`, and asks where the effort went and what the evidence already on
disk says. Every figure marked *probe* was computed for this plan from data and LLM output
already on file. The scratch scripts live outside the repository (AGENTS.md: no scripts under
`plan/`). Phase 1 of the work plan recomputes them inside the package, under a protocol
fixed in advance.

## 1. What was promised

The proposal's research question is **"does more sophisticated candidate–job matching give
a meaningful improvement in screening quality, and at what point does the extra complexity
stop paying for itself?"** It commits to:

- a **matching ladder** run on common pairs: rule/skill overlap → TF-IDF → Sentence-BERT →
  cross-encoder, with an LLM as an optional extra rung;
- an **extraction component** (spaCy rules vs fine-tuned BERT NER), evaluated separately;
- a **ranking and evidence layer** (matched requirements, gaps, evidence, confidence);
- a **conversational recruiter UI**;
- an **in-domain end-to-end evaluation** against human judgements;
- 12 weeks and about 30 team-days. The proposal is dated 16 Aug 2026, so today (26 Sep) is
  week 7. About 30 days remain (answer to Q50).

## 2. What exists

| Deliverable (proposal §6) | State at `b39fce0` |
|---|---|
| De-identified datasets, label guide, evaluation protocol | **Mostly done.** 11 sources verified, leak-free A1 split, pools, a 600-pair in-domain set with 1,034 human labels and 1,800 LLM labels. **The final step is missing**: the annotations were never reconciled into one label set that systems can be scored against |
| Extraction: rules vs BERT NER | **Not started.** Task 3.1 (DataTurks repair) and 3.5 (vocabulary) were never done, and `data/vocab/` is empty |
| Stage 1: rule-based skill overlap | **Not built as specified.** "The baseline" is TF-IDF with an 8-regex keyword boost and a one-feature logistic regression |
| Stage 2: TF-IDF | **Done** (plus BM25). Frozen, guarded by `--check`, tested |
| Stage 3: Sentence-BERT | **Not started** |
| Stage 4: cross-encoder | **Not started** |
| Comparative evaluation | Stage 2 only, and only on A1 |
| Ranking and evidence layer | **Not started** |
| Conversational UI | **Not started.** The only UI is the annotation UI |
| Report, slides, demo | Not started |

Measured against the proposal's own timeline, the project sits at the end of week 5's
content (classical baseline) in week 7. Extraction (weeks 3–4) and neural matching (weeks
6–7) have not begun. The last commit was 10 Sep, 16 days ago.

## 3. Where the effort went

| | Annotation and LLM judge | Everything else |
|---|---|---|
| Source lines (`src/`) | **4,566 (53%)** | data 2,776 · baselines 675 · evaluation 638 |
| Test lines | **2,705 (73%)** | 1,014 |
| Test runtime | **~3.5 of 4.5 min** (`test_llm_recheck.py`) | ~1 min |
| Plan words | **~53,600 of 78,300 (68%)**, 10 of 13 plan directories | ~24,700 |
| `AGENTS.md` words | **7,088 of 8,460 (84%)**: the "Evaluation and annotation" section | 1,372 |
| Commits since 29 Aug | **37 of 41** (annotation, judge, pool bias) | 4. Of those, **1** builds a model (Phase 4 extracts the existing baseline) |

D25 capped manual annotation at **one session of ~250 judgements** so the project would
not spend its budget on the instrument. The campaign reached 1,034 labels, and then three
plans of LLM-judge calibration followed (D33–D39).

The intent behind the LLM leg was simple: **a secondary annotator whose labels would be
merged with the humans' into one final set**. The calibration plans instead kept asking
whether the judge was good enough to *replace* a human. D36 (substitution) was never made,
and the interchangeability gate (D37) licensed the judge only "as a second annotator in a
future double-labelling wave". The unapproved successor
(`plan/2026-09-10-a2-reference-design/`) proposed *choosing* among the references rather
than merging them.

So the one artefact the annotation work existed to produce, a final label per pair, was
never built. Every downstream step waited on it.

`AGENTS.md` is loaded into every agent session through `CLAUDE.md`, so 7,000 words of
judge history are an onboarding cost paid on every turn. They also bury the ~15 rules that
are still live.

## 4. What the annotation work did establish (kept)

1. **A1's labels do not reproduce (A13 fails).**
   - Two blind judges agree with each other at kappa 0.35 and with A1 at chance (0.08).
   - *Probe, new:* against A1's own labels, TF-IDF reaches per-JD AUC **0.66**, zero-shot
     SBERT **0.68** [0.61, 0.74], and the LLM (sonnet, option D, 639 pairs, 63 JDs)
     **0.56** [0.53, 0.60]. The LLM figure is partly ties: 556 of its 639 labels are
     `No Fit`.
   - The strongest reader scores lowest. A ladder measured on A1 can therefore **invert**:
     it rewards whatever A1's unknown labelling process rewarded.
   - On the A1 pools, zero-shot SBERT's Recall@10 is **0.259** (strict, n=31) against
     TF-IDF's committed 0.298. The ladder is flat there.
2. **The two humans disagree systematically.**
   - On the pairs both labelled, three-class agreement is **47.6%** (generic, n=126) and
     **48.5%** (targeted, n=200), kappa 0.25 / 0.21, macro-F1 0.48 / 0.45.
   - Collapsed to binary, agreement is **73.8% / 63.0%**.
   - `van` is more generous than `yc` at both boundaries (`Good Fit` 51.6% vs 25.2% on
     generic).
3. **The LLM's labels alone cannot stand in for a human, but its score ranks well**
   (D37, D39).
4. **The proposal's absolute targets are unreachable as written.** See
   [04-success-measures.md](04-success-measures.md).

## 5. The evidence behind the edited direction (probe)

### 5.1 Scoring against one annotator at a time gives two different answers

Per-JD AUC, fit (`Good` ∪ `Potential`) vs `No Fit`, on pairs both humans labelled:

| System | generic · yc | generic · van | targeted · yc | targeted · van |
|---|---|---|---|---|
| TF-IDF cosine | 0.74 | 0.71 | **0.81** | 0.59 |
| SBERT `all-MiniLM-L6-v2`, zero-shot | **0.80** | **0.81** | 0.79 | **0.66** |

On `targeted`, TF-IDF beats SBERT against `yc` and loses to it against `van`.
- Keeping the annotators separate leaves the project with two rankings of the systems and
  no way to choose between them.
- Picking one annotator picks one of those rankings.

**The labels have to be reconciled into one set before systems are compared**, which is
what the annotation campaign was for.

### 5.2 The LLM adds information as a secondary annotator

Test: hide one human, and ask what best predicts that human's labels. The candidates are
the other human alone, or the other human combined with the LLM. The LLM is the committed
instrument (`5d986c82`, `haiku`, runs 9/10), which labelled all 600 pairs. The metric is
Spearman rho against the hidden human, on pairs all three labelled.

| Held-out human | Stratum | Other human alone | Other human + LLM label (mean) |
|---|---|---|---|
| `van` | generic (n=126) | 0.44 [0.28, 0.58] | **0.53** [0.37, 0.65] |
| `yc` | generic (n=126) | 0.44 [0.28, 0.58] | **0.57** [0.43, 0.70] |
| `van` | targeted (n=200) | 0.35 [0.23, 0.47] | **0.43** [0.32, 0.53] |
| `yc` | targeted (n=200) | 0.35 [0.23, 0.47] | **0.48** [0.37, 0.59] |

Adding the LLM moves the combined label closer to the unseen human **in all four cases**.
That is the property a secondary annotator needs.

For context, the LLM's own *label* alone scores 0.50 / 0.58 / 0.38 / 0.47 on the same
four rows. Its *score* alone scores 0.57 / 0.67 / 0.48 / 0.58. The intervals overlap.

### 5.3 How the labels are combined decides who the "truth" is

- **Dawid–Skene** is the textbook latent-class model. Its latent class is fitted from the
  labels themselves, and on this data it **collapses onto the LLM**.
  - On `targeted` its final labels agree with the LLM 90.0% of the time (kappa 0.84),
    against 54.5% / 59.0% with `van` / `yc`.
  - The reason: the LLM is the most internally consistent rater. It sends latent
    `Potential` to `Potential` 98% of the time, so the model trusts it most.
  - The result would be LLM labels as ground truth by another route, which contradicts a
    *secondary* role. **Rejected.**
- **Median of three, human-first** makes the LLM secondary by construction:
  - it never overrides two humans who agree;
  - it can only choose between the humans' labels when they differ;
  - where only one human labelled a pair, that human's label stands.

| | generic (n=400) | targeted (n=200) |
|---|---|---|
| All three agree | 39 | 61 |
| Humans agree, LLM differs (humans kept) | 21 | 36 |
| Humans disagree, LLM decides between them | 66 | 103 |
| One human, LLM agrees | 167 | 0 |
| One human, LLM disagrees (human kept, pair flagged `contested`) | 107 | 0 |
| Final distribution No / Potential / Good | 37.5 / 38.2 / 24.2 % | 41.5 / 46.0 / 12.5 % |

*Adopted later the same day (D41, D47):* the user approved the equivalent **binary** rule:
- humans agree → theirs;
- split → the LLM;
- one human → that human;
- no human → the LLM.

It yields exactly these Fit / No Fit labels, collapsed. In binary terms the tiers are 219
humans-agree, 107 LLM tie-breaks, 274 single-human and 0 LLM-only.

Two consequences the design has to state:
- On `generic`, 274 of 400 pairs had one human, so the final set there is **mostly `yc`'s
  labels**.
- The LLM decides about **half of all double-labelled pairs**: the ones where the humans
  disagree. Its influence is bounded (always between the two humans) but not small.

### 5.4 The systems, scored against that one set (preview)

Per-JD AUC, fit vs `No Fit`, against the median-of-three set:

| System | generic (39 JDs) | targeted (20 JDs) | within-family generic (30) | within-family targeted (18) |
|---|---|---|---|---|
| random | 0.53 | 0.54 | 0.51 | 0.56 |
| `Primary Keyword` match | 0.79 | 0.73 | 0.50 *(by construction)* | 0.50 |
| TF-IDF | 0.76 | 0.78 | 0.72 | 0.72 |
| SBERT zero-shot | 0.79 | 0.75 | 0.79 | 0.70 |
| keyword + TF-IDF | 0.86 [0.80, 0.92] | 0.84 [0.79, 0.90] | 0.72 | 0.72 |
| keyword + SBERT | 0.89 [0.82, 0.94] | 0.82 [0.75, 0.88] | 0.79 | 0.70 |

- **The ladder is measurable** against one set: 20–39 JDs per stratum, 59 in total.
- **Resolution is the binding constraint.** The CI half-width is about 0.05–0.10, so only
  rung-to-rung deltas of about ≥ 0.08 AUC will resolve. The study should say so up front.
- **Within-family discrimination** (candidates sharing the JD's role family, where the
  metadata cannot help) is where richer models have room to earn their place.
- **Caveat:** these configurations were looked at *before* any registration. The record
  marks them `exploratory`. Only rungs built after the rule is fixed count as held-out
  tests.

## 6. Feasibility, measured

- `sentence-transformers/all-MiniLM-L6-v2` is in the local HuggingFace cache.
  `torch 2.14` and `sentence-transformers 6.1` install **offline** from the uv cache for
  Python 3.14. The project stays on 3.12 and fetches its wheels through uv (Q51
  approved).
- CPU only (22 threads, 15 GB RAM, no GPU). Embedding 953 documents with chunked mean
  pooling took **85 s**. Fine-tuning a 6-layer MiniLM on ~4k pairs is minutes per epoch.
- SkillSpan (4,800 / 3,174 / 3,569 sentences with skill and knowledge BIO tags) and
  DataTurks (200 / 20 resumes) are on disk for extraction. ESCO (13,960 skills with
  alternative labels) is on disk for the vocabulary and for a free normalisation gold set
  (alt label → preferred label).
- `indomain-holdout.csv` reserves 13,992 eval-region CVs: the candidate pools the 600
  pairs were drawn from. That is enough for a 500-candidate pilot-scale pool per labelled
  JD without touching training data.

## 7. Keep, freeze, retire

| Item | Verdict | Why |
|---|---|---|
| Data layer, leak-free split, manifests, `verify` | **Keep** | Sound, tested, cheap to maintain |
| `baselines/` and its `--check` record | **Keep as-is** | Guarded. New rungs go beside it, not into it |
| `evaluation/metrics.py` guards (n carried, no silent k collapse, per-query bootstrap) | **Keep, extend** | The right discipline, reused for A2 |
| The 600 A2 pairs, 1,034 human labels, LLM runs 9/10 | **Keep, reconcile** into the final set, which becomes the primary benchmark | The only in-domain judgement there will be |
| The strata rule (D29) | **Amended** (Q56): generic, targeted **and a total** are reported together, never a total alone | The strata stay visible beside the pooled figure |
| "Annotator is an axis" (Phase 2) | **Narrowed**: it applies to *building* the final set, not to *evaluating* systems | Systems are scored against one set |
| "LLM is never a reference" (D37 §2) | **Revised** by D41: the LLM is a secondary annotator inside a human-first rule, never a reference on its own | The user's decision of 26 Sep 2026, backed by §5.2 |
| `annotation/` (UI, queue, sampler, `llm_recheck`) | **Freeze** | The campaign is closed and no dispatch is permitted. Its readers stay importable and its reports committed |
| D26 ceilings (0.7806 / 0.7188) and A1 as a headline benchmark | **Retire** | Built on labels that do not reproduce. Superseded in place, not deleted |
| `plan/2026-09-10-a2-reference-design/` (unapproved) | **Supersede** by D41 | It chooses a reference, and the edited direction merges them instead |
| `AGENTS.md` judge narrative | **Move** to `docs/decisions.md` plus the existing plan records | The history survives in git and `plan/`. The onboarding cost does not |

## 8. Directions considered and not taken

- **Keep scoring against each annotator separately.** Rejected (user, 26 Sep 2026). It
  yields two rankings of the systems and no conclusion (§5.1).
- **Dawid–Skene or any fitted aggregation that learns rater weights.** Rejected for the
  primary set because it makes the LLM the dominant rater (§5.3).
- **LLM as a system (ladder rung).** Deferred. The LLM is an annotator now, and scoring it
  against a set it helped build would be partly self-agreement. If it is scored later, it
  is scored against the set rebuilt *without* its labels (D43).
- **Train on A2 (cross-fitted over JDs).** Rejected as the main line, because it breaks
  proposal §5.2 (in-domain set isolated from training and tuning). It survives only as an
  optional, explicitly labelled upper-bound experiment (Q54).
- **Collect more labels or re-judge.** Excluded by the constraints of 26 Sep 2026 (no new
  data, no LLM dispatch).
