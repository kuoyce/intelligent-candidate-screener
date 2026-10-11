# 09 — Findings

The answer to the proposal's research question, the success-measure table, and the
limitations. Every number traces to a committed record: `output/annotation/`,
`output/matching/`, `output/extraction/`, `output/evidence/`.

## 1. The research question

> Does progressively increasing the sophistication of candidate–job matching provide a
> meaningful improvement in screening quality, and at what point does the additional
> complexity stop being worthwhile?

**On the in-domain set, against the final panel labels:**

1. **Cheap signals do most of the work.**
   - The keyword rule alone moves per-JD AUC from 0.556 (random) to 0.778.
   - Adding TF-IDF text similarity brings it to **0.849** [0.802, 0.891] (+0.070, resolved).
   - At 500 candidates per role, metadata + TF-IDF keeps **92.9%** of the fits in a
     50-candidate shortlist.
2. **Where a category field exists, nothing richer earns its place.**
   - Zero-shot SBERT, weakly fine-tuned SBERT, both cross-encoders and a learned
     combination are all harmful or unresolved against metadata + TF-IDF.
   - The minimal detectable effect is about ±0.02–0.03 AUC. The complexity stops paying at
     the second rung.
3. **Where it does not, one sophistication pays: in-domain weak supervision.**
   - Fine-tuning the bi-encoder on 7,548 metadata-labelled Djinni pairs lifts it from
     0.774 to **0.854**. That is +0.057 [0.009, 0.106] over TF-IDF, resolved in total and
     positive in both strata.
   - It is the choice for uploaded CVs with no category field.
4. **The cross-encoder, the proposal's Stage 4, is harmful as trained here.**
   - Trained on A1 (labels that do not reproduce), it scores 0.488, below chance on
     targeted.
   - Trained on weak pairs, it scores 0.661, below TF-IDF.
   - Re-ranking a good shortlist with either cuts Recall@10 by 0.25–0.33.

   This speaks to these training signals and a CPU budget (one to three epochs), not to
   cross-encoders in general.
5. **The public benchmark would have told a different story.**
   - On A1 the in-distribution A1 cross-encoder reaches 0.628 per-JD AUC, but TF-IDF
     (0.649) and zero-shot SBERT (0.678) still beat it.
   - The weak-trained cross-encoder falls to 0.447 there.
   - A1 measures agreement with A1's own labelling (A13), which is why it is secondary
     (D40).

The selected pipeline (D48): **metadata + TF-IDF when both sides carry a role category;
the weakly fine-tuned SBERT otherwise**, with the metadata rule's fit call. See
[07](07-selection.md).

## 2. The extraction component (a separate experiment)

**D49 splits the job by role:**
- The learned tagger is clearly better on job postings: SkillSpan union strict F1
  **0.523** vs 0.237, partial 0.837 vs 0.305.
- The rules matcher is better on resumes: DataTurks partial F1 **0.380** vs 0.224.
- So the tagger adds free-text JD requirements, and the rules matcher does every CV-side
  exact match.
- **Normalisation:** SBERT beats string matching, 0.766 vs 0.602 top-1 on held-out ESCO
  paraphrases.

See [08](08-extraction.md).

## 3. Success measures (04 §2; S0–S4 provisional, registered in 06 §5)

| Proposal target | Restated | Result | Provisional bar | Pass / miss |
|---|---|---|---|---|
| ≥ 85% agreement with human judgement | Accuracy of the fit call vs the final set | metadata rule 0.768 (floor 0.612) | S1: ≥ 0.85 | **miss** |
| Fit-classification macro-F1 > 85% | Binary macro-F1 vs the final set | metadata rule 0.721 | S1: ≥ 0.85 | **miss** |
| Retrieval Recall@10 > 80% (classical) | Recall@50 of fits, 500-candidate pools | metadata + TF-IDF **0.929** [0.882, 0.966] | S2: ≥ 0.80 | **pass** |
| Retrieval Recall@10 > 90% (advanced) | same | metadata + SBERT weak **0.936** [0.903, 0.966] | S2: ≥ 0.90 | **pass** (CI above the bar) |
| Extraction F1 > 85% | Strict span F1, SkillSpan test | tagger 0.523, rules 0.237 | S3: ≥ 0.85 | **miss** |
| Screening time 25 → 2 min per candidate | System latency | **1.7 ms per candidate**; 0.84 s per 500-candidate role | S4: report against the 2-min budget | within budget. Recruiter time is **not measured** |
| 500+ applications per role | Throughput | 500-candidate roles screened in 0.84 s | unchanged | **pass** |
| Performance gaps across groups | Per-JD AUC by JD experience band and stratum | 0.868 / 0.820 / 0.856 (0–1 / 2–3 / 4–6 yrs); 0.848 / 0.849 (generic / targeted). All CIs overlap | no target | no resolved gap (no demographic fields exist) |
| Stable core intents (UI) | One scenario test per intent | 6 intents, 17 tests, plus an `AppTest` smoke test | pass/fail gate | **pass** |
| Inter-annotator agreement | reported only (D44) | `yc`–`van` 0.672, kappa 0.353 (n=326) | no target | reported |

**Reading the misses:**
- The agreement and macro-F1 bars sit above what the annotators reach with each other on
  Fit / No Fit: 0.672 agreement, and kappa 0.353.
- Extraction's bar sits above the task's measured difficulty.
- Both were set before the data were profiled. They are kept, per S1 (a), so the proposal's
  numbers stay visible.

## 4. Limitations

- **There is no ground truth.**
  - The reference is a documented panel rule over two humans and an LLM tie-breaker.
  - On `generic`, 274 of 400 labels are one annotator's (`yc`).
  - The LLM decides 107 of the 326 double-labelled pairs.
- **Resolution.** There are 60 JDs (40 + 20); rung-to-rung deltas below about 0.03 AUC do
  not resolve.
- **The pools are designed, not natural.**
  - Each JD's 10 candidates are 8 same-family plus 2 near misses (D31), so metadata is
    strong by construction.
  - The within-family metric is the check. Text rungs matter more there (TF-IDF 0.724,
    SBERT 0.755, vs 0.540 for metadata).
- **Unjudged distractors.** Pilot-pool recall treats the 490 distractors as non-fits, which
  gives upper-bound recall.
- **Djinni** is one labour market (D11), with pre-anonymised, short CVs (751 characters
  median).
- **A1** labels do not reproduce (A13), so it is a training resource and a secondary
  benchmark only.
- **The CPU budget:**
  - cross-encoders trained for one to three epochs;
  - the tagger stopped at 6 epochs without converging;
  - hyperparameters were chosen from a two-point grid on the models' own validation
    splits.
- **Exploratory rungs:**
  - the metadata, TF-IDF, BM25, zero-shot SBERT configurations and the metadata hybrids
    were seen before registration;
  - the selected metadata + TF-IDF is among them, so its figure is not a blind test;
  - every held-out rung was compared against it, and none beat it.
- **Semantic evidence threshold.** τ = 0.28 was fixed on SkillSpan phrase-vs-sentence
  pairs, where positives contain the phrase literally, so it may be lenient for paraphrase
  support.
