# 05 — Each annotator vs each system (evidence for Q53 and Q57)

> **Decisions taken after this was shown (26 Sep 2026).** Fit / No Fit only (D47). The
> final-set rule is: humans agree → theirs; split → the LLM; one human → that human; no
> human → the LLM. Q57 is answered: the human stands. Inter-annotator agreement is reported
> with no target.
>
> The three-class tables below are kept as the record of what informed those decisions.
> They are not the protocol. The binary figures that matter now are in
> [02 §3.2 and §3.6](02-design-spec.md) and [04 §2](04-success-measures.md).

**What this is.** The systems scored against **each annotator separately** (`yc`, `van`,
the LLM) and against the final set, for `generic`, `targeted` and the total. The annotators
are also scored against each other; that is the only "human ceiling" this data has.

It was produced **to inform Q53 (success measures) and Q57 (single-human pairs)**. It is
not the evaluation protocol: systems are still scored against the final set alone (D41).
This is a scratch probe: data on file, no new LLM calls, nothing committed. P1 recomputes
anything the plan keeps.

**How to read the tables:**
- **Cells.** `generic / targeted / total`. Total means all JDs pooled, each JD weighted
  equally, so generic's 40 JDs count twice as much as targeted's 20 (Q56).
- **Coverage.**
  - `yc` and the LLM labelled all 600 pairs. `van` labelled 326: 126 generic and all 200
    targeted.
  - A figure "vs van" is therefore over fewer pairs.
  - Annotator-vs-annotator figures exist only where both labelled.
- **† = inflated by construction.** The annotator helped build the final set, so its
  agreement with the set partly measures itself. The fair human figure is annotator vs
  annotator.
- **Systems.**
  - *keyword match* is the `Primary Keyword` metadata field.
  - *TF-IDF* and *SBERT* are unsupervised text similarity.
  - *A1 baseline* is the committed TF-IDF + logistic regression, trained on A1 only.
  - *Metadata rule* is a fixed rule: keyword and experience both match → `Good`; keyword
    only → `Potential`; otherwise `No`.
  - None was fitted on A2.
- **LLM.** The committed instrument (`5d986c82`, haiku, runs 9/10).

## What the tables say

### For Q53 (success measures)

1. **Ranking: the cheap systems already rank candidates at or above human level against
   every annotator** (T1). Keyword + TF-IDF / SBERT reach per-JD AUC (total):
   - vs `yc`: **0.87 / 0.87**, where `van` reaches 0.74 and the LLM 0.78;
   - vs `van`: **0.74 / 0.77**, where `yc` reaches 0.72 and the LLM 0.73;
   - vs the LLM: **0.87 / 0.87**, where `yc` reaches 0.78 and `van` 0.71.

   Any target defined as "as good as a human" is **already met by the cheapest rung**. It
   says the system is usable; it cannot separate rungs.
2. **Classification: the cheap class-emitting systems sit at the human–human level**
   (T2-shared, total).

   | | three-class agreement | macro-F1 | kappa | binary agreement |
   |---|---|---|---|---|
   | `yc` vs `van` (the human ceiling) | 0.48 | 0.47 | 0.24 | 0.67 |
   | A1 baseline vs `yc` / vs `van` | 0.55 / 0.46 | 0.46 / 0.41 | 0.30 / 0.19 | 0.73 / 0.62 |
   | metadata rule vs `yc` / vs `van` | 0.44 / 0.48 | 0.43 / 0.46 | 0.22 / 0.20 | 0.66 / 0.76 |

   Against the final set, the two cheap systems reach 0.43–0.49 three-class, 0.70–0.77
   binary and 0.43–0.47 macro-F1.
3. **Nothing, human or machine, reaches 85% three-class agreement with any single
   annotator.**
   - The best pair on the whole table is the LLM vs `yc` on generic, at 0.65.
   - Binary agreement reaches 0.80 only for the LLM vs `yc` on generic, and 0.87 only for
     the metadata rule vs `van` on generic (126 pairs). In both cases the targeted figure
     falls to 0.67–0.70.
   - **The proposal's 85% agreement / 85% macro-F1 targets are out of reach for any
     rater.**
4. **`van` is the hardest annotator to predict, for everyone**, and `targeted` is harder
   than `generic`.
   - Against `van` on targeted, even the best system is at 0.66–0.70, and `yc` predicts
     `van` at 0.66.
   - This is the annotator difference the final set exists to absorb.
5. **The systems' ordering depends on which annotator you ask** (T1, targeted):
   - TF-IDF 0.81 vs SBERT 0.79 against `yc`;
   - TF-IDF 0.59 vs SBERT 0.66 against `van`.

   The final set gives one ordering: 0.78 vs 0.75 on targeted, and 0.77 vs 0.77 in total.
6. **One number sits suspiciously close to the proposal's.** The cheap hybrids score
   **0.85 / 0.86 AUC against the final set** (total, CI about ±0.04). Adopting "AUC ≥ 0.85"
   as *the* target now would mean choosing the metric after seeing that it lands on the
   line, and this file records that it was seen first.

**What that means for the options in [04](04-success-measures.md):**

| Option | On this evidence |
|---|---|
| **A (verbatim)** | Every agreement and macro-F1 target is missed by every system, and would also be missed by either human |
| **C (85% of the human ceiling)** | The floor would be about 0.40 macro-F1, about 0.57 binary agreement and "≥ a human" on ranking. **All are already cleared by rung 1–2.** It answers "is the system at human level?" (yes, from the start), and it does not choose between rungs |
| **B (relative)** | The only question still open: does anything beat keyword + TF-IDF at 0.85 [0.81, 0.89]? That is what the §5.4 rule tests |
| **D (mixed)** | In practice B decides, C is reported once ("human level reached at rung X"), and A goes in an appendix with this table as the reason |

### For Q57 (single-human pairs where the LLM disagrees)

**The 107 contested pairs are all `yc` + LLM on generic.**

| `yc` says | LLM says | Pairs |
|---|---|---|
| `Good Fit` | `Potential Fit` | 45 (the LLM's known `Good Fit` reluctance) |
| `No Fit` | `Potential Fit` | 29 |
| `Potential Fit` | `No Fit` | 23 |
| `Potential Fit` | `Good Fit` | 7 |
| two classes apart | | 3 |

**The effect of each option on the final set:**
- **Two-step moderation** changes **3 labels**. It is immaterial, and the systems' figures
  do not move (T3).
- **LLM decides** changes all 107. Every keyword-based system then looks **better** on
  generic, with keyword + TF-IDF going from 0.86 to 0.91 (T3), and no ordering changes.
  - That is not the systems improving. The reference is moving toward the LLM, which
    agrees with keyword/lexical systems more than `yc` does (T1: 0.87–0.90 vs the LLM).
  - A rule that flatters every system in the same direction would bias the absolute
    numbers Q53 is about.

**Simulation** (T4). Take the 326 pairs where all three labelled, hide one human, and
apply each rule to the remaining human and the LLM. The measure that does not favour the
LLM is agreement with the **hidden human**. Agreement with the full panel favours the
LLM, because the panel's median contains its vote.

- **The real situation is `yc` kept and `van` missing.** Here "human stands" matches the
  hidden `van` as well as or better than "LLM decides":
  - generic 0.48 vs 0.40;
  - targeted 0.48 vs 0.47;
  - total 0.48 vs 0.44;
  - on contested pairs alone, generic 0.48 vs **0.25**.

  When `yc` and the LLM disagree, `van` sides with `yc` more often.
- **The mirror case (`van` kept, `yc` missing)** favours letting the LLM decide (0.58 vs
  0.48 total), because `van` is the idiosyncratic rater. **That case does not occur in
  the data**: every single-human pair is `yc`'s.

**Reading:** on this evidence the default, "human stands", is the better-supported rule
for the pairs that actually exist, and two-step moderation makes no practical difference.
The decision is still yours.

---

### T1 — Ranking: per-JD AUC, fit vs No Fit, each pair's full coverage (cells: generic / targeted / total)

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| random | 0.52 / 0.51 / 0.52 | 0.60 / 0.51 / 0.55 | 0.54 / 0.53 / 0.54 | 0.53 / 0.54 / 0.54 |
| keyword match | 0.79 / 0.69 / 0.75 | 0.77 / 0.65 / 0.71 | 0.81 / 0.74 / 0.79 | 0.79 / 0.73 / 0.77 |
| TF-IDF | 0.78 / 0.81 / 0.79 | 0.71 / 0.59 / 0.65 | 0.77 / 0.76 / 0.77 | 0.76 / 0.78 / 0.77 |
| SBERT (zero-shot) | 0.79 / 0.79 / 0.79 | 0.81 / 0.66 / 0.73 | 0.76 / 0.74 / 0.76 | 0.79 / 0.75 / 0.77 |
| keyword + TF-IDF | 0.88 / 0.86 / 0.87 | 0.82 / 0.66 / 0.74 | 0.89 / 0.83 / 0.87 | 0.86 / 0.84 / 0.85 |
| keyword + SBERT | 0.89 / 0.83 / 0.87 | 0.85 / 0.70 / 0.77 | 0.90 / 0.81 / 0.87 | 0.88 / 0.82 / 0.86 |
| A1 baseline score (TF-IDF, A1 vocab) | 0.79 / 0.83 / 0.80 | 0.77 / 0.61 / 0.69 | 0.79 / 0.76 / 0.78 | 0.78 / 0.77 / 0.77 |
| annotator: yc | *(self)* | 0.79 / 0.66 / 0.72 | 0.82 / 0.72 / 0.78 | 0.96 / 0.82 / 0.91 † |
| annotator: van | 0.79 / 0.69 / 0.74 | *(self)* | 0.72 / 0.69 / 0.71 | 0.81 / 0.83 / 0.82 † |
| annotator: llm | 0.80 / 0.73 / 0.78 | 0.79 / 0.67 / 0.73 | *(self)* | 0.82 / 0.87 / 0.83 † |
| *JDs scored* | 38 / 18 / 56 | 19 / 20 / 39 | 37 / 20 / 57 | 39 / 20 / 59 |

### T1-CI — the same, with 95% CIs (bootstrap over JDs), total column only

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| random | 0.52 [0.46, 0.57] | 0.55 [0.46, 0.65] | 0.54 [0.48, 0.59] | 0.54 [0.48, 0.59] |
| keyword match | 0.75 [0.71, 0.80] | 0.71 [0.63, 0.78] | 0.79 [0.74, 0.83] | 0.77 [0.72, 0.80] |
| TF-IDF | 0.79 [0.75, 0.84] | 0.65 [0.56, 0.74] | 0.77 [0.72, 0.82] | 0.77 [0.72, 0.82] |
| SBERT (zero-shot) | 0.79 [0.74, 0.83] | 0.73 [0.65, 0.81] | 0.76 [0.70, 0.81] | 0.77 [0.72, 0.82] |
| keyword + TF-IDF | 0.87 [0.83, 0.91] | 0.74 [0.64, 0.82] | 0.87 [0.82, 0.91] | 0.85 [0.81, 0.89] |
| keyword + SBERT | 0.87 [0.82, 0.91] | 0.77 [0.68, 0.85] | 0.87 [0.82, 0.91] | 0.86 [0.82, 0.90] |
| A1 baseline score (TF-IDF, A1 vocab) | 0.80 [0.75, 0.85] | 0.69 [0.59, 0.78] | 0.78 [0.73, 0.83] | 0.77 [0.72, 0.83] |
| yc | *(self)* | 0.72 [0.66, 0.79] | 0.78 [0.74, 0.83] | 0.91 [0.87, 0.95] |
| van | 0.74 [0.66, 0.82] | *(self)* | 0.71 [0.62, 0.79] | 0.82 [0.73, 0.89] |
| llm | 0.78 [0.73, 0.82] | 0.73 [0.66, 0.80] | *(self)* | 0.83 [0.80, 0.87] |

### T2-shared — Classification on the 326 pairs both humans labelled (generic 126 / targeted 200) (cells: generic / targeted / total)

**three-class agreement**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.51 / 0.57 / 0.55 | 0.52 / 0.42 / 0.46 | 0.36 / 0.40 / 0.38 | 0.45 / 0.47 / 0.46 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.51 / 0.40 / 0.44 | 0.59 / 0.41 / 0.48 | 0.35 / 0.36 / 0.36 | 0.48 / 0.39 / 0.43 |
| annotator: yc | *(self)* | 0.48 / 0.48 / 0.48 | 0.65 / 0.53 / 0.58 | 0.84 / 0.73 / 0.77 † |
| annotator: van | 0.48 / 0.48 / 0.48 | *(self)* | 0.40 / 0.47 / 0.44 | 0.57 / 0.66 / 0.62 † |
| annotator: llm | 0.65 / 0.53 / 0.58 | 0.40 / 0.47 / 0.44 | *(self)* | 0.80 / 0.80 / 0.80 † |

**binary agreement (fit vs No Fit)**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.71 / 0.74 / 0.73 | 0.66 / 0.60 / 0.62 | 0.66 / 0.66 / 0.66 | 0.69 / 0.70 / 0.70 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.76 / 0.59 / 0.66 | 0.87 / 0.70 / 0.76 | 0.75 / 0.74 / 0.75 | 0.82 / 0.73 / 0.77 |
| annotator: yc | *(self)* | 0.74 / 0.63 / 0.67 | 0.80 / 0.67 / 0.72 | 0.90 / 0.81 / 0.84 † |
| annotator: van | 0.74 / 0.63 / 0.67 | *(self)* | 0.75 / 0.69 / 0.71 | 0.84 / 0.82 / 0.83 † |
| annotator: llm | 0.80 / 0.67 / 0.72 | 0.75 / 0.69 / 0.71 | *(self)* | 0.90 / 0.86 / 0.88 † |

**macro-F1 (reference = column)**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.46 / 0.46 / 0.46 | 0.46 / 0.37 / 0.41 | 0.32 / 0.36 / 0.34 | 0.43 / 0.42 / 0.43 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.51 / 0.38 / 0.43 | 0.56 / 0.40 / 0.46 | 0.36 / 0.37 / 0.37 | 0.50 / 0.40 / 0.44 |
| annotator: yc | *(self)* | 0.48 / 0.45 / 0.47 | 0.59 / 0.49 / 0.53 | 0.84 / 0.72 / 0.77 † |
| annotator: van | 0.48 / 0.45 / 0.47 | *(self)* | 0.40 / 0.44 / 0.42 | 0.58 / 0.63 / 0.62 † |
| annotator: llm | 0.59 / 0.49 / 0.53 | 0.40 / 0.44 / 0.42 | *(self)* | 0.74 / 0.76 / 0.75 † |

**Cohen's kappa**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.27 / 0.31 / 0.30 | 0.26 / 0.14 / 0.19 | 0.12 / 0.17 / 0.15 | 0.23 / 0.24 / 0.24 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.31 / 0.17 / 0.22 | 0.30 / 0.14 / 0.20 | 0.16 / 0.16 / 0.16 | 0.28 / 0.18 / 0.22 |
| annotator: yc | *(self)* | 0.25 / 0.21 / 0.24 | 0.45 / 0.25 / 0.32 | 0.76 / 0.57 / 0.64 † |
| annotator: van | 0.25 / 0.21 / 0.24 | *(self)* | 0.18 / 0.19 / 0.18 | 0.38 / 0.47 / 0.44 † |
| annotator: llm | 0.45 / 0.25 / 0.32 | 0.18 / 0.19 / 0.18 | *(self)* | 0.67 / 0.65 / 0.66 † |

*pairs:* yc: 126 / 200 / 326; van: 126 / 200 / 326; llm: 126 / 200 / 326; final: 126 / 200 / 326


### T2-full — Classification on each reference's full coverage (cells: generic / targeted / total)

**three-class agreement**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.52 / 0.57 / 0.54 | 0.52 / 0.42 / 0.46 | 0.38 / 0.40 / 0.39 | 0.50 / 0.47 / 0.49 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.52 / 0.40 / 0.47 | 0.59 / 0.41 / 0.48 | 0.39 / 0.36 / 0.38 | 0.51 / 0.39 / 0.47 |
| annotator: yc | *(self)* | 0.48 / 0.48 / 0.48 | 0.62 / 0.53 / 0.59 | 0.95 / 0.73 / 0.88 † |
| annotator: van | 0.48 / 0.48 / 0.48 | *(self)* | 0.40 / 0.47 / 0.44 | 0.57 / 0.66 / 0.62 † |
| annotator: llm | 0.62 / 0.53 / 0.59 | 0.40 / 0.47 / 0.44 | *(self)* | 0.67 / 0.80 / 0.71 † |

**binary agreement (fit vs No Fit)**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.73 / 0.74 / 0.73 | 0.66 / 0.60 / 0.62 | 0.71 / 0.66 / 0.69 | 0.72 / 0.70 / 0.72 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.77 / 0.59 / 0.71 | 0.87 / 0.70 / 0.76 | 0.78 / 0.74 / 0.77 | 0.79 / 0.73 / 0.77 |
| annotator: yc | *(self)* | 0.74 / 0.63 / 0.67 | 0.80 / 0.67 / 0.76 | 0.97 / 0.81 / 0.91 † |
| annotator: van | 0.74 / 0.63 / 0.67 | *(self)* | 0.75 / 0.69 / 0.71 | 0.84 / 0.82 / 0.83 † |
| annotator: llm | 0.80 / 0.67 / 0.76 | 0.75 / 0.69 / 0.71 | *(self)* | 0.83 / 0.86 / 0.84 † |

**macro-F1 (reference = column)**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.45 / 0.46 / 0.46 | 0.46 / 0.37 / 0.41 | 0.35 / 0.36 / 0.35 | 0.44 / 0.42 / 0.44 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.51 / 0.38 / 0.47 | 0.56 / 0.40 / 0.46 | 0.40 / 0.37 / 0.39 | 0.51 / 0.40 / 0.47 |
| annotator: yc | *(self)* | 0.48 / 0.45 / 0.47 | 0.59 / 0.49 / 0.56 | 0.95 / 0.72 / 0.88 † |
| annotator: van | 0.48 / 0.45 / 0.47 | *(self)* | 0.40 / 0.44 / 0.42 | 0.58 / 0.63 / 0.62 † |
| annotator: llm | 0.59 / 0.49 / 0.56 | 0.40 / 0.44 / 0.42 | *(self)* | 0.64 / 0.76 / 0.68 † |

**Cohen's kappa**

| Predictor | vs yc | vs van | vs llm | vs final set |
|---|---|---|---|---|
| A1 baseline (TF-IDF+LR, trained on A1) | 0.28 / 0.31 / 0.30 | 0.26 / 0.14 / 0.19 | 0.17 / 0.17 / 0.17 | 0.27 / 0.24 / 0.27 |
| metadata rule (kw & exp -> Good, kw -> Potential, else No) | 0.31 / 0.17 / 0.26 | 0.30 / 0.14 / 0.20 | 0.19 / 0.16 / 0.18 | 0.30 / 0.18 / 0.26 |
| annotator: yc | *(self)* | 0.25 / 0.21 / 0.24 | 0.41 / 0.25 / 0.36 | 0.92 / 0.57 / 0.81 † |
| annotator: van | 0.25 / 0.21 / 0.24 | *(self)* | 0.18 / 0.19 / 0.18 | 0.38 / 0.47 / 0.44 † |
| annotator: llm | 0.41 / 0.25 / 0.36 | 0.18 / 0.19 / 0.18 | *(self)* | 0.48 / 0.65 / 0.54 † |

*pairs:* yc: 400 / 200 / 600; van: 126 / 200 / 326; llm: 400 / 200 / 600; final: 400 / 200 / 600


### T3 — Q57: system AUC under each single-human rule (full coverage; generic / targeted / total)

| System | human stands (default) | two-step moderation | LLM decides |
|---|---|---|---|
| random | 0.53 / 0.54 / 0.54 | 0.54 / 0.54 / 0.54 | 0.53 / 0.54 / 0.53 |
| keyword match | 0.79 / 0.73 / 0.77 | 0.79 / 0.73 / 0.77 | 0.83 / 0.73 / 0.79 |
| TF-IDF | 0.76 / 0.78 / 0.77 | 0.76 / 0.78 / 0.77 | 0.78 / 0.78 / 0.78 |
| SBERT (zero-shot) | 0.79 / 0.75 / 0.77 | 0.79 / 0.75 / 0.77 | 0.78 / 0.75 / 0.77 |
| keyword + TF-IDF | 0.86 / 0.84 / 0.85 | 0.86 / 0.84 / 0.85 | 0.91 / 0.84 / 0.89 |
| keyword + SBERT | 0.88 / 0.82 / 0.86 | 0.89 / 0.82 / 0.86 | 0.92 / 0.82 / 0.88 |
| A1 baseline score (TF-IDF, A1 vocab) | 0.78 / 0.77 / 0.77 | 0.78 / 0.77 / 0.78 | 0.79 / 0.77 / 0.78 |

### T4 — Q57 simulation: hide one human on the 326 triple-labelled pairs, apply each rule to the remaining human + LLM

| Kept human (hidden) | Stratum | Rule | Agrees with full panel | …on contested pairs | Agrees with hidden human | …on contested pairs | contested n |
|---|---|---|---|---|---|---|---|
| yc (van hidden) | generic | human stands | 0.84 | 0.55 | 0.48 | 0.48 | 44 |
| yc (van hidden) | generic | two step moderation | 0.83 | 0.50 | 0.46 | 0.43 | 44 |
| yc (van hidden) | generic | llm decides | 0.80 | 0.43 | 0.40 | 0.25 | 44 |
| van (yc hidden) | generic | human stands | 0.57 | 0.29 | 0.48 | 0.28 | 76 |
| van (yc hidden) | generic | two step moderation | 0.57 | 0.29 | 0.48 | 0.28 | 76 |
| van (yc hidden) | generic | llm decides | 0.80 | 0.67 | 0.65 | 0.57 | 76 |
| yc (van hidden) | targeted | human stands | 0.73 | 0.43 | 0.48 | 0.38 | 94 |
| yc (van hidden) | targeted | two step moderation | 0.72 | 0.41 | 0.48 | 0.37 | 94 |
| yc (van hidden) | targeted | llm decides | 0.80 | 0.56 | 0.47 | 0.35 | 94 |
| van (yc hidden) | targeted | human stands | 0.66 | 0.35 | 0.48 | 0.34 | 106 |
| van (yc hidden) | targeted | two step moderation | 0.67 | 0.37 | 0.49 | 0.36 | 106 |
| van (yc hidden) | targeted | llm decides | 0.80 | 0.61 | 0.53 | 0.42 | 106 |
| yc (van hidden) | total | human stands | 0.77 | 0.46 | 0.48 | 0.41 | 138 |
| yc (van hidden) | total | two step moderation | 0.76 | 0.44 | 0.47 | 0.39 | 138 |
| yc (van hidden) | total | llm decides | 0.80 | 0.52 | 0.44 | 0.32 | 138 |
| van (yc hidden) | total | human stands | 0.62 | 0.32 | 0.48 | 0.31 | 182 |
| van (yc hidden) | total | two step moderation | 0.63 | 0.34 | 0.49 | 0.32 | 182 |
| van (yc hidden) | total | llm decides | 0.80 | 0.64 | 0.58 | 0.48 | 182 |
