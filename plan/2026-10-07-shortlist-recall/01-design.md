# 01 — Design

## 0. Answer to decision 2: what the current role score is

P7's `role_match` is **classification on both sides**: each document is classified into one
of 60 role families, and the score is the overlap of the two probability vectors
(`Σ p_jd p_cv`). Title spans are found in the text but only feed the classifier as
features. So by decision 2 a **hard** match is allowed on the family, and **fuzzy** scoring
only where we compare extracted title strings.

What changes in this plan:

| Side | P7 | This plan |
|---|---|---|
| JD | text → classifier → family probabilities | **title given** → family (lookup, then a title classifier where the lookup misses) |
| CV | text → classifier → family probabilities | **set** of families: every title span found in the CV, plus the classifier's top-*k* |
| Match | soft overlap, added to text (text can overtake it) | **hard**: JD family ∈ CV set. A level, as 1a: text orders candidates within it, never across it |

## 1. Development set (answers the "too little data" concern)

Nothing here is tuned on the final set (D46). Every choice is made on a development set
built from the **A2 train region**, from documents the role classifier never trained on:

- **Documents:** the P3 classifier's **validation part** (10,642 JDs, 15,764 CVs; never in
  its fit part), minus the 1,000-document calibration draw per side, minus every document
  in `weak-pairs.csv` (SBERT weak trained on those).
- **Pools:** 400 JDs with a non-empty `Position`, each with a 500-CV pool built like the
  pilot's: 8 CVs of the JD's keyword and 2 of another (the `generic` design), plus 490
  random CVs. Split by JD: **dev-train** 250 (fitting, decision 6) and **dev-select** 150
  (choosing among fixed options).
- **Relevance (a proxy):** a CV is relevant if its **oracle role family equals the JD's**
  (`role_family(Primary Keyword, Position)`, P3's label). This measures agreement with the
  metadata, not fit; the final set is the only test of fit. It is the right proxy for
  stage 1, whose job is to do what the keyword did (96.5% of final-set fits share the JD's
  keyword, 16 §6a).
- **Volume:** 250 × 500 = 125,000 labelled pairs for decision 6, against 600 in the final
  set. Labels are metadata, used as training labels (allowed use 1, 02-design-spec §1).
- **Metric on dev:** proxy Recall@50 and Recall@10 per JD, mean over JDs; X2 and TF-IDF
  scored on the same pools as the reference.

## 2. Components

### C1. JD title → family (decision 1)

1. Normalise the title (`title_profile.normalise_title`) and look it up in P3's alias table
   (518 entries; covers 61% of validation JDs, right 90% where it covers).
2. Where the lookup misses: a **title classifier** (hashed word and character n-grams of
   the title → family, logistic regression), trained on the fit part's JD and CV `Position`
   strings with their oracle families.
3. Output: one family, with its probability. Measured against the oracle on dev JDs and on
   the 60 final-set JDs (measurement only).

The JD long description still feeds TF-IDF and SBERT as before.

### C2. CV role set (decision 1b)

`roles(cv) = {families of all title spans in the CV} ∪ {classifier top-k families}`.

- Spans: P3's `title_spans` against the alias table, every span, not just the first.
- Option, fixed in advance, chosen on dev-select: count only spans in a **role context**
  (a line that starts with the title, or a "position / role / title:" cue, or a
  work-history header line), to drop "worked with the QA team".
- *k* ∈ {0, 1, 2, 3}; chosen on dev-select.
- Measured: oracle family ∈ roles(cv) (recall) and mean set size (looseness), on dev CVs.

### C3. Matching

- `family_match = 1 if C1(jd) ∈ C2(cv) else 0` (categorical, hard).
- `title_similarity`: max over the CV's title spans of character 3-gram cosine to the JD
  title (extracted text, fuzzy; decision 2). 0 when the CV has no span.

### C4. Main technology and weighted skills (decision 3)

- **JD main tech:** vocabulary skills in the JD title first; otherwise the skill named most
  often in the JD's requirement lines (P4's section and optional-line rules), ties to the
  earliest.
- **CV main techs:** skills weighted by frequency, by presence in `Looking For` /
  `Highlights` / the first role block, top 3.
- `main_tech_match` (hard: the JD's main tech ∈ the CV's main techs) and `weighted_skill`
  (JD skills weighted by IDF over train JDs, must-have 1, nice-to-have 0.5; share the CV
  covers).
- Checked first on dev: JD main tech vs `Primary Keyword` for non-generic keywords (they are
  mostly technologies). If that agreement is low, C4 stops at the measurement.

### C5. Related roles: one test (decision 4)

A family similarity `S(a, b)`: how often a train-fit CV whose title spans name family *a*
also names *b* (normalised), `S(a, a) = 1`. Variant: `family_match = max S(jd_fam, c)`
over the CV set, thresholded at one value fixed in advance (0.3). **One dev run.** Continue
only if dev-select Recall@50 rises by ≥ 0.02 over the hard match; otherwise record and stop.

### C6. A better role classifier (decision 7)

For the CV top-*k* in C2 and C1's fallback context. Candidates, same split and selection
rule as P3 (mean JD/CV macro-F1 on the select draw):

- (a) P3's model (incumbent);
- (b) JD boilerplate stripped before classifying (company / benefits / "we offer"
  sections, by header rules);
- (c) (b) plus title-span indicators weighted by role context (C2).

Swapped in only if it beats the incumbent by ≥ 0.01 macro-F1. The fit/validation split is
unchanged, so the dev set stays unseen.

### C7. Learned combination (decision 6)

Logistic regression over features from C1–C4: `family_match`, `title_similarity`,
`main_tech_match`, `weighted_skill`, `experience_compatible` (A1), per-JD min-max TF-IDF,
per-JD min-max SBERT weak, P7's soft `role_match`. Trained on **dev-train** pools (proxy
label), L2, C chosen from {0.1, 1, 10} on dev-select. The coefficients are recorded.

## 3. Rungs

All text-only, except the JD title (D54). All are `post-hoc`: designed after the P7 S2 miss
was seen, registered (03) before any final-set score.

| Rung | Formula | Group (S2 bar) |
|---|---|---|
| **Y1** | `2 × family_match + experience_compatible` (C1–C3, A1 experience) | classical (0.80) |
| **Y2** | Y1 + TF-IDF, the 1a+2 form (text min-max per JD × 0.999) | classical |
| **Y2t** | Y1 + the mean of per-JD min-max TF-IDF and `title_similarity`, × 0.999 | classical |
| **Y2m** | the better of Y2 / Y2t on dev, with `main_tech_match` in the level: `4 × family_match + 2 × main_tech_match + experience_compatible` (family outranks main tech outranks experience), plus the same text term | classical |
| **Y4** | Y1 + SBERT weak, the 1a+3b form | advanced (0.90) |
| **Y5** | C7, learned | classical if SBERT weak gets weight 0, else advanced |
| Y-rel | C5 variant of the dev-best classical rung; only if C5 passes its dev test | classical |

Fixed by the dev set, before the final set is read: `k`, role-context on/off, Y2 vs Y2t,
whether Y2m is scored (only if C4's check passes), C6's swap, C7's C. One configuration
per rung is frozen into 03 §2 before final scoring.
