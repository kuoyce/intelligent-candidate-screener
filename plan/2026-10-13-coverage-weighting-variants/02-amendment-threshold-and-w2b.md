# Amendment 1: embedding-threshold search (Part C) and a repaired W2 (W2b)

Registered 13 Oct 2026 at the owner's request, **before** the runs below. Exploratory (D56).
A2 final is not read by anything in this amendment (D46); the 0.7 / 0.8 / 0.9 values seen in
Part B on A2 are not used to choose.

## Part C: which embedding threshold?

Part B tried 0.7, 0.8 and 0.9 and the best AUC was at the lowest, so the optimum was not located.

- **Keys:** K3 (concept-level embedding match, `all-MiniLM-L6-v2`, unweighted matched fraction
  of the JD's concepts) at thresholds 0.50, 0.55, ..., 0.90. K0, K1 and K2 are shown as references.
  Arm R+N only.
- **Metric:** coverage-only per-JD AUC, paired delta against K0 with a JD-bootstrap CI.
- **Sets:** the 70 D50 selection JDs choose; the 30 D50 held-out JDs and the 639 A1-LLM
  pairs check. Same split as Part A (seed 0). A2 final is not computed.
- **Pick (guidance; the owner decides):** the threshold with the highest selection AUC; ties
  within 0.002 go to the **higher** (stricter) threshold. If the maximum sits on the grid edge
  (0.50) it is reported as "not located", not as an optimum. A pick that is not above K0 on
  both held-out sets is flagged as a D50-specific win.
- 9 thresholds means 9 comparisons; the curve is reported whole and the pick is a guide.

## Why W2 was repaired, and how

Part A's W2 kept the 10 heaviest JD concepts and broke ties alphabetically. Most unknown
phrases have df 0 and hit the weight cap of 5, so "top 10" was an arbitrary alphabetical slice
(R+N, D50 selection: -0.124 against W0). The finding stands as recorded for W2.

- **W2b:** top 10 by weight; ties broken by **how often the concept is mentioned in the JD**
  (more mentions first), then lexically. A mention is an unqualified SKILL or KNOWLEDGE row from
  the concept source (seed rules: an unqualified match). Mention counts are exposed by
  `ConceptExtractor.mentions` and summed over sources in a union.
- **A second blind spot, now covered.** Per-JD AUC does not change when a score is divided by a
  constant that depends only on the JD, so coverage-only per-JD AUC cannot see a dilution fix:
  it only sees which concepts are counted. But inside the learned score, the denominator sets how
  large the coverage term is against role and TF-IDF in that JD. So every variant (W0-W3, W2b) is
  also scored **in the score**:
  - fit a logistic regression on the base four features (class-balanced, as in the earlier
    plans) on the 70 D50 selection JDs' pairs, with the coverage column replaced by the
    variant's;
  - role, experience and TF-IDF features are fixed (they do not depend on the concept source);
  - score per-JD AUC on the 30 D50 held-out JDs and the 639 A1-LLM pairs (A1-LLM is not in the
    fit); paired delta against W0 with a JD-bootstrap CI.
- **Rule (unchanged in kind):** a variant is a candidate only if it is >= +0.005 over W0 on the
  coverage-only selection AUC **or** on the in-score held-out D50 AUC, and positive on the other
  held-out sets. The owner decides. Nothing reaches A2 unless frozen here first.

## Cost

Part C is minutes (embeddings for the D50 and A1-LLM phrases only). The in-score check needs one
scorer build for the role / TF-IDF / experience features (about 3 min), reused for every variant.
