# Coverage weighting variants

Status: **approved by the owner 13 Oct 2026; implementation in progress.** Exploratory under D56 (needs a one-line amendment on
approval). Follows [gliner-concepts-in-learned-weights](../2026-10-13-gliner-concepts-in-learned-weights/01-implementation-record.md).

## Question

The MiniLM tagger lifts the coverage feature on its own (R+N coverage-only per-JD AUC 0.718 on
A2 final against 0.693 for the seed rules), but no learned score beat hand-set X2 (0.850). Do
**weighting fixes in the coverage term** let the learned score beat hand-set X2 where the
unweighted version could not?

## Evidence

- 84% of MiniLM's concepts (52% of GLiNER's) are unknown to the vocabulary and carry a flat
  weight of 1.0 in `DocumentFrequency.weight`, so "english" counts as much as "kafka".
- Over the 18,721 train CVs no concept appears in more than 30% of CVs (max `html`, 15-30%).
  Generic words ("english", "testing") sit in the same 4-15% band as real skills (Python, SQL,
  JavaScript), so a frequency stoplist cannot separate them; smooth IDF can.
- 86% of MiniLM's 138k distinct phrases occur in exactly one train CV, 96% in five or fewer. A
  plain IDF would give those the maximum weight (5).
- MiniLM yields about 30 concepts per JD against 6 for the rules; coverage divides by the JD's
  concept count, so it dilutes.

## Selection discipline

I have already seen coverage-only AUC on A2 final, so **A2 final cannot choose a variant.**
Variants are chosen on D50 (A2 train region, D46-allowed), checked on held-out data, frozen
and recorded in this plan, and only then scored once on A2 final.

- D50 has 100 JDs. A deterministic split (seed 0, sorted JD ids): **70 selection JDs, 30
  held-out JDs.**
- Held-out checks: the 30 held-out D50 JDs, and the 639 A1-LLM pairs (different distribution).

## Variants (registered now; the only free numbers, 10 and 5, are fixed here, not tuned)

Weights use the existing `min(5, 1 + ln((N+1)/(df+1)))` with df from the 18,721 train CVs.

| Variant | Coverage term |
|---|---|
| W0 | Current: vocabulary concepts IDF-weighted, unknown phrases flat 1.0 |
| W1 | Unknown phrases also IDF-weighted |
| W2 | W1, and the denominator is the weight of the JD's top 10 concepts by weight (ties by lexical order); matched weight counts only those 10 |
| W3 | W1 with a rarity floor: a phrase seen in fewer than 5 train CVs keeps weight 1.0 |

Arms: **R** (seed rules, control) and **R+N**. G is dropped (failed the Stage 1 gate).
Default behaviour of `DocumentFrequency` stays W0 and is regression-tested to be unchanged.

## Procedure

1. **Select** (per arm): coverage-only per-JD AUC of each variant on the 70 selection JDs. Take
   the highest. Ties within 0.002 go to the simpler variant (order W0, W1, W3, W2).
2. **Report every variant on every set; the owner decides.** For each arm and variant, report
   coverage-only per-JD AUC (paired delta vs W0, JD-bootstrap CI) on the 70 selection JDs, the 30
   held-out D50 JDs and the 639 A1-LLM pairs, side by side. No automatic replace rule: the
   owner picks, seeing all three. Guidance only: a variant that wins on selection but not on
   both held-out sets is a D50-specific win and should not be picked. If no variant helps,
   the answer is "no change": record it and stop. A2 final is never in this table.
3. **Freeze** the variant per arm and write it into this plan before any A2 score.
4. **Stage 2 refits** with the frozen variant, as in the previous plan: notebook 18 Part 2 fits
   (feature set × A / B / C labels) and the A1-LLM and D50 learning curves (10/25/50/70 JDs,
   30 held out, 20 repetitions), against the same hand-set X2.
5. **Score A2 final once.**

## Part B: how much does exact-string matching miss? (diagnostic, owner request)

Independent of Part A, run on the same cached concept sets, R+N only. Measures what an
exact-normalised match throws away; it selects nothing and feeds no model.

| Match key | Rule |
|---|---|
| K0 | Current: casefolded, whitespace-normalised exact string |
| K1 | Alphanumeric-only key plus plural and `.js` / `js` suffix folding (`ci/cd` = `cicd`, `node.js` = `nodejs`, `apis` = `api`) |
| K2 | K1, or one phrase's token set contains the other's (`react` vs `react developer`), both >= 1 token and the shorter has >= 4 characters |
| K3 | Sentence-embedding similarity of the two phrases (`all-MiniLM-L6-v2`, local) at 0.7, 0.8 and 0.9, reported as a curve, not a choice |

Per key, on A2 final, D50 and A1-LLM, label-free: share of pairs with no overlap, concepts
matched per pair, and pairs that gain overlap versus K0. With labels (diagnostic only,
D50 and A1-LLM; A2 labels are read only to report, not to choose): Fit rate of gained pairs
against still-empty pairs, and coverage-only per-JD AUC. A listing of 30 sampled matched
phrase pairs per key (no CV text) so the false-match rate can be read by eye.

Reading: if K1/K2 recover many pairs with a Fit rate near the K0-overlap pairs, exact
matching is a material loss and belongs in a follow-up plan. If they mostly add generic
matches, it is not.

## Bar (unchanged, registered)

- A learner beats hand-set X2 only if its paired total ΔAUC has a JD-bootstrap CI above 0
  **and** `generic` and `targeted` both have a positive delta.
- A feature arm passes for a learner only if the bar is met by its final fit on the A1-LLM
  labels **and** on the D50 labels. With 2 arms × 3 learners × 2 label sources, a single
  nominal pass is not evidence.
- "More A1 LLM labels worth scoping" only if a pass occurs **and** the A1-LLM curve rises by
  more than 0.01 from 25 to 70 JDs. Otherwise no, or inconclusive.
- Reported with `generic`, `targeted`, `total`, n and a JD-bootstrap CI; `llm_free` subset for
  LLM-involved figures.

## Constraints

- No new extraction: the caches (`data/processed/extraction/concept-cache/`) already hold every
  document. No labels are read by code that builds features; labels are read by the scorer only.
- Code in the package (`evaluation/`, `matching/entity_weighted.py`), tests with inline
  synthetic corpora; each regression test is shown to fail against the code it guards. No
  scripts under `plan/`. Branch: stay on `exp/learned-weights-and-error-analysis`.
- D43/D51: covered by D56 (extend its wording on approval). The A2 final set is scored on,
  never fitted on (D46). Everything here is `exploratory`; nothing is adopted.

## Cost

Steps 1-2 are minutes (cached spans). Step 4 is about 3 min per arm (measured last time).
No new tagger inference. Part B's embedding key (K3) encodes the distinct phrases of the three
evaluation sets with a small local sentence model: a few thousand phrases, minutes on CPU.

## Out of scope

Adopting any soft or fuzzy matching in the score (Part B only measures it), learned
per-concept weights, learning-to-rank, and type-split features (the MiniLM tagger has no
KNOWLEDGE / SKILL label).

## Decisions (owner, 13 Oct 2026) and remaining questions

1. **Comparator:** hand-set X2 stays as it is, `2·role + experience + 0.999·TF-IDF` with no
   coverage term (alpha = 0). It is not changed. A coverage term enters only through the
   learned weights. (An earlier plan, `2026-10-11-entity-weighted-matching`, did mix a
   coverage term into the hand formula with a weight `alpha`; that is not part of this plan.)
2. **Floor of 5 and top 10:** accepted.
3. **Selection set:** report D50 selection, D50 held-out and A1-LLM side by side and the owner
   decides (see Procedure step 2).
4. **Exact-string mismatch:** tested in Part B.

Open: none that block approval. Interpretation to confirm: "test both" in answer 3 is read as
"show the D50 and the held-out/A1-LLM results together", not "run two selection procedures".
