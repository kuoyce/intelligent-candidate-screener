# 03 — Registration (shortlist recall)

**§1 is committed with the plan, before any Y rung is built. §2 is filled in and committed
at Q8, after the dev set has chosen every option and before the final set is read.**
`AGENTS.md` evaluation rule 8.

## 1. Fixed now

**Status of every Y rung: `post-hoc`.** They were designed after the P7 S2 miss and its
diagnosis (which read the final-set labels). Their figures are reported beside the
registered P7 result, never in place of it: "P7 registered: miss (0.779). Post-hoc Y: …".

**Reference and text.** The final set (rule v1); CV text `text_v2`; JD text `Long
Description` plus `jd_title` = the A2 JD `Position` (D54). Pools: P7's 500-candidate pools,
unchanged.

**Bars.** S2 unchanged: Recall@50 ≥ 0.80 (classical), ≥ 0.90 (advanced), point estimate;
"CI above the bar" stated separately. Recall@10 reported, no bar.

**Comparisons (paired per-JD deltas, the registration §3 verdict of the running plan,
applied to Recall@50 and to AUC; generic / targeted / total; `all` and `llm_free`):**

| # | B | A | Question |
|---|---|---|---|
| L1 | Y2 | X2 | Do the JD title and a hard role match fix P7's shortlist? |
| L2 | dev-best classical Y | TF-IDF | Against plain text |
| L3 | Y4 | SBERT weak | Advanced group |
| L4 | Y5 | dev-best fixed-formula classical Y | Does learning the weights help? |
| L5 | dev-best Y | oracle 1a+2 | The cost of having no CV metadata (expected ≤ 0; decides nothing) |

**Dev rules (the only place options are chosen):** the highest dev-select proxy Recall@50,
ties to the cheaper option; the early stops in 02 (Q3, Q4, Q7) and C6's 0.01 margin.

**Selection (Q-R1 answered yes, 7 Oct).** A post-hoc Y rung that meets its S2 bar may be
adopted as the stage-1 retriever for building the pipeline. It is always labelled post-hoc
and reported beside the registered P7 miss. If several Y rungs pass, the cheapest passing
rung is adopted unless a dearer passing rung is `justified` over it on paired Δ AUC (the
running plan's §3 verdict), as in the running plan's §5.

## 2. Frozen configuration (filled at Q8, 8 Oct 2026, before any final-set score)

Chosen on the dev set by the committed rules (`shortlist_dev.RULES`, `Z_RULES`; records
`shortlist-dev.json`, `shortlist-dev-variants.json`). Role model `44b7fff54614`, title
model `18b47dfd129f`, text `text_v2` plus `jd_title` (D54).

| Rung | Formula | Option | Dev-select R@50 | Scored at Q8? |
|---|---|---|---|---|
| Y1, Y2, Y2t, Y4 | hard role filter (01 §3) | Y2: k = 1, context on (best classical) | 0.664 | **no**: Q3 early stop (does not beat X2 on dev) |
| Y2m, Y5, Y-rel | — | — | — | no: Q4, Q5, Q7 not run (Q-R4 review) |
| Z2, Z2L, Z3 | hard filter, JD-side sets | — | ≤ 0.697 | **no**: below X2 on dev |
| **Z1** | `64 × Σ p_jd p_cv + experience_compatible + 0.999 × per-JD min-max TF-IDF`; p_jd from the JD text | w = 64 | 0.7775 | **yes** |
| **Z1L** | the same with p_jd one-hot on C1's family where the title is a lookup hit | w = 32 | 0.7767 | **yes** |

Both Z rungs are **classical** (TF-IDF text term; S2 bar 0.80). No advanced rung is scored:
none was chosen on dev, so the advanced bar (0.90) is not attempted.

**Scored at Q8, once:** Z1 and Z1L, with X2, TF-IDF and the oracle 1a+2 as references
(their P7 scores, recomputed; they must equal `stage1-metrics.json`'s `scores_sha256`).
Same 500-candidate pools as P7, `text_v2`. Figures: per-JD AUC (and within-family), and
Recall@10 / @50 on the pools. Generic, targeted and total; `all` and `llm_free`.

**Comparisons** (paired per-JD Δ, B − A; the running plan's §3 verdict, applied to AUC and,
separately, to Recall@50; for Recall@50 `llm_free` drops the pool rows of judged pairs
whose label needed the LLM):

| # | B | A | Question | Decides |
|---|---|---|---|---|
| M1 | Z1L | X2 | Do the title and a larger role weight fix P7's shortlist? | reported |
| M2 | Z1 | X2 | Does the larger role weight alone? | reported |
| M3 | Z1L | Z1 | What does the title add? | selection (below) |
| M4 | Z1L | TF-IDF | Against plain text | reported |
| M5 | Z1L | oracle 1a+2 | Cost of no CV metadata (expected ≤ 0) | nothing |

**S2:** each of Z1, Z1L passes when its total Recall@50 point estimate ≥ 0.80. "CI above the
bar" stated separately.

**Selection (Q-R1).** Among Z1 and Z1L that pass S2, the cheaper, **Z1** (no title model),
is adopted unless Z1L is `justified` over it on M3's AUC verdict. A rung that passes is
adopted as the stage-1 retriever **post-hoc**, always reported beside the registered P7
miss ("P7 registered: miss (0.779). Post-hoc Z: …"). If neither passes, nothing is adopted
and Q-P7-2 stands.

## 3. Amendments

| # | Date | Change | Why | Post-hoc? |
|---|---|---|---|---|
| Z | 7 Oct 2026 | After the Q3 early stop, five dev-only variants (user request): **Z1** soft role term with weight w ∈ {2, 4, 8, 16} (w = 2 is X2); **Z1L** the same with the JD side one-hot on a title lookup hit, else the text classifier; **Z2** hard filter on the JD text classifier's top 3; **Z2L** title family on a lookup hit, else text top 3; **Z3** title family on a lookup hit, else text top 1. TF-IDF text term throughout. Options and verdict rules in `shortlist_dev.Z_RULES`, fixed before the first Z run. No final-set scoring is implied: Q8 still waits on review | The Q3 diagnosis: a hard match on one title-derived family fails where C1 is wrong | yes |
| Z′ | 7 Oct 2026 | User chose Q-R7 (a). Z1 / Z1L weight grid extended to {2, 4, 8, 16, 32, 64, limit = 1e6} (limit: order by role agreement, then experience, then text). Same option rule over the whole grid. A paired per-JD Δ Recall@50 vs X2 (dev-select) is added to the record. After this run, §2 is frozen and Q8 scores the frozen rungs once | w = 16 was the grid's edge with recall still rising | yes |
