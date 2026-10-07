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

## 2. Frozen configuration (filled at Q8)

| Rung | `k` | role-context | text term | classifier | C (Y5) | Notes |
|---|---|---|---|---|---|---|
| Y1 | | | — | | — | |
| Y2 / Y2t | | | | | — | |
| Y2m | | | | | — | scored only if Q4's check passed |
| Y4 | | | SBERT weak | | — | |
| Y5 | | | learned | | | |
| Y-rel | | | | | — | only if Q7 passed |

## 3. Amendments

| # | Date | Change | Why | Post-hoc? |
|---|---|---|---|---|
| Z | 7 Oct 2026 | After the Q3 early stop, five dev-only variants (user request): **Z1** soft role term with weight w ∈ {2, 4, 8, 16} (w = 2 is X2); **Z1L** the same with the JD side one-hot on a title lookup hit, else the text classifier; **Z2** hard filter on the JD text classifier's top 3; **Z2L** title family on a lookup hit, else text top 3; **Z3** title family on a lookup hit, else text top 1. TF-IDF text term throughout. Options and verdict rules in `shortlist_dev.Z_RULES`, fixed before the first Z run. No final-set scoring is implied: Q8 still waits on review | The Q3 diagnosis: a hard match on one title-derived family fails where C1 is wrong | yes |
| Z′ | 7 Oct 2026 | User chose Q-R7 (a). Z1 / Z1L weight grid extended to {2, 4, 8, 16, 32, 64, limit = 1e6} (limit: order by role agreement, then experience, then text). Same option rule over the whole grid. A paired per-JD Δ Recall@50 vs X2 (dev-select) is added to the record. After this run, §2 is frozen and Q8 scores the frozen rungs once | w = 16 was the grid's edge with recall still rising | yes |
