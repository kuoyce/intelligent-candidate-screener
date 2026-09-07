# Phase 2 — the human ceiling, and what it does to every judge figure on file

**Date:** 7 Sep 2026, **completed 8 Sep 2026** · second annotator `van` · strata `generic`
and `targeted`, never pooled *(D29)* · judge instrument `75e00ea3` (post-D34), `claude-haiku-4-5`, runs 5 and 6 ·
`output/annotation/llm-a2-report.json`

**Q35 closes.** A second annotator has labelled on both in-domain strata, so this
repository has a human–human kappa for the first time. It is **lower than every judge
figure it was supposed to be the ceiling for**, and that reframes the substitution
question rather than answering it.

---

## 1. What landed

`van` arrived with the batch-2 session and also went back over part of batch 1.

| Stratum | `yc` | `van` | shared |
|---|---|---|---|
| `generic` | 400 | 126 | 126 |
| `targeted` | **200** | 200 | **200** |
| `a1_recheck` | 100 | 8 | 8 |

`targeted` closed on 8 Sep 2026. `yc`'s 200 and `van`'s 200 are the same 200 pairs, and the
same 200 the manifest holds — checked, not assumed. No pair is labelled twice by the same
person, no label falls outside A1's three classes, and no shortlist pick sits on a `No Fit`.

The 30% double-labelling of D29 worked exactly as designed: `van`'s 126 `generic` pairs
and `yc`'s original 55 `targeted` pairs are precisely the sets `needs_two_labels` marks,
with nothing outside them and nothing missing. Nobody had to be told which pairs to take.

---

## 2. The figures

Three-class, with the binary collapse beside it *(D34's standing rule)*.

### `generic` (batch 1, n=400 judged, settled)

| | n | three-class | binary |
|---|---|---|---|
| **kappa(yc, van)** — the ceiling | 126 | **0.250** [0.140, 0.365] | 0.419 |
| kappa(yc, llm) | 400 | **0.380** [0.310, 0.447] | 0.637 [0.553, 0.708] |
| kappa(van, llm) | 126 | **0.142** [0.050, 0.236] | 0.431 [0.285, 0.575] |

### `targeted` (batch 2, n=200 judged, **complete**)

| | n | three-class | binary |
|---|---|---|---|
| **kappa(yc, van)** — the ceiling | 200 | **0.214** [0.112, 0.309] | 0.292 [0.165, 0.410] |
| kappa(yc, llm) | 200 | **0.249** [0.150, 0.339] | 0.371 [0.248, 0.485] |
| kappa(van, llm) | 200 | **0.204** [0.115, 0.295] | 0.394 [0.259, 0.516] |

**Phase 1's prediction fails.** It said `targeted` would land near `generic`'s 0.380,
reasoning from the judge's marginal distribution being near-identical across the two strata
(all three proportions within 1.5 points). It came in at **0.249, and 0.380 is outside the
interval** [0.150, 0.339]. The marginal-stability argument was sound about the judge and
said nothing about the humans, which is where the movement was: `yc`'s own `No Fit` rate
goes 0.402 → 0.580 between the strata while the judge's barely moves, 0.370 → 0.395.
**Marginal stability on one side of a comparison does not predict agreement**, and this is
the counter-example. Recorded because the prediction was made in advance and in writing.

The interim readings drifted a long way down the leg: 0.377 (n=55, the hash subset),
0.335 (n=62), **0.249 (n=200)**. Split at the sitting boundary the two halves are 0.377
[0.217, 0.543] on the first 55 and 0.197 [0.082, 0.314] on the last 145, with `yc`'s
marginals nearly identical across them (`Good Fit` 0.164 against 0.166) — so this is which
pairs, not a scale drift mid-session, and the intervals overlap. It is still the reason a
partial leg must not be quoted: the n=55 figure was optimistic by 0.13 and looked settled.

### On `targeted`, the judge is indistinguishable from a second human

| | three-class |
|---|---|
| ceiling, kappa(yc, van) | 0.214 [0.112, 0.309] |
| kappa(yc, llm) | 0.249 [0.150, 0.339] |
| kappa(van, llm) | 0.204 [0.115, 0.295] |

Three intervals, all overlapping, all containing each other's point estimates. On this
stratum **the judge disagrees with each annotator about as much as they disagree with each
other**. That is not an argument for substitution — the per-class clause still refuses it,
and "as unreliable as a human" is not the claim the gate was written to test — but it is
the sharpest statement of why the relative gate has stopped measuring (§3, Q39).

---

## 3. The ceiling is below the judge, and that is not good news

The gate in [`01-design-spec.md`](01-design-spec.md) §4 is relative:

> kappa(yc, llm) ≥ 0.75 × kappa(yc, annotator2), **and** per-class agreement > 0.40 on
> every one of the three classes.

**Clause 1 passes on both strata, and passes trivially.** 0.380 against a threshold of
0.188 on `generic`. The judge agrees with `yc` half again as much as `van` does.

**Clause 2 fails, on both strata, against both annotators.**

| per class | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| `generic`, yc vs llm | **0.119** | 0.797 | 0.745 |
| `generic`, van vs llm | **0.062** | 0.541 | 0.917 |
| `targeted`, yc vs llm | **0.121** | 0.745 | 0.560 |
| `targeted`, van vs llm | **0.067** | 0.662 | 0.653 |

For scale, the two *humans* do not clear 0.40 on `Good Fit` against each other either —
0.338 on `generic`, 0.300 on `targeted`. The clause is absolute and stays absolute; but a
threshold no pair of readers on this instrument reaches is measuring the instrument as much
as the judge.

**Substitution is refused**, three-class, on both strata. That is Phase 0's refusal
unchanged and clause 2 is why it exists: A1-vs-llm scored 0.90 / 0.27 / 0.067 per class
and still produced a positive pooled kappa.

The uncomfortable part is clause 1. It was written expecting a ceiling *above* the judge,
so that 0.75 of it was a real bar. At a ceiling of 0.250 the bar is 0.188 and nearly
anything clears it — including `van`, whose agreement with the judge (0.142) is below the
ceiling and would fail its own gate. **A relative gate divided by a number this small is
not a gate.** It should not be read as passed; it should be read as not yet measuring
anything, which is the same position D33 was in for the opposite reason.

What the low ceiling actually indicts is the **instrument**. Two people who both read
`docs/annotation-guide.md` agree at **0.250** three-class on `generic` and **0.214** on
`targeted`.

**And the `generic` ceiling is measured across two guides.** Dated from the commits that
carry the labels: `yc`'s 400 `generic` rows landed 30 Aug 2026 (`eaa1138`, `907a633`,
`7ad6330`), D34 rewrote the guide on 5 Sep (`513a8fc`), and every one of `van`'s rows plus
all of `yc`'s `targeted` rows landed 7–8 Sep. So kappa(yc, van) on `generic` is
`yc`-under-`842d7b0e` against `van`-under-`75e00ea3` — **not a clean human–human figure**,
and this is exactly the retroactivity **Q33** flags, now load-bearing rather than
hypothetical.

The `targeted` ceiling **is** clean: both annotators, one guide, post-D34, n=200. It comes
out at **0.214** [0.112, 0.309] against `generic`'s 0.250 [0.140, 0.365] — near-identical,
intervals almost coincident. So the guide edit is **not** what makes two humans disagree;
two readers of the *same* guide land in the same place. That is worth having: it moves the
low ceiling from "an artefact of Q33" to "a property of the task", which is the more
serious of the two readings. Quote the `targeted` ceiling where one number is needed.

---

## 4. An annotator is an axis a figure may not be pooled across

On the same 126 `generic` pairs, the same judge scores **0.380 against `yc` and 0.142
against `van`**. There is no "kappa(human, llm)" on this corpus. There are two numbers,
they differ by more than either differs from zero, and which one you get depends on who
you ask.

Until 7 Sep 2026 `a2_frames` did not know this. It built one series named `human` with
`drop_duplicates("pair_id")` — correct while every A2 label was `yc`'s, and silently wrong
the moment it was not, because `drop_duplicates` keeps whichever row comes first in
`judgements.csv`. **That is file order, not a rule.** It resolved to all-`yc` on `generic`
and **all-`van` on `targeted`**, so `a2_report` would have emitted 0.380 and 0.204 under
one field name that AGENTS.md and this plan both read as kappa(yc, llm).

Same shape as D35's `run1_vs_run2` and the three baseline bugs: no exception, no row count
that looks wrong, a plausible number. Fixed — `a2_frames` returns `humans`, a dict keyed by
annotator, there is no pooled `human` key, and `a2_report` emits `vs_llm` keyed by name
plus a `human_ceiling` block. Four regression tests, each shown to fail against the code
it guards.

---

## 5. The judge is `yc`'s scale shifted down, not a different scale

The disagreement has a shape, and it is the one Phase 1 predicted from the marginals alone.

**Marginals.** The judge issues `Good Fit` at 4.0% (`generic`) and 3.0% (`targeted`).
`yc` issues it at 25.2% and 14.5%; `van` at 51.6% and 30.0%. Both humans bracket the judge
from far above, on both strata.

**Is llm `Good Fit` a subset of human `Good Fit`?** Largely.

| | llm `Good` | of which human `Good` | lift over base rate |
|---|---|---|---|
| `generic` / yc | 16 | 12 (**75.0%**) | 2.97× |
| `generic` / van | 5 | 4 (80.0%) | 1.55× |
| `targeted` / yc | 6 | 4 (66.7%) | 4.04× |
| `targeted` / van | 6 | 4 (66.7%) | 2.22× |

**Is human `Good Fit` inside llm `Good + Potential`?** On `yc`, almost entirely.

| | human `Good` | survives as llm `Good`/`Potential` | falls to llm `No Fit` |
|---|---|---|---|
| `generic` / yc | 101 | **96.0%** | 4 |
| `generic` / van | 65 | 78.5% | 14 |
| `targeted` / yc | 33 | **93.9%** | 2 |
| `targeted` / van | 60 | 85.0% | 9 |

**Ordinal monotonicity.** P(llm ≤ human) is 0.887 (`generic`/yc), 0.976 (`generic`/van),
0.875 (`targeted`/van) — and **0.740 on `targeted`/yc**, which is where it breaks.

### The strictness reading does not survive `targeted`

`Good Fit` behaves the same on both strata: the judge issues it rarely, enriched, and
almost nothing `yc` calls `Good Fit` falls all the way to `No Fit` (93.9% survives). What
changes is the **other** boundary. `yc` gets much harsher on `targeted` — `No Fit` 0.402
→ 0.580 — while the judge barely moves, 0.370 → 0.395. So the judge now sits *above* `yc`
on a quarter of the pairs, and the largest single cell in the confusion is **50 of `yc`'s
116 `No Fit`s coming back `Potential Fit`** (43%, against 41 of 161 = 25% on `generic`).

So "the judge is `yc`'s scale shifted down one notch" is a `generic`-only statement. On
`targeted` the two disagree at **both** boundaries in opposite directions, which is why
three-class kappa falls to 0.249 and why the binary collapse falls with it (0.637 →
0.371): `No Fit ↔ Potential Fit` **does** cross the fit/no-fit line, and that is the
disagreement `targeted` is made of. Binary was healthier than three-class on `generic`
because the disagreement there was `Good ↔ Potential`; the same collapse buys almost
nothing here.

**Reading.** Against `yc` this is a threshold, not a different notion of fit: the judge
reserves `Good Fit` for a ~4%-of-corpus slice that is 3× enriched in `yc`'s `Good Fit`,
and 96% of `yc`'s `Good Fit` survives one notch down. That is why the binary collapse
(0.637) is so much healthier than three-class (0.380) — `Good → Potential` does not cross
the fit/no-fit line. Against `van` it is weaker: 14 of 65 `generic` `Good Fit`s come back
`No Fit`, which a pure shift cannot produce. Some of that is `van`'s own scale — 51.6%
`Good Fit` is twice `yc`'s — but not all of it is.

**This sharpens Q38 and does not close it.** Two humans bracketing the judge from above
makes "the judge's `Good`/`Potential` threshold is set too high" the economical reading,
where one annotator could not distinguish it from "`yc` is generous". It is still not
proof: `van` is more generous than `yc`, so "both humans are generous" is not excluded by
two annotators any more than by one. What has changed is that the *humans* now visibly
disagree about where the boundary is, so a D34-style guide edit aimed at the judge alone
would be fixing the wrong end of it.

---

## 5b. Q14's shortlist question is not double-labelled anywhere

Found while checking the completed session, not looked for. Shortlist picks per JD:

| stratum / annotator | JDs | with a pick | group sizes seen |
|---|---|---|---|
| `generic` / yc | 40 | 38 | 10 |
| `generic` / van | 39 | **0** | 1–6 |
| `targeted` / van | 20 | 20 | 10 |
| `targeted` / yc | 20 | **1** | 10 |

This is `Group.asks_shortlist` working exactly as written: the question is refused on a
**partial** group, because a top-1 pick made over part of a field is indistinguishable
afterwards from one made over all of it. The second annotator on a JD is almost always
served a partial group, so they are almost never asked.

`targeted`/`yc` is the sharp case. Every one of those JDs eventually got all 10 labels from
`yc`, but in **two sittings** — the hash-designated 55 first, the rest afterwards — and the
second sitting was a partial re-serve. The one JD with a pick is the one where `yc` had none
of the 55 (checked: 19 JDs with prior labels → 0 picks; 1 JD without → 1 pick).

**Consequence:** there is no inter-annotator agreement on the shortlist pick, on either
stratum, and widening `targeted` to full double-labelling did not create one. It could
not: the 55 already-labelled pairs fragmented all 20 JDs before the widening. Had the
stratum been declared fully double-labelled *before* `yc`'s first sitting, all 20 groups
would have been complete and the figure would exist.

Nothing is wrong in the data and no rule was broken. It is a gap in what the session can
answer, and it is only cheaply fixable going forward — an ordering constraint, not a code
change. Filed as **Q41**.

---

## 6. What this licenses

- **Three-class substitution: refused**, both strata, both annotators. Clause 2, absolute:
  `Good Fit` agreement is 0.119 / 0.062 / 0.121 / 0.067 against a bar of 0.40.
- **Binary substitution: no longer open on `targeted`.** On `generic` the binary collapse
  was the healthy view (0.637 against a human binary ceiling of 0.419). On `targeted` it
  is 0.371 (yc) and 0.394 (van) against a ceiling of 0.292 — the judge is still above the
  ceiling, but every figure has fallen and the ceiling has fallen with it. Binary was
  carrying `generic` because the disagreement there did not cross the fit/no-fit line;
  on `targeted` it does. There is no version of this where a binary substitution is
  defensible on both strata, and D29 forbids choosing the stratum that suits.
- **Phase 1's advance prediction is falsified** and recorded as such (§2).
- **Nothing about A1.** `judge_frames`, `llm-recheck-report.json` and every D33/D34 figure
  are untouched and share no code path with this.
- **D36 is not reachable from here.** Replacement was gated on a ceiling that would make
  0.75 of it a real bar. The measurement came back and it does not.

---

## 7. Open questions after this phase

| # | Question | Status |
|---|---|---|
| Q35 | kappa(yc, annotator2)? | **Closed.** `generic` 0.250 [0.140, 0.365] n=126 (across two guide versions); `targeted` **0.214** [0.112, 0.309] n=200, single guide — quote this one |
| Q38 | Does the `Good`/`Potential` boundary need a D34-style pass? | **Sharpened, still open.** Both humans sit far above the judge; the humans also disagree with each other about the boundary (per-class `Good Fit` 0.338 yc-vs-van). A guide edit is a new judge and no figure would pool with runs 5-6 |
| Q33 | The guide edit is retroactive against labels collected under `842d7b0e` | **Load-bearing and partly answered.** `yc`'s `generic` labels are 30 Aug (pre-D34); everything else is 7–8 Sep. The two ceilings — one cross-guide (0.250), one single-guide (0.214) — are near-identical, so the guide edit is **not** the source of human disagreement. `judgements.csv` still records no guide sha, and should |
| **Q39** | *(new)* Is a relative gate usable at all when the ceiling is 0.21–0.25? | Open, and sharper at full n: on `targeted` the judge (0.249 / 0.204) and the ceiling (0.214) are one overlapping cluster, so 0.75 × ceiling separates nothing. The gate needs re-deriving, not re-running |
| **Q40** | *(new)* Does the binary collapse have a defensible gate? | **Open, and less attractive than it was.** Binary held up `generic` (0.637) and does not hold up `targeted` (0.371 / 0.394), because the disagreement there crosses the fit/no-fit line |
| **Q41** | *(new)* No inter-annotator agreement exists on Q14's shortlist pick, either stratum | Open. `asks_shortlist` refuses partial groups and the second annotator is almost always served one. Fixable only by ordering — declare a stratum fully double-labelled *before* the first sitting |
| **Q42** | *(new)* Does `yc`'s `No Fit` rate really move 0.402 → 0.580 between strata, or is that D34? | Open. `yc`'s `generic` labels are pre-D34 and their `targeted` labels post-D34, so the stratum comparison and the guide comparison are the same comparison. `van` labelled both post-D34 and moved 0.190 → 0.360, a shift of the same sign and larger — so the stratum is doing real work, but the two are not separable from what is on file |
| Q37 | May `yc` and `llm` labels coexist in one evaluation layer? | Still deferred to Phase 3, which §6 says is not reachable |
| Q34 | Option D's 639 pre-D34 rows | Untouched, still deferred, no run number reserved |

---

## 8. Changes made in this phase

| Change | Why |
|---|---|
| `session.FULL_DOUBLE_STRATA = {"targeted"}` | The prospective stratum's ceiling was n=55, CI [0.039, 0.444]. The gate divides by that. Full coverage takes it to n=200 |
| `session.labels_wanted(pairs)` | One function deciding what a pair is owed, so `outstanding`, `batch_overview`, `build_queue` and `verify --derived` cannot disagree |
| `build_queue` drops *finished*, not *judged* | A pair wanting two labels and holding one is outstanding work. Dropping on the first label would have removed 200 `targeted` pairs from the key that `yc` still owes |
| `verify --derived` resume check follows the same rule | It failed on the merge otherwise, demanding the queue drop work that is not done |
| `a2_frames` returns `humans`, a dict; no pooled `human` | §4 |
| `a2_report` emits `vs_llm` keyed by annotator + `human_ceiling` | §3, §4 |
| `queue.progress` reports `labels_have` / `labels_wanted` | It said `targeted` was 200/200 while 145 labels were outstanding |

Eight regression tests, each run against a mutated version of the code it guards and shown
to fail: four in `tests/test_llm_recheck.py` (annotator separation, named judge figures,
the ceiling block, the no-overlap note) and four in `tests/test_annotation_ui.py` (a full
stratum owed entire, no orphaning of hash-designated labels, the other stratum keeps 30%,
no self-double-labelling).

**8 Sep 2026, session closed.** `uv run pytest` (218), `data.verify --derived` (all
checks passed) and `baselines.run --check` all pass, on a queue key rebuilt after the last
label. The key now holds 21 pairs, all `a1_recheck`; the in-domain campaign is finished —
600 pairs, 1,034 labels, both strata fully double-labelled.
