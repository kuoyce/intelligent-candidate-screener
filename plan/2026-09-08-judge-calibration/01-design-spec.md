# Design Specification — judge calibration

## 1. The reference is an axis a figure may not be pooled across

`agent_sha256`, `draw`, `stratum` and `annotator` are already axes: a figure names one
value of each, and averaging across is forbidden because the average describes no
population. **The reference a judge is scored against is a fifth** (proposed **D37**).

Three references now exist on the A2 corpus:

| reference | `generic` | `targeted` | what it is |
|---|---|---|---|
| `yc` | n=400 | n=200 | one annotator's reading of the guide |
| `van` | n=126 | n=200 | another annotator's reading |
| `consensus` | n=60 | n=97 | the pairs where the two agree |

**A consensus figure may never be compared to a single-annotator figure as though one
improved on the other.** They are computed over different sets of pairs. kappa(consensus,
llm) = 0.401 on `targeted` against kappa(yc, llm) = 0.249 is not the judge getting better;
it is a different question asked of a different half of the corpus.

**Consensus coverage travels with every consensus figure**, the way D26's ceiling travels
with every Precision@k. 47.6% on `generic`, 48.5% on `targeted`. A reader who sees 0.401
without "on the 48.5% of pairs where two annotators agreed" has been told something
false by omission.

**The consensus subset is not a random subsample.** It selects pairs both humans found
unambiguous. A judge scores higher there for the same reason a student scores higher on
the easy half of an exam, and the figure is therefore an **upper** estimate of the judge's
performance on the corpus, not an unbiased one. Stated wherever it is quoted.

## 2. Model is an axis too, and the code does not know it yet

`a2_frames` currently builds `llm` as:

```python
live = frame[frame.agent_sha256.astype(str) == current]
llm = live.groupby("pair_id").label.apply(lambda s: majority(list(s))).dropna()
```

`majority` returns `None` on a tie, and `.dropna()` removes it. Two runs of the same
guide under two models is a two-element majority: **every pair the models disagree on is
silently dropped**, and the survivors are the pairs two models agreed on. The n falls,
nobody ordered it, and the kappa rises because the sample got easier.

This has never fired — runs 5 and 6 are the only A2 runs and both are `haiku`. Move 2
fires it on its first dispatch.

**Fix M1** (proposed **D38**): `a2_frames` keys the live series on `(agent_sha256, model)`
and takes the committed instrument under **one** model, defaulting to the model of the
most recent run of that instrument. Other models are returned under `per_model`, the way
`judge_frames` returns `per_instrument`. `a2_report` quotes each model separately and has
no field that blends two.

The self-consistency figure has the same constraint one level down: `self_consistency_kappa`
may only compare runs sharing **both** `agent_sha256` and `model`. Runs 5 and 7 are a
model comparison, not a reliability check, and must not be reported as one.

## 3. Consensus solves measurement. It does not solve substitution

The distinction matters enough to state as a rule.

**Measurement** asks: how good is the judge? A reference on 48% of pairs answers it, with
the selection bias of §1 stated.

**Substitution** asks: may the judge produce the labels? That needs a label on 100% of
pairs — and the 52% where the two humans disagreed is **exactly the population a
substituting judge would be labelling alone**, with no reference, no appeal and no way to
detect being wrong. It is also the harder half by construction.

So: consensus figures may be quoted in the evaluation, may be used to compare instruments
and models, and may **not** be used to license replacement. D36 stays unmade.

## 4. The gate is rewritten: interchangeability, not ground truth

Q39 said the relative clause stopped measuring. The deeper problem is that both clauses
assume a ground truth the corpus does not have. A gate that asks "does the judge match
the true label" is unanswerable when the two humans match each other at kappa 0.214.

**The replaceable question**: if the judge took one annotator's seat, would the pair of
labellers be as consistent as the two humans were? Measured on the **same pairs**,
8 Sep 2026:

| | partner | kappa(human, human) | kappa(human, llm) | 3-class | binary |
|---|---|---|---|---|---|
| `generic` n=126 | vs `yc` | 0.250 | **0.455** [0.328, 0.571] | pass | 0.690 vs 0.419 pass |
| `generic` n=126 | vs `van` | 0.250 | **0.142** [0.053, 0.234] | fail | 0.431 vs 0.419 tie |
| `targeted` n=200 | vs `yc` | 0.214 | **0.249** [0.150, 0.339] | tie | 0.371 vs 0.292 pass |
| `targeted` n=200 | vs `van` | 0.214 | **0.204** [0.106, 0.298] | tie | 0.394 vs 0.292 pass |

Read: the judge is a **good replacement for `van`** (it partners `yc` at least as well as
`van` does) and a **poor replacement for `yc`** (it partners `van` worse than `yc` does),
on `generic`; on `targeted` it is indistinguishable from either. **Binary passes all
four.** Every interval overlaps every other, so nothing here is *separated* at n=126-200
— the honest reading is "not distinguishable from a second human", not "better than one".

**What passing this gate licenses**: the judge may serve as a **second annotator** in a
future double-labelling wave — a role in which being as unreliable as a human is the
specification. It does **not** license the judge as an answer key, which is a claim about
correctness that no agreement figure between two fallible raters can support.

**The per-class clause is retained and rescoped.** It stays absolute at 0.40 but is now
applied to the **human pair as well as the judge**, and a class both humans fail on is
declared *unidentified by the instrument* rather than failed by the judge. `Good Fit` is
unidentified today: 0.338 `generic`, 0.300 `targeted`, human against human. That is a
finding about the guide, and it is what move 3 acts on.

## 5. Move 2 — the model control (Q32)

One variable. The instrument is the committed `75e00ea3`, byte-identical; the pairs are
the same 600; the humans are the same labels. Only the model changes.

**Two run numbers, not one.** D35 gives a run exactly one draw, and `a2_generic` and
`a2_targeted` are two draws — which is why runs 5 and 6 are separate. So:

- **run 7** — `a2_generic`, 400 pairs, `claude-sonnet-5`
- **run 8** — `a2_targeted`, 200 pairs, `claude-sonnet-5`

Both land in the existing draw files beside runs 5 and 6, distinguished by the `model`
column that `collect()` already preserves per row. **This is safe only after M1.**

**What each outcome licenses.** Registered in advance, before the run, because a result
that can be read either way afterwards measures nothing:

- **Sonnet moves the `Good Fit` cell materially** (per-class agreement against consensus
  clears 0.40, or the `Good -> Potential` collapse falls below half): the failure was
  model capacity. Move 3 is not needed; commit `sonnet` as the judge's model and re-quote.
- **Sonnet does not move it** (intervals overlap `haiku`'s on both strata): the failure is
  the instrument or the class definition. Move 3 proceeds, and Q32 closes as "no material
  model effect on this corpus", which also retroactively licenses comparing runs 1-3 with
  runs 4+ on the A1 recheck.
- **Sonnet is materially *worse***: report and stop. The committed model stays `haiku`.

## 6. Move 3 — a graded instrument

Gated on move 2 returning "no material model effect".

The edit has two parts and they are separable:

**(a) The boundary.** D34 scoped the tie-break to the `Good`/`Potential` boundary and
barred it from reaching `No Fit`; that fix held — the transitive collapse into `No Fit`
is gone. What remains is that the scoped tie-break still resolves downward almost
unconditionally: the judge issues `Good Fit` at 4.0% / 3.0% against `yc`'s 25.2% / 16.5%
and `van`'s 51.6% / 30.0%. A near-total absence of a class is a mis-set threshold, not
scattered judgement. The edit names a positive condition for `Good Fit` the way D34 named
one for `No Fit`.

**(b) The score.** The judge emits an integer 0-100 beside its class, and the class
becomes a function of the score with the cut points **in the report, not in the guide**.
This is the part worth paying for: after it, moving a boundary is a re-scoring of existing
labels rather than a new `agent_sha256` and 600 spawns. The cut points are then fitted
against the **consensus** reference, per stratum, with the coverage and the selection bias
of §1 attached, and a fitted cut point is a fitted parameter — it may not be quoted as an
out-of-sample result on the pairs it was fitted on.

Both parts are one guide edit, one new `agent_sha256`, one full re-judge: **run 9**
(`a2_generic`) and **run 10** (`a2_targeted`). No figure from runs 9/10 pools with 5-8.

`verify --derived` already fails if the committed judge is not what
`docs/annotation-guide.md` renders; the guide edit and the regenerated agent are one
commit, as they were for D34.

## 7. What this plan does not do

- **It does not make D36.** Nothing here licenses LLM labels as evaluation ground truth.
- **It does not re-open Q34** (option D's 639 pre-D34 labels). No run number is reserved
  for it, and runs 7-10 take the numbers they need.
- **It does not touch `judge_frames`, `RECHECK_CSV` or the committed D33/D34 figures.**
  The A1 recheck report is a different file and does not move.
- **It does not resolve Q42.** `yc`'s `No Fit` shift between strata is confounded with the
  D34 guide edit, and separating them needs `yc` to re-label `generic` under the current
  guide — human time this project does not have. Standing limitation.
- **It does not fix Q41.** The shortlist question has no inter-annotator agreement on
  either stratum and cannot acquire one without re-running a sitting. The ordering rule
  (declare a stratum fully double-labelled before its first sitting) is recorded for the
  next campaign, not applied to this one.
