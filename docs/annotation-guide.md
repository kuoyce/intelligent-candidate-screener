# Annotation guide — Good Fit / Potential Fit / No Fit

**For:** the single labelling session defined by *(decision D25)* — ~250 pairs, one sitting.
**Scheme:** A1's own 3 classes *(decision D17)*, so public and in-domain results are
interpretable together and the same label doubles as nDCG's graded relevance
(`Good` = 2, `Potential` = 1, `No` = 0).

## What you are being asked

For each pair you see one **job description** and one **candidate CV or resume**. Answer one
question:

> **If this candidate applied for this role, what should happen next?**

| Label | Means | The test |
|---|---|---|
| **Good Fit** | Shortlist. Send to the hiring manager | You would defend this shortlisting to a manager who has read the JD |
| **Potential Fit** | Worth a screening call. Something is missing or unclear, but the evidence is real | You would not send it straight to the manager, and you would not reject it either |
| **No Fit** | Reject at first pass | You would need the JD to be substantially different for this to be worth anyone's time |

Roughly: **Good Fit** ≈ "yes", **Potential Fit** ≈ "maybe, ask them", **No Fit** ≈ "no".

## Who is labelling, and what that means for the report

Annotators are business-analytics master's students with software and data-science
backgrounds — **domain-literate, not professional recruiters.** What this set contains is
team-adjudicated judgement, not recruiter ground truth, and every result computed on it is
reported that way. This is a limitation to state, not a defect to apologise for: the
alternative was no in-domain evidence at all.

## Worked examples

Each is a sketch of the *shape* of a decision, not a real document. Real pairs are longer,
messier, and often missing the field you most want.

### Good Fit

1. **JD:** mid-level Python backend, 3+ years, Django, PostgreSQL, REST APIs.
   **CV:** 4 years building Django services against Postgres, ships REST APIs, names the
   ORM and migration tooling.
   → Core stack matches, seniority matches, evidence is specific. **Good Fit.**
2. **JD:** data analyst, SQL, dashboarding, stakeholder reporting, 2+ years.
   **CV:** 3 years in analytics, writes SQL daily, built Tableau dashboards for finance,
   describes the reporting cadence.
   → Every named requirement has evidence behind it. **Good Fit.**
3. **JD:** DevOps engineer, AWS, Terraform, CI/CD, Kubernetes.
   **CV:** 5 years platform engineering on AWS, Terraform modules in production, migrated
   a CI pipeline, ran EKS.
   → Different job title, same work. Title is not the requirement. **Good Fit.**

### Potential Fit

1. **JD:** senior data engineer, 5+ years, Spark, Airflow.
   **CV:** 2 years, strong Airflow and dbt, no Spark, clearly capable and clearly junior.
   → Real relevant evidence, wrong seniority. **Potential Fit** — the gap is one screening
   question, not a rejection.
2. **JD:** full-stack, React and Node.
   **CV:** three years of solid React, one small Node project mentioned in passing.
   → Half the role is well evidenced, half is thin. **Potential Fit.**
3. **JD:** QA automation engineer, Selenium, Python.
   **CV:** manual QA for four years, lists Python as a skill with no project behind it.
   → The domain is right and the automation claim is unevidenced. **Potential Fit** — ask
   about the Python.

### No Fit

1. **JD:** backend Java engineer.
   **CV:** graphic designer, six years, no engineering content.
   → Different profession. **No Fit.**
2. **JD:** senior ML engineer, PyTorch, production model serving.
   **CV:** recent graduate, one coursework classifier notebook, no production experience.
   → The gap is a career stage, not a screening question. **No Fit.**
3. **JD:** recruiter, technical hiring, 3+ years agency experience.
   **CV:** software engineer who once helped interview candidates.
   → Adjacent exposure is not the job. **No Fit.**

## The tie-breaking rule

When you cannot decide between two labels, **choose the lower one** — `Potential` over
`Good`, `No` over `Potential`.

The reason is asymmetric cost, and it is worth understanding rather than just following. A
`Good` label that should have been `Potential` inflates every system's measured precision:
the answer key now says a mediocre match was a shortlist, and any system that ranks it
highly is rewarded. A `Potential` that should have been `Good` costs one point of recall on
one query. The first error corrupts the instrument; the second makes it slightly
conservative.

**Do not use "I'd need to think about it" as a reason to go up.** That is precisely what
`Potential Fit` means.

## What not to consider

These feel relevant and are not. Each one, used as a signal, would put a bias into the
answer key that every downstream model then learns to reproduce.

- **Company prestige.** A brand-name employer is not evidence of skill, and its absence is
  not evidence of its lack.
- **English level as a proxy for competence.** Djinni CVs carry an `English Level` field and
  many are written by non-native speakers. Fluency is a requirement only when the JD states
  it as one — and even then, judge it against what the JD asks for, not against your own.
- **CV length or polish.** A1 resumes run ~5,100 characters; Djinni CVs ~1,500. That is a
  property of the two platforms, not of the two candidates. A short CV is not a weak one.
- **Formatting, typos, layout.** Unless the role is specifically about written communication.
- **Gaps, age, gender, nationality, university, photograph** — anything you would not put in
  writing to a candidate as a reason for rejection.
- **Your guess at salary expectations or notice period.** Not a fit question.
- **Whether you personally would enjoy working with them.**

If a JD names something in this list as an actual requirement, then it is a requirement.
The rule is about what you add, not about what the JD asks for.

## Two corpora, one scheme *(assumption A16)*

The queue mixes documents from two sources, and you will notice the difference:

| | Djinni (A2) | A1 |
|---|---|---|
| Market | Ukraine / Eastern Europe IT | US, general industry |
| Median CV length | ~1,500 chars | ~5,100 chars |
| Format | Structured profile fields, concatenated | Free-text resume document |

**Apply the same three definitions to both.** The question — shortlist, call, or reject —
does not change with document length or market. But the *evidence density* does: an A1
resume gives you more to read, and a Djinni profile can be a genuine Good Fit while saying
much less. Do not reward a candidate for having written more.

This is recorded as an explicit assumption because it may turn out to be wrong. Agreement
(κ) is therefore reported **per corpus**, not pooled — if we are materially less consistent
on one of them, that is a finding about this guide, and it should be visible rather than
averaged away.

## How the pairs reach you

Most of the work happens in a small tool that runs on your own machine:

```bash
uv run python -m candidate_screener.annotation.ui --annotator <your name>
```

It opens a page listing the batches. Pick one and start; your name is stamped on every
label you save. **You can stop at any point** — each screen is saved when you press *Save
and next*, and re-running the command picks up exactly where you left off. There is no
progress file to keep, no half-finished spreadsheet, and nothing to send anyone.

One screen is **one job description and its candidates** — usually ten, sometimes fewer.
You give each candidate a label, answer the shortlist question if it appears, and move on.

Most of the ten will be from the job's own field. That is deliberate: an earlier version of
this set drew candidates from the whole corpus and nearly everything was an obvious `No Fit`,
which is quick to label and tells us nothing. Expect to have to think about most of them, and
expect a couple per screen that look plausible at a glance and are from the wrong field
entirely — those are there on purpose.

- **Label each candidate against the standard in this guide, not against the other four.**
  This is the one habit the tool makes easy to lose. Three weak candidates on a screen make
  the least weak of them look good; it is still a `No Fit` if it is a `No Fit`. The classes
  have to mean the same thing on a screen full of strong candidates as on a screen full of
  poor ones, or nothing can be compared across job descriptions afterwards. The **only**
  comparison you are asked to make is the final shortlist pick.
- **You will not see a rank, a score, a system name, or why a pair was selected.** This is
  method, not withholding: if you knew which pairs something had already flagged as good,
  you would anchor on the answer you expected, entirely unconsciously.
- **You may skip a document** that is unreadable, truncated, or in a language you cannot
  read — put the reason in the notes box. A skipped pair is data; a guessed pair is noise.
- **Documents are redacted** — emails, phone numbers, URLs and long ID numbers appear as
  `[EMAIL]`, `[PHONE]`, `[URL]`, `[NUMBER]`. Names are **not** removed: a name is not a
  regex. These are real people's documents. They are not to be redistributed,
  screenshotted, or pasted anywhere outside the labelling session. The tool listens only on
  your own machine.

### Screens with fewer candidates

Some screens show fewer than ten. There are two reasons, and you are told which only in the
first case:

- A screen headed **second opinion** is one the other annotator has already worked through.
  The candidates left are the ones set aside for an independent second judgement, which is
  how we measure whether the two of us apply the three classes the same way. You will not be
  shown their labels.
- Otherwise it is simply a shorter set. Judge it exactly as you would a full one.

**The shortlist question is not asked on short screens.** A pick made over two candidates is
not the same answer as a pick made over ten, and afterwards the two would be
indistinguishable.

### Both corpora come through the tool

Some screens are Ukrainian/EE tech roles with short CVs; others are US job descriptions with
much longer resumes — see *Two corpora, one scheme* above. They are mixed together in an
order that is different for each of you. Judge them all by the same three classes; the
difference in length is a property of the corpus, never a reason to grade one harder.

## One extra question per job description

After you have labelled all the candidates for one JD:

> **Of the candidates you marked Good or Potential for this role, which one would you
> shortlist first?**

One pick, no ranking of the rest. It takes about thirty seconds.

The reason: with five candidates and three marked `Good`, nDCG cannot separate a system that
orders those three well from one that orders them badly — they carry identical gain. Your
single pick is the only thing that breaks that tie. A full human ranking would cost far more
per JD, has no clean agreement statistic at five items, and neither Precision@k nor nDCG
consumes one.

If nothing was Good or Potential, record "none" and move on.

## How the work is split

**Two annotators, both covering every job title.** You do not each take your own set of
titles. The tool splits the queue for you: it visits job descriptions in an order keyed to
your name, so the two of you start in different places and rarely land on the same screen.
If you do meet in the middle nothing breaks — you simply both label a job description that
only needed one of you, which costs a screen out of the session's budget. Say hello before
you start and agree who begins where if you are working at the same time.

That is a deliberate choice and worth one sentence of why, because the alternative looks more
efficient. If each person owned their own titles, then "DevOps scored lower than Data Science"
would have two explanations that cannot be told apart — the system really is worse at DevOps,
or that annotator is simply stricter. Covering everything between you means a difference
between titles is a fact about the system, not about who happened to read them.

## Double labelling and adjudication

30% of the in-domain pairs are labelled by **both** of you; the rest are labelled once. This is
what makes agreement measurable — Cohen's κ for two annotators — and it is **reported per
corpus**, Djinni and A1 separately (**A16**).

Where we disagree, we meet and adjudicate. **Both the original labels and the adjudicated
one are kept.** The disagreements are not noise to be cleaned up: they are the most
informative thing in the set, because they show where the three definitions are actually
ambiguous rather than where we merely worried they might be.

Once adjudicated, the set is **frozen**. It is never used for training or tuning — the
moment it is, it stops being an evaluation set.
