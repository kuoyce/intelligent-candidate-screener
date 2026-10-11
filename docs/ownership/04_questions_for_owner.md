# 04 — Questions you should be able to answer

*Written 11 Oct 2026. Each answer is my current best answer, from the code, records and
probes cited in `01`–`03` and the walkthrough notebook. **Confidence:** High = read from a
committed record or verified by running it; Medium = solid reading but partly inferred;
Low = my judgement, needs your or the team's view. Context you gave me (11 Oct): the app
is a draft and the project is in its research phase; the within-family AUC may become the
headline; human agreement is a known, accepted gap.*

## A. The research question and the headline

| # | Question | Best current answer | Conf. |
|---|---|---|---|
| A1 | What is the research question, in one sentence? | Does more sophisticated matching (rules → TF-IDF/BM25 → SBERT → cross-encoder) earn its complexity when scored against human Fit / No Fit judgements on in-domain IT job pairs? (`AGENTS.md`, proposal v3) | High |
| A2 | So, does it? | **Partly.** A weakly fine-tuned MiniLM bi-encoder beats TF-IDF on text alone (+0.057 AUC, justified, D48). Adding extracted role + experience to TF-IDF (X2) beats TF-IDF too (+0.061, K1). Nothing beyond that is resolved: cross-encoders were harmful and LLM-supervised rerankers did not beat SBERT weak (P8). The honest summary: *one cheap fine-tune and an explicit role signal earn their place; heavier models did not, at this data size and with this one base model* | High (figures) / Medium (wording) |
| A3 | What is the headline number? | Per-JD AUC ≈ **0.85** [0.81, 0.89] for X2/X4 on text_v2, n = 59 JDs. If you switch to **within-family AUC** (being considered): **0.74–0.75** [≈0.67, 0.82], n = 48 JDs. I'd report both, with within-family first, because 96.5% of Fits share the JD's keyword by sampling design | High |
| A4 | What does AUC 0.85 mean to a recruiter? | Pick a random suitable and a random unsuitable candidate for the same job: the system scores the suitable one higher 85% of the time. Within the right role it is ~75%. It says nothing about the top-5 a recruiter sees. That is closer to precision@k, which is 0.82–0.87 at k = 1 against a 61% base rate (notebook §7b) | High |
| A5 | Is the difference between the top systems real? | No. X4, X2, metadata + TF-IDF and SBERT weak sit at 0.83–0.85 with overlapping CIs. The paired deltas are "not resolved" (K2–K4, half-widths 0.03–0.04) | High |
| A6 | Why is the cheapest system selected when others score marginally higher? | The registered rule (D48): adopt the cheapest rung that no *justified* (CI-excluding-zero, both strata, LLM-free) more expensive rung beats. That guards against chasing noise | High |
| A7 | What is "the system" right now, for the report? | Stage 1: **no rung adopted**, because X2 missed the registered Recall@50 bar (0.779 < 0.80). Stage 2: **SBERT weak** ordering. The app still runs the pre-D53 metadata pipeline, a known draft gap (P9 not done) | High |

## B. Data and labels

| # | Question | Best current answer | Conf. |
|---|---|---|---|
| B1 | What data is this evaluated on? | Djinni (A2): public, pre-anonymised Ukrainian / Eastern-European IT JDs and CVs, English only. 600 pairs = 60 JDs × 10 CVs, 40 `generic` + 20 `targeted` JDs | High |
| B2 | Why should results generalise beyond Djinni? | They may not: one market, IT only, English-filtered, and an enriched 61% Fit rate. Treat it as a comparison instrument between systems, not a production estimate (README says so) | High |
| B3 | Who labelled it, and how reliable is that? | Two team members (`yc`, `van`) and claude-haiku-4-5 as a tie-breaker. Human κ = 0.35 (fair); 46% of pairs have one human; 18% were decided by the LLM. Accepted gap (your note). The `llm_free` subset is reported beside every figure, and figures **rise** there (X2 0.900) | High |
| B4 | Could the LLM labels bias the results toward LLM-like systems? | Small risk now: no adopted system is LLM-trained (R1–R3 were rejected). The LLM also read `Position`, which systems no longer do. Verdicts must hold on `llm_free`, which the rule requires | Medium |
| B5 | Was anything trained on the test data? | No, by id. Weak pairs, IDF, role model and the LLM batch all come from the A2 **train region**. I verified **0** final-set documents in the weak pairs and the LLM batch. Not checked: near-duplicate CVs re-posted under new ids | High (id) / Low (near-dups) |
| B6 | Was anything *tuned* on the test data? | No weights or thresholds (D46, AST tests). But the same 600 pairs have been scored by ~30 systems over 5 rounds, and one post-hoc diagnosis read the final labels before the Z rungs were designed. Some adaptive overfitting is likely in the 0.85s. I can't size it | Medium |
| B7 | Why only 10 CVs per JD, and why 8 from the same keyword? | Budget (~1,000 labels in total) and the D31 pilot: a random draw gave 28 / 30 No Fit. The cost of the fix is that "same role" became most of the signal | High |
| B8 | What is A1 and why is it secondary? | A public "resume–JD fit" dataset whose labels do not reproduce: κ(A1, human) ≈ 0.01. Kept as a training resource; every A1 figure carries the A13 caveat (D40) | High |

## C. Method

| # | Question | Best current answer | Conf. |
|---|---|---|---|
| C1 | How is a CV scored, exactly (X2)? | `2 × role_match + experience_compatible + 0.999 × per-JD min-max TF-IDF`. `role_match` = Σ p_jd(k)·p_cv(k) over 60 role families from a text classifier; experience = extracted band compatibility, with unknown counted as compatible | High |
| C2 | How are skills, experience and education weighted? | **Skills**: no explicit weight; they act only through text similarity (explicit skill overlap added +0.002, K5). **Experience**: one 0/1 step, but unknown for ~half the documents, so it is 1 for 95% of pairs. **Education**: not modelled | High |
| C3 | Why MiniLM and not a bigger or multilingual model? | CPU cost and "the cached MiniLM unless more are approved" (reset design spec). No other embedding model was tried, so every "SBERT" conclusion is conditional on MiniLM-L6 | High |
| C4 | Why did the cross-encoders fail? | Likely the start point: a bi-encoder checkpoint with an untrained one-logit head, trained on ~800 pairs. They were weak on their own validation split too (0.62–0.64), so this is not a transfer gap. A proper MS MARCO cross-encoder was not cached, and switching now would be post-hoc (P8 §6) | Medium |
| C5 | Why did X2 miss the retrieval bar? | The role extractor is wrong for ~1 in 3 JDs. `role_match` then sits near 0 for real same-role CVs, and other-role CVs with similar wording fill the top 50 (stage-1 §6a; notebook §3). Where the JD's family is predicted right, Recall@50 is 0.84 | High |
| C6 | Where do the 0.80 / 0.90 retrieval bars come from? | The proposal's Recall@**10** targets, re-applied to Recall@**50** at registration. The numbers carried over, the metric changed | High |
| C7 | Is the evaluation deterministic? | Yes. Every record has a `--check` that rebuilds and byte-diffs; the warm and cold tiers (D55) separate cached-output replay from retraining. The cold `stage2` check passed on 9 Oct per a local log, though the plan file still says it is owed | High |

## D. Risks, robustness and fairness

| # | Question | Best current answer | Conf. |
|---|---|---|---|
| D1 | Can a candidate game it? | Yes. Pasting the JD's skills into a CV beats the JD's best real CV in 22–33% of cases depending on the system. A sentence *negating* those skills raises rank just as much (notebook §8) | High (probe) |
| D2 | Does it understand synonyms? | TF-IDF: no. SBERT: partly, at document level. The skill vocabulary: badly for IT. `ML` → the ML language, `.NET` → Visual Basic, and no `JS`, `k8s`, `LLM`, `QA` (`03` §3) | High (probe) |
| D3 | Could a candidate's name affect their rank? | For SBERT weak, yes, arbitrarily. Adding a name line moves the CV's position among its 10 peers for ~150–160 of 600 CVs, with no consistent direction by name origin (2 names per group; out-of-distribution text). TF-IDF: no (names are outside its vocabulary). X2: 4–5 of 600 (notebook §9) | High (probe) / Low (group conclusions) |
| D4 | Was fairness audited? | Not for protected attributes: Djinni has no such fields. `group_gaps` reports AUC by experience band and stratum only. Among Fit CVs, SBERT weak scores intermediate-English and 0–1-year CVs lower; that may be legitimate or a proxy, and length is not controlled | High (facts) / Low (interpretation) |
| D5 | Are the evidence quotes trustworthy? | They are **verbatim** (an invariant). Whether they are **correct** is not validated: negation passes as "matched", and "supported" (τ = 0.28) fires often (notebook §4). Treat evidence as unvalidated until hand-checked | Medium |
| D6 | What about PDFs, scanned CVs, other languages? | Not evaluated. The research path reads Djinni text fields; the app uses pypdf with no OCR and no language check. A mixed Ukrainian/English JD passed the English filter (notebook §2) | High |
| D7 | Is personal data protected? | Djinni is pre-anonymised. `redact.py` masks URLs, emails, phones and long digit runs, but **not names**, and it also masks date and salary ranges by mistake. The proposal promises name / age / gender / nationality masking; that is not built | High |

## E. Process and ownership

| # | Question | Best current answer | Conf. |
|---|---|---|---|
| E1 | How do I reproduce the numbers? | `uv sync --all-groups`, pre-download MiniLM (undocumented), then `data.fetch` → `data.verify` → `data.build --all --seed 0` → `checks --cold --all` (~50 min, plus ~4 h for the cold stage 2) | Medium (I ran pytest and the notebook, not a fresh clone) |
| E2 | What does the test suite protect? | The evaluation arithmetic and past bugs (639 tests, ~55 s). The product path and robustness are barely tested; there is no CI | High |
| E3 | What is left in the plan? | P9: wire the pipeline and UI to extracted fields only; unify two title normalisers and two v2 CV-text builders; findings record; README refresh. Open questions Q-A1b, Q-B2, Q-D2, Q-D3, Q-P7-1, Q-P7-2 (`09-progress.md`) | High |
| E4 | What would most improve the research claim with the time left? | (1) Make within-family AUC the headline and report precision@k. (2) Label the top-ranked unjudged CVs for a few JDs, so retrieval is judged rather than assumed. (3) Try one stronger or multilingual embedding model as a registered rung. (4) Add stuffing, negation and name-swap invariants to the test suite | Low (judgement) |
| E5 | What would you *not* do? | Score more variants on the final set. Each one spends the remaining credibility of the 600 pairs. Iterate on the A2-train dev set (`dev_pools`), and score on the final set once, at the end | Medium |

## Questions I could not answer: please confirm

1. Is the intended use ranking **within** one role family (the recruiter already filters by role) or **across** roles? That decides which AUC is the headline.
2. Is stage 1 (retrieval from ~500) part of the claimed system, or is the claim only "ordering a given shortlist"?
3. Who chose the `yc`-only cell for the stage-2 verdicts (amendment A3)? Was a both-humans cell considered?
4. Was TREC-style pooling (label what the systems disagree on) ever considered for the annotation design?
5. Are the 490 random distractors per pool meant to represent a real applicant pool? If so, their role mix should match one, and it currently does not.
