# 08 — Extraction component: implementation record (P5; tasks 2.1, 2.2, 5.1–5.5)

**Status:** implemented 26 Sep 2026 on `feat/top-down-reset`, uncommitted. D49 is
**proposed** here (§9), not made; the orchestrator decides.
**Design:** [02-design-spec.md](02-design-spec.md) §6, §9. **Tasks:** [03-work-plan.md](03-work-plan.md)
rows 2.1, 2.2 and 5.1–5.5.
**Record:** `output/extraction/extraction-metrics.json` (every figure below is read from it).

```bash
uv run python -m candidate_screener.extraction.vocab --build     # data/vocab/skills.csv
uv run python -m candidate_screener.extraction.vocab --check     # rebuild in memory, compare sha256
uv run python -m candidate_screener.extraction.vocab --tune      # re-run the filter selection
uv run python -m candidate_screener.extraction.ner --train --seed 0          # ~13 min CPU, cache only
uv run python -m candidate_screener.extraction.evaluate --seed 0             # write the record
uv run python -m candidate_screener.extraction.evaluate --seed 0 --check     # retrain + diff
uv run python -m candidate_screener.extraction.evaluate --seed 0 --check --reuse-weights
uv run pytest tests/test_extraction_*.py
```

## 1. What was built

| File | Task | What |
|---|---|---|
| `src/candidate_screener/extraction/vocab.py` | 2.1 | Builds `data/vocab/skills.csv`; `digest()`, `--build/--check/--tune` |
| `src/candidate_screener/extraction/rules.py` | 2.2 | `Match`, `SkillMatcher`, `default_matcher()`: the fixed interface |
| `src/candidate_screener/data/dataturks.py` | 5.1 | Loader, offset repair, PII drop + in-place masking, BIO |
| `src/candidate_screener/extraction/skillspan.py` | 5.2 | SkillSpan loading, BIO → spans, token grid, union training target |
| `src/candidate_screener/extraction/metrics.py` | 5.2, 5.3 | Strict / partial span P/R/F1, unit bootstrap, paired delta |
| `src/candidate_screener/extraction/ner.py` | 5.3 | MiniLM token classifier: `train`, `load_ner()`, `predict_spans(text)` |
| `src/candidate_screener/extraction/normalise.py` | 5.4 | Held-out ESCO alt labels; string cascade vs SBERT |
| `src/candidate_screener/extraction/evaluate.py` | 5.5 | The record and its `--check` |
| `tests/test_extraction_{rules,vocab,dataturks,skillspan,metrics,ner,normalise,evaluate}.py` | all | Synthetic, run with no `data/` |
| `data/vocab/skills.csv` | 2.1 | 9,124 aliases; sha256 `63d781ee8ed0a84c…` (committable: `.gitignore` carries `!data/vocab/`) |

`extraction/metrics.py` and `extraction/skillspan.py` are not in 02 §9's list. They are
the shared pieces that both the rules and the tagger evaluation read, split out so the
tests need neither torch nor data.

### The fixed interface (confirmed)

```python
@dataclass(frozen=True)
class Match: start: int; end: int; text: str; canonical: str   # char offsets into the input
class SkillMatcher:
    def __init__(self, vocab: pd.DataFrame): ...               # columns canonical, alias
    def find(self, text: str) -> list[Match]: ...              # sorted by start, non-overlapping
    def canonicals(self, text: str) -> set[str]: ...
def default_matcher() -> SkillMatcher: ...                     # loads data/vocab/skills.csv (lru_cached)
```

Semantics, all tested:
- Text is split into maximal letter/digit runs, with every other non-space character as
  its own token.
- An alias must reproduce its token sequence *and* whether each pair of tokens touches or
  is space-separated. So `C++`, `C#`, `.NET`, `Node.js` and `CI/CD` match, `C + +` does not,
  and `java` never matches inside `javascript`.
- Matching is leftmost-longest and non-overlapping, and case-insensitive.
- A whitespace gap that contains a line break never continues a phrase: in a resume, a
  line break separates list items.
- Offsets index the caller's string, including after characters whose lowercase is longer
  (e.g. `İ`).
- When two rows share an alias pattern, the first row wins.
- `load_vocab` reads every cell as a string, so `NA` and `null` stay aliases.

## 2. Vocabulary (task 2.1)

### Sources and scope

| Source | Contribution |
|---|---|
| ESCO v1.2.1, **ICT scope** | 1,489 concepts: those in the digital skills collection (1,284), or below skill group `S5` *working with computers*, or below knowledge field `06` *ICT* (through `broaderRelationsSkillPillar`). Preferred label = canonical. Aliases are the preferred and alternative labels, plus each label with a trailing parenthetical qualifier removed (`Python (computer programming)` → `Python`). Hidden labels are not used: they are misspellings and version strings (`KDevelop 4.7.0`) |
| `data_jobs` | 252 posting skills, parsed through `profile.parse_skill_cell`. Posting counts are kept in `data_jobs_postings` |
| Baseline regex families | The 8 families of `baselines/skills.py`, expanded to 45 literals (`node\.?js` → `node.js`, `nodejs`) |

The ICT restriction is a design choice made *before* any measurement. The target corpus is
IT JDs and CVs (A2), and ESCO outside ICT is dominated by competences such as *manage
musical staff*.

### Merge policy

- One row per matcher pattern. The canonical comes from the highest-priority contribution:
  1. a concept's own name;
  2. then ESCO over `data_jobs` over regex;
  3. then alphabetical order.
- **A `data_jobs` or regex term joins an ESCO canonical only through that concept's own
  label**, never through an alternative label. ESCO's alternative labels list narrower
  tools under broad competences (`tensorflow`, `keras`, `matlab` under *computer vision*;
  `OpenCV`, `bash`, `Lua` under *computer programming*; `git` under *tools for software
  configuration management*). Folding a named tool into a competence would erase what a
  recruiter screens on.
- Two hand-written tables, both documented in code:
  - `DATA_JOBS_SYNONYMS` / `REGEX_SYNONYMS` fold spellings of one tool together (`mongo` →
    `mongodb`, `golang` → `go`, `postgres` → `postgresql`, …).
  - `ESCO_HOMONYMS = ("spark",)`. ESCO's `SPARK` is the Java micro-framework, while
    `data_jobs`' `spark` (114,609 postings) is Apache Spark, so the ESCO label is withheld.
    That is also why the vocabulary holds 1,488 ESCO concepts, not 1,489: `SPARK` had no
    other alias.
- Result: **9,124 aliases, 1,691 canonicals** (8,927 carry ESCO, 251 `data_jobs`, 44
  regex). The build is deterministic, and `--check` rebuilds it byte-identically.

### Filters, tuned on SkillSpan (train for statistics, validation for selection)

The grid is `min_chars ∈ {1,2,3}` × `max_tokens ∈ {∞,6,4,3,2}` × `stopwords ∈ {off, on}` ×
13 blacklist rules, 390 configurations in all:
- `min_chars` spares aliases carrying punctuation (`C#`, `F#`);
- `stopwords` is sklearn's 318-word English list;
- a blacklist rule drops an alias matched ≥ *m* times on SkillSpan **train** whose strict
  or partial hit rate is below *p*.

Configurations are selected on SkillSpan **validation**. **Test was never used.**

| | strict P | strict R | strict F1 | partial F1 | aliases |
|---|---|---|---|---|---|
| No filters | 0.404 | 0.143 | 0.2115 | 0.3074 | 9,129 |
| **Chosen:** stopwords on, blacklist (partial, 10, 0.25) = {`email`, `it`} | 0.485 | 0.143 | **0.2207** | 0.3061 | 9,124 |

- **Only the stopword filter matters.** It removes `it` → *computer technology*, which
  lifts strict precision from 0.404 to 0.480. The blacklist adds 0.0005 of strict F1, which
  is one span. It is kept because the selection rule fixed in advance chose it, not because
  the difference is resolved.
- `max_tokens` and `min_chars` change nothing measurable. Long ESCO phrases rarely occur
  verbatim. The single letters `R` and `C` survive because validation favours keeping
  them.
- **Objective: strict F1, not partial** (deviation V3). Partial was tried first and
  rejected on validation. Overlap credit rewards common words that fall inside long
  soft-skill spans (`it`, `work`), so under partial F1 the *unfiltered* vocabulary won,
  with `it` kept. Strict F1 is also what the tagger early-stops on, so both systems are
  selected on the same criterion.
- `evaluate` re-runs the tuning and refuses to write the record if the committed
  `FILTERS` / `BLACKLIST_RULE` differ from what the tuning selects.

## 3. DataTurks repair (task 5.1)

**The defect's cause is HTML escaping.** The card's diagnosis ("inclusive/exclusive `end`
plus stray whitespace") is wrong.
- `content` holds `&amp;` (177), `&lt;` (2) and `&gt;` (2), while the offsets index the
  *unescaped* text.
- Every entity before a span therefore shifts it by 4 or 3 characters. All 222 broken
  spans are off by a positive multiple of 4 (+4 to +36; 74 of them at +4).

| Step | Spans |
|---|---|
| total | 3,556 |
| `intact` (stored text = raw slice) | 3,334 |
| `unescape` (stored text = slice of the unescaped content) | **222** |
| `search` (nearest occurrence within 64 chars: the planned strategy, kept as fallback) | 0 |
| dropped, unrepairable | **0** |
| dropped, `Name` / `Email Address` | 476 |
| unlabelled annotations | 2 |

- Every surviving span is half-open, whitespace-trimmed, and satisfies
  `doc.text[start:end] == text`. This is tested on synthetic entity shifts, arbitrary
  drift, an out-of-window span and an empty span, and checked on the real files.
- PII: `Name` and `Email Address` spans are dropped. Their characters, and anything
  `annotation.redact.PATTERNS` detects, are overwritten in place with `X`, so offsets
  survive. 4 `Skills` spans had characters masked this way, and each keeps its (masked)
  text.
- `to_bio()` converts a document to whitespace-token BIO.

## 4. Evaluation protocol (tasks 5.2, 5.3)

- **SkillSpan:**
  - The **union** of the `skill` and `knowledge` layers is the headline label `SKILL`.
    As gold, it keeps both layers' spans. They overlap in ~2% of spans, so a
    non-overlapping system has a strict-recall ceiling slightly below 1.
  - Scoring is on SkillSpan's token grid. A character-level rules match is snapped
    outward to the tokens it touches, so `C#` and `.NET` inside the token `C#/.NET` become
    one token span. The tagger predicts on that grid directly.
  - Neither system types its spans. In the per-layer rows, the other layer's correct spans
    therefore count as false positives: **read per-layer recall, not per-layer
    precision**.
- **DataTurks `Skills`:** character-level, after the §3 repair.
  - Two parts: `test` (the 20 shipped documents, as planned) and `all` (220 documents).
    Neither system is fitted on DataTurks, so every document is out-of-sample.
  - The card forbids a single F1 on the 20-document split. Its 220-document
    cross-validation advice is for a model *trained* on DataTurks, which neither system is.
  - **20 documents decide nothing.**
  - Gold `Skills` spans are often whole skills sections, so strict F1 is bounded far
    below 1 for any skill-level extractor.
  - Skills named outside an annotated span (in the experience text, typically) count as
    false positives, which understates precision for both systems.
- **Metrics:**
  - **strict** means exact boundaries;
  - **partial** means precision counts predictions that overlap any gold span, and recall
    counts gold spans overlapped by any prediction.
  - The 95% CI is a percentile bootstrap over **sentences** (SkillSpan) or
    **documents** (DataTurks), 10,000 resamples, seed 0.
  - **NER − rules** is a paired bootstrap over the same resampled units.

## 5. Learned tagger (task 5.3)

- **Model:** `AutoModelForTokenClassification` from the cached
  `all-MiniLM-L6-v2` snapshot (`c9745ed1…`, loaded offline).
- **Labels:** 3 (`O`/`B`/`I`) over the SKILL union. Where the layers overlap in the
  training target, the longer span is kept.
- **Training:**
  - sentences longer than 126 subwords are split into word chunks, never truncated;
  - AdamW, lr 1e-4, weight decay 0.01, 10% linear warmup, batch 32 (length-bucketed);
  - seed 0, 4 threads, `torch.use_deterministic_algorithms(True)`.
- **Early stopping** on validation strict F1 (union), patience 2, **max 6 epochs** (the
  CPU budget).
- **Runtime:** 792 s wall on this machine.
- **Deployment detail:** `predict_spans(text)` tags each line separately, splits long
  lines by the same subword limit, and trims edge punctuation.
- **Hyperparameters were not tuned.** lr, batch size and epochs are one fixed choice (a
  small model, few epochs), not a search.

| epoch | train loss | val strict F1 | val partial F1 |
|---|---|---|---|
| 1 | 0.4618 | 0.3492 | 0.7685 |
| 2 | 0.1793 | 0.4540 | 0.8028 |
| 3 | 0.1119 | 0.4726 | 0.8048 |
| 4 | 0.0758 | 0.4954 | 0.8156 |
| 5 | 0.0540 | 0.5009 | 0.8249 |
| **6** | 0.0442 | **0.5060** | 0.8231 |

**Early stopping never fired.** The best epoch is the last, and validation strict F1 was
still rising (+0.005 at epoch 6). So the tagger is **budget-limited, not converged**, and
its figures are a lower bound on what this architecture reaches (deviation V7).

## 6. Results

### SkillSpan test (job postings; 3,569 sentences, SKILL union gold = 2,265 spans)

| System | scope | n pred | strict P | strict R | **strict F1** [95% CI] | partial P | partial R | **partial F1** [95% CI] |
|---|---|---|---|---|---|---|---|---|
| rules | **union** | 600 | 0.567 | 0.150 | **0.237** [0.207, 0.269] | 0.732 | 0.193 | **0.305** [0.272, 0.337] |
| NER | **union** | 2,447 | 0.504 | 0.544 | **0.523** [0.499, 0.548] | 0.811 | 0.865 | **0.837** [0.821, 0.852] |
| rules | skill (1,091) | | | 0.004 | 0.005 [0.001, 0.010] | | 0.028 | 0.037 [0.025, 0.052] |
| NER | skill | | | 0.448 | 0.276 [0.252, 0.301] | | 0.840 | 0.539 [0.509, 0.568] |
| rules | knowledge (1,174) | | | 0.286 | 0.379 [0.338, 0.419] | | 0.345 | 0.459 [0.419, 0.497] |
| NER | knowledge | | | 0.634 | 0.411 [0.379, 0.442] | | 0.888 | 0.574 [0.541, 0.605] |

Paired **NER − rules**, F1:

| scope | strict Δ [95% CI] | partial Δ [95% CI] |
|---|---|---|
| **union** | **+0.286** [0.256, 0.316] | **+0.532** [0.499, 0.564] |
| skill | +0.272 [0.247, 0.297] | +0.501 [0.469, 0.532] |
| knowledge | +0.032 [−0.006, 0.069] (**not resolved**) | +0.114 [0.074, 0.156] |

- The rules matcher finds almost none of the soft-skill layer (strict recall 0.004). Its
  vocabulary is ICT-scoped by design, and SkillSpan's skill spans are phrases ("work
  independently in a fast-paced team"), not labels.
- On the knowledge layer (tools, technologies) rules are precise, but the tagger recalls
  more than twice as much (0.634 against 0.286 strict). The strict F1 gap there is not
  resolved, because the untyped tagger's knowledge precision is diluted by its skill-layer
  predictions (§4).
- Validation, for reference only (seen during selection): rules 0.221 / 0.306, NER
  0.506 / 0.823 (strict / partial F1, union).
- **S3 reference bar** (0.85 strict F1 on the union; S3 is still open, 04 §3): **both
  systems miss it** (0.237, 0.523). The SkillSpan paper's published figure is **to be
  verified against the paper** (Zhang et al., NAACL 2022). It is not quoted here from
  memory, and this phase had no network access.

### DataTurks `Skills` (resumes)

| part | n docs | gold spans | System | n pred | strict F1 [95% CI] | partial P | partial R | **partial F1** [95% CI] |
|---|---|---|---|---|---|---|---|---|
| test | 20 | 55 | rules | 619 | 0.027 [0.000, 0.073] | 0.236 | 0.527 | 0.326 [0.226, 0.448] |
| test | 20 | 55 | NER | 1,598 | 0.009 [0.000, 0.025] | 0.137 | 0.582 | 0.222 [0.141, 0.324] |
| **all** | **220** | 468 | rules | 4,990 | 0.040 [0.011, 0.077] | 0.265 | 0.673 | **0.380** [0.334, 0.425] |
| **all** | **220** | 468 | NER | 16,854 | 0.011 [0.003, 0.022] | 0.134 | 0.697 | **0.224** [0.190, 0.260] |

Paired **NER − rules** on `all`: strict −0.029 [−0.056, −0.008], **partial −0.156
[−0.190, −0.124]**. On `test`: strict −0.018 [−0.056, 0.004], partial −0.104 [−0.168,
−0.048].

- On resumes the ranking flips.
- Both systems touch about the same share of gold skills spans (partial recall 0.673
  against 0.697).
- The tagger emits 3.4× as many spans, most of them outside the annotated skills sections.
  It learned SkillSpan's convention of tagging requirement phrases, and resume experience
  text is full of such phrases that DataTurks does not annotate.
- Part of that is gold incompleteness rather than error, but the gap is resolved and in
  the rules' favour.

### Normalisation (task 5.4)

Gold: a seeded sample of **1,000** ESCO alternative labels, each owned by exactly one
concept and not equal (after normalisation) to any preferred label, mapped back to its
concept. Held-out labels are removed from every index.

| Setting | concepts | index labels | string (exact → normalised → token Jaccard) | **SBERT** (MiniLM, cosine) | SBERT − string, paired |
|---|---|---|---|---|---|
| **ICT scope, preferred labels** (headline) | 1,489 | 1,489 | 0.602 [0.571, 0.631], coverage 0.943 | **0.766** [0.740, 0.792] | **+0.164** [0.138, 0.191] |
| ICT scope, + non-held-out alt labels | 1,489 | 8,007 | 0.734 [0.706, 0.762], coverage 0.956 | **0.864** [0.842, 0.885] | +0.130 [0.105, 0.155] |
| all ESCO, preferred labels | 13,939 | 13,939 | 0.582 [0.551, 0.612], coverage 0.987 | **0.782** [0.756, 0.808] | +0.200 [0.170, 0.231] |

- `exact` and `normalised` score 0 **by construction**: no held-out label equals an index
  label after normalisation. They stay in the record as a check that the gold set has no
  trivial hits.
- A string-then-SBERT hybrid would therefore equal SBERT, and is not reported.
- n = 1,000 queries per setting; CIs are percentile bootstraps over queries (10,000
  resamples, seed 0), and the delta is paired over the same queries.
- **SBERT wins in every setting, resolved.** This measures surface-form recovery for ESCO's own
  paraphrases, not the mapping of free-text spans a tagger emits.

## 7. Determinism and the `--check` tolerance (measured, as Q23 was)

| Measurement | Result |
|---|---|
| Two full trainings (seed 0, 4 threads, deterministic algorithms). Run A: 792 s on a quiet machine. Run B: 1,962 s while another process held ~8 cores | **Bit-identical.** Same state-dict sha256 `4977c40864fb…`, same `model.safetensors` sha256 `5800d486…`, identical loss and validation F1 at every epoch |
| Tagger inference on SkillSpan test at 1, 4 and 8 threads (same weights) | **Identical tags on 3,569 / 3,569 sentences.** Strict F1 0.523345 at every thread count |
| Full record rebuilt with run B's weights (`--check --weights …/ner-check`) | **OK: 0 failures, 0 notes** (§7.1) |

**Policy: `TOLERANCE = 0.0`.**
- Every figure is a ratio of integer counts, rounded to 4 places. Any real change
  therefore exceeds 1e-4, and a non-zero tolerance would only hide it.
- The training thread count is pinned at 4 (`NerConfig.threads`) and recorded.
- Not covered: another CPU or torch build, since bit-identity was measured on one machine.
  If `--check` fails there, run `--check --reuse-weights` first: that retrains nothing, so
  it separates an inference difference from a training one.
- Other items:
  - `weights_sha256` is informational in the diff; the figures are the contract.
  - `ner.config` is always compared exactly.
  - The record holds no timestamps or timings.
- Cost:
  - a full `--check` is one training plus one evaluation, ~13 + ~8 min on a quiet
    machine;
  - `--check --reuse-weights` is ~8 min.

### 7.1 Full-record check against run B

`uv run python -m candidate_screener.extraction.evaluate --seed 0 --check --weights
data/processed/extraction/ner-check`:
- **`extraction record OK (0 failures, 0 notes)`**, 7 min 22 s wall on a loaded machine.
- The check rebuilt the whole record with run B's independently trained weights, against
  the record written from run A's:
  - the vocabulary digest;
  - the filter re-tune;
  - every SkillSpan and DataTurks figure and paired delta;
  - normalisation, including SBERT;
  - `weights_sha256`.
- Every value matched exactly, so the zero tolerance is met, not merely assumed.
- The default `--check` retrains into `data/processed/extraction/ner-check/` (~13 min on a
  quiet machine). This run reused run B's weights because the machine was then running
  another job's training.

## 7.2 Tests

- `uv run pytest tests/test_extraction_*.py`: **77 passed** with data, the cached MiniLM
  and the trained tagger present.
- **Fresh-clone simulation:** a scratch pytest plugin pointed every `config` data path at
  an empty directory, and `HF_HUB_CACHE` at an empty cache. Result: **72 passed, 5
  skipped**, each skip carrying its reason:
  - DataTurks raw files;
  - the MiniLM cache;
  - the trained tagger;
  - `data/vocab/skills.csv`;
  - ESCO raw files.
- Full suite, final run (27 Sep, `uv run pytest -q`): **387 passed, 0 failed, 0
  skipped, exit 0**. That covers every test file in the worktree, including other tracks'
  tests present at the time.
- Named regressions were each shown to **fail with their guard removed** (mutation run):
  - entity-unescape repair;
  - line break stops a phrase;
  - punctuation tokenisation (`C#/.NET`);
  - held-out labels excluded from the normalisation index;
  - a named tool not folded into an ESCO competence;
  - regex literals checked against the unanchored alternative (the `c++` defect).

## 8. Deviations from the plan

| # | Deviation | Why |
|---|---|---|
| V1 | DataTurks is repaired by **unescaping HTML entities**, with the planned nearest-occurrence search kept only as a fallback (it repairs 0) | The root cause was found. Unescaping repairs all 222 with no guess about which occurrence was meant, and drops none |
| V2 | The per-alias blacklist statistics come from SkillSpan **train**, and the selection from validation | This reads "tune on validation only, never test" as ruling out test. Fitting a statistic on train and selecting on validation is the same split discipline the tagger follows (fit on train, early-stop on validation), so both systems see the same data. Flagged for confirmation |
| V3 | Filter objective is **strict** F1 | Partial F1 kept the noise it was meant to remove (§2) |
| V4 | Hand tables: spelling synonyms, one ESCO homonym (`spark`); `data_jobs`/regex terms map to ESCO only through a concept's own label | ESCO alternative labels list narrower tools under broad competences (§2) |
| V5 | DataTurks is scored on **all 220** documents as well as the planned 20 | Neither system is fitted on DataTurks, so all 220 are out-of-sample, and 20 decide nothing |
| V6 | SkillSpan is scored on the token grid (rules matches snapped to tokens) | Scores both systems on the same units |
| V7 | Tagger capped at 6 epochs; early stopping never fired | The ~30 min CPU budget covers two trainings (the build and the check) |
| V8 | Normalisation reports three settings. `exact`/`normalised` are 0 by construction; no hybrid | §6 |
| V9 | Paired NER − rules deltas and a paired SBERT − string delta were added | Repository convention: comparisons are paired |
| V10 | Two extra modules: `extraction/metrics.py`, `extraction/skillspan.py` | Shared by both systems' evaluation, and testable without torch or data |

## 9. D49: the selected extractor — adopted 27 Sep 2026 as proposed

*Orchestrator decision:* adopted. Implemented in `evidence/requirements.py` (free-text requirements capped at 6 words) and `pipeline.screen(tagger=...)`. V2's reading of "validation only" is confirmed: per-alias statistics from SkillSpan train and selection on validation, test untouched.

**Proposal: split by role. The learned tagger extracts JD requirements; the rules matcher
extracts and canonicalises CV skills and serves every exact / alias `matched` check.**

1. **JD side: the tagger.** SkillSpan is the job-posting domain, and there the tagger is
   resolved better on the union:
   - strict F1 +0.286 [0.256, 0.316];
   - partial F1 +0.532 [0.499, 0.564].

   The rules matcher misses the soft-skill half of what postings require (skill-layer
   recall 0.004), so an evidence table built on it would list only tools as requirements.
2. **CV side: the rules matcher.**
   - On resumes the rules matcher is resolved better: DataTurks `all` partial F1 −0.156
     [−0.190, −0.124] for the tagger, at about equal recall.
   - It emits canonical labels directly, which 02 §7's `matched` ("an exact or alias hit,
     with the CV span") presupposes.
   - It is deterministic, costs microseconds, and every span is a verbatim alias. That
     makes the faithfulness invariant trivial on that side.
3. **Linking the two.** A tagger requirement span is canonicalised by running the rules
   matcher inside it. When the span contains no vocabulary alias, it stays as free text and
   can only be `supported` (SBERT sentence similarity, 02 §7) or `missing`. The
   normalisation experiment shows SBERT recovers held-out ESCO paraphrases better than
   string matching (0.766 against 0.602 top-1, ICT scope). That is the evidence for using
   SBERT, not string overlap, in the `supported` step. It is not evidence that
   free-text-span normalisation is accurate.
4. **Rung 1b stays on the rules matcher.** It is a fixed formula, and a learned extractor
   inside it would make it a trained rung.

**What would change this:** a resolved win for either system on the other side's domain,
or P6 finding the tagger's JD spans too long to link (its strict precision is 0.504).
**Alternative, if the orchestrator prefers one extractor:** rules everywhere. That is
simpler and fully deterministic, at the cost of dropping soft-skill requirements from the
evidence table.

## 10. Open issues for the orchestrator

1. **`data/vocab/skills.csv`** is now committable: the orchestrator added `!data/vocab/`
   to `.gitignore`. Commit it together with the record, since the record carries its sha256.
2. **The B1 card and catalog are wrong about the DataTurks defect's cause.** It is HTML
   entities (§3), not inclusive/exclusive `end`. The card also says the defect "blocks BIO
   conversion"; after `unescape` it does not. AGENTS.md wants the card and
   `data-catalog.md` updated in the same commit (outside ownership).
3. **ESCO `skills_en.csv` has 13,960 rows but 13,939 distinct concepts.** 21 concepts
   appear twice with differing fields. The catalog's "13,960 skills" is a row count (card
   update, outside ownership).
4. **A frozen-baseline defect, recorded and not fixed:** `SKILL_PATTERNS`' `\bc\+\+\b`
   cannot match `c++` before a space, punctuation or the end of the text. It only fires on
   `c++11`-style tokens. The Phase 4 baseline is frozen (`run --check`), so it stays.
   `vocab.regex_terms` checks literals against the unanchored alternative for this reason,
   with a named regression test.
5. **Noise that SkillSpan cannot tune away**, relevant to rung 1b on A2:
   - `R` and `C` are kept, so `R&D` and `C-level` will match.
   - `data_jobs` tool names that are English words (`word`, `flow`, `express`, `chef`,
     `shell`, `slack`, `notion`, `zoom`, `unity`, `swift`) are kept, because too few
     occurrences exist in SkillSpan train to blacklist them.
   - `go` is dropped by the stopword filter, so Go is found only as `golang`.

   Any change must be tuned on a non-final-set split (D46).
6. **Tagger convergence** (§5): a longer schedule would likely add a little. It needs CPU
   budget, and the check doubles the cost.
7. **V2's reading of "validation only"** needs confirmation.
