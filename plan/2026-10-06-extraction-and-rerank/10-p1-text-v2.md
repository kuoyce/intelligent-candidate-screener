# 10 — P1 record: `text_v2` and re-scoring the existing ladder

**Phase:** P1 (tasks 1.1–1.4) of [03-work-plan.md](03-work-plan.md). **Decision:** D53.
**Status of every figure here:** `exploratory` (02 §2: these rung configurations were
seen on `text_v1`). Implemented 6–7 Oct 2026, not committed at the time of writing.

## 1. What was built

| Task | Change | Where |
|---|---|---|
| 1.1 | `text_variant` (`"v1"` default, `"v2"` = CV text without `Position`). The field list is mirrored in `CV_FIELDS_BY_VARIANT` because `annotation/` is frozen (D42). `v1` delegates to `annotation.sample.cv_text`, so it is byte-identical by construction | `matching/corpus.py`: `build_cv_text`, `a2_pairs(text_variant=…)`, `a2_train_documents(…, text_variant=…)`, `_djinni` |
| 1.1 | The variant flows into everything that builds CV text for a system | `data/weak_pairs.with_text(frame, text_variant)`, `matching/lexical.py` (`a2_vectorizer`, `a2_bm25`, `Tfidf(text_variant)`, `Bm25(text_variant)`), `matching/sbert.py` (`SbertZeroShot(text_variant)`, `SbertWeak(text_variant)`) |
| 1.1 | Cache names carry the variant: `a2-pairs-v2-…`, `a2-train-{jd,cv}-v2-…`, `minilm-zero-shot-v2`, and `sbert-weak-3255167debe9` (the `v2` config adds `"text_variant": "v2"` to the fingerprint) | same files |
| 1.2 | Residual-leak measure | `evaluation/text_leak.py` (CLI prints it), and stored in the `v2` record under `residual_leak` |
| 1.3 | Retrained on `v2`: the TF-IDF IDF and BM25 IDF/avgdl (20k + 20k A2 train-region documents) and **SBERT weak (3b)**, same `FT_CONFIG`, weak-pair train split with `v2` CV text. Fit 1,471 s on CPU | weights under `data/processed/matching/sbert-weak-3255167debe9/` (git-ignored) |
| 1.4 | `--text-variant v2` on the in-domain evaluation writes a **new** record; `v1` record untouched | `evaluation/indomain.py`, `matching/ladder.text_v2_rungs()`, `output/matching/indomain-metrics-v2.json` (+ `indomain-costs-v2.json`) |

The `v2` sample keeps the **same documents** as `v1`: the English and non-empty filters and
the seeded draw all run on the `v1` text, and only the CV text is swapped afterwards. So the
committed `weak-pairs.csv` resolves under both variants, and the final-set pairs are the same
600. None of the 600 final-set CVs has an empty `v2` text (`residual_leak.cv.*.empty_v2_text = 0`).

### Tests (`tests/test_text_variant.py`, 11 tests, no `data/` needed)

- **Named invariant** `test_v2_cv_text_never_contains_the_position`: a synthetic CV whose
  `Position` is "Zyxquorbian Flux Wrangler" contains it under `v1` and never under `v2`.
  **Shown to fail** with `build_cv_text` mutated to ignore the variant:
  `AssertionError: assert 'Zyxquorbian Flux Wrangler' not in 'Zyxquorbian...am\n\nRemote'`.
- `test_train_documents_apply_the_variant_with_the_same_ids` and
  `test_a2_pairs_apply_the_variant` (monkeypatched loaders, temp cache): `v2` drops the
  title, keeps the ids, and writes its own cache files. **Both shown to fail** with the
  `v2` text swap in `_djinni` / `a2_pairs` mutated out (`if False:`).
- Drift guard: `CV_FIELDS_BY_VARIANT["v1"] == annotation.sample.CV_FIELDS` and `v1` output
  equals `sample.cv_text`; `v2` equals `sample.cv_text` with `Position` blanked.
- Cache keys differ by variant; `v1` keys are the pre-D53 ones (`minilm-zero-shot`,
  `sbert-weak-b5bf5501d1e8`).
- Leak normalisation and matching (seniority stripping, whole-word match, empty titles
  skipped) and the report's per-stratum counts.

## 2. Residual leak (task 1.2)

From `output/matching/indomain-metrics-v2.json` → `residual_leak`. A title is lower-cased,
the tokens junior, jr, middle, mid, senior, sr, lead, head, trainee, intern removed,
whitespace collapsed and edge punctuation trimmed. A match means the normalised title
appears as a whole word in the lower-cased, whitespace-collapsed text. No title normalised
to empty, so none was skipped. Nothing is scrubbed. 95% percentile bootstrap, 10,000
resamples.

| Share whose text still contains its own title | generic | targeted | total |
|---|---|---|---|
| **CVs** (removed `Position` in `v2` CV text; CI resamples CVs) | 0.203 [0.163, 0.243], 81/400 | 0.275 [0.215, 0.335], 55/200 | **0.227 [0.193, 0.260]**, 136/600 |
| CV leak per pair, mean per JD (CI resamples **JDs**) | 0.203 [0.165, 0.240], 40 JDs | 0.275 [0.215, 0.340], 20 JDs | 0.227 [0.193, 0.262], 60 JDs |
| **JDs** (JD `Position` in `Long Description`; CI resamples JDs) | 0.225 [0.100, 0.350], 9/40 | 0.400 [0.200, 0.600], 8/20 | **0.283 [0.167, 0.400]**, 17/60 |

All 600 final-set CVs are distinct, so the CV share and the per-JD pair share coincide in
their point estimates. Reading: about **one CV in four still names its own title in
free text** after the field is removed, more often in `targeted`. `text_v2` removes the
field, not the information. Every `text_v2` figure should be read beside this rate.

## 3. v1 vs v2 (task 1.4)

Per-JD AUC against the final set (rule_version from the record), subset `all`. Total with
its CI and *n* JDs, strata beside it. Within-family AUC (candidates sharing the JD's
`Primary Keyword`) total, n = 48 JDs. v1 from `indomain-metrics.json`, v2 from
`indomain-metrics-v2.json`. Every v2 row is `exploratory`.

| System | Variant | AUC total [95% CI] (n=59) | generic (n=39) | targeted (n=20) | within-family total [95% CI] (n=48) |
|---|---|---|---|---|---|
| random | both | 0.556 [0.505, 0.606] | 0.558 | 0.552 | 0.580 [0.511, 0.648] |
| keyword match (metadata, reference) | both | 0.766 [0.724, 0.804] | 0.786 | 0.726 | 0.500 [0.500, 0.500] |
| metadata rules (reference) | both | 0.778 [0.732, 0.819] | 0.776 | 0.784 | 0.540 [0.489, 0.591] |
| TF-IDF | v1 | 0.797 [0.750, 0.843] | 0.792 | 0.808 | 0.724 [0.648, 0.800] |
| TF-IDF | **v2** | **0.789** [0.741, 0.835] | 0.780 | 0.807 | 0.714 [0.637, 0.788] |
| BM25 | v1 | 0.636 [0.584, 0.687] | 0.613 | 0.681 | 0.625 [0.554, 0.693] |
| BM25 | **v2** | **0.632** [0.579, 0.682] | 0.610 | 0.673 | 0.616 [0.546, 0.684] |
| SBERT (zero-shot) | v1 | 0.774 [0.725, 0.822] | 0.789 | 0.746 | 0.755 [0.682, 0.825] |
| SBERT (zero-shot) | **v2** | **0.737** [0.686, 0.784] | 0.740 | 0.731 | 0.731 [0.668, 0.795] |
| SBERT (weak fine-tuned) | v1 (held-out) | 0.854 [0.817, 0.890] | 0.872 | 0.820 | 0.755 [0.679, 0.827] |
| SBERT (weak fine-tuned) | **v2** | **0.834** [0.796, 0.870] | 0.853 | 0.796 | 0.752 [0.688, 0.813] |

Paired deltas inside the v2 record (total, all, n = 59 JDs, CI over JDs):

| Id | B vs A (both v2) | ΔAUC [95% CI] |
|---|---|---|
| C3 | SBERT zero-shot vs TF-IDF | −0.053 [−0.111, 0.004] |
| C5 | SBERT weak vs TF-IDF | +0.045 [−0.006, 0.095] |
| E2 | TF-IDF vs metadata rules | +0.011 [−0.052, 0.077] |
| E3 | BM25 vs TF-IDF | −0.158 [−0.211, −0.105] |
| E6 | SBERT weak vs SBERT zero-shot | +0.097 [0.064, 0.133] |

Readings (exploratory):
- Every text rung loses a little AUC without `Position`: TF-IDF −0.008, BM25 −0.004,
  SBERT weak −0.020, SBERT zero-shot −0.037 (point estimates; the v1→v2 change is **not**
  given a paired CI here, see §5). Every v2 CI overlaps its v1 CI.
- SBERT weak stays the best text rung on v2 (0.834). Its lead over TF-IDF (C5) was
  +0.057 [0.009, 0.106] on v1 and is **not resolved** on v2 (+0.045 [−0.006, 0.095]).
- TF-IDF vs the metadata rule (E2) was unresolved on v1 (+0.019 [−0.042, 0.084]) and
  stays unresolved on v2 (+0.011).
- Within-family AUC barely moves (all |Δ| ≤ 0.024), as expected: inside one keyword
  family the title is less informative than across families.
- No selection was run on v2 (the record says so): every v2 rung is exploratory.

## 4. Deviations from the plan and from the brief

1. **`v1` cache keys omit the variant.** The brief asked that every cache key and
   fingerprint include it. `v1` keeps the pre-D53 names (`a2-pairs-<digest>`,
   `minilm-zero-shot`, `sbert-weak-b5bf5501d1e8`, …) so the committed `v1` records
   reproduce without retraining. `v2` names carry `-v2` or the `text_variant` key, so the
   two never share a cache. A test pins both facts.
2. **Seniority list** extended beyond the plan's examples (junior/middle/senior/lead/head/
   trainee) with jr, sr, mid and intern. It is stated in the record (`residual_leak.method`).
3. **Leak CIs.** The CV and JD shares resample the unit counted (CVs, JDs). A per-JD
   pair-level CV share, which resamples JDs per AGENTS.md rule 4, is also reported.
4. **`v2` record contents.** Rungs: random, keyword match, metadata rules (unchanged
   references), TF-IDF, BM25, SBERT zero-shot, SBERT weak. **Not included:** metadata +
   text hybrids (D53 forbids metadata at inference), cross-encoders and the learned hybrid
   (out of P1 per the brief), skill overlap (not in the 1.4 list), and the metadata
   classifier. `selection` holds a `not_run` note. The record leads with `text_variant`,
   `text_variant_definition` and `residual_leak`. The `v1` record layout is unchanged.
5. **Memory.** The first two `v2` runs were OOM-killed (exit 137, about 6.4 GB RSS) while
   other agents' test runs held about 5 GB. SBERT weak v2 was then trained in its own
   process: 51 min wall, peak RSS 6.5 GB. The evaluation then ran from the cached weights
   (1 min, 1.2 GB). The result does not depend on this, because the weights are
   deterministic for their fingerprint, but a fresh `v2 --check` on a busy 16 GB machine
   may need the same two steps.

## 5. Open items

- No **paired** v1→v2 delta per rung (same JDs, scores from both variants). The table
  compares point estimates and unpaired CIs only. Adding it needs both score sets in one
  record. Left for P7 if the comparison is needed.
- The residual leak (23% of CVs, 28% of JDs) is measured, not removed, as specified.
  Whether a scrubbed variant is wanted is a user decision, not taken here.

## 6. Reproduce

```bash
uv sync --all-groups
uv run pytest tests/test_text_variant.py                       # 11 tests, no data
uv run python -m candidate_screener.evaluation.text_leak      # the leak report alone
# first run trains SBERT weak v2 (~25 min fit on CPU, ~6.5 GB); on a busy machine pre-train:
uv run python -c "from candidate_screener.matching import sbert; sbert.SbertWeak('v2').fit()"
uv run python -m candidate_screener.evaluation.indomain --text-variant v2           # write
uv run python -m candidate_screener.evaluation.indomain --text-variant v2 --check   # PASS
uv run python -m candidate_screener.evaluation.indomain --check                     # v1 PASS
uv run python -m candidate_screener.baselines.run --check                           # PASS
uv run pytest                                                   # 562 passed (7 Oct 2026)
```
