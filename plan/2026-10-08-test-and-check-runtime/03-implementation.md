# 03 — Implementation record (8 Oct 2026)

Branch `chore/test-check-runtime`, cut from `feat/shortlist-recall` and merged back into it.
Implemented by subagents, one batch per workstream group (W1+W9a, W2–W5, W9b+W7+W8), then
reviewed here. **No committed record changed:** all 21 records reproduce byte for byte under
`checks --cold --all`, and `git status` was clean after every run.

## Commits

| Commit | W | Change |
|---|---|---|
| `154fe0a` | W1 | numpy kappa in `llm_recheck`: same arithmetic as sklearn, so bit-identical (tested with `==` against sklearn on 80 seeded cases); sklearn's ValueErrors kept |
| `81b10b8` | W9a | pytest `--tb=short`; per-test filters for the by-design All-NaN CI warnings |
| `cbf871c` | — | D55 (cold/warm tiers); `logs/` git-ignored |
| `a48d781` | W2 | `trigram_cosine` `lru_cache(2**20)` and a per-string 3-gram Counter memo |
| `9dd5795` | W3 | `indomain.auc_memo()`: per-run memo of `per_jd_auc`, keyed by a SHA-256 of its inputs; applied to `indomain.evaluate` and `stage1.evaluate` |
| `5fb5b78` | W4 | call-level memo of `CrossEncoder._predict` (same weights, `max_seq_length`, batch size and ordered texts only) |
| `08fbabb` | W5 | `role_family`: `title_spans` memo per text and alias table; `predict_proba` replayed only for an identical sorted batch |
| `defd5a6` | W9b | `neural.quiet_loading()` before every model load; the "Loading weights" bars are gone (indomain v1 output 25 → 1 line) |
| `6f30d41` | W7 | `--check --warm` on indomain, a1_secondary, retrieval_pilot (persistent CE score store), extraction.evaluate (alias of `--reuse-weights`), role_family (`build(train=False)`, writes nothing), dev_pools (`force=False`). `--warm` without `--check` is an argparse error |
| `cb128de` | W8 | `candidate_screener.checks`: registry, `--affected` via a static `ast` import closure, scheduler (`-j 2`, run-alone checks), `logs/checks/`, `summary.tsv` |
| (this commit) | review | atomic write of the SBERT embedding cache (`sbert.Encoder`); docs |

## Measured outcome

| What | Before | After |
|---|---:|---:|
| `uv run pytest` | 290 s (594 tests) | **55 s** (615 tests) |
| all checks, cold | 5,169 s serial | **3,000 s** (51 min) via `checks --cold --all -j 2`; all 21 PASS |
| all live checks, warm | n/a | **927 s** (15.4 min), `checks --warm --all`; 17 PASS |
| superseded checks, warm (`-j 1`) | n/a | 210 s; 4 PASS |
| `--affected` with a clean tree | n/a | 0 checks, 0 s |
| `--affected` after editing `role_filter.py` | 21 checks if run by hand | 3 checks (shortlist_dev, `--variants`, shortlist_final), 568 s |

Per check (cold figures under `-j 2` share the cores, so e.g. extraction took 886 s here
against 640 s alone):

| Check | Before, serial cold (s) | Cold, kernels only, alone (s) | Warm (s) |
|---|---:|---:|---:|
| role_family | 781 | (refit, unchanged) | **44** |
| a1_secondary † | 773 | 729 | **32** |
| shortlist_dev | 672 | **288** | 286 |
| extraction | 640 | (retrains NER, unchanged) | **218** |
| shortlist_dev --variants | 583 | **279** | 317 (at `-j 2`) |
| retrieval_pilot † | 523 | 465 | **66** |
| indomain v1 † | 453 | **146** | 50 |
| stage1 | 181 | **119** | 132 (at `-j 2`) |
| shortlist_final | 163 | 149 | 157 |
| dev_pools | 79 | (rebuild, unchanged) | **40** |
| the other 11 | 287 | ≈ same | ≈ same |

† superseded (Q3): cold, or when `--affected` selects them.

## Deviations from the plan

1. **W6 dropped** (Q6).
2. **Kernel verification.** The plan called for the cold checks of each change to run after
   that change. W2–W5 were verified together on the checks that use them (nine cold checks,
   run one at a time), then everything ran again in `checks --cold --all`. Only
   `shortlist_dev` was timed for W2 alone (372 s). The other savings are attributed to W2–W5
   jointly.
3. **W5 is narrower than planned.** Probabilities are not memoised per text: computing a row
   in a different batch could change its float, which would break bit-identity. Only spans
   are per text; probabilities are replayed for an identical batch. The saving was 84 s, not
   the ~120 s estimated.
4. **W3 also covers `stage1.evaluate`** (one decorator line).
5. **Run-alone set** is measured, not assumed: role_family (4.9 GB), shortlist_dev (6.2–7.5
   GB), dev_pools (4.8 GB), scoring_batch (4.7 GB). `shortlist_dev --variants` peaked at 4.05
   GB, so it may share the machine.
6. **`quiet_loading`** sets `TOKENIZERS_PARALLELISM=false` (`setdefault`) in the extraction
   processes too. This does not affect numerics: the cold extraction check passed with 0 notes.
7. **Added in review:** `sbert.Encoder` now writes its `.npz` cache atomically. `-j 2` made
   two concurrent writers possible. Before, a reader could have seen a half-written file; a
   lost update is still possible but is only a later cache miss.
8. **Docs:** the `AGENTS.md` cost table uses measured times, not the plan's estimates. Pytest came in at 55 s
   against the 35–45 s estimate, because 21 new tests were added (the kappa
   bit-identity test alone takes ~6 s).

## Known limitations

- **SBERT is not cold.** `sbert.Encoder` keeps a persistent embedding cache that both tiers
  read. This was true before this plan and is out of scope.
- **Warm shortlist checks are not faster.** shortlist_dev, `--variants` and shortlist_final
  have no cached model step to skip, so their warm time equals their cold time (~5 min
  each). The remaining cost is TF-IDF pair scoring (W6, dropped) and role-family
  featurisation.
- **`--affected` is static.** It follows imports, not data flow. A change to a committed
  file under `output/` or `docs/data/` selects every check, so it errs toward running more.
