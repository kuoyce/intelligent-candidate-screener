# Using the system from an app: the inference API

This page covers `candidate_screener.inference`, the API that serves the reported system
(D63, and D64 with the reranker) to an app such as the Hiring Desk: setting it up once per
machine, the `Screener` calls, and what each option and output means. How the Hiring Desk
uses it is in [`hiring-desk.md`](hiring-desk.md). Moved from README.md on 1 Nov 2026.

`candidate_screener.inference` serves the reported system to an app (plan
`2026-10-23-inference-api`). It imports no Streamlit and fits nothing: every part (role model,
vocabulary, concept frequencies, TF-IDF space, SBERT weak, the frozen L1) is loaded, so a new JD
or an uploaded CV is inference only. It runs locally on a machine with `data/`; once per machine:

```bash
uv sync --all-groups
uv run python -m candidate_screener.inference --cache-parts   # saved concept df + TF-IDF space (~40 s)
uv run python -m candidate_screener.inference --demo <final-set jd_id>   # smoke: a top 10, no CV text
```

```python
from candidate_screener import inference

s = inference.Screener.load()                        # ~4 s; wrap in st.cache_resource
res = s.rank(jd_text, cvs, jd_title="QA Engineer")   # cvs: cv_id + cv_text only (D53)
for r in res.top(10):                                # what the UI shows
    print(r.rank, r.cv_id, r.score)
why = s.explain(res, res.top(1)[0].cv_id)            # for the recruiter: reasons, skills, years
```

- **System.** D63 by default: the frozen L1 (`output/rerank/l1-chat3-model.json`) on F12 ranks
  every CV passed in. `rerank=True` is D64: L1's top 50 reordered by min-max L1 + 0.25 × min-max
  C2 (Qwen3-Reranker-0.6B, zero-shot). C2 costs ~2.5-7 s of CPU per pair, so up to a few minutes
  per JD; `progress(done, total)` reports it. `system="X2"` gives the baseline.
- **Uploads only.** CVs come from the caller; there is no CV store or search.
- **Scaling (`scaling="background"`, the default).** `tfidf_scaled` is min-max scaled over the
  CVs scored together, and L1 was trained on 500-CV pools. The CVs passed in are therefore padded
  to 500 with a fixed sample of A2 train-region CVs that are scored but never returned. Scaling
  over the uploads alone (`scaling="pool"`) reorders small pools: at 10 CVs the top candidate
  differs about 1 time in 7 (`output/inference/pool-scaling.json`). The first background call
  takes ~18 s (the 500 background CVs are embedded once and cached), later calls ~1 s.
- **`reference_date`.** "present" in a CV's date range: today by default (uploads); A2 documents
  use 2023-09-01 (`Screener.a2_pool` returns it).
- **Output.** Frozen dataclasses (`Screening`, `Ranked`, `JDProfile`, `CVProfile`), each with
  `as_dict()` (JSON-safe); `Screening.to_frame()`. No CV text is returned: anything an app
  displays from a CV passes `annotation.redact`.
- **Demo data.** `Screener.a2_pool(jd_id)` gives a final-set JD's text, title and 500-CV pool;
  pass `jd_id=` to `rank` so the C2 scores already cached are reused.
- **Check.** `inference --check` drives the API against the committed records (~12 min).
