# Local extraction record

`extraction-metrics.json` records the default seed-0 rule extraction run on 1,000
training CVs and 1,000 training JDs. It contains configuration, source/code digests
and aggregate coverage only. Actual entity surfaces and sample IDs live in the
git-ignored `data/processed/extraction/<run_id>/` artifacts.

These are descriptive figures from unlabelled Djinni documents, not precision,
recall, F1 or evidence of improved ranking. No gold extraction evaluation is
available in this checkout. Canonical IDs use a seed vocabulary rather than ESCO.

Rebuild local artifacts and validate their provenance/reproducibility with:

```bash
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0 --check
```

The check validates local inputs, code, artifact hashes and predictions. It does
not refresh this record or treat coverage as accuracy. Notebooks 10–12 provide
the executed input, extraction and evaluation-gate reports.
