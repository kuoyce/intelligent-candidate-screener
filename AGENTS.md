# Working in this repository

Conventions an agent (or a new team member) needs before touching anything. The project's
process rules live in [`CLAUDE.md`](CLAUDE.md); this file is the repository-specific detail.

## Environment

`uv` only — never `pip install` into the venv, never a bare `python`.

```bash
uv sync                                          # install
uv run python -m candidate_screener.data.verify --all
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0X-*.ipynb
```

The project is an installed package (`src/` layout), so `import candidate_screener` works
everywhere, including in notebook kernels.

## Where code goes

| Kind | Location |
|---|---|
| Reusable project code | `src/candidate_screener/…` — invoked as `python -m candidate_screener.<module>` |
| Exploration and visualisation | `notebooks/NN-topic.ipynb`, numbered in reading order |
| Plans and their implementation records | `plan/{date}-{plan_name}/` |
| Data documentation | `docs/data/` — the catalog, and one card per adopted source |

Do **not** put runnable scripts under `plan/`. A plan is a record of a decision; code that
outlives the decision belongs in the package. (This was deviation V1 of the acquisition phase.)

## Data rules

1. **`data/` is git-ignored in its entirety.** Everything in it must be reconstructible from
   `data/README.md` plus the fetch scripts. Traceability comes from
   `docs/data/acquisition-manifest.json` (SHA-256 per file), not from committing bytes.
2. **Add a source by adding a `Source(...)` to `src/candidate_screener/data/sources.py`** —
   it drives `fetch`, `verify` and the card. Record the expected row count so `verify` can
   fail when a publisher re-uploads.
3. **Never trust a dataset card.** Three of them were wrong (see
   `plan/2026-08-19-data-strategy/01-requirements-and-findings.md` §3). Mark figures
   **Verified** only after measuring them yourself with `profile`.
4. **PII.** DataTurks (B1) has `Name` and `Email Address` as annotated classes; livecareer
   (C1/C2) resumes are real documents. Redact before display and before committing any
   notebook output.
5. **Contamination.** 44 livecareer resumes overlap the A1 benchmark — exclude
   `data/interim/c1_a1_contamination.csv` from NER training sets and distractor pools.

## Notebooks

- Numbered, one topic each, executed before committing so the outputs are visible on GitHub.
- Redact PII **inside the notebook** rather than relying on stripping outputs later.
- Cite measured figures from `docs/data/profile-metrics.json` instead of restating prose —
  numbers then update when the profile is re-run.

## Documentation duty

Any change to what data exists or what it is known to contain updates, in the same commit:

- `docs/data/data-catalog.md` — mark a corrected figure *Superseded {date}* in place rather
  than deleting the old one; the audit trail is the point.
- the source's card in `docs/data/cards/`
- the plan's implementation record, if the change is a deviation from the approved plan
