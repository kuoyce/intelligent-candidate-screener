"""Canonical filesystem layout.

Every module resolves paths through here so that the layout committed in
`docs/data/data-catalog.md` and `data/README.md` has exactly one definition.

    data/
      raw/        as-downloaded, never edited
      interim/    parsed / de-identified / repaired
      processed/  model-ready splits and pools
      vocab/      ESCO + skill-frequency extracts
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
VOCAB = DATA / "vocab"

DOCS = PROJECT_ROOT / "docs"
DOCS_DATA = DOCS / "data"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

#: Committed model records — the counterpart of `docs/data/` for things we *fit*
#: rather than things we downloaded. Unlike `data/`, this directory is in git:
#: `baselines.run --check` only has a fixed point to diff against if it is
#: *(Phase 4, Q20)*.
OUTPUT = PROJECT_ROOT / "output"

#: Written by `candidate_screener.data.verify`; the acceptance record of a download.
MANIFEST = DOCS_DATA / "acquisition-manifest.json"
#: Written by `candidate_screener.data.profile`; the machine-readable Verified figures.
PROFILE_METRICS = DOCS_DATA / "profile-metrics.json"
#: Written by `candidate_screener.baselines.run`; the golden record of the Stage 1
#: classical baseline. Committed — see `output/baselines/README.md`.
BASELINE_METRICS = OUTPUT / "baselines" / "baseline-metrics.json"
#: Git-ignored regenerable cache: the fitted pickles and the per-pair scores *(D23)*.
BASELINE_CACHE = PROCESSED / "baselines"


def ensure_data_dirs() -> None:
    """Create the working directories. `data/` is git-ignored in its entirety."""
    for d in (RAW, INTERIM, PROCESSED, VOCAB):
        d.mkdir(parents=True, exist_ok=True)
