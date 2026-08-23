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

#: Written by `candidate_screener.data.verify`; the acceptance record of a download.
MANIFEST = DOCS_DATA / "acquisition-manifest.json"
#: Written by `candidate_screener.data.profile`; the machine-readable Verified figures.
PROFILE_METRICS = DOCS_DATA / "profile-metrics.json"


def ensure_data_dirs() -> None:
    """Create the working directories. `data/` is git-ignored in its entirety."""
    for d in (RAW, INTERIM, PROCESSED, VOCAB):
        d.mkdir(parents=True, exist_ok=True)
