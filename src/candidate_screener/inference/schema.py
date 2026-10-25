"""What the inference API returns: frozen dataclasses, JSON-safe through `as_dict()`.

Field names follow the hand-off tables (`output/matching/a2-jd-extraction.csv`,
`a2-cv-extraction.csv`, `a2-features.csv`), so a table and an API call read the same way. A2's
`stratum` is not a field: it is metadata of the benchmark, not of a document.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field

import pandas as pd


def _plain(value):
    """JSON-safe: tuples to lists, dates to ISO strings, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dt.date):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):   # numpy scalars
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class JDProfile:
    jd_id: str | None
    jd_title: str | None
    role: str
    role_prob: float
    top_roles: tuple[tuple[str, float], ...]      # (family, probability), best first
    skills: tuple[str, ...]                       # canonical, affirmative mentions
    skills_qualified: tuple[str, ...]             # only learning / negated mentions
    min_years: float | None
    max_years: float | None
    seniority: str | None                         # from the title only
    skill_spans: tuple[tuple[str, int, int], ...] = ()   # (canonical, start, end) in the JD text

    def as_dict(self) -> dict:
        return _plain(asdict(self))


@dataclass(frozen=True)
class CVProfile:
    cv_id: str | None
    role: str
    role_prob: float
    top_roles: tuple[tuple[str, float], ...]
    skills: tuple[str, ...]
    skills_qualified: tuple[str, ...]
    years: float | None
    years_source: str | None                      # "statement" | "ranges" | None
    seniority: str | None
    seniority_source: str | None
    skill_spans: tuple[tuple[str, int, int], ...] = ()   # offsets in the CV text passed in

    def as_dict(self) -> dict:
        return _plain(asdict(self))


@dataclass(frozen=True)
class Ranked:
    cv_id: str
    rank: int                                     # 1 = best; ties broken by cv_id
    score: float                                  # what the ranking used (system below)
    l1_score: float
    x2_score: float
    c2_score: float | None                        # only for CVs C2 scored (D64)
    features: dict = field(default_factory=dict)  # F12, by name
    in_top_k: bool = False

    def as_dict(self) -> dict:
        return _plain(asdict(self))


@dataclass(frozen=True)
class Screening:
    jd: JDProfile
    cvs: dict                                     # cv_id -> CVProfile
    ranked: tuple[Ranked, ...]                    # every CV passed in, best first
    system: str                                   # "D63", "D64" or "X2"
    scaling: str                                  # "pool" or "background"
    pool_size: int                                # CVs the TF-IDF term was scaled over
    top_k: int
    reference_date: dt.date                       # "present" in CV date ranges

    def top(self, k: int | None = None) -> tuple[Ranked, ...]:
        """The best `k` (default `top_k`): what the UI shows."""
        return self.ranked[:self.top_k if k is None else k]

    def get(self, cv_id: str) -> Ranked:
        for r in self.ranked:
            if r.cv_id == cv_id:
                return r
        raise KeyError(cv_id)

    def to_frame(self) -> pd.DataFrame:
        """One row per CV: rank, scores and the features."""
        return pd.DataFrame([{**{k: v for k, v in asdict(r).items() if k != "features"}, **r.features}
                             for r in self.ranked])

    def as_dict(self) -> dict:
        return _plain({"jd": self.jd.as_dict(), "cvs": {k: v.as_dict() for k, v in self.cvs.items()},
                       "ranked": [r.as_dict() for r in self.ranked], "system": self.system,
                       "scaling": self.scaling, "pool_size": self.pool_size, "top_k": self.top_k,
                       "reference_date": self.reference_date})
