"""Shared synthetic corpora.

Every fixture here is inline text. `data/` is git-ignored, so a suite that needs it
on disk would not run on a fresh clone and therefore would not be run at all — only
`test_baseline_golden.py` touches real data, and it skips visibly when it is absent.

Two properties of the corpus are load-bearing and easy to lose in an edit:

- **Six documents, not two.** `BM25Okapi`'s idf is `log(N - df + 0.5) - log(df + 0.5)`,
  which is exactly 0.0 when a term appears in half a two-document corpus. A corpus
  that small makes every BM25 assertion vacuous.
- **Every domain appears in at least two documents.** The baseline's `min_df=2` drops
  any term seen once, so a document whose vocabulary is unique to it vectorises to
  all-zeros and its cosine against *itself* is 0.0, not 1.0. That is a true property
  of the configuration, not a bug — but it would make the shared-space regression
  test pass for the wrong reason. Hence one JD per resume domain.
"""
from __future__ import annotations

import pytest

#: Six short documents across three unrelated domains, so term overlap is obvious by
#: inspection and no term is so common that its idf collapses.
RESUMES = [
    "senior python engineer sql aws docker microservices",
    "backend python django api postgres redis",
    "registered nurse clinical patient care hospital ward",
    "primary teacher classroom lesson planning curriculum",
    "line chef kitchen menu food preparation service",
    "delivery driver truck route logistics warehouse",
]

JDS = [
    "hiring a python developer with sql and aws experience",
    "ward nurse wanted for patient clinical care in a busy hospital",
    "chef required for kitchen menu and food preparation service",
    "primary teacher wanted for classroom lesson planning and curriculum",
    "delivery driver needed for truck route logistics from our warehouse",
]

#: Held out of every fit corpus on purpose — the D24 regression scores against it.
HELD_OUT_RESUME = "staff python developer sql aws kubernetes terraform"


@pytest.fixture
def resumes() -> list[str]:
    return list(RESUMES)


@pytest.fixture
def jds() -> list[str]:
    return list(JDS)


@pytest.fixture
def held_out_resume() -> str:
    return HELD_OUT_RESUME
