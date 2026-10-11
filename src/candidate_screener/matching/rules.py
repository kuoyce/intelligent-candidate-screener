"""Stage 1 rungs: the random floor, metadata rules, and skill overlap. Nothing is fitted."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from candidate_screener.matching import corpus
from candidate_screener.matching.base import BaseSystem


class Random(BaseSystem):
    """Rung 0: a seeded random order per pair, stable across runs."""
    name, rung, stage = "random", "0", "floor"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def _score(self, pairs):
        # A hash of (seed, pair_id): independent of row order, so a subset scores the same.
        return [int(hashlib.sha256(f"{self.seed}:{p}".encode()).hexdigest()[:12], 16) / 16**12
                for p in pairs.pair_id]


class KeywordMatch(BaseSystem):
    """Rung 1a (reference): the JD's and CV's `Primary Keyword` agree."""
    name, rung, stage = "keyword match", "1a", "Stage 1 — rules"
    can_explain = True

    def _score(self, pairs):
        return corpus.keyword_match(pairs)


class Metadata(BaseSystem):
    """Rung 1a: `2 × keyword match + experience compatible` — keyword first, experience
    breaks ties. A fixed formula, chosen before any held-out rung was scored."""
    name, rung, stage = "metadata rules", "1a", "Stage 1 — rules"
    can_explain = True

    def _score(self, pairs):
        return 2 * corpus.keyword_match(pairs) + corpus.experience_compatible(pairs)


class SkillOverlap(BaseSystem):
    """Rung 1b: the share of the JD's vocabulary skills that the CV also names.

    Coverage, not a count, so a long JD is not favoured. A JD naming no vocabulary skill
    scores every candidate 0 — the rung has nothing to say there, and says so.
    """
    name, rung, stage = "skill overlap", "1b", "Stage 1 — rules (as specified)"
    can_explain = True

    def __init__(self, matcher=None) -> None:
        super().__init__()
        self._matcher = matcher

    @property
    def matcher(self):
        if self._matcher is None:
            from candidate_screener.extraction.rules import default_matcher
            self._matcher = default_matcher()
        return self._matcher

    def _score(self, pairs):
        jd_skills: dict[str, set[str]] = {}
        out = []
        for jd, cv in zip(pairs.jd_text, pairs.cv_text):
            if jd not in jd_skills:
                jd_skills[jd] = self.matcher.canonicals(jd)
            wanted = jd_skills[jd]
            out.append(len(wanted & self.matcher.canonicals(cv)) / len(wanted) if wanted else 0.0)
        return out


def metadata_classifier(pairs: pd.DataFrame) -> pd.Series:
    """The metadata rule as a Fit / No Fit call: Fit iff the keyword matches."""
    return pd.Series(np.where(corpus.keyword_match(pairs) == 1, "Fit", "No Fit"),
                     index=pairs.pair_id.to_numpy())
