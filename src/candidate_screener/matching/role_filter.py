"""Stage 1 with the role as a filter: C3 and the Y rungs (plan 2026-10-07-shortlist-recall).

Inputs (D54): the JD title `jd_title`, the JD text and the CV text, nothing else.
`_inputs` hands the components a frame with only those columns (and `jd_id`, which groups
a pool), so CV `Position`, `Primary Keyword` and the experience fields cannot be read.

- **C3** `family_match = 1 if C1(jd_title) ∈ C2(cv) else 0` (hard: both sides are
  categorical, decision 2). `title_similarity`: the largest character 3-gram cosine
  between the JD's normalised title and the CV's title spans (extracted strings, so fuzzy
  is allowed); 0 when the CV has no span.
- **Y1** `2 × family_match + experience_compatible` (P4's extracted rule, unknown
  compatible, amendment A1 of the running plan). A level, as in 1a: a family match always
  outranks a mismatch.
- **Y2 / Y4** Y1 plus TF-IDF / SBERT weak, combined as 1a+2 (per-JD min-max × 0.999).
- **Y2t** Y1 plus 0.999 × the mean of per-JD min-max TF-IDF and `title_similarity`.

Options `k` (classifier top-k on the CV side) and `context` (role-context spans only) are
chosen on the development set (`evaluation.shortlist_dev`), never on the final set (D46).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from candidate_screener.matching import lexical
from candidate_screener.matching.base import BaseSystem

INPUT_COLUMNS = ("jd_id", "jd_title", "jd_text", "cv_text")
#: Per-text extraction results, shared by every rule built on the same model objects (the
#: dev set scores eight option settings over the same CVs).
_SHARED: dict[tuple[int, int], tuple[dict, dict]] = {}


def _inputs(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.loc[:, list(INPUT_COLUMNS)]


def _grams(s: str, n: int = 3) -> Counter:
    s = f" {s} "
    return Counter(s[i:i + n] for i in range(max(len(s) - n + 1, 0)))


def trigram_cosine(a: str, b: str) -> float:
    """Character 3-gram cosine (Q-R5), on the strings padded by one space each side."""
    if not a or not b:
        return 0.0
    ga, gb = _grams(a), _grams(b)
    dot = sum(v * gb[g] for g, v in ga.items() if g in gb)
    return dot / math.sqrt(sum(v * v for v in ga.values()) * sum(v * v for v in gb.values()))


def jd_title_key(title: object) -> str:
    """The JD title as `title_similarity` compares it: normalised, else lower-cased."""
    from candidate_screener.extraction.title_profile import normalise_title
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return ""
    return normalise_title(title) or " ".join(str(title).lower().split())


def per_jd_minmax(jd_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`values` scaled into [0, 1] within each JD (a constant JD gets 0)."""
    frame = pd.DataFrame({"jd": jd_ids, "t": values})
    lo = frame.groupby("jd").t.transform("min")
    span = frame.groupby("jd").t.transform("max") - lo
    return ((frame.t - lo) / span.where(span > 0, 1)).clip(0, 1).to_numpy()


class RoleFilterRule(BaseSystem):
    """Y1: `2 × family_match + experience_compatible`, with C3's features on the side."""
    name, rung, stage = "role filter", "Y1", "Stage 1 — role filter (post-hoc)"
    can_explain = True

    def __init__(self, k: int = 0, context: bool = False, role_model=None,
                 title_model=None) -> None:
        super().__init__()
        self.k, self.context = k, context
        self.role_model, self.title_model = role_model, title_model
        self._cv: dict[str, object] = {}
        self._jd: dict[str, tuple] = {}
        self.name = f"role filter (k={k}, context={'on' if context else 'off'})"

    def _fit(self, train):
        from candidate_screener.extraction import jd_title, role_family
        if self.role_model is None:
            self.role_model = role_family.load_role_model()
        if self.title_model is None:
            self.title_model = jd_title.load_title_model()
        self._cv, self._jd = _SHARED.setdefault((id(self.role_model), id(self.title_model)),
                                                ({}, {}))
        self.trained_on = (f"role-family classifier {self.role_model.fingerprint} (P3, A2 "
                           f"train fit part); title classifier {self.title_model.fingerprint} "
                           "(same fit part's Position strings); experience: rules (P4)")

    def _cv_roles(self, texts: list[str]):
        from candidate_screener.extraction import role_set
        todo = sorted({t for t in texts if t not in self._cv})
        if todo:
            self._cv.update(zip(todo, role_set.cv_roles(todo, self.role_model)))
        return [self._cv[t] for t in texts]

    def _jd_family(self, titles: list) -> list[str | None]:
        from candidate_screener.extraction import jd_title
        keys = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)
                for t in titles]
        todo = sorted({t for t in keys if t not in self._jd})
        if todo:
            fam = jd_title.jd_family(todo, self.title_model)
            self._jd.update(zip(todo, zip(fam.family, [jd_title_key(t) for t in todo])))
        return [self._jd[t] for t in keys]

    def components(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """`jd_family`, `family_match`, `title_similarity`, `experience_compatible`."""
        from candidate_screener.extraction import experience
        x = _inputs(pairs)
        cv = self._cv_roles(["" if pd.isna(t) else str(t) for t in x.cv_text])
        jd = self._jd_family(x.jd_title.tolist())
        match = [int(f is not None and f in r.families(self.k, self.context))
                 for (f, _), r in zip(jd, cv)]
        sim_cache: dict[tuple, float] = {}

        def sim(key: str, titles: tuple[str, ...]) -> float:
            if (key, titles) not in sim_cache:
                sim_cache[(key, titles)] = max((trigram_cosine(key, t) for t in titles),
                                               default=0.0)
            return sim_cache[(key, titles)]
        sims = [sim(key, r.titles(self.context)) for (_, key), r in zip(jd, cv)]
        exp = experience.experience_compatible_extracted(
            x.jd_text.tolist(), x.cv_text.tolist(), unknown_compatible=True).to_numpy()
        return pd.DataFrame({"jd_family": [f for f, _ in jd], "family_match": match,
                             "title_similarity": sims, "experience_compatible": exp})

    def level(self, pairs: pd.DataFrame) -> np.ndarray:
        c = self.components(pairs)
        return 2 * c.family_match.to_numpy() + c.experience_compatible.to_numpy()

    def _score(self, pairs):
        return self.level(pairs)

    def options(self) -> dict:
        return {"k": self.k, "context": self.context}


class RoleFilterPlus(lexical.MetadataPlus):
    """Y2 / Y4: Y1 plus a text rung, combined as 1a+2 (`MetadataPlus` with Y1 as the level)."""
    stage = "Stage 1 — role filter + text (post-hoc)"

    def __init__(self, text_system: BaseSystem, rung: str, rule: RoleFilterRule) -> None:
        super().__init__(text_system, rung)
        self.rule = rule
        self.name = f"{rule.name} + {text_system.name}"

    def fit(self, train=None):
        if self.rule.role_model is None or self.rule.title_model is None:
            self.rule.fit(None)
        super().fit(train)
        self.trained_on = f"{self.rule.trained_on}; {self.text.trained_on}"

    def _level(self, pairs) -> np.ndarray:
        return self.rule.level(pairs)

    def params(self):
        return self.text.params()


class RoleFilterTitle(RoleFilterPlus):
    """Y2t: Y1 + 0.999 × mean(per-JD min-max text score, title_similarity)."""

    def __init__(self, text_system: BaseSystem, rung: str, rule: RoleFilterRule) -> None:
        super().__init__(text_system, rung, rule)
        self.name = f"{rule.name} + mean({text_system.name}, title similarity)"

    def _score(self, pairs):
        x = _inputs(pairs)
        c = self.rule.components(pairs)
        text = per_jd_minmax(x.jd_id.to_numpy(), self.text.score(pairs).to_numpy())
        blend = (text + c.title_similarity.to_numpy()) / 2
        level = 2 * c.family_match.to_numpy() + c.experience_compatible.to_numpy()
        return level + 0.999 * blend
