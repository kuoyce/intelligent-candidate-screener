"""Stage 1 rungs built from extraction (plan 2026-10-06, P7; 02 §3.4, 06-registration §2).

Every X rung reads **JD text and CV text only** (D53): the role family and the experience
bands come from the extractors, never from `Primary Keyword`, `Exp Years`, `Experience
Years` or `Position`. `_text_only` hands the extractors a frame with nothing else in it, so
a rung cannot read a field by accident.

- **X1** `2 × role_match + experience_compatible`, the 1a formula unchanged, both terms
  extracted. `role_match` is P3's soft agreement `sum_k p_jd(k) p_cv(k)` in [0, 1];
  `experience_compatible` is P4's rule with an unknown band on either side counted as
  compatible (amendment A1).
- **X2 / X3 / X4** X1 plus TF-IDF / skill overlap / SBERT weak, combined exactly as 1a+2:
  the text score min-max scaled per JD into [0, 0.999] and added to X1.

Because `role_match` is soft, X1 is not a three-level rule as 1a is: a text term can
reorder candidates whose role agreement differs by less than half a unit. That is the
registered formula (02 §3.4: "the 1a formula unchanged"); the record states it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from candidate_screener.matching import lexical
from candidate_screener.matching.base import BaseSystem

TEXT_COLUMNS = ("jd_text", "cv_text")


def _text_only(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.loc[:, list(TEXT_COLUMNS)]


class ExtractedRule(BaseSystem):
    """X1: the 1a rule on extracted fields."""
    name, rung, stage = "extracted rule", "X1", "Stage 1 — extracted rules"
    trained_on = ("role-family classifier, A2 train region, text_v2 (P3); experience: "
                  "rules, nothing fitted (P4)")
    can_explain = True

    def __init__(self, role_model=None) -> None:
        super().__init__()
        self.role_model = role_model

    def _fit(self, train):
        if self.role_model is None:
            from candidate_screener.extraction import role_family
            self.role_model = role_family.load_role_model()
        self.trained_on = f"{ExtractedRule.trained_on}; role model {self.role_model.fingerprint}"

    def components(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """`role_match` and `experience_compatible`, row-aligned with `pairs`."""
        from candidate_screener.extraction import experience, role_family
        text = _text_only(pairs)
        jd, cv = text.jd_text.tolist(), text.cv_text.tolist()
        return pd.DataFrame({
            "role_match": role_family.role_match(jd, cv, self.role_model),
            "experience_compatible": experience.experience_compatible_extracted(
                jd, cv, unknown_compatible=True).to_numpy()})

    def _score(self, pairs):
        c = self.components(pairs)
        return 2 * c.role_match.to_numpy() + c.experience_compatible.to_numpy()


class ExtractedPlus(lexical.MetadataPlus):
    """X2–X4: X1 plus a text rung, combined as 1a+2 (`MetadataPlus` with X1 as the level)."""
    stage = "Stage 1 — extracted rules + text"

    def __init__(self, text_system: BaseSystem, rung: str, rule: ExtractedRule | None = None
                 ) -> None:
        super().__init__(text_system, rung)
        self.rule = rule or ExtractedRule()
        self.name = f"extracted rule + {text_system.name}"

    def fit(self, train=None):
        self.rule.fit(None)
        super().fit(train)
        self.trained_on = f"{self.rule.trained_on}; {self.text.trained_on}"

    def _level(self, pairs) -> np.ndarray:
        return self.rule.score(pairs).to_numpy()

    def params(self):
        return self.text.params()
