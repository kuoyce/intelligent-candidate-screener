"""What the extractors find in one JD or CV: role family, vocabulary skills, years, seniority.

One definition for the hand-off tables (`evaluation.a2_export --extract`) and for inference.
Text only (D53); the JD title is the one allowed field (D54). Skills are canonical vocabulary
names; `skill_spans` give their character offsets in the text passed in, never a quote.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from candidate_screener.inference.schema import CVProfile, JDProfile

TOP_ROLES = 3
DECIMALS = 6


def skills(text: str, matcher) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, int, int], ...]]:
    """(affirmative, qualified, spans): distinct canonical skills in reading order; a skill seen
    affirmatively anywhere is not also qualified; spans are every match (canonical, start, end)."""
    from candidate_screener.extraction.rules import _context
    yes: dict[str, None] = {}
    no: dict[str, None] = {}
    spans = []
    for m in matcher.find(text):
        (yes if _context(text, m.start, m.end, "text") == "unqualified" else no)[m.canonical] = None
        spans.append((m.canonical, m.start, m.end))
    return tuple(yes), tuple(k for k in no if k not in yes), tuple(spans)


def top_roles(proba: np.ndarray, labels, k: int = TOP_ROLES) -> list[tuple[tuple[str, float], ...]]:
    """Per row, the `k` most probable role families as (label, probability), best first."""
    top = np.argsort(-proba, axis=1)[:, :k]
    return [tuple((labels[j], float(row[j])) for j in i) for i, row in zip(top, proba)]


def _round(p: float) -> float:
    """numpy's rounding, as the hand-off tables were written with it."""
    return float(np.round(np.float64(p), DECIMALS))


def roles_text(roles: tuple[tuple[str, float], ...]) -> str:
    """'label:p;...' at 3 dp, the hand-off tables' `top_roles` column."""
    return ";".join(f"{label}:{p:.3f}" for label, p in roles)


def extract_jds(texts, titles, model, matcher, ids=None) -> list[JDProfile]:
    """JD profiles for aligned `texts` and `titles` (a title may be None)."""
    from candidate_screener.extraction import experience, role_family
    texts = list(texts)
    roles = top_roles(role_family.predict_proba(texts, "jd", model), model.labels)
    ids = list(ids) if ids is not None else [None] * len(texts)
    out = []
    for text, title, r, jd_id in zip(texts, titles, roles, ids):
        yes, qualified, spans = skills(text, matcher)
        exp = experience.jd_experience(text)
        out.append(JDProfile(
            jd_id=jd_id, jd_title=title, role=r[0][0], role_prob=_round(r[0][1]), top_roles=r,
            skills=yes, skills_qualified=qualified, min_years=exp.min_years, max_years=exp.max_years,
            seniority=experience.position_seniority(title), skill_spans=spans))
    return out


def extract_cvs(texts, model, matcher, reference: dt.date | None = None, ids=None) -> list[CVProfile]:
    """CV profiles for `texts`. `reference` is "present" in date ranges (None = A2's 2023-09-01)."""
    from candidate_screener.extraction import experience, role_family
    texts = list(texts)
    reference = reference or experience.REFERENCE_DATE
    roles = top_roles(role_family.predict_proba(texts, "cv", model), model.labels)
    ids = list(ids) if ids is not None else [None] * len(texts)
    out = []
    for text, r, cv_id in zip(texts, roles, ids):
        yes, qualified, spans = skills(text, matcher)
        exp = experience.cv_experience(text, reference)
        level, level_source = experience.text_seniority(text, exp.years)
        out.append(CVProfile(
            cv_id=cv_id, role=r[0][0], role_prob=_round(r[0][1]), top_roles=r,
            skills=yes, skills_qualified=qualified, years=exp.years, years_source=exp.source,
            seniority=level, seniority_source=level_source, skill_spans=spans))
    return out


def extract_jd(text: str, title: str | None, model, matcher) -> JDProfile:
    return extract_jds([text], [title], model, matcher)[0]


def extract_cv(text: str, model, matcher, reference: dt.date | None = None) -> CVProfile:
    return extract_cvs([text], model, matcher, reference)[0]
