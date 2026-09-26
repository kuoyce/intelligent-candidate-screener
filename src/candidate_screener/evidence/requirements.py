"""What a JD asks for: its vocabulary skills, in order of first mention, plus experience."""
from __future__ import annotations

import re
from dataclasses import dataclass

EXPERIENCE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", re.I)


@dataclass(frozen=True)
class Requirement:
    canonical: str
    text: str            # as written in the JD
    start: int
    end: int


def skill_requirements(jd_text: str, matcher) -> list[Requirement]:
    """One requirement per distinct canonical skill, keeping the first mention's span."""
    seen: dict[str, Requirement] = {}
    for m in matcher.find(jd_text):
        seen.setdefault(m.canonical, Requirement(m.canonical, m.text, m.start, m.end))
    return list(seen.values())


def years_required(jd_text: str) -> int | None:
    """The smallest 'N years' a JD states, or None when it states none."""
    found = [int(m.group(1)) for m in EXPERIENCE.finditer(jd_text)]
    return min(found) if found else None


def years_stated(cv_text: str) -> int | None:
    """The largest 'N years' a CV states (Djinni CVs usually lead with it)."""
    found = [int(m.group(1)) for m in EXPERIENCE.finditer(cv_text)]
    return max(found) if found else None
