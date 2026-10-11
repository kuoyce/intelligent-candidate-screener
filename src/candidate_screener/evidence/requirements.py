"""What a JD asks for: vocabulary skills, free-text requirements, and experience (D49).

Canonical requirements come from the rules matcher (deterministic, verbatim aliases).
The learned tagger adds the requirement phrases the vocabulary cannot name ("stakeholder
communication") as free-text requirements, which can only be `supported` or `missing`,
never `matched`. It was the better JD-side extractor on SkillSpan (08-extraction.md §6).
Tagger spans longer than `MAX_FREE_TEXT_WORDS` are dropped: its strict precision is 0.50,
and a long span makes a vague requirement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EXPERIENCE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", re.I)


MAX_FREE_TEXT_WORDS = 6


@dataclass(frozen=True)
class Requirement:
    canonical: str
    text: str            # as written in the JD
    start: int
    end: int
    kind: str = "skill"  # skill (vocabulary) | free_text (tagger span, no vocabulary alias)


def skill_requirements(jd_text: str, matcher) -> list[Requirement]:
    """One requirement per distinct canonical skill, keeping the first mention's span."""
    seen: dict[str, Requirement] = {}
    from candidate_screener.extraction.rules import _context
    for m in matcher.find(jd_text):
        if _context(jd_text, m.start, m.end, "text") == "unqualified":
            seen.setdefault(m.canonical, Requirement(m.canonical, m.text, m.start, m.end))
    return list(seen.values())


def requirements(jd_text: str, matcher, tagger=None) -> list[Requirement]:
    """Vocabulary requirements, then any tagger span holding no vocabulary alias."""
    out = skill_requirements(jd_text, matcher)
    if tagger is None:
        return out
    seen = {r.canonical.lower() for r in out}
    for start, end in tagger.predict_spans(jd_text):
        text = jd_text[start:end].strip(" ,.;:()")
        if not text or matcher.find(text) or len(text.split()) > MAX_FREE_TEXT_WORDS:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            begin = jd_text.index(text, start)
            out.append(Requirement(text, text, begin, begin + len(text), "free_text"))
    return out


def years_required(jd_text: str) -> int | None:
    """The smallest 'N years' a JD states, or None when it states none."""
    found = [int(m.group(1)) for m in EXPERIENCE.finditer(jd_text)]
    return min(found) if found else None


def years_stated(cv_text: str) -> int | None:
    """The largest 'N years' a CV states (Djinni CVs usually lead with it)."""
    found = [int(m.group(1)) for m in EXPERIENCE.finditer(cv_text)]
    return max(found) if found else None
