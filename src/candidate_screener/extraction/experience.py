"""E3 experience years and E4 seniority from text alone (plan 2026-10-06, P4).

Rules only; nothing here is fitted. Every function reads JD text (`Long Description`) or CV
text **without** `Position` (`CV`, `Highlights`, `Moreinfo`, `Looking For`, the `text_v2`
fields). The metadata fields are labels for measurement only (02-design-spec §1).

**JD requirement** (`jd_experience`). A *mention* is a number of years (or months) in one of
these shapes, tried in this order so a longer shape claims its characters first:

- range ``2-5 years`` / ``2 to 5 years`` -> (2, 5);
- prefix ``at least / minimum / no less than / over / more than / from N years`` -> (N, None);
- ``up to / less than N years`` -> (0, N);
- plus ``5+ years`` / ``5 years+`` -> (5, None);
- plain ``3 years`` -> (3, None).

Numbers are 1-2 digit integers or decimals (``2.5``, ``0,5``) or the words one..twenty. A
number of 21 or more is never a JD requirement, so a date like ``2019-2022`` is not one
(nor is anything without a ``year`` unit after it). A mention counts only in an
*experience context*: an experience cue (``experience``, ``commercial``, ``hands-on``,
``working``, ``as a``, ``in IT``, ``with``, ``of`` ...) within 40 characters before or 60
after on the same line, or, for a range / plus / prefix mention, the mention opening its
line (a bullet such as ``- 3+ years;``). A mention next to company words (``we``, ``our``,
``company``, ``market``, ``clients``, ``days`` ... within the same window) is dropped unless
its line also addresses the candidate (``you``, ``candidate``, ``someone``, ``looking for``,
``need``, ``require`` ...): "In 12 years the company has ..." is not a requirement, "We need
someone with 3+ years ..." is.

Mentions on a *nice-to-have* line, or under a nice-to-have header, are optional. The
requirement is the **largest minimum among the required mentions** (the optional ones only
if there is no required one): a posting that asks for "2+ years of X" and "4+ years of Y"
requires 4. ``max_years`` is the upper end of the mention chosen (None unless a range). A
posting with no numeric mention but "no experience required" / "without experience" gets 0.

**CV years** (`cv_experience`). An explicit statement (the same mention shapes, always with
an experience cue, never ``N years old`` / ``N years ago``) wins: the largest one is the
total. Without one, date ranges are summed: ``2019 - 2022``, ``Jan 2020 - present``,
``03.2018-05.2021``. A year alone is read as mid-year (so ``2019 - 2022`` is 3 years); a
month starts on its first day and ends on its last. ``present`` / ``now`` / ``current`` is
the ``reference`` date passed in (default `REFERENCE_DATE`, 2023-09-01: the latest
`Published` date in the A2 JDs, so no figure depends on the wall clock). Ranges are merged
where they overlap before summing, a range on (or next to) a line about education is
skipped, and a range before 1970, after the reference date, or longer than 45 years is
dropped.

**Bands.** JD years snap down to Djinni's own `Exp Years` scale (<1 ``no_exp``, 1 ``1y``,
2 ``2y``, 3-4 ``3y``, 5+ ``5y``; there is no 4-year option, so 4 is read as the ``3y`` a
recruiter could pick) and then through `JD_EXP_TO_BAND`; CV years go through
`cv_experience_band`. Both are the metadata's own functions, imported, not copied.

**Missing data.** `experience_compatible_extracted` hands the extracted bands to
`matching.corpus.experience_compatible`, the rung-1a rule itself, so the semantics are
identical by construction: an unknown band on either side scores 0. That is what the 1a
rule already does for a missing field, and the alternative (unknown = compatible) would
credit a document for saying nothing.

**Seniority** (E4). Ordered levels `LEVELS` = intern < junior < middle < senior < lead. The
tokens are `title_profile.SENIORITY` (``strong`` is a modifier, not a level). Ground truth
is the highest level among the seniority tokens of `Position` (``lead`` before
``generation`` is a sales keyword, not a level). From text, a token counts when it heads a
title: a singular role noun (``developer``, ``engineer``, ``manager`` ...) follows within
four tokens on the same line, or the token stands alone as a title (``team lead``,
``head of``, ``intern``, ``trainee``). Plural nouns (``senior engineers``) are colleagues,
not the document's own title. Without a title token, years fall back: <2 junior, 2-4
middle, 5+ senior (never intern or lead).
"""
from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from candidate_screener.data.a2_finetune import JD_EXP_TO_BAND, cv_experience_band
from candidate_screener.extraction.title_profile import SENIORITY, canonical_tokens

#: "present" in a CV date range. The latest `Published` date of an A2 JD (2023-09-01):
#: the corpus was collected 2020-2023, so a range left open is open up to then.
REFERENCE_DATE = dt.date(2023, 9, 1)

JD_MAX_YEARS = 20
CV_MAX_YEARS = 45
MIN_RANGE_YEAR = 1970

# --- numbers ---------------------------------------------------------------------------

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
          "twenty": 20}
_NUM = r"(?:\d{1,2}(?:[.,]\d{1,2})?(?!\d)|" + "|".join(_WORDS) + r")"
_UNIT = r"(?P<unit>years?|yrs?|months?)\b"
_B = r"(?<![\w.,])"


def _num(s: str) -> float:
    s = s.lower()
    return float(_WORDS[s]) if s in _WORDS else float(s.replace(",", "."))


_SHAPES: tuple[tuple[str, re.Pattern], ...] = tuple((kind, re.compile(p, re.I)) for kind, p in (
    ("range", _B + rf"(?P<lo>{_NUM})\s*(?:-|–|—|to)\s*(?P<hi>{_NUM})\s*\+?\s*" + _UNIT),
    ("prefix", r"\b(?:at\s+least|minimum(?:\s+of)?|min\.?|no\s+less\s+than|not\s+less\s+than"
               r"|over|more\s+than|from|>=?|≥)\s*" + rf"(?P<lo>{_NUM})\s*\+?\s*" + _UNIT),
    ("upto", r"\b(?:up\s+to|less\s+than|under|maximum(?:\s+of)?|max\.?)\s*"
             + rf"(?P<hi>{_NUM})\s*" + _UNIT),
    ("plus", _B + rf"(?P<lo>{_NUM})\s*\+\s*" + _UNIT),
    ("plus", _B + rf"(?P<lo>{_NUM})\s*" + _UNIT + r"\s*\+"),
    ("plain", _B + rf"(?P<lo>{_NUM})\s*(?:full\s+)?" + _UNIT),
))

_CUE = re.compile(
    r"experien|\bexp\b|expertise|background|commercial|hands[- ]on|\bwork|professional"
    r"|production|track record|develop|programming|coding|engineering|\bas\s+an?\b"
    r"|\bin\s+(?:the\s+)?(?:it|role|position|field|industry|similar|software|development"
    r"|testing|qa|sales|marketing|design|management|recruit\w*|this)\b|\bwith\b|\busing\b"
    r"|\bof\b|\bin\b", re.I)
_COMPANY = re.compile(r"\b(?:we|we're|we've|our|company|companies|us|founded|market|clients?"
                      r"|customers?|history|anniversary|vacation|days?|warranty)\b", re.I)
_CANDIDATE = re.compile(r"\b(?:you|your|candidate|someone|person|looking\s+for|need|require"
                        r"|seeking|hire|hiring|join)", re.I)
_CV_REJECT_AFTER = re.compile(r"^\W{0,3}(?:old|of\s+age|ago|later)\b", re.I)
_OPTIONAL = re.compile(r"nice[- ]to[- ]have|would\s+be\s+(?:a\s+)?(?:plus|advantage|great"
                       r"|nice)|(?:is|will\s+be)\s+a\s+(?:plus|bonus|big\s+plus|advantage)"
                       r"|advantage|preferred|preferably|desirable|bonus|optional", re.I)
_NO_EXP = re.compile(r"no\s+(?:commercial\s+|prior\s+|previous\s+|work\s+)?experience\s+(?:is\s+)?"
                     r"(?:required|needed|necessary)|without\s+(?:commercial\s+|prior\s+|work\s+)?"
                     r"experience|experience\s+is\s+not\s+(?:required|necessary)", re.I)
_BULLET = re.compile(r"^[\s\-–—•*·●▪►✅✔+>\d.)]*$")
_BULLET_CHARS = "-–—•*·●▪►✅✔+>"
_HEADER_WORDS = re.compile(r"requirement|responsibilit|offer|about|what you|benefit|must[- ]have"
                           r"|skills|qualification|duties|tasks|expect|looking for|"
                           r"nice|plus|advantage|bonus|preferred", re.I)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class Mention:
    lo: float | None
    hi: float | None
    kind: str
    span: Span
    optional: bool = False


@dataclass(frozen=True)
class JDExperience:
    min_years: float | None
    max_years: float | None
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class CVExperience:
    years: float | None
    source: str | None          # "statement" | "ranges" | None
    spans: tuple[Span, ...]


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end < 0 else end


def _raw_mentions(text: str) -> list[Mention]:
    taken: list[tuple[int, int]] = []
    out: list[Mention] = []
    for kind, pat in _SHAPES:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            scale = 1 / 12 if m.group("unit").lower().startswith("month") else 1.0
            g = m.groupdict()
            lo = _num(g["lo"]) * scale if g.get("lo") else (0.0 if kind == "upto" else None)
            hi = _num(g["hi"]) * scale if g.get("hi") else None
            taken.append((m.start(), m.end()))
            out.append(Mention(lo, hi, kind, Span(m.start(), m.end(), m.group(), kind)))
    return sorted(out, key=lambda x: x.span.start)


def _in_context(text: str, men: Mention, side: str) -> bool:
    s, e = men.span.start, men.span.end
    ls, le = _line_bounds(text, s)
    before, after = text[max(ls, s - 40):s], text[e:min(le, e + 60)]
    if re.search(r"\b(?:company|organisation|organization|business)\s+(?:has|with|of|celebrates)\s*$", before, re.I) \
            and re.match(r"\s*(?:of\s+)?(?:history|existence|operation|anniversary)\b", after, re.I):
        return False
    if side == "cv":
        if _CV_REJECT_AFTER.match(after):
            return False
    elif _COMPANY.search(before + text[s:e] + after) and not _CANDIDATE.search(text[ls:le]):
        return False
    if _CUE.search(after) or re.search(r"experien|\bexp\b", before, re.I):
        return True
    return side == "jd" and men.kind in ("range", "plus", "prefix") \
        and bool(_BULLET.match(text[ls:s]))


def _optional_lines(text: str) -> set[int]:
    """Start offsets of lines that are nice-to-have (by cue, or under such a header)."""
    out: set[int] = set()
    in_optional = False
    pos = 0
    for line in text.split("\n"):
        stripped = line.strip()
        is_header = (0 < len(stripped) <= 60 and stripped[0] not in _BULLET_CHARS
                     and not re.search(r"\d", stripped)
                     and (stripped.endswith(":") or bool(_HEADER_WORDS.search(stripped))))
        if is_header:
            in_optional = bool(_OPTIONAL.search(stripped))
        if in_optional or _OPTIONAL.search(line):
            out.add(pos)
        pos += len(line) + 1
    return out


def jd_mentions(text: str) -> list[Mention]:
    """The experience-requirement mentions of a JD, each marked optional or required."""
    text = str(text or "")
    optional = _optional_lines(text)
    out = []
    for men in _raw_mentions(text):
        if men.lo is not None and men.lo > JD_MAX_YEARS or men.hi is not None \
                and men.hi > JD_MAX_YEARS:
            continue
        if not _in_context(text, men, "jd"):
            continue
        is_opt = _line_bounds(text, men.span.start)[0] in optional
        out.append(Mention(men.lo, men.hi, men.kind, men.span, is_opt))
    return out


def jd_experience(text: str, aggregate: str = "max") -> JDExperience:
    """(min_years, max_years | None, evidence spans) of a JD's experience requirement.

    `aggregate`: ``max`` (the rule: the largest required minimum) or ``first`` (the first
    required mention; a sensitivity variant measured on A2 train only)."""
    mentions = jd_mentions(text)
    pool = [m for m in mentions if not m.optional] or mentions
    if pool:
        chosen = pool[0] if aggregate == "first" else max(pool, key=lambda m: m.lo or 0.0)
        return JDExperience(chosen.lo, chosen.hi, tuple(m.span for m in mentions))
    hit = _NO_EXP.search(str(text or ""))
    if hit:
        return JDExperience(0.0, None, (Span(hit.start(), hit.end(), hit.group(), "no_exp"),))
    return JDExperience(None, None, ())


# --- CV date ranges --------------------------------------------------------------------

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def _date(tag: str) -> str:
    return (rf"(?:(?P<mon{tag}>{_MONTH}),?\s*|(?P<mm{tag}>0?[1-9]|1[0-2])\s*[./-]\s*)?"
            rf"(?P<yr{tag}>(?:19|20)\d\d)(?!\d)")


_PRESENT = r"(?P<present>present|now|current(?:ly)?|today|till\s+now|to\s+date|nowadays|ongoing)"
_RANGE = re.compile(r"(?<![\d.])" + _date("1") + r"\s*(?:-|–|—|to|till|until)\s*(?:"
                    + _date("2") + "|" + _PRESENT + r")", re.I)
_EDU = re.compile(r"universit|bachelor|master'?s?\b|degree|institute|college|school|faculty"
                  r"|academy|course|student|education|diploma|graduat|lyceum|gymnasium", re.I)


def _point(m: re.Match, tag: str, end: bool) -> float:
    year = int(m.group(f"yr{tag}"))
    mon = m.group(f"mon{tag}")
    month = _MONTHS[mon[:3].lower()] if mon else (int(m.group(f"mm{tag}"))
                                                 if m.group(f"mm{tag}") else None)
    if month is None:
        return year + 0.5
    return year + (month if end else month - 1) / 12


def _ref_point(reference: dt.date) -> float:
    return reference.year + (reference.month - 1) / 12 + (reference.day - 1) / 365


def cv_ranges(text: str, reference: dt.date = REFERENCE_DATE
              ) -> list[tuple[float, float, Span]]:
    """Employment-like date ranges as (start, end) in fractional years, with spans."""
    text = str(text or "")
    ref = _ref_point(reference)
    lines = text.split("\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    out = []
    for m in _RANGE.finditer(text):
        i = max(k for k, s in enumerate(starts) if s <= m.start())
        context = "\n".join(lines[max(0, i - 1):i + 2])
        if _EDU.search(context):
            continue
        start = _point(m, "1", end=False)
        stop = ref if m.group("present") else _point(m, "2", end=True)
        stop = min(stop, ref)
        if int(m.group("yr1")) < MIN_RANGE_YEAR or start >= ref or stop <= start \
                or stop - start > CV_MAX_YEARS:
            continue
        out.append((start, stop, Span(m.start(), m.end(), m.group(), "cv_range")))
    return out


def merged_years(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length of a union of intervals: overlaps are counted once."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def cv_statements(text: str) -> list[Mention]:
    text = str(text or "")
    out = []
    for men in _raw_mentions(text):
        if men.kind == "upto" or men.lo is None or men.lo > CV_MAX_YEARS:
            continue
        if men.hi is not None and men.hi > CV_MAX_YEARS:
            continue
        if _in_context(text, men, "cv"):
            out.append(men)
    return out


def cv_experience(text: str, reference: dt.date = REFERENCE_DATE) -> CVExperience:
    """(years, source, evidence spans): the largest explicit statement, else merged ranges."""
    statements = cv_statements(text)
    if statements:
        best = max(m.lo for m in statements)
        return CVExperience(round(best, 4), "statement", tuple(m.span for m in statements))
    ranges = cv_ranges(text, reference)
    if ranges:
        years = merged_years((s, e) for s, e, _ in ranges)
        return CVExperience(round(years, 4), "ranges", tuple(sp for _, _, sp in ranges))
    return CVExperience(None, None, ())


# --- bands -----------------------------------------------------------------------------

def jd_exp_years_category(years: float | None) -> str | None:
    """Snap a JD minimum down to Djinni's `Exp Years` scale (no 4-year option)."""
    if years is None or pd.isna(years):
        return None
    return ("no_exp" if years < 1 else "1y" if years < 2 else "2y" if years < 3
            else "3y" if years < 5 else "5y")


def jd_band(years: float | None) -> str | None:
    cat = jd_exp_years_category(years)
    return None if cat is None else JD_EXP_TO_BAND[cat]


def cv_band(years: float | None) -> str | None:
    return cv_experience_band(years)


@lru_cache(maxsize=200_000)
def _jd_band_of(text: str, aggregate: str = "max") -> str | None:
    return jd_band(jd_experience(text, aggregate).min_years)


@lru_cache(maxsize=200_000)
def _cv_band_of(text: str, reference: dt.date = REFERENCE_DATE) -> str | None:
    return cv_band(cv_experience(text, reference).years)


def experience_compatible_extracted(jd_texts: Sequence[str], cv_texts: Sequence[str],
                                    reference: dt.date = REFERENCE_DATE,
                                    unknown_compatible: bool = False) -> pd.Series:
    """The rung-1a `experience_compatible` rule on bands extracted from text.

    Row-aligned pairs in, 0/1 out. By default unknown on either side -> 0, exactly as the
    1a rule treats a missing field (it is the same function). `unknown_compatible=True` is
    X1's registered variant (amendment A1, 06-registration §7): an unknown band on either
    side counts as compatible (1); a known pair is still the 1a comparison."""
    from candidate_screener.matching.corpus import experience_compatible
    if len(jd_texts) != len(cv_texts):
        raise ValueError("jd_texts and cv_texts must be row-aligned pairs")
    frame = pd.DataFrame({"jd_exp_band": [_jd_band_of(str(t or "")) for t in jd_texts],
                          "cv_exp_band": [_cv_band_of(str(t or ""), reference)
                                          for t in cv_texts]})
    out = experience_compatible(frame)
    if unknown_compatible:
        unknown = frame.jd_exp_band.isna() | frame.cv_exp_band.isna()
        out = out.where(~unknown.to_numpy(), 1)
    return out


# --- E4 seniority ----------------------------------------------------------------------

LEVELS = ("intern", "junior", "middle", "senior", "lead")
LEVEL_ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}
#: Every token of `title_profile.SENIORITY` mapped to a level (`strong` is a modifier).
TOKEN_LEVEL: dict[str, str | None] = {
    "trainee": "intern", "intern": "intern", "internship": "intern",
    "junior": "junior", "jr": "junior", "middle": "middle", "mid": "middle",
    "senior": "senior", "sr": "senior", "lead": "lead", "teamlead": "lead",
    "techlead": "lead", "head": "lead", "principal": "lead", "chief": "lead",
    "strong": None,
}
#: Tokens that are a title by themselves (no role noun needed).
STANDALONE = frozenset({"teamlead", "techlead", "head", "intern", "internship", "trainee"})
ROLE_NOUNS = frozenset({
    "developer", "engineer", "designer", "manager", "analyst", "specialist", "tester", "qa",
    "aqa", "architect", "devops", "scientist", "consultant", "recruiter", "administrator",
    "sysadmin", "marketer", "copywriter", "writer", "artist", "researcher", "sre", "officer",
    "lead", "teamlead", "techlead", "owner", "generalist", "expert", "sourcer", "accountant",
    "animator", "producer", "coordinator", "executive", "representative", "director",
    "professional", "position", "role", "level", "lawyer", "translator", "editor",
})
NOUN_WINDOW = 4


def _not_lead_generation(tokens: Sequence[str], k: int) -> bool:
    return not (tokens[k] == "lead" and k + 1 < len(tokens)
                and tokens[k + 1] in ("generation", "gen"))


def position_seniority(position: object) -> str | None:
    """Ground truth: the highest seniority level among the tokens of `Position`."""
    if position is None or (isinstance(position, float) and pd.isna(position)):
        return None
    toks = [t for t, _ in canonical_tokens(position)]
    levels = [TOKEN_LEVEL[t] for k, t in enumerate(toks)
              if t in TOKEN_LEVEL and TOKEN_LEVEL[t] and _not_lead_generation(toks, k)]
    return max(levels, key=LEVEL_ORDER.__getitem__) if levels else None


def seniority_spans(text: str) -> list[tuple[str, Span]]:
    """Seniority tokens of `text` that head a title, in text order."""
    text = str(text or "")
    toks = canonical_tokens(text)
    words = [t for t, _ in toks]
    out = []
    for k, (tok, off) in enumerate(toks):
        level = TOKEN_LEVEL.get(tok)
        if level is None or not _not_lead_generation(words, k):
            continue
        ok = tok in STANDALONE
        for j in range(k + 1, min(len(toks), k + 1 + NOUN_WINDOW)):
            if "\n" in text[off:toks[j][1]]:
                break
            if words[j] in ROLE_NOUNS:
                ok = True
                break
        if ok:
            end = off + len(tok)
            out.append((level, Span(off, end, text[off:end], "seniority")))
    return out


def years_to_seniority(years: float | None) -> str | None:
    if years is None or pd.isna(years):
        return None
    return "junior" if years < 2 else "middle" if years < 5 else "senior"


def text_seniority(text: str, years: float | None = None, pick: str = "first"
                   ) -> tuple[str | None, str | None]:
    """(level, source): a title token from the text (``first`` or ``max``), else years."""
    spans = seniority_spans(text)
    if spans:
        levels = [lvl for lvl, _ in spans]
        level = levels[0] if pick == "first" else max(levels, key=LEVEL_ORDER.__getitem__)
        return level, "token"
    level = years_to_seniority(years)
    return level, ("years" if level else None)


assert set(TOKEN_LEVEL) == set(SENIORITY), "TOKEN_LEVEL must cover title_profile.SENIORITY"
