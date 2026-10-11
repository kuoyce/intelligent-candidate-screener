"""Scripted recruiter intents: parsed by rules, answered from templates (proposal §4.6).

No generative model is involved. Every sentence an answer states is read off the pipeline's
results, and every quote is a verbatim evidence span. Candidates are referred to by rank
(`#2`, `candidate 2`, `2`) or by id prefix.

    top 5                      the shortlist
    with python / filter sql   who shows a skill (matched or supported)
    at least 3 years           filter by stated experience
    compare 1 and 3            side by side
    why is 1 above 3           what separates two candidates
    what is 2 missing          the gaps
    explain 2                  one candidate in full
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HELP = ("Try: `top 5` · `with python` · `at least 3 years` · `compare 1 and 3` · "
        "`why is 1 above 3` · `what is 2 missing` · `explain 2`")

REF = r"(?:#|candidate\s+|cv\s+)?([\w-]+)"
PATTERNS = [
    ("top", re.compile(r"^(?:show\s+)?(?:the\s+)?(?:top|best)(?:\s+(\d+))?", re.I)),
    ("why", re.compile(rf"^why\s+(?:is\s+)?{REF}\s+(?:ranked\s+)?(?:above|over|ahead of|higher than)\s+{REF}", re.I)),
    ("compare", re.compile(rf"^compare\s+{REF}\s+(?:and|with|vs\.?|to)\s+{REF}", re.I)),
    ("missing", re.compile(rf"^(?:what\s+(?:is|does)\s+{REF}\s+(?:missing|lack)|gaps?\s+(?:for\s+)?{REF})", re.I)),
    ("years", re.compile(r"(?:at\s+least|min(?:imum)?|>=?)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)),
    ("skill", re.compile(r"^(?:filter|with|who\s+(?:knows|has)|candidates\s+with|show\s+.*\s+with)\s+(.+)$", re.I)),
    ("explain", re.compile(rf"^(?:explain|tell me about|details?\s+(?:for|of)?)\s*{REF}", re.I)),
]


@dataclass(frozen=True)
class Intent:
    kind: str
    args: tuple


def parse(text: str) -> Intent:
    text = text.strip().rstrip("?.!")
    for kind, pattern in PATTERNS:
        m = pattern.search(text)
        if m:
            if kind == "missing":
                return Intent(kind, (m.group(1) or m.group(2),))
            return Intent(kind, tuple(g for g in m.groups() if g is not None))
    return Intent("help", ())


def find(results: list, ref: str):
    """A result by rank number or by cv_id prefix; None if unknown or ambiguous."""
    if ref.isdigit() and 1 <= int(ref) <= len(results):
        return results[int(ref) - 1]
    hits = [r for r in results if str(r.cv_id).startswith(ref)]
    return hits[0] if len(hits) == 1 else None


def _label(r) -> str:
    return f"#{r.rank} ({str(r.cv_id)[:8]})"


def _by_status(r, status: str) -> list[dict]:
    return [e for e in r.evidence if e["status"] == status]


def _cov(r) -> str:
    return "—" if r.coverage is None else f"{r.coverage:.0%}"


def answer(intent: Intent, results: list) -> str:
    if not results:
        return "No candidates are loaded yet."
    k = intent.kind
    if k == "top":
        n = int(intent.args[0]) if intent.args else 5
        lines = [f"| rank | candidate | fit call | requirement coverage | relative strength |",
                 "|---|---|---|---|---|"]
        for r in results[:n]:
            lines.append(f"| {r.rank} | {str(r.cv_id)[:8]} | {r.fit_call or 'not assessed'} | "
                         f"{_cov(r)} | {r.confidence:.0%} |")
        return "\n".join(lines)
    if k == "skill":
        skill = intent.args[0].strip().lower()
        hits = [r for r in results
                if any(skill in e["requirement"].lower() or (e["quote"] or "").lower().find(skill) >= 0
                       for e in r.evidence if e["status"] != "missing")]
        if not hits:
            return f"No candidate shows evidence for **{skill}** among the JD's requirements."
        return f"{len(hits)} candidate(s) show **{skill}**: " + ", ".join(_label(r) for r in hits)
    if k == "years":
        n = int(intent.args[0])
        hits = [r for r in results if r.years_stated is not None and r.years_stated >= n]
        unknown = sum(r.years_stated is None for r in results)
        return (f"{len(hits)} candidate(s) state at least {n} years: "
                + (", ".join(_label(r) for r in hits) or "none")
                + (f". {unknown} state no figure." if unknown else "."))
    if k in ("compare", "why"):
        a, b = find(results, intent.args[0]), find(results, intent.args[1])
        if a is None or b is None:
            return "I could not find both candidates. Use their rank, e.g. `compare 1 and 3`."
        a_has = {e["requirement"] for e in a.evidence if e["status"] != "missing"}
        b_has = {e["requirement"] for e in b.evidence if e["status"] != "missing"}
        only_a, only_b = sorted(a_has - b_has), sorted(b_has - a_has)
        head = (f"{_label(a)} is ranked {'above' if a.rank < b.rank else 'below'} {_label(b)} "
                f"(score {a.score:.3f} vs {b.score:.3f}; fit call {a.fit_call or 'n/a'} vs "
                f"{b.fit_call or 'n/a'}; coverage {_cov(a)} vs {_cov(b)}).")
        parts = [head]
        if only_a:
            parts.append(f"Only {_label(a)} shows: " + ", ".join(only_a) + ".")
        if only_b:
            parts.append(f"Only {_label(b)} shows: " + ", ".join(only_b) + ".")
        if k == "why" and a.rank > b.rank:
            parts.append(f"Note: {_label(a)} is in fact ranked below {_label(b)}.")
        if a.fit_call != b.fit_call and a.fit_call and b.fit_call:
            parts.append("The fit calls differ because the role-family (Primary Keyword) match differs.")
        return " ".join(parts)
    if k == "missing":
        r = find(results, intent.args[0])
        if r is None:
            return "I could not find that candidate. Use their rank, e.g. `what is 2 missing`."
        gaps = [e["requirement"] for e in _by_status(r, "missing")]
        return (f"{_label(r)} shows no evidence for: " + ", ".join(gaps) + "."
                if gaps else f"{_label(r)} shows evidence for every listed requirement.")
    if k == "explain":
        r = find(results, intent.args[0])
        if r is None:
            return "I could not find that candidate. Use their rank, e.g. `explain 2`."
        lines = [f"**{_label(r)}** — fit call: {r.fit_call or 'not assessed'}, requirement "
                 f"coverage {_cov(r)}, relative strength {r.confidence:.0%}."]
        if r.years_required is not None or r.years_stated is not None:
            lines.append(f"Experience: JD asks {r.years_required or '—'} years; CV states "
                         f"{r.years_stated or '—'}.")
        for e in r.evidence:
            if e["status"] == "matched":
                lines.append(f"- ✔ **{e['requirement']}** — “{e['quote']}”")
            elif e["status"] == "supported":
                lines.append(f"- ≈ **{e['requirement']}** (similarity {e['similarity']:.2f}) — “{e['quote']}”")
            else:
                lines.append(f"- ✘ **{e['requirement']}** — no evidence found")
        return "\n".join(lines)
    return HELP
