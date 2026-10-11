"""Recruiter intents: one scenario per intent; answers cite only pipeline evidence."""
import re

import pytest

from candidate_screener.app import intents
from candidate_screener.pipeline import Result


def _results():
    ev_a = [{"requirement": "Python", "status": "matched", "quote": "Python", "start": 0, "end": 6, "similarity": None},
            {"requirement": "SQL", "status": "supported", "quote": "queried Postgres daily", "start": 10, "end": 32, "similarity": 0.61},
            {"requirement": "Docker", "status": "missing", "quote": None, "start": None, "end": None, "similarity": None}]
    ev_b = [{"requirement": "Python", "status": "missing", "quote": None, "start": None, "end": None, "similarity": None},
            {"requirement": "SQL", "status": "matched", "quote": "SQL", "start": 3, "end": 6, "similarity": None},
            {"requirement": "Docker", "status": "missing", "quote": None, "start": None, "end": None, "similarity": None}]
    return [Result("aaaa1111", 1, 2.9, "Fit", 1.0, 2 / 3, 3, 5, ev_a),
            Result("bbbb2222", 2, 1.2, "No Fit", 0.5, 1 / 3, 3, None, ev_b)]


@pytest.mark.parametrize("text, kind", [
    ("top 5", "top"), ("show the best 3", "top"), ("with python", "skill"),
    ("at least 3 years", "years"), ("compare 1 and 2", "compare"),
    ("why is 1 above 2?", "why"), ("what is 2 missing", "missing"), ("gaps for 1", "missing"),
    ("explain 2", "explain"), ("hello", "help")])
def test_parse(text, kind):
    assert intents.parse(text).kind == kind


def _quotes_are_evidence(answer, results):
    quoted = re.findall(r"“([^”]+)”", answer)
    allowed = {e["quote"] for r in results for e in r.evidence if e["quote"]}
    assert set(quoted) <= allowed


def test_top():
    out = intents.answer(intents.parse("top 1"), _results())
    assert "aaaa1111" in out and "bbbb2222" not in out and "Fit" in out


def test_skill_filter_counts_supported_evidence():
    out = intents.answer(intents.parse("with sql"), _results())
    assert "2 candidate(s)" in out


def test_years_filter_reports_unknowns():
    out = intents.answer(intents.parse("at least 4 years"), _results())
    assert "#1" in out and "1 state no figure" in out


def test_compare_and_why_name_what_separates_them():
    out = intents.answer(intents.parse("why is 1 above 2"), _results())
    assert "Only #1 (aaaa1111) shows: Python" in out and "role-family" in out
    out = intents.answer(intents.parse("why is 2 above 1"), _results())
    assert "in fact ranked below" in out


def test_missing_lists_gaps_only_from_evidence():
    out = intents.answer(intents.parse("what is 2 missing"), _results())
    assert "Python" in out and "Docker" in out and "SQL" not in out


def test_explain_quotes_only_verbatim_evidence():
    results = _results()
    out = intents.answer(intents.parse("explain 1"), results)
    _quotes_are_evidence(out, results)
    assert "“queried Postgres daily”" in out and "no evidence found" in out


def test_unknown_candidate_and_empty_state():
    assert "could not find" in intents.answer(intents.parse("explain 9"), _results())
    assert intents.answer(intents.parse("top 3"), []) == "No candidates are loaded yet."
