"""`data.dataturks`: offset repair on synthetic broken offsets, PII handling, BIO."""
from __future__ import annotations

import json

import pytest

from candidate_screener.data import dataturks as dt


def _line(content: str, points: list[tuple[str, int, int, str]]) -> str:
    """A DataTurks JSON line; `end` is inclusive, as in the shipped files."""
    return json.dumps({"content": content, "annotation": [
        {"label": [label], "points": [{"start": s, "end": e, "text": t}]}
        for label, s, e, t in points]})


def test_entity_shift_is_repaired_by_unescaping():
    # Offsets index the unescaped text "R&D team. Skills: Python", but content is escaped.
    unescaped = "R&D team. Skills: Python"
    s = unescaped.index("Python")
    content = unescaped.replace("&", "&amp;")
    assert content[s:s + 6] != "Python"  # the defect: shifted by 4
    docs, led = dt.parse([_line(content, [("Skills", s, s + 5, "Python")])], "test")
    (span,) = docs[0].spans
    assert led["unescape"] == 1 and docs[0].text[span.start:span.end] == "Python"
    assert "&amp;" not in docs[0].text


def test_arbitrary_drift_is_repaired_by_nearest_search():
    content = "Skills: Java, SQL and Java again"
    s = content.index("SQL")
    docs, led = dt.parse([_line(content, [("Skills", s - 3, s - 1, "SQL")])], "test")
    (span,) = docs[0].spans
    assert led["search"] == 1 and (span.start, span.end) == (s, s + 3)


def test_nearest_occurrence_wins_over_the_first():
    content = "Java " + "x" * 40 + " Java"
    s = content.rindex("Java")
    docs, _ = dt.parse([_line(content, [("Skills", s - 4, s - 1, "Java")])], "test")
    assert docs[0].spans[0].start == s


def test_unfindable_or_empty_span_is_dropped_and_counted():
    content = "Skills: Java"
    docs, led = dt.parse([_line(content, [("Skills", 0, 3, "Kotlin"), ("Skills", 0, 1, "  ")])],
                         "test")
    assert docs[0].spans == [] and led["dropped_unrepairable"] == 2


def test_outside_the_window_is_dropped():
    content = "Go " + "x" * (dt.SEARCH_WINDOW + 10) + " Rust"
    docs, led = dt.parse([_line(content, [("Skills", 0, 1, "Rust")])], "test")
    assert docs[0].spans == [] and led["dropped_unrepairable"] == 1


def test_whitespace_is_trimmed_and_every_span_equals_its_slice():
    content = "Skills\n\nPython, SQL\n"
    s = content.index("\nPython")
    docs, led = dt.parse([_line(content, [("Skills", s, s + 12, "\nPython, SQL\n")])], "test")
    (span,) = docs[0].spans
    assert led["intact"] == 1 and span.text == "Python, SQL"
    assert docs[0].text[span.start:span.end] == span.text


def test_pii_spans_are_dropped_and_masked_in_place():
    content = "Jane Roe\njane@example.com\nSkills: Python"
    lines = [_line(content, [("Name", 0, 7, "Jane Roe"),
                             ("Email Address", 9, 24, "jane@example.com"),
                             ("Skills", 34, 39, "Python")])]
    docs, led = dt.parse(lines, "test")
    d = docs[0]
    assert led["dropped_pii"] == 2 and {s.label for s in d.spans} == {"Skills"}
    assert "Jane" not in d.text and "example" not in d.text
    assert len(d.text) == len(content)  # offsets survive masking
    assert d.text[34:40] == "Python"
    raw, _ = dt.parse(lines, "test", masked=False)
    assert "Jane" in raw[0].text and {s.label for s in raw[0].spans} == {"Skills"}


def test_unlabelled_and_malformed_are_counted():
    docs, led = dt.parse(['{"content": "x", "annotation": [{"label": [], "points": []}]}',
                          "{not json"], "test")
    assert led["dropped_unlabelled"] == 1 and led["malformed_lines"] == 1 and len(docs) == 1


def test_to_bio_over_skills():
    content = "Skills: Python, SQL. Worked at Acme"
    s = content.index("Python")
    docs, _ = dt.parse([_line(content, [("Skills", s, s + 10, "Python, SQL"),
                                        ("Companies worked at", 31, 34, "Acme")])], "test")
    tags = [(t, tag) for t, _, _, tag in dt.to_bio(docs[0])]
    assert tags == [("Skills:", "O"), ("Python,", "B"), ("SQL.", "I"), ("Worked", "O"),
                    ("at", "O"), ("Acme", "O")]


@pytest.mark.skipif(not (dt.DATATURKS / "traindata.json").is_file(),
                    reason="needs data/raw/dataturks (candidate_screener.data.fetch)")
def test_shipped_files_repair_completely():
    docs, led = dt.load()
    assert len(docs) == 220 and led["unescape"] == 222 and led.get("dropped_unrepairable", 0) == 0
    assert all(d.text[s.start:s.end] == s.text for d in docs for s in d.spans)
    assert not any(s.label in dt.PII_LABELS for d in docs for s in d.spans)
