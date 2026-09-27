"""`extraction.rules`: the longest-match phrase matcher and its fixed interface."""
from __future__ import annotations

import pandas as pd
import pytest

from candidate_screener.extraction import rules
from candidate_screener.extraction.rules import Match, SkillMatcher

VOCAB = pd.DataFrame(
    [
        ("C++", "C++"), ("C++", "cpp"), ("C#", "C#"), (".NET", ".NET"),
        ("ASP.NET", "ASP.NET"), ("node.js", "Node.js"), ("node.js", "nodejs"),
        ("CI/CD", "CI/CD"), ("Java", "Java"), ("JavaScript", "JavaScript"),
        ("machine learning", "machine"), ("machine learning", "machine learning"),
        ("deep learning", "deep learning"), ("SQL", "SQL"), ("SQL Server", "SQL Server"),
        ("Python", "Python"),
    ],
    columns=["canonical", "alias"],
)


@pytest.fixture(scope="module")
def m() -> SkillMatcher:
    return SkillMatcher(VOCAB)


def spans(m, text):
    return [(x.text, x.canonical) for x in m.find(text)]


@pytest.mark.parametrize("skill", ["C++", "C#", ".NET", "Node.js", "CI/CD"])
def test_punctuation_bearing_skills_match_with_exact_offsets(m, skill):
    text = f"Must know {skill}, ideally."
    (hit,) = m.find(text)
    assert hit.text == skill and text[hit.start:hit.end] == skill
    assert hit.start == len("Must know ")


def test_punctuation_skills_at_end_of_text_and_before_punctuation(m):
    """The baseline regex `\\bc\\+\\+\\b` misses these; a token pattern must not."""
    assert spans(m, "c++") == [("c++", "C++")]
    assert spans(m, "I write c++.") == [("c++", "C++")]


def test_spaced_punctuation_is_not_the_skill(m):
    assert m.find("grade C + + and C #") == []


def test_word_boundary_java_is_not_javascript(m):
    assert spans(m, "JavaScript developer") == [("JavaScript", "JavaScript")]
    assert spans(m, "Java developer") == [("Java", "Java")]


def test_case_insensitive(m):
    assert spans(m, "PYTHON and python") == [("PYTHON", "Python"), ("python", "Python")]


def test_longest_match_wins_and_output_does_not_overlap(m):
    out = m.find("machine learning with SQL Server")
    assert [(x.text, x.canonical) for x in out] == [
        ("machine learning", "machine learning"), ("SQL Server", "SQL Server")]
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))


def test_leftmost_match_consumes_its_tokens(m):
    """`ASP.NET` is one match, not `ASP.NET` and `.NET` overlapping."""
    assert spans(m, "ASP.NET MVC") == [("ASP.NET", "ASP.NET")]


def test_two_skills_inside_one_slash_token(m):
    assert spans(m, "C#/.NET stack") == [("C#", "C#"), (".NET", ".NET")]


def test_alias_normalisation_maps_to_canonical(m):
    assert m.canonicals("nodejs and cpp") == {"node.js", "C++"}


def test_whitespace_run_matches_but_line_break_does_not(m):
    assert spans(m, "machine    learning") == [("machine    learning", "machine learning")]
    # A line break separates resume list items: `machine` alone matches, not the phrase.
    assert spans(m, "machine\nlearning") == [("machine", "machine learning")]


def test_offsets_survive_characters_whose_lowercase_is_longer():
    """'İ'.lower() is two characters; offsets must still index the caller's string."""
    m = SkillMatcher(VOCAB)
    text = "İİİ Python"
    (hit,) = m.find(text)
    assert text[hit.start:hit.end] == "Python"


def test_first_row_wins_for_a_duplicated_alias():
    df = pd.DataFrame({"canonical": ["A", "B"], "alias": ["tool", "TOOL"]})
    assert SkillMatcher(df).canonicals("tool") == {"A"}


def test_match_is_frozen_and_sorted(m):
    out = m.find("SQL then Python then C#")
    assert [x.start for x in out] == sorted(x.start for x in out)
    with pytest.raises(Exception):
        out[0].start = 3  # type: ignore[misc]
    assert isinstance(out[0], Match)


def test_empty_and_no_match(m):
    assert m.find("") == [] and m.canonicals("horticulture and bees") == set()


def test_load_vocab_keeps_na_like_aliases_as_strings(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("canonical,alias\nNA,NA\nnull,null\n")
    df = rules.load_vocab(p)
    assert df.alias.tolist() == ["NA", "null"]


@pytest.mark.skipif(not rules.VOCAB_CSV.is_file(),
                    reason="data/vocab/skills.csv not built (extraction.vocab --build)")
def test_default_matcher_reads_the_committed_vocabulary():
    m = rules.default_matcher()
    assert {"C++", "C#", "PostgreSQL"} <= m.canonicals("C++, C# and Postgres")
