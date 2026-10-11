"""`extraction.rules`: the longest-match phrase matcher and its fixed interface."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from candidate_screener.extraction import rules, extract_document, score_exact_spans, validate_entity
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


def document(text, field="Moreinfo"):
    return {"doc_id": "synthetic-1", "source": "synthetic", "doc_type": "cv", "fields": {field: text}}


def test_punctuation_sensitive_aliases_and_longest_overlap():
    rows = extract_document(document("C++ C# .NET Node.js React.js PostgreSQL"))
    assert [r["surface"] for r in rows] == ["C++", "C#", ".NET", "Node.js", "React.js", "PostgreSQL"]
    assert len({r["entity_id"] for r in rows}) == len(rows)


def test_ordinary_go_and_r_require_programming_context():
    assert not extract_document(document("We Go to work. R is an initial. go home."))
    assert [r["surface"] for r in extract_document(document("Go programming; R statistical analysis; Golang"))] == ["Go", "R", "Golang"]


def test_negation_and_learning_are_clause_bounded():
    rows = extract_document(document("No Python experience; learning Kubernetes; SQL. No AWS but Docker"))
    assert [(r["surface"], r["context_status"]) for r in rows] == [
        ("Python", "negated"), ("Kubernetes", "learning"), ("SQL", "unqualified"),
        ("AWS", "negated"), ("Docker", "unqualified")]
    assert extract_document(document("Python", "Looking For"))[0]["context_status"] == "learning"
    for text in ("not only Python", "no problem with Python"):
        assert extract_document(document(text))[0]["context_status"] == "unqualified"
    assert extract_document(document("No experience with Python"))[0]["context_status"] == "negated"


def test_original_unicode_offsets_and_separate_fields():
    doc = document("🙂 café Python and C++")
    doc["fields"]["Position"] = "Software engineer"
    original = deepcopy(doc)
    rows = extract_document(doc)
    assert doc == original
    assert {r["label"] for r in rows} == {"KNOWLEDGE", "TITLE"}
    for row in rows:
        assert doc["fields"][row["field"]][row["start"]:row["end"]] == row["surface"]
    assert rows == extract_document(doc)


def test_null_fields_and_cross_field_phrases():
    doc = document("software")
    doc["fields"].update({"Position": "engineer", "Highlights": None})
    assert extract_document(doc) == []
    with pytest.raises(ValueError, match="strings"):
        extract_document(document(3))


def test_offset_validation_refuses_shifted_evidence():
    text = "Python"
    row = extract_document(document(text))[0]
    row["start"] += 1
    with pytest.raises(ValueError):
        validate_entity(row, text)


def test_exact_span_label_scoring_keeps_documents_and_layers_distinct():
    gold = [{"doc_id": "a", "field": "body", "start": 0, "end": 6, "label": "KNOWLEDGE"}]
    assert score_exact_spans(gold, gold)["micro"]["f1"] == 1.0
    wrong = [{**gold[0], "end": 5}, {**gold[0], "doc_id": "b"}, {**gold[0], "label": "SKILL"}]
    scores = score_exact_spans(gold, wrong)
    assert scores["micro"]["true_positive"] == 0
    assert scores["per_label"]["KNOWLEDGE"]["support"] == 1
    assert score_exact_spans(gold, [{**gold[0], "layer": "other"}])["micro"]["f1"] == 0
    assert score_exact_spans(gold, gold + gold)["micro"]["precision"] == 0.5


def test_rule_confidence_null_and_seed_namespace():
    rows = extract_document(document("Master of Science; data analysis; SQL"))
    assert {r["label"] for r in rows} == {"DEGREE", "SKILL", "KNOWLEDGE"}
    assert all(r["confidence"] is None and r["canonical_id"].startswith("seed:") for r in rows)


def test_phrase_matcher_and_document_extractor_coexist_without_changing_context(m):
    text = "No Python experience; learning C++"
    assert m.canonicals(text) == {"Python", "C++"}
    records = rules.extract_document(document(text))
    assert [(row["surface"], row["context_status"]) for row in records] == [
        ("Python", "negated"), ("C++", "learning")]
    assert all(text[row["start"]:row["end"]] == row["surface"] for row in records)
