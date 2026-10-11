"""`extraction.vocab`: sources, merge priorities, filters and the deterministic render."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from candidate_screener.baselines.skills import SKILL_PATTERNS
from candidate_screener.extraction import vocab
from candidate_screener.extraction.rules import SkillMatcher
from candidate_screener.extraction.vocab import Filters, Sources


def _sources() -> Sources:
    skills = pd.DataFrame([
        ("u1", "Python (computer programming)", "Python\nPython 3K", "knowledge"),
        ("u2", "computer vision", "tensorflow\nOCR", "knowledge"),
        ("u3p", "use office software", "", "skill/competence"),
        ("u3", "use spreadsheets software", "", "skill/competence"),
        ("u4", "manage musical staff", "manage music staff", "skill/competence"),
        ("u5", "computer technology", "IT\ninformation technology", "knowledge"),
        ("u6", "SPARK", "", "knowledge"),
    ], columns=["conceptUri", "preferredLabel", "altLabels", "skillType"])
    groups = pd.DataFrame([("g_s5", "S5"), ("g_06", "06"), ("g_0611", "0611"), ("g_s1", "S1")],
                          columns=["conceptUri", "code"])
    broader = pd.DataFrame([("u2", "g_0611"), ("g_0611", "g_06"), ("u3p", "g_s5"),
                            ("u3", "u3p"), ("u4", "g_s1")], columns=["conceptUri", "broaderUri"])
    digital = pd.DataFrame({"conceptUri": ["u1", "u5", "u6"]})
    data_jobs = {"python": 100, "tensorflow": 50, "mongo": 5, "mongodb": 20, "spark": 70,
                 "excel": 30, "r": 40}
    return Sources(skills, digital, broader, groups, data_jobs,
                   ["python", "c#", "node.js", "nodejs", "postgres"])


NO_FILTERS = Filters()


def _row(df, alias):
    r = df[df.alias.str.lower() == alias.lower()]
    assert len(r) == 1, (alias, r)
    return r.iloc[0]


def test_esco_scope_is_digital_plus_descendants_of_s5_and_06():
    assert vocab.esco_scope(_sources()) == {"u1", "u2", "u3p", "u3", "u5", "u6"}


def test_preferred_alt_and_parenthetical_variants_are_aliases():
    df = vocab.build(_sources(), NO_FILTERS)
    for alias in ("Python (computer programming)", "Python", "Python 3K"):
        assert _row(df, alias).canonical == "Python (computer programming)"
    assert "manage music staff" not in set(df.alias)  # outside the ICT scope


def test_sources_are_recorded_per_alias():
    df = vocab.build(_sources(), NO_FILTERS)
    assert _row(df, "python").source == "esco|data_jobs|regex"
    assert _row(df, "python").data_jobs_postings == "100"


def test_a_named_tool_is_not_folded_into_a_broad_esco_competence():
    """ESCO lists `tensorflow` as an alt label of *computer vision*; it stays itself."""
    df = vocab.build(_sources(), NO_FILTERS)
    row = _row(df, "tensorflow")
    assert row.canonical == "tensorflow" and row.source == "esco|data_jobs"
    assert row.esco_uri == ""


def test_synonyms_fold_spellings_onto_one_canonical():
    df = vocab.build(_sources(), NO_FILTERS)
    assert _row(df, "mongo").canonical == _row(df, "mongodb").canonical == "mongodb"
    assert _row(df, "nodejs").canonical == _row(df, "node.js").canonical == "node.js"
    assert _row(df, "postgres").canonical == "postgresql"


def test_esco_homonym_is_withheld():
    df = vocab.build(_sources(), NO_FILTERS)
    assert _row(df, "spark").canonical == "spark" and _row(df, "spark").source == "data_jobs"


def test_filters():
    df = vocab.build(_sources(), Filters(min_chars=3, max_tokens=2, stopwords=True,
                                         blacklist=("EXCEL",)))
    aliases = set(df.alias.str.lower())
    assert "it" not in aliases                       # stopword
    assert "r" not in aliases and "c#" in aliases    # min_chars spares punctuation
    assert "use spreadsheets software" not in aliases  # max_tokens
    assert "excel" not in aliases                    # blacklist, case-insensitive
    assert "information technology" in aliases


def test_render_is_deterministic_and_digest_is_of_the_bytes(tmp_path):
    a = vocab.render(vocab.build(_sources()))
    b = vocab.render(vocab.build(_sources()))
    assert a == b
    p = tmp_path / "skills.csv"
    p.write_bytes(a)
    import hashlib
    assert vocab.digest(p) == hashlib.sha256(a).hexdigest()
    assert a.decode().splitlines()[0] == ",".join(vocab.COLUMNS)


def test_one_row_per_matcher_pattern():
    df = vocab.build(_sources(), NO_FILTERS)
    m = SkillMatcher(df)
    assert m.n_aliases == len(df)


def test_regex_families_expand_to_literals():
    terms = vocab.regex_terms()
    assert len(terms) == 45
    assert {"c++", "node.js", "nodejs", "scikit-learn"} <= set(terms)
    for t in terms:
        assert any(re.search(p, f" {t}1") or re.fullmatch(p.replace(r"\b", ""), t)
                   for p in SKILL_PATTERNS), t


def test_regression_baseline_cpp_pattern_cannot_match_cpp_before_a_space():
    """Why `regex_terms` checks the unanchored alternative: `\\bc\\+\\+\\b` is dead on
    `c++ ` (no word boundary after `+`). The literal must still reach the vocabulary."""
    assert re.findall(SKILL_PATTERNS[0], "i know c++ well") == []
    assert "c++" in vocab.regex_terms()


def test_regex_terms_refuses_an_unexpected_shape():
    with pytest.raises(ValueError):
        vocab.regex_terms([r"(python|java)"])


@pytest.mark.skipif(not (vocab.ESCO_DIR.is_dir() and vocab.VOCAB_CSV.is_file()),
                    reason="needs data/raw ESCO + data_jobs and a built data/vocab/skills.csv")
def test_committed_vocabulary_rebuilds_byte_identically():
    assert vocab.check()
