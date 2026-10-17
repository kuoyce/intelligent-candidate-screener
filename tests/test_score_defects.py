"""Score defects (plan 2026-10-17-score-defects): synthetic text only."""
import pandas as pd

from candidate_screener.evaluation import score_defects as sd


def test_negated_clause_is_dropped_and_affirmed_clause_kept():
    t = "Built SPAs in React. No Angular/Vue projects. Python daily"
    assert sd.negated_clauses(t) == [" No Angular/Vue projects"]
    out = sd.strip_negated(t)
    assert "Angular" not in out and "React" in out and "Python" in out


def test_negation_without_a_usage_word_is_not_a_negated_skill():
    t = "Not only backend, also frontend with Angular"
    assert sd.negated_clauses(t) == [] and sd.strip_negated(t) == t


def test_no_professional_experience_phrases():
    assert sd.states_no_experience("I have no commercial experience yet")
    assert sd.states_no_experience("I don't have any professional experience")
    assert not sd.states_no_experience("5 years of commercial experience")


def test_f2_zeroes_only_when_the_jd_asks_for_years():
    base = pd.Series([1.0, 1.0, 1.0], index=list("abc"))
    jd = pd.Series(["3+ years of experience with Python", "Python developer", "3+ years of experience"], index=list("abc"))
    cv = pd.Series(["No commercial experience", "No commercial experience", "4 years of commercial experience"], index=list("abc"))
    assert list(sd.f2_experience(base, jd, cv)) == [0.0, 1.0, 1.0]
