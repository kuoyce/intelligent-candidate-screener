"""E3 experience years and E4 seniority rules (plan 2026-10-06, P4): synthetic text only.

The named invariants (task 4.3), each shown to fail against a naive implementation in the
P4 record (13-p4-experience.md):

- overlapping date ranges are not double-counted (naive: sum of range lengths);
- "5+ years" is a minimum with no maximum (naive: one `N years` regex giving (N, N));
- an open range uses the reference date, not today (naive: `date.today()`);
- a year range in a JD is not years of experience (naive: `(\\d+)\\s*-\\s*(\\d+)`);
- bands match `JD_EXP_TO_BAND` / `cv_experience_band` (naive: one numeric banding for both);
- unknown extracted band scores 0, as the 1a rule does (naive: unknown = compatible).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from candidate_screener.data.a2_finetune import JD_EXP_TO_BAND, cv_experience_band
from candidate_screener.extraction import experience as ex
from candidate_screener.extraction.title_profile import SENIORITY


# --- invariant: overlapping ranges ------------------------------------------------------

def test_overlapping_date_ranges_are_not_double_counted():
    text = "Developer, Acme\n2018 - 2021\nConsultant, Beta\n2020 - 2022\n"
    got = ex.cv_experience(text, reference=dt.date(2023, 9, 1))
    assert got.source == "ranges"
    # mid-2018 .. mid-2022 is 4 years; the plain sum would be 3 + 2 = 5
    assert got.years == pytest.approx(4.0)
    assert ex.merged_years([(0, 3), (2, 4), (10, 11)]) == pytest.approx(5.0)


def test_nested_range_adds_nothing():
    assert ex.merged_years([(2010, 2020), (2012, 2014)]) == pytest.approx(10.0)


# --- invariant: "5+ years" is a minimum -------------------------------------------------

@pytest.mark.parametrize("text, lo", [
    ("- 5+ years of experience with Java", 5.0),
    ("5 years+ of commercial experience", 5.0),
    ("At least 3 years of experience in QA", 3.0),
    ("Experience: from 2 years", 2.0),
    ("three years of experience as a designer", 3.0),
])
def test_plus_and_at_least_parse_as_minimum_without_maximum(text, lo):
    got = ex.jd_experience(text)
    assert got.min_years == lo
    assert got.max_years is None


def test_range_keeps_both_ends():
    got = ex.jd_experience("2-5 years of commercial experience")
    assert (got.min_years, got.max_years) == (2.0, 5.0)


def test_requirement_is_the_largest_required_minimum_and_skips_nice_to_have():
    text = ("Requirements:\n- 2+ years of experience with Python\n"
            "- At least 4 years of experience in backend development\n"
            "Nice to have:\n- 7+ years of experience with Go\n")
    got = ex.jd_experience(text)
    assert got.min_years == 4.0
    assert len(got.spans) == 3                       # the optional one is still evidence
    assert ex.jd_experience(text, aggregate="first").min_years == 2.0


def test_evidence_spans_point_at_the_text():
    text = "We need someone with 3+ years of experience in QA."
    span = ex.jd_experience(text).spans[0]
    assert text[span.start:span.end] == span.text and "3" in span.text


# --- invariant: open range uses the reference date -------------------------------------

def test_open_range_uses_the_reference_date_not_today():
    text = "Backend developer\nJan 2020 - present\n"
    assert ex.cv_experience(text, reference=dt.date(2022, 1, 1)).years == pytest.approx(2.0)
    assert ex.cv_experience(text, reference=dt.date(2023, 1, 1)).years == pytest.approx(3.0)
    default = ex.cv_experience(text).years
    assert default == ex.cv_experience(text, reference=ex.REFERENCE_DATE).years
    assert default == pytest.approx(3 + 8 / 12, abs=1e-3)     # 2023-09-01, not the clock


def test_month_ranges_and_numeric_months():
    got = ex.cv_experience("QA engineer 03.2018–05.2021", reference=dt.date(2023, 9, 1))
    assert got.years == pytest.approx(3 + 3 / 12, abs=1e-3)


def test_range_ending_after_the_reference_is_capped():
    got = ex.cv_experience("Engineer\n2022 - 2030\n", reference=dt.date(2023, 1, 1))
    assert got.years == pytest.approx(0.5)


# --- invariant: a date range in a JD is not experience ---------------------------------

@pytest.mark.parametrize("text", [
    "The project ran 2019-2022 and is now in support.",
    "Roadmap 2019 – 2022: migrate the platform.",
    "In 12 years the company has grown to 300 people.",
    "With over 10 years of experience, we have built 200 products.",
    "24 calendar days of vacation per year",
])
def test_dates_and_company_age_are_not_a_requirement(text):
    assert ex.jd_experience(text).min_years is None


def test_no_experience_required_is_zero():
    assert ex.jd_experience("No experience required, we will teach you.").min_years == 0.0


# --- CV statements ----------------------------------------------------------------------

def test_cv_explicit_statement_wins_over_ranges():
    text = "I have 6+ years of experience in web development.\nDev\n2021 - 2022\n"
    got = ex.cv_experience(text)
    assert (got.years, got.source) == (6.0, "statement")


@pytest.mark.parametrize("text", ["I am 25 years old.", "Graduated 3 years ago."])
def test_age_and_ago_are_not_experience(text):
    assert ex.cv_experience(text).years is None


def test_education_ranges_are_skipped():
    assert ex.cv_experience("Bachelor of Science, Some University\n2010 - 2014").years is None


def test_months_are_fractions_of_a_year():
    assert ex.cv_experience("8 months of commercial experience").years == pytest.approx(8 / 12,
                                                                                       abs=1e-3)


# --- invariant: band mapping is the metadata's ------------------------------------------

@pytest.mark.parametrize("category, years", [("no_exp", 0), ("1y", 1), ("2y", 2),
                                             ("3y", 3), ("5y", 5)])
def test_jd_band_matches_jd_exp_to_band(category, years):
    assert ex.jd_exp_years_category(years) == category
    assert ex.jd_band(years) == JD_EXP_TO_BAND[category]


def test_jd_four_years_snaps_down_to_3y():
    """Djinni has no 4-year option; 4 reads as `3y` (band 2-3), never the CV's 4-6."""
    assert ex.jd_band(4) == "2-3"
    assert ex.jd_band(12) == "4-6"            # JD bands never reach 7+
    assert ex.jd_band(None) is None


def test_cv_band_is_cv_experience_band():
    for y in np.arange(0, 13, 0.25):
        assert ex.cv_band(float(y)) == cv_experience_band(float(y))
    assert ex.cv_band(None) is None


# --- invariant: experience_compatible on extracted bands --------------------------------

def test_experience_compatible_extracted_is_the_1a_rule_with_unknown_as_zero():
    from candidate_screener.matching.corpus import experience_compatible
    jds = ["3+ years of experience with Python", "3+ years of experience with Python",
           "We value curiosity.", "1+ year of experience in QA"]
    cvs = ["I have 5 years of experience in backend.", "I have 1 year of experience.",
           "I have 5 years of experience in backend.", "Hobbies: chess."]
    got = ex.experience_compatible_extracted(jds, cvs)
    assert list(got) == [1, 0, 0, 0]          # unknown on either side -> 0
    bands = pd.DataFrame({"jd_exp_band": ["2-3", "2-3", None, "0-1"],
                          "cv_exp_band": ["4-6", "0-1", "4-6", None]})
    assert list(experience_compatible(bands)) == list(got)


def test_experience_compatible_extracted_requires_aligned_pairs():
    with pytest.raises(ValueError):
        ex.experience_compatible_extracted(["a"], ["b", "c"])


# --- E4 seniority -----------------------------------------------------------------------

def test_token_levels_cover_the_shared_seniority_list():
    assert set(ex.TOKEN_LEVEL) == set(SENIORITY)
    assert {v for v in ex.TOKEN_LEVEL.values() if v} == set(ex.LEVELS)


@pytest.mark.parametrize("position, level", [
    ("Senior Java Developer", "senior"), ("Junior/Middle QA", "middle"),
    ("Team Lead .NET", "lead"), ("Head of Sales", "lead"), ("Trainee", "intern"),
    ("Lead Generation Manager", None), ("Python Developer", None), (None, None),
])
def test_position_seniority(position, level):
    assert ex.position_seniority(position) == level


def test_text_seniority_needs_a_title_not_a_colleague():
    assert ex.text_seniority("We are looking for a Senior Python Developer.") == ("senior", "token")
    assert ex.text_seniority("You will work with senior engineers.") == (None, None)
    assert ex.text_seniority("Lead generation for B2B clients.") == (None, None)


def test_text_seniority_pick_and_years_fallback():
    text = "Junior developer at A.\nLater a Senior developer at B."
    assert ex.text_seniority(text, pick="first")[0] == "junior"
    assert ex.text_seniority(text, pick="max")[0] == "senior"
    assert ex.text_seniority("Hobbies: chess.", years=6) == ("senior", "years")
    assert ex.text_seniority("Hobbies: chess.", years=1) == ("junior", "years")
    assert ex.text_seniority("Hobbies: chess.", years=None) == (None, None)


# --- evaluation helpers -----------------------------------------------------------------

def test_band_block_counts_unknown_as_wrong_and_skips_missing_oracle():
    from candidate_screener.extraction.experience_eval import band_block
    b = band_block(["0-1", "2-3", "2-3", None], ["0-1", None, "4-6", "0-1"])
    assert b["n"] == 3
    assert b["coverage"]["k"] == 2
    assert b["accuracy_all"]["k"] == 1 and b["accuracy_covered"]["n"] == 2
    assert b["confusion_oracle_by_extracted"]["2-3"] == {"0-1": 0, "2-3": 0, "4-6": 1,
                                                         "7+": 0, "none": 1}


def test_pair_agreement_reports_both_strata_and_total():
    from candidate_screener.extraction.experience_eval import pair_agreement
    frame = pd.DataFrame({"jd_id": ["a", "a", "b", "b", "c", "c"],
                          "stratum": ["generic"] * 4 + ["targeted"] * 2,
                          "extracted": [1, 0, 1, 1, 0, 0], "oracle": [1, 1, 1, 1, 0, 1],
                          "both_known": [True, False, True, True, True, False]})
    out = pair_agreement(frame)
    assert set(out) == {"generic", "targeted", "total"}
    assert out["total"]["n_pairs"] == 6 and out["total"]["agreement"] == pytest.approx(4 / 6,
                                                                                   abs=1e-4)
    assert out["generic"]["n_jds"] == 2 and out["targeted"]["extracted_0_oracle_1"] == 1
    assert out["total"]["both_bands_extracted"] == 4
    assert out["total"]["agreement_where_both_extracted"] == pytest.approx(1.0)
