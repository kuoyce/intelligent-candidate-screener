"""The P2 title-profile measurements (plan 2026-10-06): synthetic data only."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from candidate_screener.extraction import title_profile as tp


@pytest.mark.parametrize("raw, norm", [
    ("Senior Java Developer", "java developer"),
    ("Middle+ Python Engineer (Django)", "python developer"),
    ("Front-End Developer", "frontend developer"),
    ("front end developer", "frontend developer"),
    ("Frontend developer", "frontend developer"),
    ("UX/UI designer", "uiux designer"),
    ("UI/UX Designer", "uiux designer"),
    ("Junior Node.js developer, remote", "nodejs developer"),
    ("Senior .NET Developer", "dotnet developer"),
    ("Strong Junior React Native Dev", "reactnative developer"),
    ("Senior Software Engineer - Fintech", "software developer"),
    ("QA Engineer II", "qa engineer"),
    ("Head of Sales", "sales"),
    ("Trainee", ""),
    ("Golang developer", "go developer"),
    ("Data Engineer", "data engineer"),      # not a development qualifier: noun kept
    ("DevOps Engineer", "devops engineer"),
])
def test_normalise_title(raw, norm):
    assert tp.normalise_title(raw) == norm


def test_lead_kept_in_lead_generation_but_stripped_as_seniority():
    """`Lead Generation` is a keyword; `Lead` before a role is a seniority marker."""
    assert tp.normalise_title("Lead Generation Specialist") == "lead generation specialist"
    assert tp.normalise_title("Lead Python Developer") == "python developer"
    assert tp.normalise_title("Team Lead .NET") == "dotnet"


def test_normalise_title_handles_missing():
    assert tp.normalise_title(None) == ""
    assert tp.normalise_title(float("nan")) == ""


def test_role_function_drops_languages_not_platforms():
    assert tp.role_function("java developer") == "developer"
    assert tp.role_function("ios developer") == "ios developer"
    assert tp.role_function("frontend developer") == "frontend developer"


def test_find_title_uses_the_same_rule_as_titles():
    grams = tp.ngram_offsets(tp.canonical_tokens("We hire a Senior Front-End Engineer to ..."))
    off = tp.find_title(grams, tp.normalise_title("Middle Frontend Developer"))
    assert off is not None and off < 30
    assert tp.find_title(grams, "python developer") is None
    assert tp.find_title(grams, "") is None


def test_find_title_is_token_bounded():
    """`java developer` must not be found inside `javascript developer`."""
    grams = tp.ngram_offsets(tp.canonical_tokens("JavaScript developer wanted"))
    assert tp.find_title(grams, "java developer") is None
    assert tp.find_title(grams, "javascript developer") == 0


def test_find_any_returns_the_earliest_alias():
    text = "x " * 150 + "experienced data analyst and later a project manager"
    grams = tp.ngram_offsets(tp.canonical_tokens(text))
    off = tp.find_any(grams, {"project manager", "data analyst"})
    assert off == text.index("data analyst")
    assert tp.find_any(grams, {"chef de partie"}) is None
    assert tp._where(off) == "later" and tp._where(10) == "first" and tp._where(None) == "none"


def test_first_alias_prefers_longest_at_same_offset():
    grams = tp.ngram_offsets(tp.canonical_tokens("qa automation engineer needed"))
    assert tp.first_alias(grams, {"qa automation", "qa automation engineer"}) == \
        "qa automation engineer"


def test_esco_labels_maps_alt_labels_to_preferred():
    occ = pd.DataFrame({"preferredLabel": ["software developer", "chef"],
                        "altLabels": ["Senior Programmer\nsoftware engineer", None],
                        "iscoGroup": [2512, 5120]})
    table = tp.esco_labels(occ)
    assert table["software developer"] == "software developer"
    assert table["chef"] == "chef"
    assert table["developer"] == "software developer"   # `Senior Programmer`: synonym + seniority
    assert set(tp.esco_labels(occ, ict_only=True).values()) == {"software developer"}


def test_data_jobs_labels_min_count_and_modal_class():
    dj = pd.DataFrame({
        "job_title": ["Senior Data Engineer"] * 3 + ["Data Engineer - Remote"] * 2
                     + ["Rare Unicorn Wrangler"],
        "job_title_short": ["Data Engineer"] * 5 + ["Data Scientist"]})
    every = tp.data_jobs_labels(dj)
    assert every["data engineer"] == "Data Engineer"
    assert "rare unicorn wrangler" in every
    kept = tp.data_jobs_labels(dj, min_count=2)
    assert "rare unicorn wrangler" not in kept
    assert "data scientist" in kept       # the class name itself is always an alias


def test_tb_families_needs_both_sides():
    jd = pd.Series(["java developer"] * 3 + ["qa engineer"] * 3 + ["cto"] * 5)
    cv = pd.Series(["java developer"] * 3 + ["qa engineer"] * 1 + [""] * 9)
    assert tp.tb_families(jd, cv, 3) == {"java developer"}


def test_fanout_counts_families_per_keyword():
    kw = pd.Series(["Python"] * 6 + ["QA"] * 2)
    fam = pd.Series(["python developer"] * 3 + ["data engineer"] * 2 + [""] + ["qa engineer"] * 2)
    out = tp.fanout(kw, fam, ["Python", "QA", "Absent"])
    assert out["Python"]["families"] == 2
    assert out["Python"]["share_in_a_family"] == round(5 / 6, 4)
    assert out["Python"]["top_families"]["python developer"] == 0.5
    assert out["QA"]["families_for_80pct_of_mapped"] == 1
    assert "Absent" not in out


def test_purity_is_one_when_families_nest_in_keywords():
    kw = pd.Series(["A", "A", "B", "B"])
    assert tp.purity(kw, pd.Series(["x", "x", "y", "y"]))["doc_weighted"] == 1.0
    assert tp.purity(kw, pd.Series(["x", "x", "x", "x"]))["doc_weighted"] == 0.5


def test_coverage_counts_docs_and_distinct_strings():
    norm = pd.Series(["java developer", "java developer", "cto", ""])
    c = tp.coverage(norm, {"java developer": "x"})
    assert c["docs"]["k"] == 2 and c["docs"]["n"] == 3
    assert c["distinct_strings"] == {"k": 1, "n": 2, "share": 0.5}


def test_wilson_interval():
    w = tp.wilson(50, 100)
    assert w["share"] == 0.5 and w["ci95"][0] < 0.5 < w["ci95"][1]
    assert tp.wilson(0, 0)["share"] is None
    assert tp.wilson(0, 10)["ci95"][0] == 0.0


def test_seeded_sample_ignores_row_order():
    df = pd.DataFrame({"id": [f"d{i}" for i in range(50)], "v": range(50)})
    a = tp.seeded_sample(df, 10, 0)
    b = tp.seeded_sample(df.sample(frac=1, random_state=3), 10, 0)
    assert list(a.id) == list(b.id)


def test_v2_cv_text_never_contains_position():
    cv = pd.DataFrame({"Position": ["Unique Title Zeta"], "CV": ["body"], "Highlights": [None],
                       "Moreinfo": ["more"], "Looking For": ["remote"]})
    text = tp.v2_cv_text(cv).iloc[0]
    assert "Zeta" not in text and "body" in text and "remote" in text


def test_title_in_text_end_to_end():
    texts = ["Python Developer needed for backend work", "We need someone for sales",
             "x " * 120 + "join as a python developer"]
    positions = ["Senior Python Developer", "Account Manager", "Python Developer"]
    out = tp.title_in_text(texts, positions, {"python developer"},
                           {"any": {"python developer", "account manager"}})
    assert out["own_position_exact"]["found"]["k"] == 2
    assert out["own_position_exact"]["found_in_first_200"]["k"] == 1
    assert out["own_family_title"]["found"]["n"] == 2
    assert out["any_alias_title"]["any"]["found"]["k"] == 2
    assert out["earliest_family_title_is_own_family"]["k"] == 2


def test_profile_merges_partial_runs(tmp_path, monkeypatch):
    """`profile --check X` must not discard the other sources' figures."""
    from candidate_screener.data import profile
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"generated": "old", "checks": {"a": 1, "b": 2}}))
    monkeypatch.setattr(profile, "CHECKS", {"a": lambda: 10, "b": lambda: 20})
    monkeypatch.setattr(sys, "argv", ["profile", "--check", "b", "--metrics", str(path)])
    assert profile.main() == 0
    rec = json.loads(path.read_text())
    assert rec["checks"] == {"a": 1, "b": 20}
    assert rec["generated"] == "old" and set(rec["updated"]) == {"b"}
