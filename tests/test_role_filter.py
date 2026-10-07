"""Shortlist recall, C1-C3 and the Y rungs (plan 2026-10-07, Q2): synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candidate_screener.extraction import jd_title, role_family as rf, role_set
from candidate_screener.matching import role_filter as yf
from conftest import reads_final_labels

LABELS = ("Java", "Python", "QA")
ALIASES = {"java developer": "Java", "python developer": "Python", "qa engineer": "QA"}


@pytest.fixture(scope="module")
def models():
    texts = (["java spring hibernate backend services"] * 6
             + ["python django pandas services"] * 6
             + ["manual testing test cases regression bugs"] * 6)
    y = ["Java"] * 6 + ["Python"] * 6 + ["QA"] * 6
    sides = {s: rf.fit_tfidf(texts, y, ALIASES, LABELS) for s in rf.SIDES}
    role = rf.RoleModel("tfidf", LABELS, dict(ALIASES), sides, "test")
    clf = jd_title.fit_title_classifier(
        ["java engineer", "java dev", "python engineer", "python dev", "tester", "qa"] * 3,
        ["Java", "Java", "Python", "Python", "QA", "QA"] * 3)
    title = jd_title.TitleModel(LABELS, dict(ALIASES), clf, "test")
    return role, title


def _frame(jd_titles, cv_texts, jd_text="python django services, 3 years of experience"):
    n = len(cv_texts)
    return pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(n)], "jd_id": ["j"] * n,
        "cv_id": [f"c{i}" for i in range(n)], "jd_title": list(jd_titles),
        "jd_text": [jd_text] * n, "cv_text": list(cv_texts),
        "jd_keyword": ["Python"] * n, "cv_keyword": ["Java"] * n,
        "jd_exp_band": ["2-3"] * n, "cv_exp_band": ["0-1"] * n,
        "jd_position": ["x"] * n, "cv_position": ["Python developer"] * n})


def test_jd_title_lookup_then_classifier(models):
    _, title = models
    out = jd_title.jd_family(["Senior Python Developer", "Java guru", "", None], title)
    assert list(out.source) == ["lookup", "classifier", "none", "none"]
    assert list(out.family[:2]) == ["Python", "Java"]
    assert out.family[2] is None


def test_scores_never_read_cv_position_keyword_or_experience_fields(models):
    """Q2 acceptance (D53/D54): scrambling every metadata column leaves every Y score
    unchanged; only jd_title, jd_text and cv_text are inputs."""
    from candidate_screener.matching import rules
    role, title = models
    cvs = ["Java developer\nspring hibernate", "QA engineer\nregression testing",
           "Python developer 2019 - 2021\ndjango pandas", "cook"]
    frame = _frame(["Python Developer"] * 4, cvs)
    rule = yf.RoleFilterRule(k=1, context=True, role_model=role, title_model=title)
    rule.fit(None)
    y2 = yf.RoleFilterPlus(rules.SkillOverlap(), "Y2", rule)
    y2.fit(None)
    y2t = yf.RoleFilterTitle(rules.SkillOverlap(), "Y2t", rule)
    y2t.fit(None)
    before = [s.score(frame).to_numpy() for s in (rule, y2, y2t)]
    rng = np.random.default_rng(0)
    scrambled = frame.assign(**{c: rng.permutation(frame[c].to_numpy())
                                for c in ("cv_keyword", "jd_keyword", "cv_exp_band",
                                          "jd_exp_band", "cv_position", "jd_position")})
    scrambled["cv_position"] = ["QA engineer", "Java developer", "cook", "cto"]
    after = [s.score(scrambled).to_numpy() for s in (rule, y2, y2t)]
    for b, a in zip(before, after):
        assert np.array_equal(b, a)
    with pytest.raises(KeyError):                       # the title is an input; nothing else
        rule.score(frame.drop(columns="jd_title"))


def test_a_cv_with_two_role_titles_matches_a_jd_of_either(models):
    """Q2 acceptance (decision 1b): the CV side is a set of families."""
    role, title = models
    cv = "Java developer at Acme 2015 - 2018\n\nPython developer at Beta 2018 - 2021"
    frame = _frame(["Java Developer", "Python Developer", "QA Engineer"], [cv] * 3)
    rule = yf.RoleFilterRule(k=0, context=False, role_model=role, title_model=title)
    rule.fit(None)
    c = rule.components(frame)
    assert list(c.jd_family) == ["Java", "Python", "QA"]
    assert list(c.family_match) == [1, 1, 0]


def test_role_context_drops_mentions_outside_a_role_line():
    text = ("Summary\nI worked closely with the QA engineer on releases.\n"
            "- Java developer\nPosition: Python developer\n"
            "Acme Corp, QA engineer, 2019 - 2021")
    starts = {m: text.index(m) for m in ("QA engineer on", "Java developer", "Python developer")}
    assert not role_set.in_role_context(text, starts["QA engineer on"])
    assert role_set.in_role_context(text, starts["Java developer"])          # (a) line start
    assert role_set.in_role_context(text, starts["Python developer"])        # (b) cue
    assert role_set.in_role_context(text, text.rindex("QA engineer"))        # (c) header line


def test_cv_role_set_options(models):
    role, _ = models
    cv = "I helped the QA engineer.\nPython developer\npython django pandas services"
    r = role_set.cv_roles([cv], role)[0]
    assert r.families(0, False) == {"QA", "Python"}
    assert r.families(0, True) == {"Python"}
    assert r.families(1, True) == {"Python"}                  # top-1 is Python as well
    assert len(r.families(3, True)) == 3
    with pytest.raises(ValueError):
        r.families(4, True)


def test_family_match_is_a_level_text_never_crosses_it(models):
    """Y1 is a level (decision 2): the best non-matching CV ranks below the worst match."""
    from candidate_screener.matching import rules
    role, title = models
    cvs = ["Python developer\n", "Java developer\npython django pandas services python django"]
    rule = yf.RoleFilterRule(k=0, context=True, role_model=role, title_model=title)
    rule.fit(None)
    y2 = yf.RoleFilterPlus(rules.SkillOverlap(), "Y2", rule)
    y2.fit(None)
    s = y2.score(_frame(["Python Developer"] * 2, cvs)).to_numpy()
    assert s[0] > s[1]


def test_trigram_cosine():
    assert yf.trigram_cosine("python developer", "python developer") == pytest.approx(1.0)
    assert yf.trigram_cosine("", "python developer") == 0.0
    a = yf.trigram_cosine("python developer", "python engineer")
    b = yf.trigram_cosine("python developer", "account manager")
    assert 0 < b < a < 1


def test_components_never_read_final_labels_or_judgements():
    for module in (jd_title, role_set, yf):
        assert reads_final_labels(module) == []
