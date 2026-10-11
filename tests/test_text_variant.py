"""`text_v2` (D53): the CV's `Position` field is out of the text a system reads, and the
two variants never share a cache. Plus the residual-leak measure (plan 2026-10-06 P1.2)."""
import pandas as pd
import pytest

from candidate_screener.annotation import sample
from candidate_screener.evaluation import text_leak
from candidate_screener.matching import corpus

TITLE = "Zyxquorbian Flux Wrangler"


def _cvs():
    return pd.DataFrame({
        "Position": [TITLE, "QA Engineer", None],
        "CV": ["Built pipelines in Spark.", "", "Manual testing"],
        "Highlights": ["Led a team", None, ""],
        "Moreinfo": ["", "Selenium, pytest", "Jira"],
        "Looking For": ["Remote", "Product company", None],
    })


def test_v2_cv_text_never_contains_the_position():
    """Named invariant (P1.1). Fails if the variant is ignored and `v1` text comes back."""
    v1 = corpus.build_cv_text(_cvs(), "v1")
    v2 = corpus.build_cv_text(_cvs(), "v2")
    assert TITLE in v1.iloc[0]
    assert TITLE not in v2.iloc[0]
    assert "QA Engineer" not in v2.iloc[1]
    assert "Built pipelines in Spark." in v2.iloc[0] and "Remote" in v2.iloc[0]


def test_v1_is_the_annotation_builder_exactly():
    """`annotation/` is frozen (D42): the mirror must not drift from it."""
    assert corpus.CV_FIELDS_BY_VARIANT["v1"] == sample.CV_FIELDS
    assert corpus.CV_FIELDS_BY_VARIANT["v2"] == tuple(f for f in sample.CV_FIELDS
                                                      if f != "Position")
    assert corpus.build_cv_text(_cvs(), "v1").equals(sample.cv_text(_cvs()))


def test_v2_joins_like_v1_without_the_first_field():
    """Same separators and blank handling as the annotation builder, minus `Position`."""
    no_title = _cvs().assign(Position=None)
    assert corpus.build_cv_text(_cvs(), "v2").equals(sample.cv_text(no_title))


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="text_variant"):
        corpus.build_cv_text(_cvs(), "v3")
    with pytest.raises(ValueError, match="text_variant"):
        corpus.variant_tag("V2")


def test_cache_names_differ_by_variant_and_v1_keeps_its_pre_d53_names():
    assert corpus.variant_tag("v1") == "" and corpus.variant_tag("v2") == "-v2"
    sbert = pytest.importorskip("candidate_screener.matching.sbert")
    assert sbert.SbertZeroShot.cache_key("v1") == "minilm-zero-shot"
    assert sbert.SbertZeroShot.cache_key("v2") != sbert.SbertZeroShot.cache_key("v1")
    assert sbert.SbertWeak.cache_key("v2") != sbert.SbertWeak.cache_key("v1")
    assert "text_variant" not in sbert.SbertWeak.config("v1")
    assert sbert.SbertWeak.config("v2")["text_variant"] == "v2"


def test_lexical_rungs_carry_their_variant():
    from candidate_screener.matching import lexical
    assert lexical.Tfidf().text_variant == "v1" and lexical.Tfidf("v2").text_variant == "v2"
    assert lexical.Tfidf().trained_on == lexical.Tfidf.trained_on
    assert "text_v2" in lexical.Bm25("v2").trained_on


# --- residual leak ---------------------------------------------------------------------

def test_title_normalisation_strips_seniority_and_edges():
    assert text_leak.normalise_title("Senior  Python Developer") == "python developer"
    assert text_leak.normalise_title("Middle/Senior QA") == "qa"
    assert text_leak.normalise_title("Team Lead, Java") == "team, java"
    assert text_leak.normalise_title("Senior") == ""


def test_title_match_is_whole_word_and_skips_empty_titles():
    assert text_leak.contains_title("I am a QA engineer", "Junior QA") is True
    assert text_leak.contains_title("aqua marine", "QA") is False        # not a substring hit
    assert text_leak.contains_title("python\n\ndeveloper here", "Python Developer") is True
    assert text_leak.contains_title("anything", "Lead") is None


def test_leak_report_counts_cvs_and_jds_per_stratum():
    pairs = pd.DataFrame({
        "stratum": ["generic"] * 2 + ["targeted"] * 2, "jd_id": ["j1", "j1", "j2", "j2"],
        "cv_id": ["a", "b", "c", "d"],
        "jd_text": ["We need a Python Developer"] * 2 + ["QA role"] * 2,
        "jd_position": ["Senior Python Developer"] * 2 + ["Data Scientist"] * 2,
        "cv_text": ["python developer, 5 years", "java", "data", ""],
        "cv_position": ["Python Developer", "Java Dev", "Senior", "QA"]})
    r = text_leak.report(pairs)
    assert r["cv"]["total"]["n"] == 3 and r["cv"]["total"]["positives"] == 1
    assert r["cv"]["total"]["skipped_empty_title"] == 1
    assert r["cv"]["total"]["empty_v2_text"] == 1
    assert r["cv"]["generic"]["share"] == 0.5 and r["cv"]["targeted"]["share"] == 0.0
    assert r["jd"]["generic"]["share"] == 1.0 and r["jd"]["targeted"]["share"] == 0.0
    assert r["cv_pairs_per_jd"]["total"]["n"] == 2


# --- the frames systems read ---------------------------------------------------------

def _raw():
    jd = pd.DataFrame({"id": ["j1", "j2"], "Long Description_lang": ["en", "en"],
                       "Long Description": ["Need a flux wrangler", "QA role"],
                       "Exp Years": ["2y", "5y"], "Primary Keyword": ["Flux", "QA"],
                       "Position": ["Flux Wrangler", "QA"]})
    cv = _cvs().assign(id=["c1", "c2", "c3"], CV_lang="en", **{"Experience Years": [2, 5, 1],
                       "Primary Keyword": ["Flux", "QA", "QA"]})
    return jd, cv


def test_train_documents_apply_the_variant_with_the_same_ids(tmp_path, monkeypatch):
    """What TF-IDF's IDF and SBERT weak train on: v2 drops the title, keeps the documents,
    and caches under its own name. Fails if `_djinni` ignores the variant."""
    jd, cv = _raw()
    monkeypatch.setattr(corpus, "CACHE", tmp_path)
    monkeypatch.setattr(corpus, "load_split", lambda name, split: jd if name == "djinni-jd" else cv)
    monkeypatch.setattr(corpus, "load_train_region", lambda: (set(jd["id"]), set(cv["id"])))
    corpus.a2_train_documents.cache_clear()
    try:
        _, cv1 = corpus.a2_train_documents(2, 3, 0, text_variant="v1")
        _, cv2 = corpus.a2_train_documents(2, 3, 0, text_variant="v2")
    finally:
        corpus.a2_train_documents.cache_clear()
    assert list(cv1["id"]) == list(cv2["id"])
    t1, t2 = dict(zip(cv1["id"], cv1.cv_text)), dict(zip(cv2["id"], cv2.cv_text))
    assert TITLE in t1["c1"] and TITLE not in t2["c1"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "a2-train-cv-3-0.parquet", "a2-train-cv-v2-3-0.parquet",
        "a2-train-jd-2-0.parquet", "a2-train-jd-v2-2-0.parquet"]


def test_a2_pairs_apply_the_variant(tmp_path, monkeypatch):
    """What every rung scores. Fails if `a2_pairs` ignores the variant."""
    jd, cv = _raw()
    jd = jd.assign(jd_text=jd["Long Description"], exp_band="2-3")
    cv = cv.assign(cv_text=sample.cv_text(cv), exp_band="2-3")
    manifest = tmp_path / "pairs.csv"
    pd.DataFrame({"pair_id": ["j1__c1", "j2__c2"], "stratum": ["generic", "targeted"],
                  "jd_id": ["j1", "j2"], "cv_id": ["c1", "c2"]}).to_csv(manifest, index=False)
    monkeypatch.setattr(corpus, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(corpus.final_set, "PAIRS_CSV", manifest)
    monkeypatch.setattr(sample, "load_english", lambda: (jd.copy(), cv.copy()))
    corpus.a2_pairs.cache_clear()
    try:
        v1, v2 = corpus.a2_pairs("v1"), corpus.a2_pairs(text_variant="v2")
    finally:
        corpus.a2_pairs.cache_clear()
    assert TITLE in v1.cv_text.iloc[0] and TITLE not in v2.cv_text.iloc[0]
    assert v2.cv_position.iloc[0] == TITLE            # kept for measurement only
    assert len(list((tmp_path / "cache").iterdir())) == 2
