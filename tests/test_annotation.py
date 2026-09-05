"""The annotation instrument's guards *(decision D25)*.

Everything here protects one session of ~250 human judgements that cannot be re-run
cheaply. Three failures would each waste it silently, and none of them raises on its own:

- a pair reaching an annotator with PII still in it (`AGENTS.md` data rule 4);
- the dispatch file carrying `selection_reason`, which tells the annotator which pairs A1
  has already labelled and lets them anchor on the expected answer — that would void the
  A1 recheck, the only test of assumption **A13**;
- the candidate draw being uniform rather than banded, which returns ~5 `No Fit` per JD
  and makes P@5 identically zero for every system;
- **a later batch re-drawing an earlier one's pairs**, which orphans every label already
  collected against them. This one was real: seeding each JD by its *position* meant
  growing the campaign from 40 JDs to 60 changed 190 of 200 `pair_id`s *(D28)*.

All on inline synthetic corpora: the suite must run on a fresh clone with no `data/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candidate_screener.annotation import queue, redact, sample


# --- redaction -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("reach me at a.b+c@example.co.uk", "[EMAIL]"),
    ("call +380 44 123 4567 any time", "[PHONE]"),
    ("portfolio at https://jane.dev/cv", "[URL]"),
    ("employee id 123456789", "[NUMBER]"),
])
def test_redaction_masks_each_detectable_class(text, expected):
    assert expected in redact.redact_text(text)


def test_redaction_is_idempotent():
    """The queue builder redacts, and `verify --derived` re-reads. A second pass over
    already-masked text must not corrupt the placeholders."""
    once = redact.redact_text("mail a@b.co or call +380 44 123 4567")
    assert redact.redact_text(once) == once


def test_redaction_leaves_ordinary_numbers_alone():
    """A pattern loose enough to eat years and salaries damages the document the
    annotator has to read, which is a different way of wasting the session."""
    kept = redact.redact_text("5 years experience, 2019-2024, Python 3.12")
    assert "5 years" in kept and "Python 3.12" in kept


def test_assert_clean_raises_on_surviving_pii():
    with pytest.raises(AssertionError, match="survived redaction"):
        redact.assert_clean(pd.Series(["contact person@example.com"]), "test corpus")


def test_assert_clean_passes_after_redaction():
    redact.assert_clean(redact.redact(pd.Series(["contact person@example.com"])), "test")


# --- CV text construction (Q12) -------------------------------------------

def test_cv_text_concatenates_all_five_fields_not_the_cv_column():
    """Q12 measured the median rising 751 -> 1,525 chars from this alone. Using the `CV`
    column by itself is the single largest avoidable part of the length gap against A1."""
    cv = pd.DataFrame([{"Position": "Data Engineer", "CV": "built pipelines",
                        "Highlights": "led a team", "Moreinfo": "remote only",
                        "Looking For": "senior role"}])
    text = sample.cv_text(cv).iloc[0]
    for field in ("Data Engineer", "built pipelines", "led a team", "remote only",
                  "senior role"):
        assert field in text


def test_cv_text_drops_empty_fields_rather_than_writing_nan():
    cv = pd.DataFrame([{"Position": "QA", "CV": None, "Highlights": "",
                        "Moreinfo": None, "Looking For": "remote"}])
    text = sample.cv_text(cv).iloc[0]
    assert "nan" not in text.lower()
    assert text.startswith("QA") and text.endswith("remote")


# --- candidate banding -----------------------------------------------------

def synthetic_candidates() -> pd.DataFrame:
    """Nine CVs: three obviously on-topic, three adjacent, three unrelated."""
    texts = ["python sql airflow data pipelines warehouse",
             "python pandas sql etl analytics dashboards",
             "sql spark data engineering python modelling",
             "java spring backend microservices rest api",
             "javascript react frontend css components",
             "kubernetes docker terraform aws devops",
             "registered nurse patient clinical ward care",
             "chef kitchen menu food service preparation",
             "truck driver route logistics warehouse delivery"]
    return pd.DataFrame({"id": [f"cv_{i}" for i in range(len(texts))],
                         "cv_text": texts, "Primary Keyword": "Data Engineer"})


def test_banding_spans_the_similarity_range_rather_than_taking_the_top_five():
    """The choice that decides whether 200 judgements measure anything.

    Taking the top 5 by score gives 5 near-duplicates and no negatives; a uniform draw
    gives 5 negatives. Banding is what puts both in the set.
    """
    jd = pd.Series({"id": "jd_0",
                    "jd_text": "hiring a data engineer for python sql pipelines"})
    banded = sample.band_candidates(jd, synthetic_candidates(),
                                    np.random.SeedSequence(0), len(sample.SAME_BANDS))
    assert list(banded.band) == list(sample.SAME_BANDS)
    assert banded.id.nunique() == len(sample.SAME_BANDS)      # no CV drawn twice
    assert banded[banded.band == "high"].lexical_score.min() >= \
        banded[banded.band == "low"].lexical_score.max()


def test_banding_is_deterministic_at_a_fixed_seed():
    jd = pd.Series({"id": "jd_0", "jd_text": "python sql data pipelines"})
    first = sample.band_candidates(jd, synthetic_candidates(), np.random.SeedSequence(0), 5)
    second = sample.band_candidates(jd, synthetic_candidates(), np.random.SeedSequence(0), 5)
    assert list(first.id) == list(second.id)


# --- the queue -------------------------------------------------------------

def test_queue_id_is_stable_and_pair_specific():
    assert queue.queue_id("a1", "j_1", "r_1") == queue.queue_id("a1", "j_1", "r_1")
    assert queue.queue_id("a1", "j_1", "r_1") != queue.queue_id("a2", "j_1", "r_1")
    assert queue.queue_id("a1", "j_1", "r_1") != queue.queue_id("a1", "j_1", "r_2")


def test_dispatch_columns_exclude_everything_that_could_anchor_an_annotator():
    """`selection_reason` tells the reader which pairs A1 already labelled.

    An annotator who knows a pair is a recheck can anchor on the label they expect A1 to
    have given, and the measured agreement then says nothing about A13.
    """
    for leaky in ("selection_reason", "a1_label", "corpus", "position", "lexical_score"):
        assert leaky not in queue.DISPATCH_COLUMNS


def test_judgement_schema_has_no_relevance_column(tmp_path):
    """D20: our labels overlay `pools.csv`, they never become it.

    `pools.csv` is a pure function of (`data/raw/`, seed) and `verify --derived` check 6
    asserts it reproduces byte-for-byte. A human judgement is not a function of a seed.
    """
    assert "relevance" not in queue.JUDGEMENT_COLUMNS
    assert "pair_id" in queue.JUDGEMENT_COLUMNS
    assert "selection_reason" in queue.JUDGEMENT_COLUMNS


def test_recheck_strata_cover_all_three_a1_classes():
    """A recheck that samples only positives cannot detect a false-positive bias in A1's
    labelling, which is half of what A13 is about."""
    assert set(queue.RECHECK_STRATA) == {"Good Fit", "Potential Fit", "No Fit"}
    assert sum(queue.RECHECK_STRATA.values()) == 100


# --- the batch campaign (D28) ---------------------------------------------

BATCH_1 = {"batch": 1, "n_jds": 4, "per_jd": 5, "seed": 0, "keywords": None,
           "reuse_jds": False}


def test_document_seed_depends_on_the_id_not_the_position():
    """The bug D28 fixes, at its root.

    `SeedSequence(seed).spawn(n)` zipped positionally with a sorted list gives each JD a
    seed determined by how many JDs precede it. Insert one JD and every later JD is
    re-seeded, so its candidate pool is re-drawn and its `pair_id`s change — silently,
    with no exception and no row-count difference.
    """
    a = sample.document_seed("jd_zzz", 0).generate_state(4)
    b = sample.document_seed("jd_zzz", 0).generate_state(4)
    c = sample.document_seed("jd_aaa", 0).generate_state(4)
    d = sample.document_seed("jd_zzz", 1).generate_state(4)
    assert list(a) == list(b)          # stable for the same (id, seed)
    assert list(a) != list(c)          # different documents differ
    assert list(a) != list(d)          # the campaign seed still moves it


def test_the_draw_does_not_depend_on_corpus_row_order():
    """Two fixes meet here, and the second is the one that is easy to miss.

    `document_seed` removed the batch's dependence on a JD's *position*; sorting inside
    `choose_jds` removes its dependence on the order `load_split` returns rows in. Without
    the second, a publisher re-upload that shuffled the parquet would select a different 40
    JDs while every seed, count and config stayed identical — invisible drift arriving
    through the corpus rather than through the code.
    """
    jd = pd.DataFrame({"id": [f"jd_{i:02d}" for i in range(20)],
                       "exp_band": ["0-1", "2-3", "4-6", "0-1"] * 5,
                       "Primary Keyword": "Java"})
    straight = list(sample.choose_jds(jd, 8, 0)["id"])
    shuffled = list(sample.choose_jds(jd.sample(frac=1, random_state=3), 8, 0)["id"])
    assert straight == shuffled


def test_choose_jds_skips_what_an_earlier_batch_took():
    jd = pd.DataFrame({"id": [f"jd_{i}" for i in range(20)],
                       "exp_band": ["0-1", "2-3", "4-6", "0-1"] * 5,
                       "Primary Keyword": "Java"})
    first = set(sample.choose_jds(jd, 6, 0)["id"])
    second = set(sample.choose_jds(jd, 6, 0, exclude=first)["id"])
    assert not (first & second)
    assert len(second) == 6


def test_band_candidates_honours_per_jd():
    """A `reuse_jds` batch may ask for a different depth than the 5 `BANDS` describe."""
    jd = pd.Series({"id": "jd_0", "jd_text": "python sql data pipelines"})
    for per_jd in (3, 5, 7):
        out = sample.band_candidates(jd, synthetic_candidates(),
                                     np.random.SeedSequence(0), per_jd)
        assert len(out) == per_jd
        assert out.id.nunique() == per_jd


def test_add_batch_numbers_and_defaults_its_seed():
    specs = [dict(BATCH_1)]
    number = max(s["batch"] for s in specs) + 1
    assert number == 2
    # `seed=None` -> the batch number, so two batches never share a seed by accident
    assert sample.DEFAULT_CAMPAIGN[0]["batch"] == 1


def test_stratum_is_generic_only_when_no_titles_were_imposed():
    assert sample.stratum_of({"keywords": None}) == "generic"
    assert sample.stratum_of({}) == "generic"
    assert sample.stratum_of({"keywords": ["Data Science"]}) == "targeted"


def test_every_targeted_batch_pools_into_one_stratum():
    """D29. Batch is the operational unit; stratum is the reporting unit, and is coarser.

    Three keyword-scoped batches are one `targeted` figure, because they are drawn the
    same way and differ only in which titles they cover — so the pooled number describes a
    real population: the families we chose to cover.
    """
    specs = [dict(BATCH_1)] + [
        {"batch": n, "n_jds": 1, "per_jd": 5, "seed": n, "keywords": [k],
         "reuse_jds": False}
        for n, k in ((2, "Data Science"), (3, "DevOps"), (4, "QA"))]
    # Derived from the specs, not written by hand — otherwise this tests `summarise`'s
    # groupby and not the pooling decision, and a `stratum_of` that refused to pool
    # (one stratum per title list) would still pass.
    pairs = pd.DataFrame({
        "pair_id": list("abcd"), "batch": [s["batch"] for s in specs],
        "stratum": [sample.stratum_of(s) for s in specs],
        "jd_id": list("jklm"), "cv_id": list("wxyz"),
        "primary_keyword": ["Java", "Data Science", "DevOps", "QA"],
        "exp_band": ["0-1"] * 4, "lexical_band": ["high"] * 4,
        "keyword_match": [True] * 4})
    holdout = pd.DataFrame({"doc_id": ["j"], "doc_type": ["jd"], "reason": ["q"]})

    summary = sample.summarise(pairs, holdout, specs)
    assert set(summary["strata"]) == {"generic", "targeted"}
    assert summary["strata"]["targeted"]["batches"] == [2, 3, 4]
    assert summary["strata"]["targeted"]["pairs"] == 3
    assert summary["strata"]["generic"]["pairs"] == 1


def test_stratum_is_stamped_on_the_pair_not_looked_up_from_the_spec():
    """So that editing a batch's keywords later cannot re-stratify collected judgements.

    A stratum recorded only in the config is a stratum that changes retroactively the
    first time someone widens a batch's title list.
    """
    assert "stratum" in queue.JUDGEMENT_COLUMNS
    assert "batch" in queue.JUDGEMENT_COLUMNS


# --- resume (D28) ----------------------------------------------------------

def test_judged_pair_ids_is_empty_when_the_layer_is_header_only(tmp_path, monkeypatch):
    path = tmp_path / "judgements.csv"
    path.write_text(",".join(queue.JUDGEMENT_COLUMNS) + "\n")
    monkeypatch.setattr(queue, "JUDGEMENTS", path)
    assert queue.judged_pair_ids() == set()


def test_judged_pair_ids_reads_the_layer_and_nothing_else(tmp_path, monkeypatch):
    """The whole resume mechanism. No progress file means no way for two to disagree."""
    path = tmp_path / "judgements.csv"
    rows = pd.DataFrame([{c: "" for c in queue.JUDGEMENT_COLUMNS} for _ in range(2)])
    rows["pair_id"] = ["j1__c1", "j2__c2"]
    rows.to_csv(path, index=False)
    monkeypatch.setattr(queue, "JUDGEMENTS", path)
    assert queue.judged_pair_ids() == {"j1__c1", "j2__c2"}


def test_judgement_schema_carries_the_batch():
    """Without it, a figure cannot be split by stratum after the fact."""
    assert "batch" in queue.JUDGEMENT_COLUMNS


# --- on-category enrichment (D31) -----------------------------------------

def mixed_corpus(per_family: int = 30) -> pd.DataFrame:
    """CVs across three role families, so a draw can be on- or off-category."""
    vocab = {"Data Engineer": "python sql airflow pipelines warehouse etl spark",
             "JavaScript": "javascript react frontend css components node webpack",
             "QA": "testing selenium regression defects automation qa cases"}
    rows = []
    for family, words in vocab.items():
        terms = words.split()
        for i in range(per_family):
            rows.append({"id": f"{family[:2].lower()}_{i}",
                         "Primary Keyword": family,
                         "cv_text": " ".join(terms[i % len(terms):] + terms[:i % len(terms)])})
    return pd.DataFrame(rows)


def a_jd(family: str = "Data Engineer") -> pd.Series:
    return pd.Series({"id": "jd_0", "Primary Keyword": family,
                      "jd_text": "python sql airflow pipelines warehouse etl spark"})


def test_eight_of_ten_candidates_share_the_jds_role_family():
    """Mutation: draw all 10 from one undifferentiated pool, as the first cut did.

    That is not a hypothetical: the first cut ran at **5.0%** on-category and the pilot
    labelled 28 of its first 30 pairs `No Fit`. A set where almost every pair is an
    obvious negative gives no system anything to be right or wrong about, and costs the
    same human hours as one that does.
    """
    picks, pool = sample.candidates_for_jd(
        a_jd(), mixed_corpus(), np.random.SeedSequence(0),
        per_jd=sample.PER_JD, same_min=sample.SAME_KEYWORD_MIN)

    assert len(picks) == sample.PER_JD
    assert int(picks.keyword_match.sum()) == sample.SAME_KEYWORD_MIN
    assert (picks[picks.keyword_match]["Primary Keyword"] == "Data Engineer").all()
    assert not picks[~picks.keyword_match]["Primary Keyword"].eq("Data Engineer").any()
    assert picks.id.nunique() == sample.PER_JD           # no CV drawn twice
    assert set(picks.id) <= set(pool.id)                 # every pick is held out (D18)


def test_the_off_category_pair_is_a_near_miss_not_a_random_cv():
    """Mutation: give the off-category draw `("low", "low")`, or draw it uniformly.

    An off-category CV picked at random is a trivial `No Fit` that every system already
    ranks last — it separates nothing and wastes two of the ten slots. The two that are
    kept off-category are taken from the *top* of their own pool, because a lexically
    similar CV from the wrong role family is the mistake a screening system actually
    makes.
    """
    assert sample.OTHER_BANDS[0] == "high"
    picks, _ = sample.candidates_for_jd(
        a_jd(), mixed_corpus(), np.random.SeedSequence(0),
        per_jd=sample.PER_JD, same_min=sample.SAME_KEYWORD_MIN)
    off = picks[~picks.keyword_match]
    assert list(off.band) == list(sample.OTHER_BANDS)


def test_a_thin_role_family_takes_the_shortfall_off_category_and_says_so():
    """Three of the 41 families have fewer than 100 CVs in the eval region (Rust has 40),
    and a targeted batch can name one. The quota must degrade visibly, not silently."""
    thin = mixed_corpus()
    thin = pd.concat([thin[thin["Primary Keyword"] != "Data Engineer"],
                      thin[thin["Primary Keyword"] == "Data Engineer"].head(3)])

    picks, _ = sample.candidates_for_jd(
        a_jd(), thin, np.random.SeedSequence(0), per_jd=10, same_min=8)
    assert len(picks) == 10
    assert int(picks.keyword_match.sum()) == 3
    assert "keyword_match" in picks, "the shortfall has to be readable off the pair"


def test_a_batch_cannot_ask_for_more_on_category_than_it_has_candidates():
    spec = {"batch": 9, "n_jds": 1, "per_jd": 5, "seed": 9, "keywords": None,
            "same_keyword_min": 8, "reuse_jds": False}
    jd = pd.DataFrame([{"id": "jd_0", "Primary Keyword": "QA", "exp_band": "0-1",
                        "jd_text": "qa automation"}])
    with pytest.raises(ValueError, match="exceeds per_jd"):
        sample.draw_batch(spec, jd, mixed_corpus(), set(), set(), set())


def test_a_rebuild_that_orphans_a_collected_label_is_refused(tmp_path, monkeypatch):
    """Mutation: return early from `assert_labels_survive`.

    D31 re-cut batch 1 from 5 candidates per JD to 10, which changed every `pair_id` in
    it. That was safe only because `judgements.csv` was empty at the time — a precondition
    that is true when someone writes the change and false when someone repeats it later.
    """
    judgements = tmp_path / "judgements.csv"
    judgements.write_text("pair_id,annotator,label\njd_0__cv_gone,alice,No Fit\n")
    monkeypatch.setattr(queue, "JUDGEMENTS", judgements)

    survives = pd.DataFrame({"jd_id": ["jd_0"], "cv_id": ["cv_gone"]})
    sample.assert_labels_survive(survives)               # still there: no complaint

    orphans = pd.DataFrame({"jd_id": ["jd_0"], "cv_id": ["cv_other"]})
    with pytest.raises(AssertionError, match="orphans 1 collected judgement"):
        sample.assert_labels_survive(orphans)
