"""The annotation instrument's guards *(decision D25)*.

Everything here protects one session of ~250 human judgements that cannot be re-run
cheaply. Three failures would each waste it silently, and none of them raises on its own:

- a pair reaching an annotator with PII still in it (`AGENTS.md` data rule 4);
- the dispatch file carrying `selection_reason`, which tells the annotator which pairs A1
  has already labelled and lets them anchor on the expected answer — that would void the
  A1 recheck, the only test of assumption **A13**;
- the candidate draw being uniform rather than banded, which returns ~5 `No Fit` per JD
  and makes P@5 identically zero for every system.

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
                                    np.random.SeedSequence(0))
    assert list(banded.band) == list(sample.BANDS)
    assert banded.id.nunique() == len(sample.BANDS)      # no CV drawn twice
    assert banded[banded.band == "high"].lexical_score.min() >= \
        banded[banded.band == "low"].lexical_score.max()


def test_banding_is_deterministic_at_a_fixed_seed():
    jd = pd.Series({"id": "jd_0", "jd_text": "python sql data pipelines"})
    first = sample.band_candidates(jd, synthetic_candidates(), np.random.SeedSequence(0))
    second = sample.band_candidates(jd, synthetic_candidates(), np.random.SeedSequence(0))
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
    assert sum(queue.RECHECK_STRATA.values()) == 50
