"""The labelling session's rules. Inline synthetic pairs — no `data/` needed.

Each test here guards a property that has no visible symptom when it breaks: a group
that leaks the sampled band looks like a normal group, a double-label designation that
re-rolls looks like a normal queue, and a shortlist pick recorded against a `No Fit`
looks like a normal row. Per `AGENTS.md`, each was run against a mutated version of the
code it guards and shown to fail; the mutations are named in the docstrings.
"""
from __future__ import annotations

import pandas as pd
import pytest

from candidate_screener.annotation import sample, session, ui


def make_pairs(n_jds: int = 3, per_jd: int = 5, batch: int = 1,
               stratum: str = "generic") -> pd.DataFrame:
    rows = [{"pair_id": f"jd{j}__cv{j}_{c}", "jd_id": f"jd{j}", "cv_id": f"cv{j}_{c}",
             "batch": batch, "stratum": stratum, "primary_keyword": "Python",
             "lexical_band": ("high", "high", "mid", "mid", "low")[c % 5]}
            for j in range(n_jds) for c in range(per_jd)]
    return pd.DataFrame(rows)


def texts(pairs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    jd = pd.Series({j: f"job description for {j}, python engineer"
                    for j in pairs.jd_id.unique()})
    cv = pd.Series({c: f"cv of candidate {c}, five years python"
                    for c in pairs.cv_id.unique()})
    return jd, cv


def empty_judgements() -> pd.DataFrame:
    from candidate_screener.annotation.queue import JUDGEMENT_COLUMNS
    return pd.DataFrame(columns=list(JUDGEMENT_COLUMNS))


@pytest.fixture
def judgements_file(tmp_path, monkeypatch):
    path = tmp_path / "judgements.csv"
    monkeypatch.setattr(session, "JUDGEMENTS", path)
    return path


# --- double labelling ------------------------------------------------------

def test_double_label_designation_survives_a_new_batch():
    """Mutation: designate by taking the first 30% of the current pair list.

    That version passes every other test here and still destroys the kappa sample —
    appending a batch renumbers the list, so pairs already labelled once stop being
    double-labelled and pairs already finished start needing a second label.
    """
    before = make_pairs(n_jds=4)
    after = pd.concat([before, make_pairs(n_jds=4, batch=2, stratum="targeted")
                       .assign(jd_id=lambda d: d.jd_id + "_b2",
                               cv_id=lambda d: d.cv_id + "_b2",
                               pair_id=lambda d: d.jd_id + "__" + d.cv_id)])
    designated = {p for p in before.pair_id if session.needs_two_labels(p)}
    still = {p for p in after.pair_id
             if p in set(before.pair_id) and session.needs_two_labels(p)}
    assert designated == still
    assert 0 < len(designated) < len(before), "30% should be a strict subset"


def test_a_pair_is_never_served_twice_to_the_same_annotator(judgements_file):
    """Mutation: drop the `~pairs.pair_id.isin(mine)` clause in `outstanding`."""
    pairs = make_pairs(n_jds=1)
    doubles = [p for p in pairs.pair_id if session.needs_two_labels(p)]
    assert doubles, "fixture must contain at least one double-labelled pair"

    judged = pd.DataFrame({"pair_id": doubles, "annotator": "alice"})
    owed = session.outstanding(pairs, judged, "alice")
    assert not set(owed.pair_id) & set(doubles)


def test_a_double_labelled_pair_stays_open_for_the_other_annotator(judgements_file):
    """Mutation: make `wanted` always 1 — kappa then has nothing to compute on."""
    pairs = make_pairs(n_jds=1)
    doubles = [p for p in pairs.pair_id if session.needs_two_labels(p)]
    judged = pd.DataFrame({"pair_id": doubles, "annotator": "alice"})

    owed = session.outstanding(pairs, judged, "bob")
    assert set(doubles) <= set(owed.pair_id)

    twice = pd.concat([judged, judged.assign(annotator="bob")])
    assert not set(session.outstanding(pairs, twice, "carol").pair_id) & set(doubles)


# --- what reaches the screen ----------------------------------------------

def test_a_served_group_is_redacted():
    """Mutation: drop the `redact.redact` call in `serve_group`."""
    pairs = make_pairs(n_jds=1)
    jd, cv = texts(pairs)
    cv.iloc[0] = "reach me at alice@example.com or +380 50 123 4567, see https://x.io/cv"

    group = session.serve_group(pairs, empty_judgements(), "alice", jd, cv)
    blob = " ".join(c.text for c in group.candidates)
    assert "alice@example.com" not in blob and "50 123 4567" not in blob
    assert "https://x.io/cv" not in blob
    assert "[EMAIL]" in blob and "[PHONE]" in blob and "[URL]" in blob


def test_a_served_group_carries_nothing_that_anchors():
    """Mutation: add `lexical_band` to `Group.as_dict`.

    The band is which similarity third the sampler drew the candidate from. An annotator
    who saw it would be told the sampler's own guess before making theirs, and the
    in-domain agreement figure would be measuring the sampler, not the scheme.
    """
    pairs = make_pairs(n_jds=1)
    jd, cv = texts(pairs)
    served = session.serve_group(pairs, empty_judgements(), "alice", jd, cv).as_dict()

    assert set(served) == {"jd_id", "batch", "stratum", "jd_text", "candidates",
                           "complete"}
    for candidate in served["candidates"]:
        assert set(candidate) == {"cv_id", "text"}
    assert "lexical" not in str(served) and "selection_reason" not in str(served)


def test_a_group_is_one_jd_and_is_finished_before_the_next_is_started():
    """Mutation: serve `owed.head(5)` instead of grouping by `jd_id`.

    The top-1 shortlist question is asked once per JD over a complete set of candidates.
    A screen mixing two JDs cannot ask it, and `queue.progress` would report every JD as
    partial for the whole session.
    """
    # 3 candidates per JD, not 5, so that "the next 5 owed pairs" would necessarily
    # straddle two JDs. At 5-per-JD the mutation and the correct code agree on the
    # first screen and the test passes against both, which is no test at all.
    pairs = make_pairs(n_jds=3, per_jd=3)
    jd, cv = texts(pairs)
    group = session.serve_group(pairs, empty_judgements(), "alice", jd, cv)

    assert len(group.candidates) == 3
    served = {c.cv_id for c in group.candidates}
    assert served == set(pairs[pairs.jd_id == group.jd_id].cv_id)
    assert not served & set(pairs[pairs.jd_id != group.jd_id].cv_id)


def test_a_document_with_no_text_is_refused():
    pairs = make_pairs(n_jds=1)
    jd, cv = texts(pairs)
    cv.iloc[2] = ""
    with pytest.raises(AssertionError, match="cannot read"):
        session.serve_group(pairs, empty_judgements(), "alice", jd, cv)


# --- recording -------------------------------------------------------------

def label_all(group, value="No Fit"):
    return {c.cv_id: value for c in group.candidates}


def a_group(pairs=None):
    pairs = make_pairs(n_jds=1) if pairs is None else pairs
    jd, cv = texts(pairs)
    return session.serve_group(pairs, empty_judgements(), "alice", jd, cv)


def test_record_appends_and_never_rewrites(judgements_file):
    """Mutation: open `judgements.csv` with mode 'w'."""
    group = a_group()
    session.record("alice", group, label_all(group))
    session.record("bob", group, label_all(group))

    rows = pd.read_csv(judgements_file)
    assert len(rows) == 10
    assert set(rows.annotator) == {"alice", "bob"}
    assert set(rows.columns) == set(session.JUDGEMENT_COLUMNS)
    assert set(rows.stratum) == {"generic"} and set(rows.corpus) == {"a2"}


def test_record_refuses_a_shortlist_pick_that_was_not_relevant(judgements_file):
    """Mutation: drop the eligibility check. The pick then breaks no tie in the
    relevant set, which is the only thing it exists to do."""
    group = a_group()
    labels = label_all(group)
    labels[group.candidates[0].cv_id] = "Good Fit"
    with pytest.raises(ValueError, match="not among the candidates"):
        session.record("alice", group, labels,
                       shortlist_pick=group.candidates[1].cv_id)


def test_record_refuses_none_when_something_was_relevant(judgements_file):
    group = a_group()
    labels = label_all(group)
    labels[group.candidates[0].cv_id] = "Potential Fit"
    with pytest.raises(ValueError, match="not an answer"):
        session.record("alice", group, labels, shortlist_pick="none")


def test_record_accepts_none_when_nothing_was_relevant(judgements_file):
    group = a_group()
    assert session.record("alice", group, label_all(group), "none") == 5
    assert set(pd.read_csv(judgements_file).shortlist_pick) == {0}


def test_record_refuses_a_partial_group(judgements_file):
    group = a_group()
    labels = label_all(group)
    labels.pop(group.candidates[0].cv_id)
    with pytest.raises(ValueError, match="a label for each"):
        session.record("alice", group, labels)


def test_record_refuses_a_label_outside_a1s_scheme(judgements_file):
    group = a_group()
    labels = label_all(group)
    labels[group.candidates[0].cv_id] = "Maybe"
    with pytest.raises(ValueError, match="3-class scheme"):
        session.record("alice", group, labels)


def test_record_refuses_an_anonymous_judgement(judgements_file):
    group = a_group()
    with pytest.raises(ValueError, match="annotator is required"):
        session.record("  ", group, label_all(group))


def test_recorded_pairs_stop_being_served(judgements_file):
    pairs = make_pairs(n_jds=2)
    jd, cv = texts(pairs)
    first = session.serve_group(pairs, session.load_judgements(), "alice", jd, cv)
    session.record("alice", first, label_all(first))

    second = session.serve_group(pairs, session.load_judgements(), "alice", jd, cv)
    assert second.jd_id != first.jd_id


def test_the_second_pass_over_a_double_labelled_jd_is_partial(judgements_file):
    """Mutation: hard-code `complete=True` in `serve_group`.

    The second annotator on a double-labelled JD sees only the shared subset — two of
    five, say. Marked complete, they would be asked which of *those* they would
    shortlist first, and the answer would be indistinguishable in `judgements.csv` from
    a pick made over the whole field.
    """
    pairs = make_pairs(n_jds=1)
    jd, cv = texts(pairs)

    first = session.serve_group(pairs, session.load_judgements(), "alice", jd, cv)
    assert first.complete and len(first.candidates) == 5
    session.record("alice", first, label_all(first))

    second = session.serve_group(pairs, session.load_judgements(), "bob", jd, cv)
    assert second is not None, "the double-labelled subset must still be servable"
    assert not second.complete
    assert 0 < len(second.candidates) < 5
    assert all(session.needs_two_labels(f"{second.jd_id}__{c.cv_id}")
               for c in second.candidates)


def test_a_partial_group_may_not_carry_a_shortlist_pick(judgements_file):
    """Mutation: drop the `not group.complete` guard in `record`."""
    pairs = make_pairs(n_jds=1)
    jd, cv = texts(pairs)
    first = session.serve_group(pairs, session.load_judgements(), "alice", jd, cv)
    session.record("alice", first, label_all(first))
    second = session.serve_group(pairs, session.load_judgements(), "bob", jd, cv)

    labels = label_all(second)
    labels[second.candidates[0].cv_id] = "Good Fit"
    with pytest.raises(ValueError, match="partial re-serve"):
        session.record("bob", second, labels,
                       shortlist_pick=second.candidates[0].cv_id)

    # ...and 'none' is accepted even though a candidate was relevant, which on a
    # complete group would be refused.
    assert session.record("bob", second, labels, "none") == len(second.candidates)


# --- creating a batch ------------------------------------------------------

def test_an_unknown_job_title_is_refused_rather_than_drawing_nothing():
    """Mutation: return early from `validate_keywords`.

    Without it a typo scopes the batch to zero JDs, `indomain-batches.json` gains a batch
    that produces no pairs, and nothing anywhere says so.
    """
    known = ["Python", "DevOps", "Data Science"]
    sample.validate_keywords(["Python", "DevOps"], known)
    sample.validate_keywords(None, known)
    with pytest.raises(ValueError, match="unknown Primary Keyword"):
        sample.validate_keywords(["Python", "Data science"], known)


def test_selecting_every_title_is_a_generic_batch_not_a_targeted_one(monkeypatch):
    """Selecting all titles and selecting none are the same batch: unscoped.

    Passing all 41 through as `keywords` would stamp `targeted` on the pairs and put them
    in the wrong reporting stratum permanently (D29) — `stratum` is written onto the pair
    and never recomputed.
    """
    known = [{"title": t, "jds": 1} for t in ("Python", "DevOps", "QA")]
    monkeypatch.setattr(ui, "titles", lambda: known)
    captured = {}
    monkeypatch.setattr(sample, "add_batch",
                        lambda **kw: captured.update(kw) or {"summary": {}})

    ui.create_batch(["Python", "DevOps", "QA"], 10, 5)
    assert captured["keywords"] is None
    assert sample.stratum_of({"keywords": captured["keywords"]}) == "generic"

    ui.create_batch([], 10, 5)
    assert captured["keywords"] is None

    ui.create_batch(["DevOps"], 10, 5)
    assert captured["keywords"] == ["DevOps"]
    assert sample.stratum_of({"keywords": captured["keywords"]}) == "targeted"


def test_a_new_batch_cannot_be_absurdly_large(monkeypatch):
    monkeypatch.setattr(ui, "titles", lambda: [{"title": "Python", "jds": 1}])
    with pytest.raises(ValueError, match="n_jds must be"):
        ui.create_batch([], 5000, 5)


# --- the page --------------------------------------------------------------

def test_the_page_loads_nothing_from_the_network():
    """The page renders real CVs. It must work with no internet and must not be able
    to send a document anywhere — so no CDN, no font host, no external form action."""
    html = ui.PAGE.read_text(encoding="utf-8")
    for marker in ("http://", "https://", "//cdn", "<form"):
        assert marker not in html, f"page references {marker!r}"


# --- what `verify --derived` catches after the session ---------------------

def collected(rows: list[dict]) -> dict[str, bool]:
    from candidate_screener.data.verify_derived import check_collected_labels
    from candidate_screener.annotation.queue import JUDGEMENT_COLUMNS
    frame = pd.DataFrame(rows, columns=list(JUDGEMENT_COLUMNS))
    return {message: ok for ok, message in check_collected_labels(frame)}


def judgement(pair_id, **kw) -> dict:
    jd_id, cv_id = pair_id.split("__")
    row = {"pair_id": pair_id, "batch": 1, "stratum": "generic", "corpus": "a2",
           "query_id": jd_id, "doc_id": cv_id, "selection_reason": "indomain_banded",
           "annotator": "alice", "label": "No Fit", "shortlist_pick": 0, "notes": ""}
    return row | kw


@pytest.fixture
def real_pair(monkeypatch):
    """One `pair_id` that exists in the committed manifest, without loading a corpus."""
    pairs = make_pairs(n_jds=2)
    monkeypatch.setattr(session, "load_pairs", lambda: pairs)
    return pairs


def failures(report: dict) -> str:
    return " | ".join(m for m, ok in report.items() if not ok)


def test_verify_accepts_a_clean_session(real_pair):
    report = collected([judgement("jd0__cv0_0", shortlist_pick=1),
                        judgement("jd0__cv0_1"),
                        judgement("jd0__cv0_1", annotator="bob")])
    assert all(report.values()), failures(report)


def test_verify_catches_a_stratum_that_drifted_from_the_manifest(real_pair):
    """The D29 failure mode: `stratum` is stamped at labelling time, so a stale value
    files a judgement under a population it was not drawn from. Every figure in the
    report is quoted per stratum, so this is not cosmetic."""
    report = collected([judgement("jd0__cv0_0", stratum="targeted", shortlist_pick=1)])
    assert not report["collected labels: 1 judgement(s) whose `stratum` disagrees with "
                      "the manifest (D29 — it is stamped, never recomputed)"]


def test_verify_catches_a_label_outside_a1s_scheme(real_pair):
    report = collected([judgement("jd0__cv0_0", label="Maybe")])
    assert not report["collected labels: ['Maybe'] labels outside A1's 3-class scheme "
                      "['Good Fit', 'Potential Fit', 'No Fit'] (D17)"]


def test_verify_catches_one_person_labelling_the_same_pair_twice(real_pair):
    report = collected([judgement("jd0__cv0_0"), judgement("jd0__cv0_0")])
    assert not report["collected labels: 1 pair(s) labelled twice by the same annotator "
                      "— an overlap of one person with themselves is not an agreement "
                      "measurement"]


def test_verify_catches_two_shortlist_picks_for_one_jd(real_pair):
    report = collected([judgement("jd0__cv0_0", shortlist_pick=1),
                        judgement("jd0__cv0_1", shortlist_pick=1)])
    assert not report["collected labels: 1 JD(s) where one annotator recorded more than "
                      "one top-1 shortlist pick (Q14 asks for exactly one)"]


def test_verify_catches_a_pair_that_names_no_row_in_the_manifest(real_pair):
    report = collected([judgement("jd9__cv9_9")])
    assert not report["collected labels: 1 in-domain pair(s) name no row in "
                      "indomain-pairs.csv"]


def test_two_annotators_do_not_start_on_the_same_job_description():
    """Mutation: drop the `_seat` sort and take `owed.jd_id.iloc[0]`.

    Both annotators then open the same JD and label it in parallel. Nothing is corrupted
    — `record` appends and `verify --derived` catches a pair labelled twice by the *same*
    person — but the session is budgeted at ~250 judgements and duplicated screens come
    straight out of that.
    """
    pairs = make_pairs(n_jds=12)
    jd, cv = texts(pairs)
    empty = empty_judgements()
    starts = {who: session.serve_group(pairs, empty, who, jd, cv).jd_id
              for who in ("alice", "bob")}
    assert starts["alice"] != starts["bob"]


def test_each_annotator_still_reaches_every_job_description(judgements_file):
    """The per-annotator order must be a permutation, not a filter — the second
    annotator has to arrive at the double-labelled pairs on JDs the first has finished."""
    pairs = make_pairs(n_jds=6)
    jd, cv = texts(pairs)
    for _ in range(6):
        group = session.serve_group(pairs, session.load_judgements(), "alice", jd, cv)
        session.record("alice", group, label_all(group))

    seen, guard = set(), 0
    while (group := session.serve_group(
            pairs, session.load_judgements(), "bob", jd, cv)) and guard < 20:
        seen.add(group.jd_id)
        session.record("bob", group, label_all(group))
        guard += 1
    shared = set(pairs[pairs.pair_id.map(session.needs_two_labels)].jd_id)
    assert shared, "fixture must contain double-labelled pairs"
    assert seen == shared, "bob must reach every JD carrying a shared pair, and no other"
