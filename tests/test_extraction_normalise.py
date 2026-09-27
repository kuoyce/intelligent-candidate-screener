"""`extraction.normalise`: the free gold set and the string / SBERT methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candidate_screener.extraction import normalise as nz

SKILLS = pd.DataFrame([
    ("u1", "Python (computer programming)", "Python 3\npython scripting"),
    ("u2", "JavaScript", "client-side scripting\nJS"),
    ("u3", "manage databases", "database administration\nDBA\nshared label"),
    ("u4", "database design", "design databases\nshared label"),
    ("u5", "javascript", "ECMAScript"),  # normalises to u2's preferred label
], columns=["conceptUri", "preferredLabel", "altLabels"])


def test_gold_excludes_ambiguous_and_preferred_equal_labels():
    g = nz.gold_candidates(SKILLS)
    assert "shared label" not in set(g.alt_label)
    assert g.key.is_unique and len(g) == 8


def test_held_out_is_seeded():
    a, b = nz.held_out(SKILLS, n=4, seed=0), nz.held_out(SKILLS, n=4, seed=0)
    assert a.equals(b) and len(a) == 4
    assert not a.equals(nz.held_out(SKILLS, n=4, seed=3))


def test_held_out_labels_never_reach_the_index():
    q = nz.held_out(SKILLS, n=8)
    idx = nz.build_index(SKILLS, q, "aliases")
    assert not set(idx.label.map(nz.norm)) & set(q.key)
    assert len(nz.build_index(SKILLS, q, "preferred")) == 5


def test_string_cascade_uses_token_overlap_as_last_resort():
    idx = pd.DataFrame({"label": ["manage databases", "database design"],
                        "concept": ["u3", "u4"]})
    p = nz.string_predictions(["manage databases", "Manage-Databases!", "design of database",
                               "zzz"], idx)
    assert p["exact"] == ["u3", None, None, None]
    assert p["normalised"] == ["u3", "u3", None, None]
    assert p["string"] == ["u3", "u3", "u4", None]


def test_sbert_path_with_a_stub_encoder():
    def enc(texts):  # bag of letters, L2-normalised: enough to exercise nearest-neighbour
        m = np.array([[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts],
                     dtype=float)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    idx = pd.DataFrame({"label": ["javascript", "database"], "concept": ["u2", "u3"]})
    assert nz.sbert_predictions(["JavaScripts", "databases"], idx, enc) == ["u2", "u3"]
    out = nz.evaluate(SKILLS, "preferred", enc, n=6)
    assert set(out["methods"]) == set(nz.METHODS)
    assert out["methods"]["exact"]["top1"] == 0.0  # by construction


def test_accuracy_ci_brackets_the_point():
    a = nz.accuracy(np.array([1, 0, 1, 1] * 25), n_boot=2000)
    assert a["n"] == 100 and a["top1"] == pytest.approx(0.75)
    assert a["top1_ci"][0] <= a["top1"] <= a["top1_ci"][1]


def test_paired_accuracy_delta():
    a, b = np.array([1, 0, 0, 1] * 10), np.array([1, 1, 0, 1] * 10)
    d = nz.paired_accuracy_delta(a, b, n_boot=2000)
    assert d["delta"] == pytest.approx(0.25) and d["delta_ci"][0] <= 0.25 <= d["delta_ci"][1]
    assert nz.paired_accuracy_delta(a, a, n_boot=1000)["delta_ci"] == [0.0, 0.0]
