"""`extraction.skillspan`: BIO decoding, the token grid and the union training target."""
from __future__ import annotations

from candidate_screener.extraction import skillspan as ss


def test_bio_spans_including_an_orphan_inside_tag():
    assert ss.bio_spans(["B", "I", "O", "I", "I", "B", "B"]) == [(0, 2), (3, 5), (5, 6), (6, 7)]
    assert ss.bio_spans([]) == []


def test_join_tokens_offsets_index_the_joined_text():
    text, offs = ss.join_tokens(["Knows", "C++", "."])
    assert text == "Knows C++ ." and [text[s:e] for s, e in offs] == ["Knows", "C++", "."]


def test_snap_expands_to_tokens_and_merges_overlaps():
    text, offs = ss.join_tokens(["Uses", "C#/.NET", "daily"])
    c = text.index("C#")
    # C# and .NET are two char spans inside one token: one token span.
    assert ss.snap([(c, c + 2), (c + 3, c + 7)], offs) == {(1, 2)}
    # Adjacent tokens stay separate spans.
    assert ss.snap([(0, 4), (c, c + 2)], offs) == {(0, 1), (1, 2)}


def test_gold_union_keeps_both_layers():
    row = {"tokens": list("abcd"), "tags_skill": ["B", "I", "I", "O"],
           "tags_knowledge": ["O", "B", "O", "B"]}
    assert ss.gold_spans(row, "union") == {(0, 3), (1, 2), (3, 4)}
    assert ss.gold_spans(row, "knowledge") == {(1, 2), (3, 4)}


def test_training_tags_resolve_a_cross_layer_overlap_to_the_longer_span():
    row = {"tokens": list("abcd"), "tags_skill": ["B", "I", "I", "O"],
           "tags_knowledge": ["O", "B", "O", "B"]}
    assert ss.training_tags(row) == ["B", "I", "I", "B"]
