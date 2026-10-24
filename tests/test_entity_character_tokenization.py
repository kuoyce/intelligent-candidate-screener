import copy
import re
import pytest

from candidate_screener.entity_annotation.character_tokenization import (
    BASE_PATTERN, BoundaryTokenizer, derive_tokenizer, validate_manifest,
)
from candidate_screener.entity_annotation.finetune_data import prepare_training
from candidate_screener.entity_annotation.text import digest


def base(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(BASE_PATTERN, text)]


def row(text, spans, doc_id="one"):
    return {"doc_id": doc_id, "text": text, "source_sha256": digest(text),
            "split": "train", "reviewed_all": True, "revision": 1,
            "campaign_id": "synthetic", "categories": ["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"],
            "spans": [{"start": a, "end": b, "surface": text[a:b], "label": "KNOWLEDGE"} for a, b in spans]}


def test_original_tokenizer_fails_but_frozen_rules_preserve_exact_human_boundaries():
    text = " \tHighlightsExcel  \n😀x\r"
    sample = row(text, [(0, 2), (12, 18), (18, 20), (21, 22)])
    original = base(text)
    assert any(s["start"] not in {w[1] for w in original} or s["end"] not in {w[2] for w in original} for s in sample["spans"])
    policy = derive_tokenizer([sample], base)
    units = list(BoundaryTokenizer(policy)(text))
    assert units[0] == ("space", 0, 1) and units[1] == ("tab", 1, 2)
    newline = text.index("\n")
    assert any(unit == ("newline", newline, newline + 1) for unit in units)
    assert all(a[2] == b[1] for a, b in zip(units, units[1:]))
    assert units[0][1] == 0 and units[-1][2] == len(text)
    assert all(s["start"] in {u[1] for u in units} and s["end"] in {u[2] for u in units} for s in sample["spans"])


def test_global_rule_union_and_unseen_surfaces_no_gold_argument_at_inference():
    rows = [row("fusedword", [(0, 3)]), row("fusedword", [(5, 9)], "two")]
    policy = derive_tokenizer(rows, base)
    assert policy["cuts"] == {"fusedword": [3, 5]}
    tokenizer = BoundaryTokenizer(policy)
    assert [u[0] for u in tokenizer("fusedword otherword")] == ["fus", "ed", "word", "space", "otherword"]
    policy["cuts"]["fusedword"] = [1]
    assert [u[0] for u in tokenizer("fusedword")] == ["fus", "ed", "word"]


def test_each_unicode_whitespace_codepoint_has_source_unit():
    sample = row("x\u00a0\u2003\t\n\ry", [(1, 6)])
    units = list(BoundaryTokenizer(derive_tokenizer([sample], base))(sample["text"]))
    assert [u[0] for u in units] == ["x", "whitespace", "whitespace", "tab", "newline", "carriagereturn", "y"]
    assert all(b - a == 1 for _, a, b in units)


def test_heldout_changed_snapshot_and_tampered_policy_rejected():
    sample = row("text", [(0, 2)])
    policy = derive_tokenizer([sample], base)
    changed = dict(sample, revision=2)
    with pytest.raises(ValueError, match="snapshot changed"):
        validate_manifest(policy, rows=[changed])
    damaged = copy.deepcopy(policy)
    damaged["cuts"]["text"] = [1]
    with pytest.raises(ValueError, match="digest"):
        BoundaryTokenizer(damaged)
    with pytest.raises(ValueError, match="training"):
        derive_tokenizer([dict(sample, split="test")], base)


def test_snapshot_serialized_digest_matches_export_exactly():
    import json
    sample = row("😀 text", [(0, 1)])
    policy = derive_tokenizer([sample], base)
    assert policy["snapshot_sha256"] == digest(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")


def test_character_units_align_all_gold_once_and_keep_whitespace_negatives():
    from types import SimpleNamespace
    sample = row(" leftRight \t", [(0, 5), (5, 10), (10, 11)])
    policy = derive_tokenizer([sample], base)
    class Tokenizer:
        model_max_length = 512
        def __call__(self, prepared, **kwargs):
            return {"input_ids": [[0] * (20 + len(words)) for words in prepared]}
    processor = SimpleNamespace(words_splitter=BoundaryTokenizer(policy), transformer_tokenizer=Tokenizer(),
                                prepare_inputs=lambda words, labels: (words, None))
    model = SimpleNamespace(data_processor=processor, config=SimpleNamespace(max_len=384, max_width=policy["max_width_units"]))
    result = prepare_training([sample], model, tokenizer_manifest=policy, sentence_detector=lambda text: [(0, len(text))])
    assert not result["diagnostics"]["blocked"]
    assert result["diagnostics"]["supervised_spans"] == 3
    assert result["records"][0]["tokenized_text"] == ["space", "left", "Right", "space", "tab"]
    assert result["windows"][0]["start"] == 0 and result["windows"][-1]["end"] == len(sample["text"])
