"""Frozen train-derived lexical cuts and complete raw-offset whitespace units.

The runtime tokenizer accepts text only. Its private vocabulary is learned only
from reviewed train boundaries; no source text or human spans are consulted at
inference. Unknown lexical surfaces keep the pinned base split policy.
"""
from __future__ import annotations
import hashlib
import json
import re

VERSION = "train-boundary-lexicon-v1"
POLICY = "global-exact-surface-cut-union; one-unit-per-whitespace-codepoint"
BASE_PATTERN = r"\w+(?:[-_]\w+)*|\S"
LEXICAL = re.compile(BASE_PATTERN)
SURROGATES = {" ": "space", "\t": "tab", "\n": "newline", "\r": "carriagereturn"}


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_fingerprint(rows: list[dict]) -> str:
    return _digest(_canonical(sorted(rows, key=lambda row: row["doc_id"])))


def validate_manifest(manifest: dict, *, rows: list[dict] | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Tokenizer manifest must be an object")
    payload = {key: value for key, value in manifest.items() if key != "tokenizer_sha256"}
    if _digest(_canonical(payload)) != manifest.get("tokenizer_sha256"):
        raise ValueError("Tokenizer manifest digest mismatch")
    if (manifest.get("version") != VERSION or manifest.get("policy") != POLICY or manifest.get("base_pattern") != BASE_PATTERN or
            manifest.get("surrogates") != SURROGATES or manifest.get("other_whitespace") != "whitespace"):
        raise ValueError("Unsupported frozen tokenizer policy")
    cuts = manifest.get("cuts")
    if not isinstance(cuts, dict):
        raise ValueError("Tokenizer cuts must be a lexical vocabulary")
    for surface, positions in cuts.items():
        if (not isinstance(surface, str) or not surface or LEXICAL.fullmatch(surface) is None or
                not isinstance(positions, list) or not positions or
                any(type(p) is not int or not 0 < p < len(surface) for p in positions) or
                positions != sorted(set(positions))):
            raise ValueError("Invalid lexical cut rule")
    if (type(manifest.get("max_width_units")) is not int or manifest["max_width_units"] < 1 or
            type(manifest.get("max_gold_units")) is not int or manifest["max_gold_units"] < 0 or
            manifest["max_width_units"] != max(1, manifest["max_gold_units"])):
        raise ValueError("Invalid trained tokenizer span width")
    sources = manifest.get("training_sources")
    if (not isinstance(sources, list) or not sources or
            any(not isinstance(s, dict) or not isinstance(s.get("doc_id"), str) or not s["doc_id"] for s in sources) or
            len({s.get("doc_id") for s in sources}) != len(sources)):
        raise ValueError("Tokenizer requires unique training source provenance")
    for source in sources:
        if (type(source.get("revision")) is not int or source["revision"] < 1 or
                not isinstance(source.get("campaign_id"), str) or not source["campaign_id"] or
                any(not isinstance(source.get(key), str) or re.fullmatch(r"[a-f0-9]{64}", source[key]) is None
                    for key in ("source_sha256", "spans_sha256"))):
            raise ValueError("Invalid tokenizer training source provenance")
    if any(not isinstance(manifest.get(key), str) or re.fullmatch(r"[a-f0-9]{64}", manifest[key]) is None
           for key in ("snapshot_sha256", "snapshot_fingerprint")):
        raise ValueError("Invalid tokenizer snapshot identity")
    if rows is not None and snapshot_fingerprint(rows) != manifest.get("snapshot_fingerprint"):
        raise ValueError("Tokenizer training snapshot changed")


class BoundaryTokenizer:
    def __init__(self, manifest: dict):
        validate_manifest(manifest)
        # Copy the small vocabulary: callers cannot change frozen behavior by
        # mutating the dictionary passed to this instance after validation.
        self.cuts = {surface: tuple(positions) for surface, positions in manifest["cuts"].items()}
        self.surrogates = dict(manifest["surrogates"])
        self.other_whitespace = manifest["other_whitespace"]
        self.tokenizer_sha256 = manifest["tokenizer_sha256"]

    def __call__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Tokenizer input must be original source text")
        cursor = 0
        for match in LEXICAL.finditer(text):
            for i in range(cursor, match.start()):
                if not text[i].isspace():
                    raise ValueError("Base lexical policy left a nonwhitespace source gap")
                yield self.surrogates.get(text[i], self.other_whitespace), i, i + 1
            surface = match.group()
            offsets = (0, *self.cuts.get(surface, ()), len(surface))
            for left, right in zip(offsets, offsets[1:]):
                yield surface[left:right], match.start() + left, match.start() + right
            cursor = match.end()
        for i in range(cursor, len(text)):
            if not text[i].isspace():
                raise ValueError("Base lexical policy left a nonwhitespace source gap")
            yield self.surrogates.get(text[i], self.other_whitespace), i, i + 1


def derive_tokenizer(rows: list[dict], base_splitter) -> dict:
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Tokenizer needs reviewed train documents")
    if len({row["doc_id"] for row in rows}) != len(rows):
        raise ValueError("Tokenizer training document identities must be unique")
    cuts, sources = {}, []
    for row in rows:
        if row.get("split") != "train" or row.get("reviewed_all") is not True:
            raise ValueError("Tokenizer derivation is restricted to reviewed training documents")
        text = row["text"]
        if not isinstance(text, str) or _digest(text) != row.get("source_sha256"):
            raise ValueError("Tokenizer source digest mismatch")
        words = list(base_splitter(text))
        expected = [(m.group(), m.start(), m.end()) for m in LEXICAL.finditer(text)]
        if [tuple(w) for w in words] != expected:
            raise ValueError("Base splitter differs from the pinned lexical policy")
        boundaries = set()
        for span in row["spans"]:
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or
                    span.get("surface") != text[start:end]):
                raise ValueError("Tokenizer received invalid original gold offsets/surface")
            boundaries.update((start, end))
        for surface, start, end in words:
            internal = {point - start for point in boundaries if start < point < end}
            if internal:
                cuts.setdefault(surface, set()).update(internal)
        sources.append({"doc_id": row["doc_id"], "source_sha256": row["source_sha256"],
                        "revision": row.get("revision"), "campaign_id": row.get("campaign_id"),
                        "spans_sha256": _digest(_canonical(row["spans"]))})
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    manifest = {"version": VERSION, "base_pattern": BASE_PATTERN,
                "policy": POLICY,
                "surrogates": dict(SURROGATES), "other_whitespace": "whitespace",
                "cuts": {surface: sorted(points) for surface, points in sorted(cuts.items())},
                "training_sources": sorted(sources, key=lambda source: source["doc_id"]),
                "snapshot_sha256": _digest(encoded), "snapshot_fingerprint": snapshot_fingerprint(rows),
                "max_gold_units": 0, "max_width_units": 1}
    manifest["tokenizer_sha256"] = _digest(_canonical(manifest))
    tokenizer = BoundaryTokenizer(manifest)
    maximum = 0
    for row in rows:
        units = list(tokenizer(row["text"]))
        starts = {unit[1]: i for i, unit in enumerate(units)}
        ends = {unit[2]: i + 1 for i, unit in enumerate(units)}
        for span in row["spans"]:
            if span["start"] not in starts or span["end"] not in ends:
                raise ValueError("Derived tokenizer cannot represent a human gold boundary")
            maximum = max(maximum, ends[span["end"]] - starts[span["start"]])
    manifest.update(max_gold_units=maximum, max_width_units=max(1, maximum))
    manifest.pop("tokenizer_sha256")
    manifest["tokenizer_sha256"] = _digest(_canonical(manifest))
    validate_manifest(manifest, rows=rows)
    return manifest
