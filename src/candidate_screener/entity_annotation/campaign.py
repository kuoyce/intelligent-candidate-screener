"""Deterministic document-only entity campaign manifests; no corpus defaults."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from candidate_screener.config import PROJECT_ROOT
from .text import digest, prepare_text


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_hash(text: str) -> str:
    return digest(" ".join(text.casefold().split()))


def public_document(document: dict) -> dict:
    """An explicit allowlist prevents original text/provenance reaching the UI."""
    return {"doc_id": document["doc_id"], "side": document["side"],
            "split": document["split"], "text": document["display_text"],
            "display_sha256": document["display_sha256"],
            "redaction_version": document["redaction_version"]}


def build_manifest(documents: list[dict], *, labels: list[str], counts: dict | None = None,
                   seed: int = 0, guideline_version: str = "entity-guide-v1") -> dict:
    counts = {"train": 80, "dev": 20, "test": 20} if counts is None else dict(counts)
    if not labels or len(labels) != len(set(labels)) or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("An explicit unique category allowlist is required")
    if set(counts) != {"train", "dev", "test"} or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 or n % 2 for n in counts.values()):
        raise ValueError("Split counts must be nonnegative even integers for balanced CV/JD allocation")
    unique = {}
    for doc in documents:
        if doc["side"] not in {"cv", "jd"} or not isinstance(doc["text"], str):
            raise ValueError("Documents require cv/jd side and original string text")
        key = (str(doc["source"]), doc["side"], str(doc["doc_id"]))
        if key in unique and unique[key]["text"] != doc["text"]:
            raise ValueError("Document identity has conflicting source text")
        if key in unique and any(unique[key].get(k) != doc.get(k) for k in
                ("source_split", "eligible_splits", "group_id", "previously_reviewed")):
            raise ValueError("Document identity has conflicting eligibility metadata")
        unique[key] = dict(doc)
    # Union shared document groups and normalized text across every supplied row.
    items = sorted(unique.items())
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, (key, doc) in enumerate(items):
        aliases = [("text", normalized_hash(doc["text"]))]
        if doc.get("group_id") is not None:
            aliases.append(("group", key[0], str(doc["group_id"])))
        for alias in aliases:
            if alias in seen:
                parent[root(i)] = root(seen[alias])
            seen[alias] = i
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(root(i), []).append(item)
    candidates = []
    for members in groups.values():
        # Duplicate source content counts once. Distinct paired group documents
        # remain together. Cross-original-split overlap is excluded altogether.
        if len({d.get("source_split") for _, d in members} - {None}) > 1:
            continue
        dedup = {}
        for key, doc in members:
            dedup.setdefault((doc["side"], normalized_hash(doc["text"])), (key, doc))
        members = list(dedup.values())
        group_hash = digest(_canonical([list(key) + [digest(doc["text"])] for key, doc in members]))
        candidates.append((digest(str(seed) + group_hash), group_hash, members))
    candidates.sort()
    selected, used = [], set()
    for split in ("test", "dev", "train"):
        remaining = {side: counts[split] // 2 for side in ("cv", "jd")}
        for _, group_hash, members in candidates:
            if group_hash in used:
                continue
            if any(split not in doc.get("eligible_splits", ["test"] if doc.get("source_split") == "test" else ["train", "dev"] if doc.get("source_split") == "train" else ["train", "dev", "test"]) for _, doc in members):
                continue
            if split == "test" and any(doc.get("previously_reviewed", False) for _, doc in members):
                continue
            needed = {side: sum(doc["side"] == side for _, doc in members) for side in remaining}
            if any(needed[s] > remaining[s] for s in remaining):
                continue
            used.add(group_hash)
            for key, doc in members:
                prepared = prepare_text(doc["text"])
                selected.append({"doc_id": digest(_canonical(key))[:24], "source": key[0],
                    "source_id": key[2], "side": doc["side"], "split": split,
                    "group_sha256": group_hash, "normalized_sha256": normalized_hash(doc["text"]),
                    "source_split": doc.get("source_split"),
                    "eligible_splits": list(doc.get("eligible_splits", ["test"] if doc.get("source_split") == "test" else ["train", "dev"] if doc.get("source_split") == "train" else ["train", "dev", "test"])),
                    "previously_reviewed": doc.get("previously_reviewed", False),
                    "source_text": doc["text"], **prepared})
            for side in remaining:
                remaining[side] -= needed[side]
        if any(remaining.values()):
            raise ValueError(f"Insufficient eligible unique groups for {split}: remaining {remaining}")
    payload = {"schema_version": 1, "selection_policy": "balanced-grouped-document-v1",
               "seed": seed, "labels": list(labels), "counts": counts,
               "guideline_version": guideline_version, "documents": selected}
    payload["campaign_id"] = digest(_canonical(payload))[:24]
    validate_manifest(payload)
    return payload


def validate_manifest(manifest: dict) -> None:
    payload = {k: v for k, v in manifest.items() if k != "campaign_id"}
    if digest(_canonical(payload))[:24] != manifest["campaign_id"]:
        raise ValueError("Manifest identity mismatch")
    labels = manifest.get("labels")
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("Manifest requires a unique nonempty category allowlist")
    counts = manifest.get("counts", {})
    if (set(counts) != {"train", "dev", "test"} or
            any(type(n) is not int or n < 0 or n % 2 for n in counts.values())):
        raise ValueError("Manifest split counts must be balanced nonnegative even integers")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Manifest documents must be a list")
    ids = [doc.get("doc_id") for doc in documents]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest document IDs must be unique nonempty strings")
    groups, texts = {}, {}
    actual = {(split, side): 0 for split in counts for side in ("cv", "jd")}
    for doc in documents:
        split, side = doc.get("split"), doc.get("side")
        if (split, side) not in actual:
            raise ValueError("Manifest requires known split and cv/jd side")
        actual[split, side] += 1
        if doc.get("source_split") == "test" and split in {"train", "dev"}:
            raise ValueError("Original test documents cannot enter entity train/dev")
        if type(doc.get("previously_reviewed", False)) is not bool:
            raise ValueError("Previously reviewed flag must be boolean")
        if split == "test" and doc.get("previously_reviewed", False):
            raise ValueError("Previously reviewed documents cannot enter blind test")
        eligible = doc.get("eligible_splits", ["train", "dev", "test"])
        if not isinstance(eligible, list) or split not in eligible or any(s not in counts for s in eligible):
            raise ValueError("Selected split is outside source eligibility")
        prepared = prepare_text(doc["source_text"])
        if any(doc.get(k) != v for k, v in prepared.items()):
            raise ValueError("Source/display provenance mismatch")
        if doc.get("normalized_sha256") != normalized_hash(doc["source_text"]):
            raise ValueError("Normalized source digest mismatch")
        for registry, key in [(groups, doc["group_sha256"]), (texts, normalized_hash(doc["source_text"]))]:
            if key in registry and registry[key] != doc["split"]:
                raise ValueError("Cross-split document/group leakage")
            registry[key] = doc["split"]
    if any(actual[split, side] != counts[split] // 2 for split, side in actual):
        raise ValueError("Manifest split/side inventory differs from frozen counts")
    if "annotators" in manifest or "assignments" in manifest:
        annotators, assignments = manifest.get("annotators"), manifest.get("assignments")
        if (not isinstance(annotators, list) or not 1 <= len(annotators) <= 2 or
                any(not isinstance(a, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", a) for a in annotators) or
                len(set(annotators)) != len(annotators)):
            raise ValueError("Register one or two unique annotator IDs")
        if not isinstance(assignments, dict) or set(assignments) != set(annotators):
            raise ValueError("Assignments must match registered annotators")
        for assigned in assignments.values():
            if (not isinstance(assigned, list) or any(not isinstance(i, str) for i in assigned) or
                    len(set(assigned)) != len(assigned) or not set(assigned).issubset(ids)):
                raise ValueError("Assignments require unique campaign document IDs")
        if set(assignments[annotators[0]]) != set(ids):
            raise ValueError("First annotator must receive every document")


def save_manifest(manifest: dict, path: str | Path) -> Path:
    """Atomic exclusive creation; callers select an ignored local data path."""
    validate_manifest(manifest)
    path = Path(path)
    resolved = path.resolve()
    synthetic = all(doc["source"] in {"synthetic", "synthetic-entity-demo"} for doc in manifest["documents"])
    data_root = (PROJECT_ROOT / "data").resolve()
    if not resolved.is_relative_to(data_root) and not (synthetic and "data" in resolved.parts):
        raise ValueError("Source manifests must remain under ignored data/")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest
