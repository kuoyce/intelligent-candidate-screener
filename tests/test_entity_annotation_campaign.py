import copy
import pytest
from candidate_screener.entity_annotation.campaign import build_manifest, public_document, save_manifest, load_manifest, validate_manifest


def documents():
    return [{"source": "synthetic", "doc_id": f"{side}{i}", "side": side,
             "text": f"{side} technology {i} person@example.com"}
            for side in ("cv", "jd") for i in range(4)]


def make(docs=None):
    return build_manifest(documents() if docs is None else docs, labels=["TECH"], counts={"train": 2, "dev": 2, "test": 2})


def test_deterministic_unique_balanced_manifest_and_private_display():
    manifest = make()
    assert manifest == make(list(reversed(documents())))
    validate_manifest(manifest)
    assert len({d["doc_id"] for d in manifest["documents"]}) == 6
    for split in ("train", "dev", "test"):
        assert sorted(d["side"] for d in manifest["documents"] if d["split"] == split) == ["cv", "jd"]
    public = public_document(manifest["documents"][0])
    assert "source_text" not in public and "boundary_map" not in public
    assert "person@example.com" not in public["text"]


def test_immutable_manifest_and_tamper(tmp_path):
    manifest = make()
    path = tmp_path / "data" / "manifest.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest
    with pytest.raises(FileExistsError):
        save_manifest(manifest, path)
    modified = copy.deepcopy(manifest)
    modified["documents"][0]["source_text"] += "tamper"
    with pytest.raises(ValueError, match="identity"):
        validate_manifest(modified)


def test_duplicate_content_cannot_leak_or_count_twice():
    docs = documents()
    for doc in docs:
        doc["text"] = "same content"
    with pytest.raises(ValueError, match="Insufficient"):
        make(docs)


def test_cross_original_split_duplicate_excluded():
    docs = documents()
    docs[0]["source_split"] = "train"
    duplicate = dict(docs[0], doc_id="duplicate", source_split="test")
    manifest = make(docs + [duplicate])
    assert not any(d["source_id"] in {docs[0]["doc_id"], "duplicate"} for d in manifest["documents"])


def test_group_members_stay_together_or_insufficient():
    docs = documents()
    for doc in docs:
        doc["group_id"] = "one-big-group"
    with pytest.raises(ValueError, match="Insufficient"):
        make(docs)


def test_conflicting_same_id_eligibility_rejected():
    docs = documents()
    with pytest.raises(ValueError, match="eligibility"):
        make(docs + [dict(docs[0], source_split="test")])


def reidentify(manifest):
    from candidate_screener.entity_annotation.campaign import _canonical
    from candidate_screener.entity_annotation.text import digest
    manifest["campaign_id"] = digest(_canonical({k: v for k, v in manifest.items() if k != "campaign_id"}))[:24]
    return manifest


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(labels=[]), "allowlist"),
    (lambda m: m["documents"][0].update(doc_id=m["documents"][1]["doc_id"]), "unique"),
    (lambda m: m["documents"][0].update(side="other"), "cv/jd"),
    (lambda m: m.update(counts={"train": 4, "dev": 2, "test": 2}), "inventory"),
    (lambda m: next(d for d in m["documents"] if d["split"] == "train").update(source_split="test"), "Original test"),
    (lambda m: next(d for d in m["documents"] if d["split"] == "test").update(previously_reviewed=True), "Previously reviewed"),
    (lambda m: m["documents"][0].update(normalized_sha256="wrong"), "Normalized"),
])
def test_semantics_rejected_even_with_recomputed_manifest_identity(change, match):
    manifest = make()
    change(manifest)
    with pytest.raises(ValueError, match=match):
        validate_manifest(reidentify(manifest))


def test_assignments_are_immutable_validated_and_registered():
    manifest = make()
    ids = [d["doc_id"] for d in manifest["documents"]]
    manifest.update(annotators=["primary", "second"], assignments={"primary": ids, "second": ids[:2]})
    validate_manifest(reidentify(manifest))
    manifest["assignments"]["primary"] = ids[:-1]
    with pytest.raises(ValueError, match="every document"):
        validate_manifest(reidentify(manifest))
    manifest["assignments"]["primary"] = ids
    manifest["assignments"]["second"] = ["outside"]
    with pytest.raises(ValueError, match="campaign document"):
        validate_manifest(reidentify(manifest))


def test_selected_source_eligibility_provenance_preserved():
    docs = documents()
    for doc in docs:
        doc.update(source_split="train", eligible_splits=["train", "dev", "test"], previously_reviewed=False)
    manifest = make(docs)
    assert all(d["source_split"] == "train" and d["eligible_splits"] == ["train", "dev", "test"] and d["previously_reviewed"] is False for d in manifest["documents"])


def test_real_source_cannot_save_outside_project_data(tmp_path):
    manifest = make()
    for doc in manifest["documents"]:
        doc["source"] = "a1"
    with pytest.raises(ValueError, match="ignored data"):
        save_manifest(reidentify(manifest), tmp_path / "data" / "manifest.json")
