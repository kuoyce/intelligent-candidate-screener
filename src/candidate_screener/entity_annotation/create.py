"""Create a frozen entity campaign only from explicitly supplied batch choices."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candidate_screener.config import PROCESSED
from .campaign import build_manifest, save_manifest

KNOWN_REVIEWED = {"r_4d6943fc4289", "j_365a7813a56a", "r_006c891d237a", "j_3acb4cc033cc"}
ALLOWED_LABELS = {"KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS"}


def a1_documents(root: Path = PROCESSED / "fit"):
    values, memberships = {}, {}
    for split in ("train", "test"):
        frame = pd.read_parquet(root / f"{split}.parquet")
        for side, identity, field in (("cv", "resume_id", "resume_text"),
                                      ("jd", "jd_id", "job_description_text")):
            for doc_id, text in frame[[identity, field]].itertuples(index=False, name=None):
                if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str):
                    raise ValueError("A1 document identities and text must be strings")
                key = (side, doc_id)
                if key in values and values[key] != text:
                    raise ValueError("Same document ID has conflicting source text")
                values[key] = text
                memberships.setdefault(key, set()).add(split)
    return [{"source": "a1-fit", "side": side, "doc_id": identity, "text": text,
             "source_split": next(iter(memberships[(side, identity)])),
             "previously_reviewed": identity in KNOWN_REVIEWED}
            for (side, identity), text in sorted(values.items())
            if len(memberships[(side, identity)]) == 1]


def a2_documents(seed=0):
    from candidate_screener.extraction.inputs import prepare_inputs
    # This existing selector verifies training membership and never reads final labels.
    payload = prepare_inputs(sample_size=2000, seed=seed)
    documents = []
    for doc in payload["documents"]:
        fields = doc["fields"]
        names = [name for name in fields if doc["doc_type"] == "jd" or name != "Position"]
        text = "\n\n".join(f"{name}:\n{fields[name]}" for name in names if fields[name])
        documents.append({"source": "a2-train", "side": doc["doc_type"], "doc_id": doc["doc_id"],
                          "text": text, "eligible_splits": ["train", "dev", "test"]})
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("a1", "a2-train"), required=True)
    parser.add_argument("--labels", nargs=4, required=True, choices=sorted(ALLOWED_LABELS))
    parser.add_argument("--counts", type=int, nargs=3, metavar=("TRAIN", "DEV", "TEST"), required=True)
    parser.add_argument("--annotators", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if sum(args.counts) > 120 or len(set(args.labels)) != 4:
        parser.error("D56 allows at most120documents and four distinct confirmed categories")
    from .service import _annotator
    for annotator in args.annotators:
        _annotator(annotator)
    if len(args.annotators) not in {1, 2} or len(set(args.annotators)) != len(args.annotators):
        parser.error("Choose one or two distinct annotator IDs")
    documents = a1_documents() if args.source == "a1" else a2_documents(args.seed)
    counts = dict(zip(("train", "dev", "test"), args.counts))
    manifest = build_manifest(documents, labels=args.labels, counts=counts, seed=args.seed)
    # Annotator assignment is part of immutable campaign identity.
    import json
    from .text import digest
    manifest["annotators"] = args.annotators
    manifest["assignments"] = {args.annotators[0]: [d["doc_id"] for d in manifest["documents"]]}
    if len(args.annotators) == 2:
        assigned = []
        for split in counts:
            for side in ("cv", "jd"):
                candidates = sorted((d for d in manifest["documents"] if d["split"] == split and d["side"] == side),
                                    key=lambda d: digest(str(args.seed) + d["doc_id"]))
                number = max(1, round(len(candidates) * .2)) if candidates else 0
                assigned.extend(d["doc_id"] for d in candidates[:number])
        manifest["assignments"][args.annotators[1]] = assigned
    manifest.pop("campaign_id")
    manifest["campaign_id"] = digest(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")))[:24]
    path = PROCESSED / "entity-annotation/campaigns" / manifest["campaign_id"] / "manifest.json"
    save_manifest(manifest, path)
    print(f"Frozen campaign: {path}")
    print(f"Start: uv run python -m candidate_screener.entity_annotation.ui --campaign {path} --annotator {args.annotators[0]}")


if __name__ == "__main__":
    main()
