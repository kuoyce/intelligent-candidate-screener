"""Synthetic-only annotation demo; never selects real corpus documents."""
from candidate_screener.config import PROCESSED
from .campaign import build_manifest, load_manifest, save_manifest


def create_demo():
    documents = []
    for side in ("cv", "jd"):
        for index in range(4):
            text = (f"Synthetic {side.upper()} example {index + 1}.\n"
                    "Python developer 😀 with project management skills.\n"
                    "Bachelor of Science in Computer Science.\n"
                    f"Contact example{index}@example.org.\n"
                    + ("Built PostgreSQL applications." if side == "cv" else "Seeking a Software Engineer with SQL knowledge."))
            documents.append({"source": "synthetic-entity-demo", "side": side,
                              "doc_id": f"demo-{side}-{index}", "text": text})
    manifest = build_manifest(documents, labels=["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"],
                              counts={"train": 4, "dev": 2, "test": 2})
    path = PROCESSED / "entity-annotation/demo" / manifest["campaign_id"] / "manifest.json"
    if path.exists():
        if load_manifest(path) != manifest:
            raise ValueError("Synthetic demo manifest differs")
    else:
        save_manifest(manifest, path)
    return path
