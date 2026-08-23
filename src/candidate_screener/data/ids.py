"""Content-addressed document identity.

A1's parquet files carry only `resume_text`, `job_description_text`, `label` — no
publisher ID (verified). Every downstream manifest needs stable document identity,
so it is minted from the text itself:

    doc_id = "r_" | "j_" + sha256(normalise(text)).hexdigest()[:12]
    normalise(t)      = re.sub(r"\\s+", " ", t).strip()

Content addressing is what makes the manifests reproducible across a re-download,
makes duplicate detection free (the 643/351 unique-document counts fall straight
out of the ID set), and keeps document text out of the committed manifests.

See `plan/2026-08-23-derived-artefacts/01-design-spec.md` §2.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

_WS = re.compile(r"\s+")

#: Prefix per document type. The prefix is cosmetic — it makes a manifest row
#: readable without a join — but it is part of the ID and must not be changed.
PREFIX = {"resume": "r_", "jd": "j_"}

#: 12 hex characters = 48 bits. Over ~10^3 documents the birthday collision
#: probability is ~10^-9; `mint_ids` asserts zero collisions regardless.
DIGEST_CHARS = 12


def normalise(text: object) -> str:
    """Collapse all whitespace runs to a single space and strip.

    Deliberately conservative: case, punctuation and word order are preserved, so
    two IDs are equal only for documents that differ purely in layout whitespace.
    """
    return _WS.sub(" ", str(text)).strip()


def doc_id(text: object, doc_type: str) -> str:
    digest = hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()
    return PREFIX[doc_type] + digest[:DIGEST_CHARS]


def mint_ids(texts: pd.Series, doc_type: str) -> pd.Series:
    """Map a text column to IDs, asserting the hash is injective over the corpus.

    A collision would silently merge two documents into one manifest row and
    corrupt every disjointness guarantee built on top, so it fails loudly.
    """
    unique = pd.Series(texts.dropna().astype(str).unique())
    ids = unique.map(lambda t: doc_id(t, doc_type))
    if ids.nunique() != len(unique):
        clashes = ids[ids.duplicated(keep=False)]
        raise AssertionError(
            f"{doc_type}: {len(unique)} distinct texts collapsed to {ids.nunique()} IDs — "
            f"sha256 collision at {DIGEST_CHARS} hex chars on {clashes.tolist()[:4]}"
        )
    lookup = dict(zip(unique, ids))
    return texts.map(lambda t: lookup[str(t)])
