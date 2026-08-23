"""Acceptance test on the *derived* artefacts — `verify --derived`.

`verify --all` asks "are the right bytes on disk?"; this asks "is what we built from
them still sound?". Every check corresponds to a numbered acceptance criterion in
`plan/2026-08-23-derived-artefacts/01-design-spec.md` §4, and each one has a
failure mode that would otherwise be invisible: a leaked document does not raise, it
just quietly inflates a score.

Checks 3.3-3.5 arrive with their tasks; the registry below is deliberately open.

Usage:
    uv run python -m candidate_screener.data.verify --derived
"""
from __future__ import annotations

import json

import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.data import fit_split as fs
from candidate_screener.data.ids import normalise


def _probe_c4_overlap(splits: dict[str, pd.DataFrame]) -> tuple[list[dict], str]:
    """Locate the 7 C4 resumes whose text is embedded in an A1 resume.

    These documents are legitimately in A1 — the guard is not against them, it is
    against a future re-import of the rejected `vm-structured-onet` set landing the
    *same* people in a training split while they are also being scored in test.
    Recording which split they fell into is what makes that exclusion applyable
    later rather than merely remembered.
    """
    path = PROCESSED / "vm-structured-onet" / "resumes_raw_structured.json"
    if not path.exists():
        return [], "skipped — rejected C4 derivatives not on disk"

    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = list(records.values())
    probes = {}
    for r in records:
        objective = normalise(r.get("career_objective") or "")
        if len(objective) >= 60:
            probes[objective[:60]] = r.get("resume_id") or r.get("id")

    hits = []
    for name, df in splits.items():
        texts = [(rid, normalise(t)) for rid, t in zip(df.resume_id, df.resume_text)]
        seen = set()
        for probe, c4_id in probes.items():
            for rid, text in texts:
                if rid not in seen and probe in text:
                    hits.append({"doc_id": rid, "split": name, "c4_resume_id": c4_id})
                    seen.add(rid)
                    break
    return hits, f"{len({h['doc_id'] for h in hits})} A1 resumes overlap the rejected C4 set"


def check_fit_split() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    if not fs.SPLIT_MANIFEST.exists():
        return [(False, "fit-split.csv missing — run `build --task fit-split`")]

    manifest = pd.read_csv(fs.SPLIT_MANIFEST)
    yields = json.loads(fs.YIELD_MANIFEST.read_text(encoding="utf-8"))

    # 1. Every manifest ID resolves to exactly one document in data/raw/.
    pairs, _ = fs.load_pooled()
    raw_ids = {"resume": set(pairs.resume_id), "jd": set(pairs.jd_id)}
    unresolved = sum(
        len(set(g.doc_id) - raw_ids[t]) for t, g in manifest.groupby("doc_type"))
    results.append((unresolved == 0 and manifest.doc_id.is_unique,
                    f"identity: {len(manifest)} manifest IDs, {unresolved} unresolved, "
                    f"{'no' if manifest.doc_id.is_unique else 'DUPLICATE'} repeats"))

    # 2. Split disjointness — the guarantee the whole artefact exists to provide.
    multi = int((manifest.groupby("doc_id").split.nunique() > 1).sum())
    built = {n: pd.read_parquet(p) for n in ("train", "val", "test")
             if (p := fs.FIT_OUT / f"{n}.parquet").exists()}
    overlaps = []
    names = sorted(built)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for col in ("resume_id", "jd_id"):
                n = len(set(built[a][col]) & set(built[b][col]))
                if n:
                    overlaps.append(f"{a}/{b} share {n} {col}")
    results.append((multi == 0 and not overlaps,
                    f"disjointness: {multi} documents in >1 split; "
                    f"{'; '.join(overlaps) if overlaps else 'no cross-split document overlap'}"))

    # 3. The discarded pairs are accounted for, not silently lost.
    c = yields["counts"]
    total = sum(c[k]["pairs"] for k in ("train", "val", "test", "discarded"))
    usable = yields["pooling"]["usable_pairs"]
    results.append((total == usable,
                    f"accounting: {c['discarded']['pairs']} discarded + "
                    f"{total - c['discarded']['pairs']} assigned = {total} of {usable} usable"))

    # 4. Byte-reproducibility: rebuild the manifest in memory and diff it.
    assigned, r_split, j_split = fs.split_pairs(pairs, yields["test_frac"],
                                                yields["val_frac"], yields["seed"])
    rebuilt = pd.concat([
        pd.DataFrame({"doc_id": r_split.index, "doc_type": "resume", "split": r_split.to_numpy()}),
        pd.DataFrame({"doc_id": j_split.index, "doc_type": "jd", "split": j_split.to_numpy()}),
    ]).sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True)
    same = rebuilt.equals(manifest.reset_index(drop=True))
    results.append((same, f"determinism: rebuild at seed {yields['seed']} "
                          f"{'reproduces' if same else 'DIFFERS FROM'} the committed manifest"))

    # 5. Contamination guard against the rejected C4 derivatives.
    hits, note = _probe_c4_overlap(built)
    if hits:
        pd.DataFrame(hits).sort_values(["split", "doc_id"]).to_csv(
            fs.MANIFESTS / "fit-c4-overlap.csv", index=False, lineterminator="\n")
    results.append((True, f"contamination guard: {note}"
                          f"{' -> fit-c4-overlap.csv' if hits else ''}"))
    return results


#: Task -> its acceptance checks. 3.1, 3.3, 3.4 and 3.5 register here as they land.
CHECKS = {"fit-split": check_fit_split}


def main(tasks: list[str] | None = None) -> int:
    failed = 0
    for name, fn in CHECKS.items():
        if tasks and name not in tasks:
            continue
        print(f"\n=== {name}")
        for ok, message in fn():
            print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
            failed += not ok
    print(f"\n{'all derived checks passed' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0
