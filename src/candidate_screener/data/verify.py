"""Acceptance test on the download: is every adopted source present and unchanged?

For each registered source this checks that the expected files exist, counts the
rows in them, and compares against the **Verified** figures recorded on
19 Aug 2026. A drift is a failure, not a warning — it means a publisher has
re-uploaded and `docs/data/data-catalog.md` needs re-verification before any
number in it is trusted.

The result is written to `docs/data/acquisition-manifest.json` (file sizes and
SHA-256 digests included) so the exact bytes the experiments ran on stay
reproducible even though `data/` is git-ignored.

Usage:
    uv run python -m candidate_screener.data.verify --all
    uv run python -m candidate_screener.data.verify --source fit esco
    uv run python -m candidate_screener.data.verify --all --no-hash   # faster
    uv run python -m candidate_screener.data.verify --derived          # built artefacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from candidate_screener.config import MANIFEST, RAW
from candidate_screener.data.sources import SOURCES, Artifact, Source


def count_rows(path: Path) -> int | None:
    """Row count for the formats in `data/raw/`; None where rows are not meaningful."""
    match path.suffix.lower():
        case ".parquet":
            return pq.ParquetFile(path).metadata.num_rows
        case ".csv":
            return len(pd.read_csv(path, low_memory=False))
        case ".json":  # DataTurks ships JSON-lines despite the extension
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        case _:
            return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _check_artifact(root: Path, art: Artifact, with_hash: bool) -> tuple[dict, list[str]]:
    files = sorted(root.glob(art.glob))
    problems: list[str] = []
    if len(files) < art.min_files:
        problems.append(f"{art.glob}: found {len(files)} file(s), expected >= {art.min_files}")

    rows = 0
    counted = False
    entries = []
    # Hashing and row-counting thousands of PDFs is pointless; record the group instead.
    detail = len(files) <= 32
    for f in files:
        n = count_rows(f) if detail else None
        if n is not None:
            rows += n
            counted = True
        if detail:
            entries.append({
                "path": str(f.relative_to(RAW)),
                "bytes": f.stat().st_size,
                "rows": n,
                **({"sha256": sha256(f)} if with_hash else {}),
            })

    if art.rows is not None:
        if not counted:
            problems.append(f"{art.glob}: expected {art.rows} rows but nothing countable was found")
        elif rows != art.rows:
            problems.append(f"{art.glob}: {rows} rows, expected {art.rows} (publisher re-upload?)")

    return {
        "glob": art.glob,
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "rows": rows if counted else None,
        "rows_expected": art.rows,
        "entries": entries,
    }, problems


def verify_source(src: Source, with_hash: bool = True) -> dict:
    root = RAW / src.directory
    record: dict = {
        "catalog_id": src.catalog_id,
        "title": src.title,
        "tier": src.tier,
        "licence": src.licence,
        "pii": src.pii,
        "locator": src.locator,
        "path": str(root.relative_to(RAW.parent)),
        "present": root.is_dir(),
        "artifacts": [],
        "problems": [],
    }
    if not root.is_dir():
        record["problems"].append(f"missing directory {root} — run fetch")
        return record

    for art in src.artifacts:
        detail, problems = _check_artifact(root, art, with_hash)
        record["artifacts"].append(detail)
        record["problems"].extend(problems)
    record["bytes"] = sum(a["bytes"] for a in record["artifacts"])
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", choices=sorted(SOURCES), default=[])
    ap.add_argument("--all", action="store_true", help="every adopted source")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 digests")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--derived", action="store_true",
                    help="acceptance test on the Phase 3 derived artefacts instead")
    ap.add_argument("--task", nargs="+", default=[],
                    help="with --derived: check only these tasks")
    args = ap.parse_args()

    if args.derived:
        from candidate_screener.data.verify_derived import main as verify_derived
        return verify_derived(args.task or None)

    keys = [k for k, s in SOURCES.items() if s.adopted] if args.all else args.source
    if not keys:
        ap.error("pass --all or --source <key> [...]")

    records, failed = {}, []
    for key in keys:
        rec = verify_source(SOURCES[key], with_hash=not args.no_hash)
        records[key] = rec
        status = "OK" if not rec["problems"] else "FAIL"
        size = f"{rec.get('bytes', 0) / 1e6:8.1f} MB"
        rows = sum(a["rows"] or 0 for a in rec["artifacts"])
        print(f"[{status:4}] {key:<16}{rec['catalog_id']:<4}{size}  rows={rows:>9}  {rec['licence']}")
        for p in rec["problems"]:
            print(f"         ! {p}")
            failed.append(key)

    manifest = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "hashed": not args.no_hash,
        "total_bytes": sum(r.get("bytes", 0) for r in records.values()),
        "sources": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{manifest['total_bytes'] / 1e6:.0f} MB verified -> manifest {args.manifest}")

    if failed:
        print(f"FAILED: {', '.join(sorted(set(failed)))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
