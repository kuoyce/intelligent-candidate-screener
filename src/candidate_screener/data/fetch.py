"""Download every automated source in the registry into `data/raw/`.

Sources are pulled from the Hugging Face datasets-server parquet conversion, which
needs no token because every adopted HF source is public and ungated (verified
19 Aug 2026). Two adopted sources cannot be automated and are reported as manual
steps: C1 (Kaggle, needs an API token) and D1 ESCO (registration + browser download).

Usage:
    uv run python -m candidate_screener.data.fetch --all
    uv run python -m candidate_screener.data.fetch --source fit djinni-jd
    uv run python -m candidate_screener.data.fetch --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from candidate_screener.config import RAW, ensure_data_dirs
from candidate_screener.data.sources import AUTOMATED, MANUAL, SOURCES, Source

DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    skip (exists) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"    {dest.name}  {dest.stat().st_size / 1e6:.1f} MB")


def fetch_hf(src: Source) -> None:
    """Pull the auto-converted parquet for every config/split of a public dataset."""
    meta = _get_json(f"{DATASETS_SERVER}/parquet?dataset={urllib.parse.quote(src.locator, safe='')}")
    files = meta.get("parquet_files", [])
    if not files:
        raise RuntimeError(f"no parquet exposed for {src.locator}: {meta}")
    for f in files:
        name = f"{f['config']}-{f['split']}-{Path(f['url']).name}"
        _download(f["url"], RAW / src.directory / name)


def fetch_github(src: Source) -> None:
    base = f"https://raw.githubusercontent.com/{src.locator}/master"
    for fn in (a.glob for a in src.artifacts):
        _download(f"{base}/{fn}", RAW / src.directory / fn)


def fetch_kaggle(src: Source) -> None:
    """Use the Kaggle CLI when a token is configured; otherwise print the manual step."""
    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not installed — see the manual instructions below")
    dest = RAW / src.directory
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", src.locator, "-p", str(dest), "--unzip"],
        check=True,
    )
    # A browser/WSL download leaves one NTFS alternate-stream marker per extracted file.
    for junk in dest.rglob("*:Zone.Identifier"):
        junk.unlink()


def manual_instructions(src: Source) -> str:
    if src.kind == "kaggle":
        return (
            f"  kaggle datasets download -d {src.locator} \\\n"
            f"    -p data/raw/{src.directory} --unzip\n"
            "  (needs a Kaggle account + ~/.kaggle/kaggle.json; extract the WHOLE archive — "
            "the PDFs are the point of this source)"
        )
    return (
        f"  1. open {src.locator}\n"
        "  2. register (free), accept the terms, choose CSV / English\n"
        f"  3. extract into data/raw/{src.directory}/ (use 7-Zip on Windows)"
    )


def fetch(key: str) -> None:
    src = SOURCES[key]
    print(f"[{key}] {src.catalog_id} {src.title}")
    handler = {"hf": fetch_hf, "github": fetch_github, "kaggle": fetch_kaggle}.get(src.kind)
    if handler is None:
        raise RuntimeError("manual source — no automated path")
    handler(src)


def _list() -> None:
    print(f"{'key':<16}{'cat':<5}{'tier':<6}{'auto':<8}licence")
    for key, s in SOURCES.items():
        tier = f"T{s.tier}" + ("" if s.adopted else "*")
        print(f"{key:<16}{s.catalog_id:<5}{tier:<6}{'yes' if s.automated else 'MANUAL':<8}{s.licence}")
    print("\n* tier 3 = evaluated and rejected; registered for traceability, never fetched by --all")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", choices=sorted(SOURCES), default=[])
    ap.add_argument("--all", action="store_true", help="every adopted, automatable source")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    args = ap.parse_args()

    if args.list:
        _list()
        return 0

    wanted = AUTOMATED if args.all else args.source
    if not wanted:
        ap.error("pass --all, --source <key> [...] or --list")

    ensure_data_dirs()
    failed = []
    for key in wanted:
        try:
            fetch(key)
        except Exception as exc:  # keep going; report at the end
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(key)

    print(f"\nraw data -> {RAW}")
    manual_todo = [k for k in MANUAL if not any((RAW / SOURCES[k].directory).glob("*"))]
    for key in manual_todo:
        print(f"\nMANUAL STEP — {SOURCES[key].catalog_id} {SOURCES[key].title}")
        print(manual_instructions(SOURCES[key]))
    if not manual_todo:
        print("manual sources (C1 Kaggle PDFs, D1 ESCO) are already in place")
    if failed:
        print(f"\nfailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nnext: uv run python -m candidate_screener.data.verify --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
