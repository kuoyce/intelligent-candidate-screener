"""Download the sources adopted in ../02-data-catalog.md into data/raw/.

Every source here is public and ungated; no HF token is required. The Kaggle PDF
corpus (C1) and ESCO (D1) are manual steps -- see ../03-acquisition-action-plan.md.

Usage:
    uv run python plan/2026-08-19-data-strategy/scripts/fetch_sources.py --all
    uv run python plan/2026-08-19-data-strategy/scripts/fetch_sources.py --source fit djinni-jd
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

# catalog id -> (kind, locator)
SOURCES: dict[str, tuple[str, str]] = {
    "fit":          ("hf", "cnamuangtoun/resume-job-description-fit"),   # A1
    "djinni-jd":    ("hf", "lang-uk/recruitment-dataset-job-descriptions-english"),  # A2
    "djinni-cv":    ("hf", "lang-uk/recruitment-dataset-candidate-profiles-english"),  # A2
    "skillspan":    ("hf", "jjzha/skillspan"),        # B2
    "green":        ("hf", "jjzha/green"),            # B3
    "livecareer":   ("hf", "opensporks/resumes"),     # C2
    "resume-atlas": ("hf", "ahmedheakl/resume-atlas"),  # C3
    "data-jobs":    ("hf", "lukebarousse/data_jobs"),  # D2
    "dataturks":    ("github", "DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy"),  # B1
}

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
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"    {dest.name}  {dest.stat().st_size / 1e6:.1f} MB")


def fetch_hf(repo: str, key: str) -> None:
    """Pull the auto-converted parquet for every config/split of a public dataset."""
    meta = _get_json(f"{DATASETS_SERVER}/parquet?dataset={urllib.parse.quote(repo, safe='')}")
    files = meta.get("parquet_files", [])
    if not files:
        raise RuntimeError(f"no parquet exposed for {repo}: {meta}")
    for f in files:
        name = f"{f['config']}-{f['split']}-{Path(f['url']).name}"
        _download(f["url"], RAW / key / name)


def fetch_github(repo: str, key: str) -> None:
    base = f"https://raw.githubusercontent.com/{repo}/master"
    for fn in ("traindata.json", "testdata.json"):
        _download(f"{base}/{fn}", RAW / key / fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", choices=sorted(SOURCES), default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    wanted = sorted(SOURCES) if args.all else args.source
    if not wanted:
        ap.error("pass --all or --source <id> [...]")

    failed = []
    for key in wanted:
        kind, locator = SOURCES[key]
        print(f"[{key}] {locator}")
        try:
            (fetch_hf if kind == "hf" else fetch_github)(locator, key)
        except Exception as exc:  # keep going; report at the end
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(key)

    print(f"\nraw data -> {RAW}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nStill manual: C1 Kaggle PDFs (needs token), D1 ESCO (needs registration).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
