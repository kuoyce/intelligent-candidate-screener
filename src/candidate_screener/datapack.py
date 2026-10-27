"""The Docker app's data folder, `data/datapack/` (plan 2026-10-27-docker, `04-app-data-folder.md`).

    uv run python -m candidate_screener.datapack create [--with-workspace]   # fill it from data/
    uv run python -m candidate_screener.datapack verify [--full]             # check it
    uv run python -m candidate_screener.datapack handover DEST               # a copy to pass on
    uv run python -m candidate_screener.datapack purge [--yes]               # empty it
    docker compose --profile tools run --rm tools <command> ...              # the same, in Docker

The folder holds what the app reads (the S4 run-time audit, 27 Oct 2026): the CSV-backed review
tables and the skill vocabulary from the repository's committed copies, and from a source (a
`data/` directory or a received handover copy, symbolic links followed) the A2 text parquets, the
role-family model, SBERT weak and the inference parts. `expected()` derives every name from the
code that reads it, so `verify` names an input the code now expects and the folder lacks.

The app also writes its workspace (accounts, reviews: `processed/hr-portal/`) and its caches into
the folder. `create` keeps the workspace and empties the caches; `handover` leaves both out;
`purge --yes` deletes everything. Every command runs by hand, never at app start.

File copies only: importing this module and computing the list import no torch, so it runs in the
main environment, natively or in the `tools` service, whose `SCREENER_DATAPACK_SOURCE` and
`SCREENER_DATAPACK_DEST` give the defaults there. The project's own `data/` is never filled or
emptied, except where `SCREENER_DATAPACK_DEST` names it as the folder (the `tools` container mounts
the folder there).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import os
import posixpath
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path, PurePosixPath

from candidate_screener import config

FORMAT = 1
MANIFEST = "pack-manifest.json"
SOURCE_ENV, DEST_ENV = "SCREENER_DATAPACK_SOURCE", "SCREENER_DATAPACK_DEST"
#: The text the app reads (D53): `inference/screener.py` and `app/portal_inference.py` pass "v2" to
#: `corpus.a2_pairs` and `corpus.a2_train_documents`.
TEXT_VARIANT = "v2"
#: The CSV-backed reviews `app/portal_data.py` reads, copied from the committed `output/matching/`.
REVIEW_TABLES = ("a2-cv-extraction.csv", "a2-jd-extraction.csv", "a2-x2-scores.csv")
#: The suffix of a file or folder being written; the next `create` removes a leftover one.
TMP = ".datapack-tmp"
FILE_MODE, DIR_MODE, PRIVATE_FILE, PRIVATE_DIR = 0o644, 0o755, 0o600, 0o700
CHUNK = 1 << 20
_FINGERPRINT = re.compile(r"[0-9a-f]{6,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: The review screens spell the A2 text cache out: `PROCESSED / "matching/a2-pairs-v2-<fp>.parquet"`.
_A2_LITERAL = re.compile(r"PROCESSED\s*/\s*[\"']([^\"']*a2-pairs[^\"']*\.parquet)[\"']")


class DatapackError(Exception):
    """A refusal, or a source that lacks an input: the command stops before writing anything."""


@dataclasses.dataclass(frozen=True)
class Pointer:
    """A `LATEST` file naming the fitted part the code loads: `target` with `{}` = its text,
    relative to the LATEST's folder."""
    target: str
    reader: str
    needs: tuple[str, ...] = ()          # a folder target: the files the code reads in it
    record: str | None = None            # the fingerprint the code's committed record names
    provenance: tuple[str, dict] | None = None   # (file in the target, {field: input it must match})


@dataclasses.dataclass(frozen=True)
class Expected:
    """What the app reads from a data folder (relative POSIX paths) and what it writes there."""
    committed: dict          # input -> the repository's committed file it is copied from
    files: dict              # input copied from the source -> the code that reads it
    folders: dict            # folder copied whole from the source -> (the code, files it needs)
    pointers: dict           # LATEST input -> Pointer
    digests: dict            # input -> (SHA-256 the code expects, where that comes from)
    fingerprints: dict       # recorded in the manifest
    workspace: str
    caches: tuple
    workspace_versions: dict  # workspace file -> the format version the code reads or writes


@dataclasses.dataclass
class Report:
    folder: Path
    full: bool
    problems: list
    lines: list
    inputs: int = 0
    size: int = 0


# --------------------------------------------------------------------------- what the code reads

def _rel(path) -> str:
    """`path` relative to the data root, as the manifest spells it."""
    return Path(path).relative_to(config.DATA).as_posix()


def _shown(path) -> str:
    try:
        return Path(path).relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _record(path: Path, *keys):
    """A value of a committed JSON record, or None when it is absent."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in keys:
            value = value[key]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value


def _version_in(path: Path, pattern: str):
    """The format version a module's source spells out, or None when it cannot be found."""
    try:
        match = re.search(pattern, Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(match.group(1)) if match else None


def expected() -> Expected:
    """What the app reads from its data folder, each name derived from the module that reads it."""
    from candidate_screener.app import portal_data, portal_store
    from candidate_screener.evaluation import final_set
    from candidate_screener.extraction import evaluate, role_family, rules
    from candidate_screener.inference import parts
    from candidate_screener.matching import corpus, sbert
    from candidate_screener.rerank import cross

    tag = corpus.variant_tag(TEXT_VARIANT)
    pairs = corpus._digest(final_set.PAIRS_CSV)
    files = {_rel(corpus.CACHE / f"a2-pairs{tag}-{pairs}.parquet"): f"corpus.a2_pairs({TEXT_VARIANT!r})"}
    sample = inspect.signature(corpus.a2_train_documents).parameters
    for side in ("cv", "jd"):
        name = f"a2-train-{side}{tag}-{sample[f'n_{side}'].default}-{sample['seed'].default}.parquet"
        files[_rel(corpus.CACHE / name)] = f"corpus.a2_train_documents(text_variant={TEXT_VARIANT!r})"
    app = Path(portal_data.__file__).parent
    for module in sorted(app.glob("*.py")):
        for literal in _A2_LITERAL.findall(module.read_text(encoding="utf-8")):
            rel = _rel(config.PROCESSED / literal)
            files[rel] = ", ".join(filter(None, (files.get(rel), f"app/{module.name}")))
    weak = sbert.SbertWeak.cache_key(TEXT_VARIANT)
    folders = {_rel(sbert.WEIGHTS / weak): (f"sbert.SbertWeak({TEXT_VARIANT!r})", ("config.json",))}

    reviews = Path(inspect.signature(portal_data.PortalDataset.from_folder).parameters["folder"].default)
    committed = {_rel(reviews / name): config.OUTPUT / "matching" / name for name in REVIEW_TABLES}
    vocabulary = _rel(rules.VOCAB_CSV)
    committed[vocabulary] = rules.VOCAB_CSV
    digests = {rel: (_sha256(path), f"the committed {_shown(path)}")
               for rel, path in committed.items() if rel != vocabulary and path.is_file()}
    vocabulary_sha = _record(evaluate.RECORD, "vocab", "sha256")
    if vocabulary_sha:
        digests[vocabulary] = (vocabulary_sha, f"the vocabulary {_shown(evaluate.RECORD)} records")

    role = _record(role_family.RECORD, "fingerprint")
    role_latest = _rel(role_family.CACHE / "LATEST")
    pointers = {
        role_latest: Pointer("role-model-{}.pkl", "role_family.load_role_model()", record=role),
        _rel(parts.ROOT / "LATEST"): Pointer(
            "{}", "inference.parts.load_parts()", needs=("parts.json", *parts.FILES),
            provenance=("parts.json", {"role_model": role_latest, "vocabulary_sha256": vocabulary})),
    }
    versions = {"accounts.json": _version_in(app / "portal_auth.py", r'"version":\s*(\d+),\s*"users"'),
                portal_store.DEFAULT_PATH.name: _version_in(
                    Path(portal_store.__file__), r"portal_store_migrations VALUES\((\d+)\)")}
    return Expected(
        committed=committed, files=files, folders=folders, pointers=pointers, digests=digests,
        fingerprints={"a2_pairs": pairs, "sbert_weak": weak, "role_model": role,
                      "vocabulary_sha256": vocabulary_sha},
        workspace=_rel(portal_store.DEFAULT_PATH.parent),
        caches=(_rel(sbert.EMBED_CACHE), _rel(cross.ROOT)), workspace_versions=versions)


def _layout(exp: Expected, root: Path) -> tuple[dict, dict, dict, list]:
    """The inputs against a data root, whose LATEST files name the fitted parts: (required file ->
    the code that reads it, folder copied whole -> the code, LATEST -> its value, problems)."""
    required = {rel: f"committed {_shown(path)}" for rel, path in exp.committed.items()}
    required.update(exp.files)
    folders = {}
    for folder, (reader, needs) in exp.folders.items():
        folders[folder] = reader
        required.update({f"{folder}/{name}": reader for name in needs})
    values, problems = {}, []
    for latest, pointer in exp.pointers.items():
        required[latest] = pointer.reader
        try:
            with open(Path(root) / latest, encoding="utf-8") as stream:
                value = stream.read(256).strip()
        except OSError:
            problems.append(f"missing: {latest}")
            continue
        if not _FINGERPRINT.fullmatch(value):
            problems.append(f"{latest} does not name a fingerprint")
            continue
        values[latest] = value
        target = posixpath.join(posixpath.dirname(latest), pointer.target.format(value))
        if pointer.needs:
            folders[target] = pointer.reader
            required.update({f"{target}/{name}": pointer.reader for name in pointer.needs})
        else:
            required[target] = pointer.reader
    return required, folders, values, problems


# ---------------------------------------------------------------------------------- file helpers

def _sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mb(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def _files(count: int) -> str:
    return f"{count} file{'' if count == 1 else 's'}"


def _under(rel: str, prefixes) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def _inside(path: Path, folder: Path) -> bool:
    return folder in path.parents


def _linked(base: Path, rel: str) -> bool:
    """Whether a folder on the way from `base` to `base/rel` (rel included) is a symbolic link."""
    current = base
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _clean(rel) -> str:
    """A manifest path: relative, normalised, POSIX, inside the folder."""
    path = PurePosixPath(rel) if isinstance(rel, str) else None
    if (path is None or not rel or path.is_absolute() or "\\" in rel or path.as_posix() != rel
            or any(part in {".", ".."} for part in path.parts)):
        raise DatapackError(f"{MANIFEST} lists an unusable path: {rel!r}")
    return rel


def _tree(root: Path) -> tuple[list[str], list[str]]:
    """The files and folders under `root`, relative and sorted, symbolic links followed."""
    files, dirs, seen = [], [], set()

    def fail(error):
        raise DatapackError(f"cannot read {error.filename}: {error.strerror}")

    for current, _, names in os.walk(root, followlinks=True, onerror=fail):
        stat = os.stat(current)
        if (stat.st_dev, stat.st_ino) in seen:
            raise DatapackError(f"{current} is reached twice (a symbolic-link loop)")
        seen.add((stat.st_dev, stat.st_ino))
        here = Path(current).relative_to(root)
        if here.parts:
            dirs.append(here.as_posix())
        files += [(here / name).as_posix() for name in names]
    return sorted(files), sorted(dirs)


def _walk_files(root: Path, skip) -> list[str]:
    """The files and symbolic links under `root` outside `skip`, links not followed (a linked
    folder is listed as one entry)."""
    found = []
    for current, dirs, names in os.walk(root):
        here = Path(current).relative_to(root)
        kept = []
        for name in dirs:
            rel = (here / name).as_posix()
            if _under(rel, skip):
                continue
            if (Path(current) / name).is_symlink():
                found.append(rel)
            else:
                kept.append(name)
        dirs[:] = kept
        found += [rel for rel in ((here / name).as_posix() for name in names) if not _under(rel, skip)]
    return sorted(found)


def _make_dirs(base: Path, rel: str, mode: int) -> Path:
    """`base/rel` as real folders set to `mode`; refuses a symbolic link on the way, so nothing is
    written outside `base`."""
    current = base
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise DatapackError(f"{current} is a symbolic link; a data folder holds real folders only")
        if not current.is_dir():
            current.mkdir()
        os.chmod(current, mode)
    return current


def _copy(source: Path, target: Path, mode: int) -> tuple[int, str]:
    """Copy `source`'s bytes (links followed) to `target` through a temporary file created with
    `mode`; return (bytes, SHA-256)."""
    temporary = target.with_name(f".{target.name}{TMP}")
    temporary.unlink(missing_ok=True)
    digest, size = hashlib.sha256(), 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        with open(source, "rb") as reader, os.fdopen(os.open(temporary, flags, mode), "wb") as writer:
            while chunk := reader.read(CHUNK):
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def _remove(path: Path) -> None:
    """Delete a file, a link (never its target) or a folder tree."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _count(path: Path) -> int:
    if path.is_symlink() or not path.is_dir():
        return 1
    return sum(len(names) for _, _, names in os.walk(path))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}{TMP}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, FILE_MODE)
    os.replace(temporary, path)


def _commit() -> str:
    try:
        result = subprocess.run(["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    head = result.stdout.strip()
    return head if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", head) else "unknown"


# ------------------------------------------------------------------------------- folder checks

def _destination(path: Path) -> Path:
    """`path` resolved, unless it is the project, one of its parents, or the project's `data/`
    (allowed only where SCREENER_DATAPACK_DEST names it: the tools container mounts the folder
    at /app/data)."""
    real = Path(path).resolve()
    project, data = Path(config.PROJECT_ROOT).resolve(), Path(config.DATA).resolve()
    if real in {project, *project.parents, *data.parents}:
        raise DatapackError(f"{path} is the project folder or one of its parents; refusing")
    named = os.environ.get(DEST_ENV)
    if real == data and not (named and Path(named).resolve() == data):
        raise DatapackError(f"{path} is the project's data/ folder (research data); the Docker "
                            "data folder lives below it (data/datapack); refusing")
    return real


def _read_manifest(folder: Path) -> tuple[dict, dict]:
    """(manifest, {input path: entry}); raises when the folder has none or it is unusable."""
    path = Path(folder) / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatapackError(f"{folder} has no {MANIFEST}: not a data folder (run create)") from None
    except (OSError, ValueError) as error:
        raise DatapackError(f"{path} is unreadable: {error}") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise DatapackError(f"{path} is not a data folder manifest")
    if manifest.get("format") != FORMAT:
        raise DatapackError(f"{path} has format {manifest.get('format')!r}; this code reads format "
                            f"{FORMAT} (purge the folder and create it again)")
    entries = {}
    for item in manifest["inputs"]:
        if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256", "origin"}
                or type(item["bytes"]) is not int or item["bytes"] < 0
                or not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"])
                or item["origin"] not in {"committed", "copied"}):
            raise DatapackError(f"{path} has a malformed input entry: {item!r}")
        rel = _clean(item["path"])
        if rel in entries or rel == MANIFEST:
            raise DatapackError(f"{path} lists {rel} twice")
        entries[rel] = item
    manifest["empty_directories"] = [_clean(rel) for rel in manifest.get("empty_directories") or []]
    return manifest, entries


def _private(exp: Expected) -> tuple[str, ...]:
    return (exp.workspace, *exp.caches)


def _refuse_private(exp: Expected, entries) -> None:
    listed = [rel for rel in entries if _under(rel, _private(exp))]
    if listed:
        raise DatapackError(f"{MANIFEST} lists {listed[0]}, inside the workspace or a cache; refusing")


def _check_fillable(real: Path, exp: Expected) -> None:
    """`create` fills a new or empty folder, a data folder, or one holding only what the app (or an
    interrupted create) writes; never a folder of other things."""
    if not real.exists():
        return
    if not real.is_dir():
        raise DatapackError(f"{real} is not a folder")
    names = {entry.name for entry in real.iterdir()}
    if not names or MANIFEST in names:
        return
    layout = {PurePosixPath(rel).parts[0] for rel in (*exp.committed, *exp.files, *exp.folders,
                                                      *exp.pointers, *_private(exp))}
    stray = sorted(name for name in names - layout if not name.endswith(TMP))
    if stray:
        raise DatapackError(f"{real} is not empty, has no {MANIFEST} and holds {', '.join(stray[:5])}: "
                            "create fills only a new or empty folder or a data folder; refusing")


# ---------------------------------------------------------------------------------------- create

def _plan(exp: Expected, source: Path, real: Path) -> tuple[dict, list, dict, dict]:
    """({input: (file to copy, origin)}, empty folders inside copied folders, LATEST values,
    folders copied whole)."""
    required, folders, values, problems = _layout(exp, source)
    plan = {}
    for rel in required:
        repository = exp.committed.get(rel)
        # In the tools container the repository's data/ is the folder itself (mounted over it):
        # the committed copy then comes from the source, and verify compares it with the record.
        if repository is None or _inside(Path(repository).resolve(), real):
            plan[rel] = (source / rel, "copied")
        else:
            plan[rel] = (Path(repository), "committed")
    empty = []
    for folder, reader in folders.items():
        if not (source / folder).is_dir():
            problems.append(f"missing: {source / folder}/ ({reader})")
            continue
        files, dirs = _tree(source / folder)
        for name in files:
            plan.setdefault(f"{folder}/{name}", (source / folder / name, "copied"))
        empty += [f"{folder}/{rel}" for rel in dirs if not any(name.startswith(rel + "/") for name in files)]
    problems += [f"missing or not a regular file: {path}" for path, _ in plan.values() if not path.is_file()]
    if problems:
        raise DatapackError("the source cannot fill the folder; nothing was changed:\n  "
                            + "\n  ".join(dict.fromkeys(problems)))
    linked = [rel for rel in [*plan, *empty] if _linked(real, posixpath.dirname(rel))]
    if linked:
        raise DatapackError(f"{real / linked[0]} is reached through a symbolic link inside the folder; refusing")
    return plan, sorted(empty), values, folders


def _workspace_plan(exp: Expected, source: Path, real: Path, replace: bool):
    """(source workspace, its files, its folders, whether one is replaced), checked before writing."""
    origin = source / exp.workspace
    if not origin.is_dir():
        raise DatapackError(f"the source has no {exp.workspace}/ to copy; nothing was changed")
    files, dirs = _tree(origin)
    odd = [name for name in files if not (origin / name).is_file()]
    if odd:
        raise DatapackError(f"{origin / odd[0]} is not a regular file; nothing was changed")
    target = real / exp.workspace
    present = target.is_symlink() or target.is_file() or (target.is_dir() and any(target.iterdir()))
    if present and not replace:
        raise DatapackError(f"{target} already holds a workspace; add --replace-workspace to replace "
                            "it with the source's; nothing was changed")
    return origin, files, dirs, present


def _old_inputs(real: Path, exp: Expected) -> list[str]:
    if not (real / MANIFEST).exists():
        return []
    _, entries = _read_manifest(real)
    _refuse_private(exp, entries)
    linked = [rel for rel in entries if _linked(real, posixpath.dirname(rel))]
    if linked:
        raise DatapackError(f"{real / linked[0]} is reached through a symbolic link inside the folder; refusing")
    return list(entries)


def _empty(path: Path) -> int:
    """Delete what a cache folder holds (a linked folder: only the link); return the files deleted."""
    if path.is_symlink():
        path.unlink()
        return 1
    if not path.is_dir():
        return 0
    deleted = 0
    for entry in path.iterdir():
        deleted += _count(entry)
        _remove(entry)
    return deleted


def _prune(real: Path, exp: Expected) -> None:
    """Remove empty folders left by old inputs; the workspace, the caches and their parents stay."""
    keep = _private(exp)
    for current, _, _ in os.walk(real, topdown=False):
        rel = Path(current).relative_to(real).as_posix()
        if rel == "." or _under(rel, keep) or any(prefix.startswith(rel + "/") for prefix in keep):
            continue
        try:
            os.rmdir(current)
        except OSError:
            pass


def _clear_leftovers(real: Path, exp: Expected) -> None:
    keep = _private(exp)
    for current, dirs, names in os.walk(real):
        here = Path(current).relative_to(real)
        dirs[:] = [name for name in dirs if not _under((here / name).as_posix(), keep)]
        for name in [*names, *dirs]:
            if name.endswith(TMP):
                _remove(Path(current) / name)
        dirs[:] = [name for name in dirs if not name.endswith(TMP)]


def _install_workspace(real: Path, exp: Expected, plan) -> str:
    origin, files, dirs, replacing = plan
    final = real / exp.workspace
    temporary = final.with_name(final.name + TMP)
    if os.path.lexists(temporary):
        _remove(temporary)
    _make_dirs(real, posixpath.dirname(exp.workspace), DIR_MODE)
    temporary.mkdir(mode=PRIVATE_DIR)
    os.chmod(temporary, PRIVATE_DIR)
    for rel in dirs:
        _make_dirs(temporary, rel, PRIVATE_DIR)
    for rel in files:
        _copy(origin / rel, temporary / rel, PRIVATE_FILE)
    if os.path.lexists(final):
        _remove(final)
    os.replace(temporary, final)
    return f"{'replaced with' if replacing else 'copied from'} the source's ({_files(len(files))}; private)"


def create(source: Path, dest: Path, given: str, with_workspace: bool = False,
           replace_workspace: bool = False, out=print) -> int:
    """Fill `dest` from `source`; return verify's exit status."""
    exp = expected()
    source = Path(source)
    if not source.is_dir():
        raise DatapackError(f"the source {given} is not a folder")
    real = _destination(dest)
    if source.resolve() == real or _inside(source.resolve(), real):
        raise DatapackError(f"the source {given} is the data folder or lies inside it; refusing")
    _check_fillable(real, exp)
    for rel in _private(exp):
        if _linked(real, posixpath.dirname(rel)):
            raise DatapackError(f"{real / rel} is reached through a symbolic link inside the folder; refusing")
    old = _old_inputs(real, exp)
    workspace = _workspace_plan(exp, source, real, replace_workspace) if with_workspace else None
    plan, empty, values, folders = _plan(exp, source, real)

    # Nothing has been written up to here.
    made = not real.exists()
    real.mkdir(parents=True, exist_ok=True)
    if made:
        os.chmod(real, DIR_MODE)
    _clear_leftovers(real, exp)
    removed = [rel for rel in old if rel not in plan]
    for rel in removed:
        (real / rel).unlink(missing_ok=True)
    emptied = {cache: _empty(real / cache) for cache in exp.caches}
    _prune(real, exp)
    inputs = []
    for rel in sorted(plan):
        path, origin = plan[rel]
        _make_dirs(real, posixpath.dirname(rel), DIR_MODE)
        size, digest = _copy(path, real / rel, FILE_MODE)
        inputs.append({"path": rel, "bytes": size, "sha256": digest, "origin": origin})
    for rel in empty:
        _make_dirs(real, rel, DIR_MODE)
    if workspace is not None:
        kept = _install_workspace(real, exp, workspace)
    else:
        target = real / exp.workspace
        kept = "kept" if target.is_dir() and any(target.iterdir()) else "none"
    _write_json(real / MANIFEST, {
        "format": FORMAT,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": given,
        "source_commit": _commit(),
        "fingerprints": exp.fingerprints,
        "latest": values,
        "inputs": inputs,
        "empty_directories": empty,
    })

    total = sum(item["bytes"] for item in inputs)
    committed = sum(item["origin"] == "committed" for item in inputs)
    out(f"create {real} from {given}")
    out(f"  inputs: {_files(len(inputs))}, {_mb(total)} ({committed} from the repository's committed "
        f"copies, {len(inputs) - committed} from the source); {len(removed)} old inputs removed")
    groups = {}
    for item in inputs:
        group = next((f"{folder}/" for folder in folders if _under(item["path"], [folder])), item["path"])
        groups.setdefault(group, []).append(item["bytes"])
    for group, sizes in groups.items():
        out(f"    {group}" + (f" ({_files(len(sizes))}, {_mb(sum(sizes))})" if group.endswith("/")
                              else f" ({_mb(sum(sizes))})"))
    out("  caches emptied: " + ", ".join(f"{cache}/ ({_files(count)})" for cache, count in emptied.items()))
    out(f"  workspace {exp.workspace}/: {kept}")
    return _print_report(check(real, full=True, exp=exp), out)


# ---------------------------------------------------------------------------------------- verify

def _recorded_version(path: Path):
    """The format version a workspace file records, None when it records none."""
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return value.get("version") if isinstance(value, dict) else "unreadable"
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "portal_store_migrations" not in tables:
                return None
            return db.execute("SELECT MAX(version) FROM portal_store_migrations").fetchone()[0]
    except (OSError, ValueError, sqlite3.Error):
        return "unreadable"


def _workspace_line(folder: Path, exp: Expected, problems: list) -> str:
    """The workspace's recorded format versions beside the code's (accounts.json must match; the
    store migrates an older database forward, so only a newer one is a problem)."""
    workspace = folder / exp.workspace
    if not workspace.is_dir() or not any(workspace.iterdir()):
        return f"workspace {exp.workspace}/: none"
    parts = []
    for name, code in exp.workspace_versions.items():
        if not (workspace / name).is_file():
            continue
        found = _recorded_version(workspace / name)
        parts.append(f"{name} format {found if found is not None else 'not recorded'} "
                     f"(the code: {code if code is not None else 'unknown'})")
        if code is None or found is None:
            continue
        exact = name.endswith(".json")
        if not isinstance(found, int) or (found != code if exact else found > code):
            problems.append(f"workspace: {exp.workspace}/{name} has format {found}; the code "
                            f"{'reads' if exact else 'writes'} {code} (a breaking change: purge, or "
                            "create --with-workspace --replace-workspace)")
    return f"workspace {exp.workspace}/: " + ("; ".join(parts) or "present, no recorded format version")


def check(folder: Path, full: bool = False, exp: Expected | None = None) -> Report:
    """Compare a data folder with its manifest and with what the code expects."""
    exp = exp or expected()
    folder = Path(folder)
    report = Report(folder, full, [], [])
    if not folder.is_dir():
        report.problems.append(f"{folder} is not a folder")
        return report
    try:
        manifest, entries = _read_manifest(folder)
    except DatapackError as error:
        report.problems.append(str(error))
        return report
    report.lines.append(f"manifest: created {manifest.get('created')} from {manifest.get('source')} "
                        f"at commit {str(manifest.get('source_commit'))[:12]}")
    report.inputs, report.size = len(entries), sum(item["bytes"] for item in entries.values())
    problems = report.problems
    private = _private(exp)
    for rel, item in sorted(entries.items()):
        path = folder / rel
        if _under(rel, private):
            problems.append(f"{MANIFEST} lists {rel}, inside the workspace or a cache")
        elif _linked(folder, rel):
            problems.append(f"linked: {rel} is (or is reached through) a symbolic link; the folder holds copies")
        elif not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.stat().st_size != item["bytes"]:
            problems.append(f"changed: {rel} ({path.stat().st_size} bytes; the manifest says {item['bytes']})")
        elif full and _sha256(path) != item["sha256"]:
            problems.append(f"changed: {rel} (its SHA-256 differs from the manifest's)")
    problems += [f"unexpected: {rel}" for rel in _walk_files(folder, (*private, MANIFEST)) if rel not in entries]

    required, folders, values, layout_problems = _layout(exp, folder)
    problems += layout_problems
    problems += [f"the code expects {rel} ({reader}); the folder lacks it"
                 for rel, reader in required.items() if rel not in entries]
    problems += [f"stale: {rel} is no longer read by the code" for rel in sorted(entries)
                 if rel not in required and not _under(rel, [*folders, *private])]
    for latest, pointer in exp.pointers.items():
        value = values.get(latest)
        if value is None:
            continue
        if pointer.record and value != pointer.record:
            problems.append(f"stale: {latest} names {value}; the code's committed record names {pointer.record}")
        if pointer.provenance:
            name, fields = pointer.provenance
            record = posixpath.join(posixpath.dirname(latest), pointer.target.format(value), name)
            if not (folder / record).is_file():
                continue
            recorded = _record(folder / record)
            recorded = recorded if isinstance(recorded, dict) else {}
            for field, ref in fields.items():
                want = values.get(ref) if ref in exp.pointers else entries.get(ref, {}).get("sha256")
                if want is not None and recorded.get(field) != want:
                    problems.append(f"stale: {record} records {field} {recorded.get(field)}, but the "
                                    f"folder's {ref} is {want}: refit them for this folder")
    problems += [f"stale: {rel} differs from {origin}" for rel, (digest, origin) in exp.digests.items()
                 if rel in entries and entries[rel]["sha256"] != digest]
    report.lines.append(_workspace_line(folder, exp, problems))
    return report


def _print_report(report: Report, out) -> int:
    checked = "sizes and SHA-256" if report.full else "sizes only (--full adds SHA-256)"
    out(f"verify {report.folder}: {checked}")
    for line in report.lines:
        out(f"  {line}")
    problems = list(dict.fromkeys(report.problems))
    if problems:
        out(f"{len(problems)} problem{'' if len(problems) == 1 else 's'}:")
        for problem in problems:
            out(f"  {problem}")
        return 1
    out(f"OK: {report.inputs} inputs ({_mb(report.size)}) match the manifest and what the code expects")
    return 0


# -------------------------------------------------------------------------------------- handover

def handover(dest: Path, folder: Path, out=print) -> int:
    """Copy `folder`'s inputs and manifest (never the workspace or caches) into a new or empty
    `dest`; return the full verification's exit status."""
    exp = expected()
    folder = Path(folder)
    manifest, entries = _read_manifest(folder)
    _refuse_private(exp, [*entries, *manifest["empty_directories"]])
    target = Path(dest)
    if os.path.lexists(target) and (target.is_symlink() or not target.is_dir() or any(target.iterdir())):
        raise DatapackError(f"{dest} is not a new or empty folder; refusing")
    real, origin = target.resolve(), folder.resolve()
    if real == origin or _inside(real, origin):
        raise DatapackError(f"{dest} lies inside the data folder; refusing")
    missing = [rel for rel in entries if not (folder / rel).is_file()]
    if missing:
        raise DatapackError(f"the folder lacks {missing[0]} (verify lists every problem); nothing was copied")
    made = not real.exists()
    real.mkdir(parents=True, exist_ok=True)
    if made:
        os.chmod(real, DIR_MODE)
    for rel in sorted(entries):
        _make_dirs(real, posixpath.dirname(rel), DIR_MODE)
        _copy(folder / rel, real / rel, FILE_MODE)
    for rel in manifest["empty_directories"]:
        _make_dirs(real, rel, DIR_MODE)
    _copy(folder / MANIFEST, real / MANIFEST, FILE_MODE)
    out(f"handover {real}: {len(entries)} inputs "
        f"({_mb(sum(item['bytes'] for item in entries.values()))}) and {MANIFEST}; "
        "no workspace, no caches")
    return _print_report(check(real, full=True, exp=exp), out)


# ----------------------------------------------------------------------------------------- purge

def purge(dest: Path, yes: bool = False, out=print) -> int:
    """Empty a data folder (workspace included); without `yes`, only list what would go."""
    exp = expected()
    path = Path(dest)
    if not path.exists():
        raise DatapackError(f"{dest} does not exist; nothing to purge")
    if not path.is_dir():
        raise DatapackError(f"{dest} is not a folder; refusing")
    real = _destination(path)
    entries = sorted(real.iterdir(), key=lambda entry: (entry.name == MANIFEST, entry.name))
    if not entries:
        out(f"purge {real}: already empty")
        return 0
    if not (real / MANIFEST).is_file():
        raise DatapackError(f"{real} is not a data folder (no {MANIFEST}, and not empty); nothing deleted")
    try:
        inputs = set(_read_manifest(real)[1])
    except DatapackError:
        inputs = set()
    groups = {"inputs and the manifest": [0, 0], f"workspace {exp.workspace}/": [0, 0],
              "caches": [0, 0], "other": [0, 0]}
    others = []
    for rel in _walk_files(real, ()):
        if rel == MANIFEST or rel in inputs:
            group = "inputs and the manifest"
        elif _under(rel, [exp.workspace]):
            group = f"workspace {exp.workspace}/"
        elif _under(rel, exp.caches):
            group = "caches"
        else:
            group = "other"
            others.append(rel)
        groups[group][0] += 1
        groups[group][1] += (real / rel).lstat().st_size
    files, size = sum(n for n, _ in groups.values()), sum(s for _, s in groups.values())
    out(f"purge {real}: {'deleting' if yes else 'would delete'} {_files(files)} ({_mb(size)})")
    for group, (count, total) in groups.items():
        out(f"  {group}: {_files(count)} ({_mb(total)})")
    for rel in others[:10]:
        out(f"    {rel}")
    if len(others) > 10:
        out(f"    ... and {len(others) - 10} more")
    if not yes:
        out("nothing deleted: add --yes to delete")
        return 0
    for entry in entries:            # the manifest last, so an interrupted purge can run again
        _remove(entry)
    out(f"purge {real}: deleted; the folder is empty")
    return 0


# ------------------------------------------------------------------------------------------- CLI

def _default_dest(given: str | None) -> Path:
    return Path(given or os.environ.get(DEST_ENV) or config.PROJECT_ROOT / "data" / "datapack")


def _parser() -> argparse.ArgumentParser:
    dest_help = (f"the data folder (default: ${DEST_ENV}, else "
                 f"{_shown(config.PROJECT_ROOT / 'data' / 'datapack')})")
    parser = argparse.ArgumentParser(
        prog="python -m candidate_screener.datapack",
        description="The Docker app's data folder (plan 2026-10-27-docker): the project files the "
                    "app reads, its workspace and its caches. File copies only; run by hand, "
                    "never at app start.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    create_ = commands.add_parser(
        "create", help="Fill the data folder from a source: copy the app's inputs, empty the "
                       "caches, keep the workspace.",
        description="Copy the app's inputs from a source (symbolic links followed) into the data "
                    "folder, replacing old ones; empty the caches; keep a workspace; write "
                    f"{MANIFEST}; then verify the folder fully. Exit 1 if verification finds a "
                    "problem; 2 if refused (nothing changed) or stopped by a file error (run "
                    "create again).")
    create_.add_argument("--source", metavar="PATH",
                         help=f"a data/ folder or a received handover copy (default: ${SOURCE_ENV}, "
                              f"else {_shown(config.DATA)}/)")
    create_.add_argument("--dest", metavar="PATH", help=dest_help)
    create_.add_argument("--with-workspace", action="store_true",
                         help="also copy the source's workspace (processed/hr-portal/: accounts, "
                              "reviews), private (0700 folders, 0600 files)")
    create_.add_argument("--replace-workspace", action="store_true",
                         help="with --with-workspace: replace a workspace the folder already holds")
    verify_ = commands.add_parser(
        "verify", help="Check the data folder against its manifest and what the code expects; "
                       "exit 1 on any problem.",
        description="Report every missing, changed, unexpected or stale input, every input the "
                    "code now expects and the folder lacks, and the workspace's format version. "
                    "The workspace and the caches are not inputs. Exit 0 when clean, 1 otherwise.")
    verify_.add_argument("--dest", metavar="PATH", help=dest_help)
    verify_.add_argument("--full", action="store_true",
                         help="also compare every input's SHA-256 (reads every input)")
    handover_ = commands.add_parser(
        "handover", help="Copy the inputs and the manifest (no workspace, no caches) to a new "
                         "folder, then verify the copy.",
        description="Make a copy of the data folder to pass on privately: its inputs and "
                    f"{MANIFEST} only, never the workspace or the caches; then verify the copy "
                    "fully. Its receiver runs `create --source DEST`.")
    handover_.add_argument("dest", metavar="DEST", help="a new or empty folder for the copy")
    handover_.add_argument("--from", dest="origin", metavar="PATH",
                           help=f"the data folder to copy (default: ${DEST_ENV}, else "
                                f"{_shown(config.PROJECT_ROOT / 'data' / 'datapack')})")
    purge_ = commands.add_parser(
        "purge", help="Empty the data folder, workspace included; without --yes, only list what "
                      "would be deleted.",
        description="Delete everything in the data folder: inputs, manifest, workspace and caches "
                    f"(for a breaking change). Refuses a folder without {MANIFEST} that is not "
                    "empty, and the project, its data/ and their parents.")
    purge_.add_argument("--dest", metavar="PATH", help=dest_help)
    purge_.add_argument("--yes", action="store_true", help="delete; without it nothing is deleted")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "create" and args.replace_workspace and not args.with_workspace:
        parser.error("--replace-workspace needs --with-workspace")
    try:
        if args.command == "create":
            given = args.source or os.environ.get(SOURCE_ENV) or str(config.DATA)
            return create(Path(given), _default_dest(args.dest), given, args.with_workspace,
                          args.replace_workspace)
        if args.command == "verify":
            return _print_report(check(_default_dest(args.dest), full=args.full), print)
        if args.command == "handover":
            return handover(Path(args.dest), _default_dest(args.origin))
        return purge(_default_dest(args.dest), yes=args.yes)
    except DatapackError as error:
        print(f"datapack {args.command}: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"datapack {args.command}: {error}"
              + ("; the folder may be partly updated: run create again" if args.command == "create" else ""),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
