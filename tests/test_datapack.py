"""The Docker data folder commands (`candidate_screener.datapack`, plan 2026-10-27-docker, 04).

Synthetic trees under `tmp_path` only, no `data/`: a repository holding the committed copies, a
source `data/` whose `processed/` is a symbolic link (as in a worktree) and holds extra files the
folder must not get, and the expected list with the real names (derived from the code) but
synthetic contents and records. The last tests pin the derived names to the code that reads them.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import pickle
import re
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from candidate_screener import config, datapack

ROLE, NEW_ROLE, OLD_ROLE = "aaaaaaaaaaaa", "eeeeeeeeeeee", "bbbbbbbbbbbb"
PARTS, NEW_PARTS, OLD_PARTS = "cccccccccccc", "ffffffffffff", "dddddddddddd"
MANIFEST = datapack.MANIFEST


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data, mode=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)
    return path


def files_under(root) -> set[str]:
    """Files and links under `root`, relative, links not followed."""
    root = Path(root)
    found = set()
    for current, dirs, names in os.walk(root):
        found |= {(Path(current) / name).relative_to(root).as_posix() for name in names}
        found |= {(Path(current) / name).relative_to(root).as_posix() for name in dirs
                  if (Path(current) / name).is_symlink()}
    return found


def snapshot(root, skip=()) -> dict:
    """Every path under `root` except `skip`: kind, mode, size, mtime and content digest."""
    skip = [Path(path) for path in skip]
    state = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if Path(current) / name not in skip]
        for name in sorted(dirs + names):
            path = Path(current) / name
            info = path.lstat()
            if path.is_symlink():
                state[str(path)] = ("link", os.readlink(path))
            elif path.is_dir():
                state[str(path)] = ("dir", info.st_mode, info.st_mtime_ns)
            else:
                state[str(path)] = ("file", info.st_mode, info.st_size, info.st_mtime_ns, sha(path))
    return state


def mode(path) -> int:
    return stat.S_IMODE(Path(path).stat().st_mode)


def run(capsys, *argv) -> tuple[int, str]:
    code = datapack.main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def ok(capsys, *argv) -> str:
    code, out = run(capsys, *argv)
    assert code == 0, out
    return out


@dataclasses.dataclass
class World:
    tmp: Path
    repo: Path
    store: Path        # where the source's processed/ link points
    source: Path
    dest: Path
    exp: datapack.Expected
    inputs: set        # what create must copy, relative

    def manifest(self, folder=None) -> dict:
        return json.loads(((folder or self.dest) / MANIFEST).read_text(encoding="utf-8"))

    def pointer(self, folder_target: bool) -> tuple[str, datapack.Pointer]:
        return next((rel, p) for rel, p in self.exp.pointers.items() if bool(p.needs) == folder_target)

    def use(self, **changes) -> None:
        """What the code now expects (the fixture's `datapack.expected` returns `self.exp`)."""
        self.exp = dataclasses.replace(self.exp, **changes)


@pytest.fixture(scope="session")
def real_expected():
    return datapack.expected()


def _write_parts(world: World, latest: str, pointer, value: str, role: str) -> None:
    folder = Path(latest).parent / pointer.target.format(value)
    for name in pointer.needs:
        write(world.source / folder / name, f"part {name} {value}\n")
    name, fields = pointer.provenance
    recorded = {field: role if ref in world.exp.pointers else sha(world.repo / ref)
                for field, ref in fields.items()}
    write(world.source / folder / name, json.dumps(recorded))
    write(world.source / latest, value + "\n")


@pytest.fixture
def world(tmp_path, monkeypatch, real_expected) -> World:
    real = real_expected
    repo, store, source = tmp_path / "repo", tmp_path / "store", tmp_path / "source"
    (store / "processed").mkdir(parents=True)
    source.mkdir()
    (source / "processed").symlink_to(store / "processed", target_is_directory=True)
    write(source / "README.md", "not an input\n")
    write(source / "raw" / "djinni.csv", "not an input\n")
    write(source / "processed" / "extraction" / "concept-cache" / "x.json", "{}")
    inputs = set()
    for rel in real.committed:
        write(repo / rel, f"committed copy of {rel}\n")
        write(source / rel, f"the source's own copy of {rel}, never used\n")
        inputs.add(rel)
    for index, rel in enumerate(real.files):
        write(source / rel, f"source bytes of {rel}\n" * (index + 2))
        inputs.add(rel)
    for folder, (_, needs) in real.folders.items():
        for name in needs:
            write(source / folder / name, f"{folder}/{name}\n")
        write(source / folder / "1_Pooling" / "config.json", "{}")
        (source / folder / "2_Normalize").mkdir()
        write(tmp_path / "blobs" / "weights.bin", b"\x00weights" * 100)
        (source / folder / "model.safetensors").symlink_to(tmp_path / "blobs" / "weights.bin")
        write(source / f"{folder}-work" / "checkpoint.bin", "training leftovers")
        inputs |= {f"{folder}/{name}" for name in (*needs, "1_Pooling/config.json", "model.safetensors")}
    pointers = {latest: dataclasses.replace(p, record=ROLE if p.record is not None else None)
                for latest, p in real.pointers.items()}
    exp = dataclasses.replace(
        real, committed={rel: repo / rel for rel in real.committed}, pointers=pointers,
        digests={rel: (sha(repo / rel), f"the test repository's {rel}") for rel in real.committed})
    built = World(tmp_path, repo, store, source, tmp_path / "pack", exp, inputs)
    for latest, pointer in pointers.items():
        folder = Path(latest).parent
        if pointer.needs:
            _write_parts(built, latest, pointer, OLD_PARTS, ROLE)
            _write_parts(built, latest, pointer, PARTS, ROLE)
            inputs |= {(folder / PARTS / name).as_posix() for name in pointer.needs}
        else:
            write(source / folder / pointer.target.format(OLD_ROLE), b"old model")
            write(source / folder / f"training-{ROLE}.json", "{}")
            write(source / folder / pointer.target.format(ROLE), b"role model " * 50)
            write(source / latest, ROLE + "\n")
            inputs.add((folder / pointer.target.format(ROLE)).as_posix())
        inputs.add(latest)
    workspace = source / exp.workspace
    write(workspace / "accounts.json", json.dumps({"version": 1, "users": {}}), mode=0o600)
    database = workspace / "workspace.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE portal_store_migrations(version INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO portal_store_migrations VALUES(2)")
    db.close()
    os.chmod(database, 0o600)
    write(workspace / "entities" / "cache.json", "{}", mode=0o644)      # widened: create narrows it
    for cache in exp.caches:
        write(source / cache / "cached.bin", b"cache")
    monkeypatch.setattr(datapack, "expected", lambda: built.exp)
    return built


def fill_workspace_and_caches(folder: Path, exp) -> None:
    write(folder / exp.workspace / "accounts.json", json.dumps({"version": 1, "users": {}}), mode=0o600)
    write(folder / exp.workspace / "drafts" / "jd.json", "{}", mode=0o600)
    for cache in exp.caches:
        write(folder / cache / "fingerprint" / "scores-0.parquet", b"scores")


# ------------------------------------------------------------------------------------- create

def test_create_copies_exactly_the_listed_inputs_following_symlinks(world, capsys):
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 0, out
    assert files_under(world.dest) == world.inputs | {MANIFEST}
    for rel in world.inputs:
        copy = world.dest / rel
        assert copy.is_file() and not copy.is_symlink(), rel
        origin = world.repo / rel if rel in world.exp.committed else world.source / rel
        assert copy.read_bytes() == origin.read_bytes(), rel
        assert mode(copy) == 0o644, rel
    folders = [Path(current) for current, _, _ in os.walk(world.dest)]
    assert all(mode(folder) == 0o755 for folder in folders)
    sbert = next(iter(world.exp.folders))
    assert (world.dest / sbert / "2_Normalize").is_dir()          # an empty folder of the model
    assert "OK: " in out


def test_create_never_modifies_the_source(world, capsys):
    before = snapshot(world.tmp, skip=[world.dest])
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    fill_workspace_and_caches(world.dest, world.exp)
    ok(capsys, "create", "--source", world.source, "--dest", world.dest,
               "--with-workspace", "--replace-workspace")
    assert snapshot(world.tmp, skip=[world.dest]) == before


def test_create_replaces_old_inputs_empties_caches_and_keeps_the_workspace(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    fill_workspace_and_caches(world.dest, world.exp)
    workspace = snapshot(world.dest / world.exp.workspace)
    # The source moves on: a retrained role model and parts refitted with it.
    role_latest, role = world.pointer(folder_target=False)
    parts_latest, parts = world.pointer(folder_target=True)
    write(world.source / Path(role_latest).parent / role.target.format(NEW_ROLE), b"new model")
    write(world.source / role_latest, NEW_ROLE + "\n")
    _write_parts(world, parts_latest, parts, NEW_PARTS, NEW_ROLE)
    world.use(pointers={**world.exp.pointers,
                                     role_latest: dataclasses.replace(role, record=NEW_ROLE)})

    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 0, out
    old_model = Path(role_latest).parent / role.target.format(ROLE)
    old_parts = Path(parts_latest).parent / PARTS
    assert not (world.dest / old_model).exists() and not (world.dest / old_parts).exists()
    assert (world.dest / Path(role_latest).parent / role.target.format(NEW_ROLE)).is_file()
    assert (world.dest / Path(parts_latest).parent / NEW_PARTS / "parts.json").is_file()
    listed = {item["path"] for item in world.manifest()["inputs"]}
    assert old_model.as_posix() not in listed and not any(p.startswith(old_parts.as_posix()) for p in listed)
    for cache in world.exp.caches:
        assert files_under(world.dest / cache) == set(), cache
    assert snapshot(world.dest / world.exp.workspace) == workspace
    assert "kept" in out


def test_with_workspace_copies_it_with_private_permissions(world, capsys):
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    assert code == 0, out
    origin, copy = world.source / world.exp.workspace, world.dest / world.exp.workspace
    assert files_under(copy) == files_under(origin) and files_under(copy)
    for rel in files_under(origin):
        assert (copy / rel).read_bytes() == (origin / rel).read_bytes()
        assert mode(copy / rel) == 0o600, rel
    assert all(mode(folder) == 0o700 for folder, _, _ in os.walk(copy))
    assert "accounts.json format 1 (the code: 1)" in out


def test_with_workspace_refuses_an_existing_workspace_unless_replaced(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    theirs = write(world.dest / world.exp.workspace / "accounts.json", '{"version": 1, "users": {"x": {}}}')
    before = snapshot(world.dest)
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    assert code == 2 and "--replace-workspace" in out
    assert snapshot(world.dest) == before
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest,
                    "--with-workspace", "--replace-workspace")
    assert code == 0, out
    assert theirs.read_bytes() == (world.source / world.exp.workspace / "accounts.json").read_bytes()
    with pytest.raises(SystemExit):
        datapack.main(["create", "--source", str(world.source), "--dest", str(world.dest), "--replace-workspace"])


def test_manifest_is_complete(world, capsys, monkeypatch):
    monkeypatch.chdir(world.tmp)
    started = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    ok(capsys, "create", "--source", "source", "--dest", "pack")
    manifest = world.manifest()
    assert set(manifest) == {"format", "created", "source", "source_commit", "fingerprints", "latest",
                             "inputs", "empty_directories"}
    assert manifest["format"] == datapack.FORMAT and manifest["source"] == "source"
    created = dt.datetime.strptime(manifest["created"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    assert started <= created <= dt.datetime.now(dt.timezone.utc)
    assert manifest["source_commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"])
    assert manifest["fingerprints"] == world.exp.fingerprints
    assert manifest["latest"] == {rel: PARTS if p.needs else ROLE for rel, p in world.exp.pointers.items()}
    assert {item["path"] for item in manifest["inputs"]} == world.inputs
    for item in manifest["inputs"]:
        copy = world.dest / item["path"]
        assert item["bytes"] == copy.stat().st_size and item["sha256"] == sha(copy)
        assert item["origin"] == ("committed" if item["path"] in world.exp.committed else "copied")
    assert manifest["empty_directories"] == [f"{folder}/2_Normalize" for folder in world.exp.folders]


def test_defaults_come_from_the_tools_settings(world, capsys, monkeypatch):
    monkeypatch.setenv(datapack.SOURCE_ENV, str(world.source))
    monkeypatch.setenv(datapack.DEST_ENV, str(world.dest))
    ok(capsys, "create")
    assert world.manifest()["source"] == str(world.source)
    ok(capsys, "verify")


def test_a_committed_copy_inside_the_folder_comes_from_the_source(world, capsys, real_expected):
    """In the tools container the folder is mounted over data/, so data/vocab/ is the folder itself."""
    from candidate_screener.extraction import rules
    vocabulary = next(rel for rel, path in real_expected.committed.items() if path == rules.VOCAB_CSV)
    world.use(committed={**world.exp.committed, vocabulary: world.dest / vocabulary})
    write(world.source / vocabulary, (world.repo / vocabulary).read_bytes())   # the committed bytes
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 0, out
    entry = next(item for item in world.manifest()["inputs"] if item["path"] == vocabulary)
    assert entry["origin"] == "copied"
    assert (world.dest / vocabulary).read_bytes() == (world.source / vocabulary).read_bytes()
    write(world.source / vocabulary, "another vocabulary\n")       # not what the code records
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 1 and f"stale: {vocabulary} differs from the test repository's {vocabulary}" in out


def test_create_refuses_unsafe_destinations(world, capsys, monkeypatch):
    documents = write(world.tmp / "documents" / "notes.txt", "mine").parent
    code, out = run(capsys, "create", "--source", world.source, "--dest", documents)
    assert code == 2 and "refusing" in out and files_under(documents) == {"notes.txt"}
    inner = world.dest / "inner"
    inner.mkdir(parents=True)
    assert run(capsys, "create", "--source", inner, "--dest", world.dest)[0] == 2
    data = world.tmp / "project" / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_ROOT", data.parent)
    monkeypatch.setattr(config, "DATA", data)
    monkeypatch.delenv(datapack.DEST_ENV, raising=False)
    for refused in (data, data.parent, world.tmp):
        assert run(capsys, "create", "--source", world.source, "--dest", refused)[0] == 2, refused
    assert not (data / "processed").exists()
    monkeypatch.setenv(datapack.DEST_ENV, str(data))            # the tools container: /app/data
    ok(capsys, "create", "--source", world.source)
    assert (data / MANIFEST).is_file()


def test_create_refuses_a_symbolic_link_inside_the_folder(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()
    folder = world.dest / next(iter(world.exp.files)).rpartition("/")[0]
    folder.rename(world.tmp / "moved")
    folder.symlink_to(elsewhere, target_is_directory=True)
    assert run(capsys, "create", "--source", world.source, "--dest", world.dest)[0] == 2
    assert list(elsewhere.iterdir()) == []


def test_create_refuses_a_source_that_lacks_an_input(world, capsys):
    missing = next(iter(world.exp.files))
    (world.source / missing).unlink()
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 2 and missing in out and not world.dest.exists()


# ------------------------------------------------------------------------------------- verify

def test_verify_passes_a_clean_folder(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    fill_workspace_and_caches(world.dest, world.exp)
    for flags in ((), ("--full",)):
        code, out = run(capsys, "verify", "--dest", world.dest, *flags)
        assert code == 0 and f"OK: {len(world.inputs)} inputs" in out, out


def test_verify_reports_missing_changed_and_unexpected_files(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    fill_workspace_and_caches(world.dest, world.exp)           # neither is an input: never reported
    gone, grown, flipped = sorted(world.exp.files)[:3]
    (world.dest / gone).unlink()
    with open(world.dest / grown, "ab") as stream:
        stream.write(b"more")
    content = bytearray((world.dest / flipped).read_bytes())
    content[0] ^= 1
    (world.dest / flipped).write_bytes(bytes(content))
    write(world.dest / "processed" / "matching" / "stray.parquet", b"stray")
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 1
    assert f"missing: {gone}" in out and f"changed: {grown}" in out
    assert "unexpected: processed/matching/stray.parquet" in out
    assert flipped not in out                                   # same size: only --full sees it
    assert world.exp.workspace not in out.replace(f"workspace {world.exp.workspace}/:", "")
    assert not any(cache in out for cache in world.exp.caches)
    code, out = run(capsys, "verify", "--dest", world.dest, "--full")
    assert code == 1 and f"changed: {flipped} (its SHA-256 differs" in out


def test_verify_reports_inputs_the_code_now_expects(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    (old, (reader, needs)), = world.exp.folders.items()
    new = old.rpartition("-")[0] + "-0123456789ab"
    role_latest, role = world.pointer(folder_target=False)
    world.use(folders={new: (reader, needs)},
              pointers={**world.exp.pointers, role_latest: dataclasses.replace(role, record=NEW_ROLE)})
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 1
    assert f"the code expects {new}/{needs[0]}" in out and f"stale: {old}/{needs[0]} is no longer read" in out
    assert f"stale: {role_latest} names {ROLE}; the code's committed record names {NEW_ROLE}" in out


def test_verify_reports_stale_committed_copies_and_parts_fitted_for_another_folder(world, capsys):
    parts_latest, parts = world.pointer(folder_target=True)
    _write_parts(world, parts_latest, parts, PARTS, OLD_ROLE)    # fitted with another role model
    code, out = run(capsys, "create", "--source", world.source, "--dest", world.dest)
    assert code == 1 and f"records role_model {OLD_ROLE}, but the folder's" in out
    committed = next(iter(world.exp.committed))
    write(world.repo / committed, "a newer commit\n")
    world.use(digests={**world.exp.digests, committed: (sha(world.repo / committed), "the new commit")})
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 1 and f"stale: {committed} differs from the new commit" in out


def test_verify_reports_the_workspace_format(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 0 and "accounts.json format 1 (the code: 1)" in out and "workspace.sqlite format 2" in out
    write(world.dest / world.exp.workspace / "accounts.json", json.dumps({"version": 9, "users": {}}))
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 1 and "accounts.json has format 9; the code reads 1" in out


def test_verify_fails_without_a_manifest(world, capsys):
    world.dest.mkdir()
    code, out = run(capsys, "verify", "--dest", world.dest)
    assert code == 1 and f"no {MANIFEST}" in out


# ----------------------------------------------------------------------------------- handover

def test_handover_copies_the_inputs_and_manifest_only_and_verifies(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    fill_workspace_and_caches(world.dest, world.exp)
    copy = world.tmp / "handover"
    code, out = run(capsys, "handover", copy, "--from", world.dest)
    assert code == 0 and "OK: " in out, out
    assert files_under(copy) == world.inputs | {MANIFEST}
    assert not (copy / world.exp.workspace).exists()
    assert not any((copy / cache).exists() for cache in world.exp.caches)
    assert (copy / MANIFEST).read_bytes() == (world.dest / MANIFEST).read_bytes()
    assert all((copy / rel).is_dir() for rel in world.manifest()["empty_directories"])
    received = world.tmp / "received"                           # the receiver's create
    ok(capsys, "create", "--source", copy, "--dest", received)


def test_handover_refuses_a_used_destination_or_one_inside_the_folder(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    used = write(world.tmp / "used" / "keep.txt", "keep").parent
    assert run(capsys, "handover", used, "--from", world.dest)[0] == 2
    assert files_under(used) == {"keep.txt"}
    assert run(capsys, "handover", world.dest / "copy", "--from", world.dest)[0] == 2
    assert not (world.dest / "copy").exists()


def test_handover_fails_when_its_copy_does_not_verify(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest)
    target = world.dest / next(iter(world.exp.files))
    content = bytearray(target.read_bytes())
    content[-2] ^= 1
    target.write_bytes(bytes(content))
    code, out = run(capsys, "handover", world.tmp / "handover", "--from", world.dest)
    assert code == 1 and "its SHA-256 differs" in out


# -------------------------------------------------------------------------------------- purge

def test_purge_without_yes_deletes_nothing(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    fill_workspace_and_caches(world.dest, world.exp)
    before = snapshot(world.dest)
    code, out = run(capsys, "purge", "--dest", world.dest)
    assert code == 0 and "would delete" in out and "nothing deleted" in out
    assert snapshot(world.dest) == before


def test_purge_with_yes_empties_the_folder(world, capsys):
    ok(capsys, "create", "--source", world.source, "--dest", world.dest, "--with-workspace")
    fill_workspace_and_caches(world.dest, world.exp)
    code, out = run(capsys, "purge", "--dest", world.dest, "--yes")
    assert code == 0, out
    assert world.dest.is_dir() and list(world.dest.iterdir()) == []


def test_purge_refuses_a_folder_that_is_not_a_data_folder(world, capsys):
    other = write(world.tmp / "other" / "keep.txt", "keep").parent
    assert run(capsys, "purge", "--dest", other, "--yes")[0] == 2
    assert files_under(other) == {"keep.txt"}
    assert run(capsys, "purge", "--dest", world.tmp / "absent", "--yes")[0] == 2


def test_purge_refuses_the_project_its_data_and_their_parents(world, capsys, monkeypatch):
    project = world.tmp / "home" / "project"
    data = project / "data"
    guarded = (data, project, project.parent, world.tmp)
    for folder in (*guarded, data / "datapack"):                 # each looks like a data folder
        write(folder / MANIFEST, "{}")
        write(folder / "keep.txt", "keep")
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    monkeypatch.setattr(config, "DATA", data)
    monkeypatch.delenv(datapack.DEST_ENV, raising=False)
    for folder in guarded:
        code, out = run(capsys, "purge", "--dest", folder, "--yes")
        assert code == 2 and "refusing" in out, folder
        assert (folder / "keep.txt").is_file() and (folder / MANIFEST).is_file()
    ok(capsys, "purge", "--dest", data / "datapack", "--yes")
    assert list((data / "datapack").iterdir()) == []
    monkeypatch.setenv(datapack.DEST_ENV, str(data))            # the tools container: /app/data
    assert run(capsys, "purge", "--dest", project, "--yes")[0] == 2
    ok(capsys, "purge", "--yes")
    assert list(data.iterdir()) == []


# ------------------------------------------------------------------------------- the whole run

def test_no_command_writes_outside_its_destination(world, capsys):
    copy = world.tmp / "handover"
    steps = [("create", "--source", world.source, "--dest", world.dest, "--with-workspace"),
             ("verify", "--dest", world.dest, "--full"),
             ("handover", copy, "--from", world.dest),
             ("purge", "--dest", world.dest),
             ("purge", "--dest", world.dest, "--yes")]
    for step in steps:
        before = snapshot(world.tmp, skip=[world.dest, copy])
        ok(capsys, *step)
        assert snapshot(world.tmp, skip=[world.dest, copy]) == before, step


PROBE = r"""
import json, sys
from pathlib import Path
HEAVY = {"torch", "sentence_transformers", "transformers"}
attempts = []

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HEAVY:
            attempts.append(name)
            raise ImportError("blocked by the test: " + name)

sys.meta_path.insert(0, Block())
from candidate_screener import datapack
exp = datapack.expected()
datapack._layout(exp, Path(sys.argv[1]))
datapack.check(Path(sys.argv[1]), full=True, exp=exp)
print(json.dumps({"attempts": attempts,
                  "loaded": sorted(m for m in sys.modules if m.partition(".")[0] in HEAVY)}))
"""


def test_importing_and_listing_import_no_torch(tmp_path):
    """The tools run in the main environment: no torch, sentence-transformers or transformers."""
    result = subprocess.run([sys.executable, "-c", PROBE, str(tmp_path)], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"attempts": [], "loaded": []}


# ------------------------------------------------- the derived names are the files the code reads

def test_a2_names_are_the_files_corpus_reads(tmp_path, monkeypatch, real_expected):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from candidate_screener.annotation import sample
    from candidate_screener.matching import corpus
    cache = datapack._rel(corpus.CACHE)
    names = [rel.rpartition("/")[2] for rel in real_expected.files if rel.rpartition("/")[0] == cache]

    def refuse(*args, **kwargs):
        raise AssertionError("corpus would build the cache from data/: the derived name is not its name")

    monkeypatch.setattr(corpus, "CACHE", tmp_path)
    monkeypatch.setattr(corpus, "_djinni", refuse)
    monkeypatch.setattr(sample, "load_english", refuse)
    for name in names:
        pd.DataFrame({"marker": [name]}).to_parquet(tmp_path / name, index=False)
    pairs = corpus.a2_pairs.__wrapped__(datapack.TEXT_VARIANT)
    jd, cv = corpus.a2_train_documents.__wrapped__(text_variant=datapack.TEXT_VARIANT)
    read = {pairs.marker[0], jd.marker[0], cv.marker[0]}
    assert len(read) == 3 and read <= set(names)


def test_review_source_text_is_a_file_the_list_holds(tmp_path, monkeypatch, real_expected):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from candidate_screener.app import portal_data
    monkeypatch.setattr(portal_data, "PROCESSED", tmp_path / "processed")
    listed = [rel for rel in real_expected.files if "a2-pairs" in rel]
    for index, rel in enumerate(listed):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"jd_id": [f"jd{index}"], "cv_id": [f"cv{index}"], "jd_text": ["j"],
                      "cv_text": ["c"]}).to_parquet(path, index=False)
    documents = portal_data.load_source_documents()
    assert documents["jd"] and set(documents["jd"]) <= {f"jd{index}" for index in range(len(listed))}


def test_role_model_file_is_the_one_load_role_model_opens(tmp_path, monkeypatch, real_expected):
    from candidate_screener.extraction import role_family
    latest, pointer = next((rel, p) for rel, p in real_expected.pointers.items() if not p.needs)
    assert latest == datapack._rel(role_family.CACHE / "LATEST")
    assert pointer.record == json.loads(role_family.RECORD.read_text(encoding="utf-8"))["fingerprint"]
    monkeypatch.setattr(role_family, "CACHE", tmp_path)
    monkeypatch.setattr(role_family, "_LOADED", {})
    (tmp_path / "LATEST").write_text("abcdef123456\n", encoding="utf-8")
    with open(tmp_path / pointer.target.format("abcdef123456"), "wb") as stream:
        pickle.dump({"kind": "tfidf", "labels": ["x"], "aliases": {}, "sides": {},
                     "fingerprint": "abcdef123456"}, stream)
    assert role_family.load_role_model().fingerprint == "abcdef123456"


def test_the_rest_of_the_list_comes_from_the_modules_that_read_it(real_expected):
    from candidate_screener.app import portal_store
    from candidate_screener.inference import parts
    from candidate_screener.matching import sbert
    from candidate_screener.rerank import cross
    assert datapack._rel(sbert.WEIGHTS / sbert.SbertWeak.cache_key("v2")) in real_expected.folders
    parts_latest = datapack._rel(parts.ROOT / "LATEST")
    assert real_expected.pointers[parts_latest].needs == ("parts.json", *parts.FILES)
    assert real_expected.workspace == datapack._rel(portal_store.DEFAULT_PATH.parent)
    assert set(real_expected.caches) == {datapack._rel(sbert.EMBED_CACHE), datapack._rel(cross.ROOT)}
    assert all(rel.startswith(("processed/", "vocab/")) for rel in
               (*real_expected.committed, *real_expected.files, *real_expected.folders, *real_expected.pointers))
