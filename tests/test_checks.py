"""The check runner (plan 2026-10-08 W8): registry coverage, import closure, selection and
the scheduler. No data: the scheduler runs tiny `python -c` sleeps in place of checks."""
import re
import sys
import time
from pathlib import Path

import pytest

from candidate_screener import checks

PKG = Path(checks.__file__).resolve().parent


def _check_modules() -> set[str]:
    """Dotted modules (relative to the package) whose source defines a `--check` flag."""
    out = set()
    for f in PKG.rglob("*.py"):
        if f.name == "checks.py" and f.parent == PKG:
            continue
        if re.search(r"""["']--check["']""", f.read_text(encoding="utf-8")):
            out.add(".".join(f.relative_to(PKG).with_suffix("").parts))
    return out


def test_every_check_module_is_registered():
    found = _check_modules()
    assert "evaluation.indomain" in found and "data.profile" in found   # the scan works
    registered = {c.module for c in checks.REGISTRY}
    missing = found - registered - checks.EXCLUDED
    assert not missing, f"--check modules missing from checks.REGISTRY: {sorted(missing)}"
    assert not (checks.EXCLUDED & registered)


def test_registry_is_consistent():
    names = [c.name for c in checks.REGISTRY]
    assert len(names) == len(set(names))
    for c in checks.REGISTRY:
        assert checks.module_file(f"{checks.PACKAGE}.{c.module}") is not None, c.module
    assert {c.name for c in checks.REGISTRY if c.superseded} == {
        "indomain_v1", "retrieval", "retrieval_pilot", "a1_secondary"}


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sub" / "__init__.py").write_text("")
    (root / "a.py").write_text("from pkg import b\n")
    (root / "b.py").write_text("import os\n\ndef f():\n    from pkg.sub import deep\n")
    (root / "sub" / "deep.py").write_text("from . import leaf\nfrom ..c import X\n")
    (root / "sub" / "leaf.py").write_text("")
    (root / "c.py").write_text("X = 1\n")
    (root / "lonely.py").write_text("import pkg.a\n")
    return root


def test_import_closure_follows_function_level_and_relative_imports(tmp_path):
    root = _package(tmp_path)
    got = {p.relative_to(root).as_posix() for p in checks.import_closure("pkg.a", root, "pkg")}
    assert got == {"__init__.py", "a.py", "b.py", "sub/__init__.py", "sub/deep.py",
                   "sub/leaf.py", "c.py"}
    assert "lonely.py" not in got
    leaf = {p.relative_to(root).as_posix()
            for p in checks.import_closure("pkg.sub.leaf", root, "pkg")}
    assert leaf == {"__init__.py", "sub/__init__.py", "sub/leaf.py"}


def test_affected_maps_files_through_the_closure(tmp_path):
    root = _package(tmp_path)
    reg = (checks.Check("A", "a", ("--check",)), checks.Check("L", "lonely", ("--check",)),
           checks.Check("C", "c", ("--check",), superseded=True))
    hit = checks.affected(["pkg/sub/leaf.py"], reg, tmp_path, root, "pkg")
    assert set(hit) == {"A", "L"}
    hit = checks.affected(["pkg/c.py", "tests/x.py", "plan/p.md"], reg, tmp_path, root, "pkg")
    assert set(hit) == {"A", "L", "C"}                  # superseded C included when hit
    assert set(checks.affected(["output/m.json"], reg, tmp_path, root, "pkg")) == {"A", "L", "C"}
    assert set(checks.affected(["docs/data/manifests/x.csv"], reg, tmp_path, root,
                               "pkg")) == {"A", "L", "C"}
    assert checks.affected(["docs/decisions.md", "README.md"], reg, tmp_path, root, "pkg") == {}


def test_the_real_role_filter_selects_its_dependents():
    hit = checks.affected(["src/candidate_screener/matching/role_filter.py"])
    assert {"shortlist_dev", "shortlist_dev_variants", "shortlist_final"} <= set(hit)
    assert "final_set" not in hit


def test_selection_by_tier():
    warm = {c.name for c in checks.select("warm", all_=True)}
    cold = {c.name for c in checks.select("cold", all_=True)}
    assert cold == set(checks.BY_NAME)
    assert warm == cold - {"indomain_v1", "retrieval", "retrieval_pilot", "a1_secondary"}
    hits = {"a1_secondary": ["x"], "stage1": ["x"]}
    for tier in ("warm", "cold"):
        assert {c.name for c in checks.select(tier, hits=hits)} == {"a1_secondary", "stage1"}
    assert [c.name for c in checks.select("warm", names=["vocab"])] == ["vocab"]
    with pytest.raises(SystemExit):
        checks.select("warm", names=["nope"])


def test_warm_arguments_are_appended_only_in_the_warm_tier():
    c = checks.BY_NAME["extraction"]
    assert c.command_args("cold") == ("--check",)
    assert c.command_args("warm") == ("--check", "--warm")
    assert checks.BY_NAME["stage1"].command_args("warm") == ("--check",)


def _sleeper(path: Path):
    """A command builder: each check appends `start name` / `end name` to `path`."""
    def command(check, tier):
        code = (f"import time,sys; f=open({str(path)!r},'a'); f.write('start {check.name}\\n');"
                f"f.flush(); time.sleep({check.cold_s}); "
                f"open({str(path)!r},'a').write('end {check.name}\\n'); "
                f"sys.exit({1 if check.name.startswith('bad') else 0})")
        return [sys.executable, "-c", code]
    return command


def _concurrency(events: list[str]) -> list[set[str]]:
    live, snapshots = set(), []
    for e in events:
        kind, name = e.split()
        (live.add if kind == "start" else live.discard)(name)
        snapshots.append(set(live))
    return snapshots


def test_scheduler_respects_run_alone_and_jobs(tmp_path):
    reg = [checks.Check("s1", "m", (), cold_s=0.3), checks.Check("s2", "m", (), cold_s=0.3),
           checks.Check("s3", "m", (), cold_s=0.3), checks.Check("heavy", "m", (), cold_s=0.4,
                                                                 run_alone=True),
           checks.Check("bad", "m", (), cold_s=0.1)]
    events = tmp_path / "events"
    lines: list[str] = []
    t0 = time.monotonic()
    res = checks.run(reg, "cold", 2, tmp_path / "logs", command=_sleeper(events),
                     cwd=tmp_path, emit=lines.append, poll=0.02)
    snaps = _concurrency(events.read_text().splitlines())
    assert max(len(s) for s in snaps) == 2                      # -j 2 is used ...
    assert all(len(s) <= 2 for s in snaps)                      # ... and never exceeded
    assert all(s == {"heavy"} for s in snaps if "heavy" in s)   # run_alone runs alone
    assert {r.check.name: r.status for r in res} == {
        "s1": "PASS", "s2": "PASS", "s3": "PASS", "heavy": "PASS", "bad": "FAIL"}
    assert sum(not ln.startswith(" ") for ln in lines) == 5     # one line per check
    assert re.match(r"PASS  s1 +\d+\.\d s  logs/s1\.log$", next(ln for ln in lines
                                                                if " s1 " in ln))
    assert time.monotonic() - t0 < 1.4 + 1.5                     # parallel, with slack
    checks.write_summary(res, "cold", tmp_path / "logs")
    assert len((tmp_path / "logs" / "summary.tsv").read_text().splitlines()) == 6


def test_scheduler_with_one_job_is_serial(tmp_path):
    reg = [checks.Check(f"s{i}", "m", (), cold_s=0.1) for i in range(3)]
    events = tmp_path / "events"
    checks.run(reg, "cold", 1, tmp_path / "logs", command=_sleeper(events), cwd=tmp_path,
               emit=lambda s: None, poll=0.02)
    assert all(len(s) <= 1 for s in _concurrency(events.read_text().splitlines()))
