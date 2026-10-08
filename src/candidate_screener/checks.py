"""One runner for the committed-record checks (plan 2026-10-08 W8, D55).

    uv run python -m candidate_screener.checks [--warm | --cold]
                                               [--all | --affected [REF] | NAME ...]
                                               [-j N] [--logs DIR] [--list] [--explain]

Tiers (D55). **Cold** runs each check's plain `--check`: it retrains and re-infers, and is
the reproducibility proof (mandatory before a commit touching model or training code,
before a plan is marked complete, and before a final-set score). **Warm** (the default)
adds each check's `--warm` where it has one: the cached NER tagger, the cached role-family
model, the cached dev pools and the persistent cross-encoder score store. A check without
a warm mode runs its cold `--check` in both tiers.

Selection. `--all --warm` runs every check but the superseded records (indomain v1,
`retrieval`, `retrieval_pilot`, `a1_secondary`; Q3); `--all --cold` runs everything.
`--affected [REF]` (the default, REF = `HEAD`, i.e. uncommitted work) takes the files of
`git diff --name-only REF` plus untracked files, and selects a check when one of them is
in the static import closure of the check's module within `candidate_screener`, or is a
file every check may read (`output/`, `docs/data/`, `pyproject.toml`, `uv.lock`, a non-`.py`
file in the package), which selects all. Superseded checks are included when affected.
`--affected` is a convenience: any doubt means `--all`.

Execution. Each check is a subprocess (`sys.executable -m <module> <argv>`, from the
repository root, with the parent's environment unchanged) writing stdout and stderr to
`<logs>/<name>.log`. At most `-j` run at once; a `run_alone` check (peak RSS >= 4.5 GB)
starts only when nothing runs, and nothing starts beside it. Longest first. One stdout
line per finished check, one summary line, and `<logs>/summary.tsv`. Exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from candidate_screener.config import PROJECT_ROOT

PACKAGE = "candidate_screener"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS = Path("logs") / "checks"
#: Modules with a `--check` flag that are not committed-record checks.
#: `data.profile --check NAME…` re-profiles raw corpora against `profile-metrics.json`
#: assertions; it is a profiler run on demand when a source changes, not a record rebuild.
EXCLUDED = {"data.profile"}
#: Changed paths (prefixes) that any check may read: a change selects every check.
READ_BY_ALL = ("output/", "docs/data/", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Check:
    name: str
    module: str                       # relative to `candidate_screener`
    argv: tuple[str, ...]             # the cold command line
    warm: tuple[str, ...] | None = None   # appended in the warm tier; None: warm == cold
    run_alone: bool = False           # peak RSS >= 4.5 GB: never shares the machine
    superseded: bool = False          # Q3: cold tier, or when --affected hits it
    cold_s: float = 0.0               # last measured wall, cold (for --list and ordering)
    warm_s: float = 0.0               # last measured wall, warm

    def command_args(self, tier: str) -> tuple[str, ...]:
        return self.argv + (self.warm if tier == "warm" and self.warm else ())

    def seconds(self, tier: str) -> float:
        return self.warm_s if tier == "warm" and self.warm_s else self.cold_s


W = ("--warm",)
#: Seconds: measured 8 Oct 2026 by `checks --cold --all` and `--warm --all` at -j 2 (two
#: checks share 22 cores, so a parallel figure can exceed a serial one); the superseded warm
#: figures by `checks --warm -j 1`. `run_alone` = measured peak RSS >= 4.5 GB in that cold run
#: (role_family 4.9, shortlist_dev 6.2, dev_pools 4.8, scoring_batch 4.7; shortlist_dev
#: --variants peaked at 4.05 and so may share the machine).
REGISTRY: tuple[Check, ...] = (
    Check("verify_derived", "data.verify", ("--derived",), cold_s=17.8),
    Check("baselines", "baselines.run", ("--check",), cold_s=14.6),
    Check("weak_pairs", "data.weak_pairs", ("--check",), cold_s=26.2),
    Check("final_set", "evaluation.final_set", ("--check",), cold_s=8.2),
    Check("indomain_v1", "evaluation.indomain", ("--check",), W, superseded=True,
          cold_s=251.6, warm_s=50.4),
    Check("indomain_v2", "evaluation.indomain", ("--check", "--text-variant", "v2"),
          cold_s=31.0),
    Check("ordering", "evaluation.ordering", ("--check",), cold_s=48.0),
    Check("a1_secondary", "evaluation.a1_secondary", ("--check",), W, superseded=True,
          cold_s=795.0, warm_s=32.2),
    Check("retrieval", "evaluation.retrieval", ("--check",), superseded=True, cold_s=67.8),
    Check("retrieval_pilot", "evaluation.retrieval_pilot", ("--check",), W, superseded=True,
          cold_s=549.7, warm_s=66.3),
    Check("stage1", "evaluation.stage1", ("--check",), cold_s=208.2),
    Check("vocab", "extraction.vocab", ("--check",), cold_s=3.8),
    Check("extraction", "extraction.evaluate", ("--check",), W, cold_s=886.3, warm_s=218.2),
    Check("experience_eval", "extraction.experience_eval", ("--check",), cold_s=8.8),
    Check("title_profile", "extraction.title_profile", ("--check",), cold_s=56.6),
    Check("role_family", "extraction.role_family", ("--check",), W, run_alone=True,
          cold_s=745.1, warm_s=44.4),
    Check("scoring_batch", "rerank.scoring_batch", ("--check",), run_alone=True, cold_s=37.0),
    Check("dev_pools", "evaluation.dev_pools", ("--check",), W, run_alone=True,
          cold_s=76.5, warm_s=40.2),
    Check("shortlist_dev", "evaluation.shortlist_dev", ("--check",), run_alone=True,
          cold_s=289.4),
    Check("shortlist_dev_variants", "evaluation.shortlist_dev", ("--variants", "--check"),
          cold_s=284.2),
    Check("shortlist_final", "evaluation.shortlist_final", ("--check",), cold_s=160.3),
)
BY_NAME = {c.name: c for c in REGISTRY}


# --- import closure -------------------------------------------------------------------

def module_file(dotted: str, root: Path = PACKAGE_DIR, package: str = PACKAGE) -> Path | None:
    """The file of `package.a.b` (`a/b.py` or `a/b/__init__.py`), or None if not ours."""
    if dotted != package and not dotted.startswith(package + "."):
        return None
    parts = dotted.split(".")[1:]
    base = root.joinpath(*parts)
    for f in (base.with_suffix(".py") if parts else None, base / "__init__.py"):
        if f is not None and f.is_file():
            return f
    return None


def _with_parents(dotted: str, package: str) -> list[str]:
    """`p.a.b` -> [`p`, `p.a`, `p.a.b`]: importing a submodule runs its packages."""
    parts = dotted.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)
            if ".".join(parts[:i]) == package or ".".join(parts[:i]).startswith(package + ".")]


def _imports(path: Path, dotted: str, package: str) -> set[str]:
    """Every `candidate_screener` module named by an import anywhere in `path` (function
    bodies included), with `from x import y` counted as `x.y` too in case y is a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_pkg = path.name == "__init__.py"
    here = dotted.split(".") if is_pkg else dotted.split(".")[:-1]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[:len(here) - (node.level - 1)] if node.level > 1 else here
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names if a.name != "*")
    return {m for m in out if m == package or m.startswith(package + ".")}


def import_closure(dotted: str, root: Path = PACKAGE_DIR,
                   package: str = PACKAGE) -> set[Path]:
    """Files under `root` reachable from module `dotted` by static imports, transitively."""
    seen: set[Path] = set()
    todo = list(_with_parents(dotted, package))
    names_done: set[str] = set()
    while todo:
        name = todo.pop()
        if name in names_done:
            continue
        names_done.add(name)
        f = module_file(name, root, package)
        if f is None or f in seen:
            continue
        seen.add(f)
        for m in _imports(f, name, package):
            todo.extend(_with_parents(m, package))
    return seen


# --- selection ------------------------------------------------------------------------

def changed_files(ref: str = "HEAD", cwd: Path = PROJECT_ROOT) -> list[str]:
    """`git diff --name-only REF` plus untracked files, relative to the repository root."""
    def git(*a: str) -> list[str]:
        out = subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                             text=True).stdout
        return [line for line in out.splitlines() if line]
    return sorted(set(git("diff", "--name-only", ref))
                  | set(git("ls-files", "--others", "--exclude-standard")))


def affected(changed: Iterable[str], registry: Sequence[Check] = REGISTRY,
             project_root: Path = PROJECT_ROOT, root: Path = PACKAGE_DIR,
             package: str = PACKAGE) -> dict[str, list[str]]:
    """`{check name: [changed files that select it]}`, for checks with at least one."""
    pkg_rel = root.resolve().relative_to(project_root.resolve()).as_posix() + "/"
    closures: dict[str, set[str]] = {}
    hits: dict[str, list[str]] = {}
    for f in changed:
        everyone = (any(f == p or f.startswith(p) for p in READ_BY_ALL)
                    or (f.startswith(pkg_rel) and not f.endswith(".py")))
        for c in registry:
            if not everyone:
                if c.module not in closures:
                    closures[c.module] = {
                        p.resolve().relative_to(project_root.resolve()).as_posix()
                        for p in import_closure(f"{package}.{c.module}", root, package)}
                if f not in closures[c.module]:
                    continue
            hits.setdefault(c.name, []).append(f)
    return hits


def select(tier: str, names: Sequence[str] = (), all_: bool = False,
           hits: dict[str, list[str]] | None = None,
           registry: Sequence[Check] = REGISTRY) -> list[Check]:
    """Named checks; or `--all` (warm skips superseded); or the affected ones (any tier)."""
    if names:
        by = {c.name: c for c in registry}
        unknown = [n for n in names if n not in by]
        if unknown:
            raise SystemExit(f"unknown check(s): {', '.join(unknown)}; see --list")
        return [by[n] for n in names]
    if all_:
        return [c for c in registry if tier == "cold" or not c.superseded]
    return [c for c in registry if c.name in (hits or {})]


# --- execution ------------------------------------------------------------------------

def default_command(check: Check, tier: str) -> list[str]:
    return [sys.executable, "-m", f"{PACKAGE}.{check.module}", *check.command_args(tier)]


@dataclass
class Result:
    check: Check
    status: str
    seconds: float
    log: Path
    peak_rss_gb: float


def _tail(path: Path, n: int = 5) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-n:]


def run(checks: Sequence[Check], tier: str, jobs: int, logs: Path,
        command: Callable[[Check, str], list[str]] = default_command,
        cwd: Path = PROJECT_ROOT, emit: Callable[[str], None] | None = None,
        poll: float = 0.2) -> list[Result]:
    """Run `checks` under the scheduler; return results in finishing order."""
    if emit is None:
        def emit(line: str) -> None:                 # flushed: progress shows when piped
            print(line, flush=True)
    logs.mkdir(parents=True, exist_ok=True)
    pending = sorted(checks, key=lambda c: -c.seconds(tier))
    running: dict[int, tuple[Check, float, Path, object]] = {}
    results: list[Result] = []
    width = max((len(c.name) for c in checks), default=0)
    while pending or running:
        alone_running = any(c.run_alone for c, *_ in running.values())
        while pending and not alone_running and len(running) < max(jobs, 1):
            pick = next((c for c in pending
                         if (not running if c.run_alone else True)), None)
            if pick is None:
                break
            pending.remove(pick)
            log = logs / f"{pick.name}.log"
            fh = log.open("w", encoding="utf-8")
            proc = subprocess.Popen(command(pick, tier), cwd=cwd, stdout=fh,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[proc.pid] = (pick, time.monotonic(), log, fh)
            alone_running = pick.run_alone
        done = False
        for pid in list(running):
            wpid, status, usage = os.wait4(pid, os.WNOHANG)
            if wpid == 0:
                continue
            check, start, log, fh = running.pop(pid)
            fh.close()
            secs = time.monotonic() - start
            code = os.waitstatus_to_exitcode(status)
            res = Result(check, "PASS" if code == 0 else "FAIL", secs, log,
                         usage.ru_maxrss / 1024 ** 2)
            results.append(res)
            shown = os.path.relpath(log, cwd) if log.is_absolute() else str(log)
            emit(f"{res.status}  {check.name:<{width}}  {secs:7.1f} s  {shown}")
            if res.status == "FAIL":
                for line in _tail(log):
                    emit(f"      {line}")
            done = True
        if not done:
            time.sleep(poll)
    return results


def write_summary(results: Sequence[Result], tier: str, logs: Path) -> None:
    rows = ["name\ttier\tstatus\tseconds\tpeak_rss_gb\tlog"]
    rows += [f"{r.check.name}\t{tier}\t{r.status}\t{r.seconds:.1f}\t{r.peak_rss_gb:.2f}\t"
             f"{os.path.relpath(r.log, PROJECT_ROOT) if r.log.is_absolute() else r.log}"
             for r in results]
    (logs / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def listing(registry: Sequence[Check] = REGISTRY) -> list[str]:
    out = [f"{'name':<24}{'cold s':>8}{'warm s':>8}  flags                    command"]
    for c in sorted(registry, key=lambda c: -c.cold_s):
        flags = ",".join(f for f, on in (("warm-mode", bool(c.warm)), ("run-alone", c.run_alone),
                                          ("superseded", c.superseded)) if on) or "-"
        warm = f"{c.warm_s:8.1f}" if c.warm_s else f"{'=cold':>8}"
        out.append(f"{c.name:<24}{c.cold_s:8.1f}{warm}  {flags:<24} "
                   f"-m {PACKAGE}.{c.module} {' '.join(c.argv)}"
                   + (f" [+ {' '.join(c.warm)}]" if c.warm else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--warm", dest="tier", action="store_const", const="warm")
    tier.add_argument("--cold", dest="tier", action="store_const", const="cold")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="every registered check")
    sel.add_argument("--affected", nargs="?", const="HEAD", default=None, metavar="REF",
                     help="checks whose import closure holds a file changed since REF "
                          "(default HEAD: uncommitted and untracked files)")
    ap.add_argument("names", nargs="*", metavar="NAME", help="run these checks (see --list)")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--explain", action="store_true",
                    help="with --affected: print which changed files select which checks")
    a = ap.parse_args(argv)
    t = a.tier or "warm"
    if a.list:
        print("\n".join(listing()))
        return 0
    if a.names and (a.all or a.affected):
        ap.error("give NAMEs, --all or --affected, not several")
    hits = None
    if not a.names and not a.all:
        hits = affected(changed_files(a.affected or "HEAD"))
        if a.explain:
            for name, files in hits.items():
                print(f"{name}: {', '.join(files)}")
    checks = select(t, a.names, a.all, hits)
    logs = a.logs if a.logs.is_absolute() else PROJECT_ROOT / a.logs
    start = time.monotonic()
    results = run(checks, t, a.jobs, logs)
    write_summary(results, t, logs)
    failed = sum(r.status == "FAIL" for r in results)
    print(f"{len(results) - failed} passed, {failed} failed in "
          f"{time.monotonic() - start:.1f} s (tier {t}, -j {a.jobs})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
