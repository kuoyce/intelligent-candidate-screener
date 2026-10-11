"""Every decision number cited anywhere in the repository resolves in `docs/decisions.md`.

D45 moved decision history out of `AGENTS.md`. A citation that resolves nowhere is a
reader sent to look for a rule that is not written down.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITATION = re.compile(r"\bD(\d{1,2})\b")  # two digits: longer runs are ids, not decisions
SCANNED = ("AGENTS.md", "README.md", "src", "docs", "plan", "output", "tests", "notebooks")


def cited_numbers() -> dict[int, str]:
    found: dict[int, str] = {}
    for top in SCANNED:
        base = ROOT / top
        files = [base] if base.is_file() else [
            p for p in base.rglob("*") if p.suffix in {".md", ".py", ".json", ".ipynb"}]
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix == ".ipynb":   # cell sources only: outputs hold base64 images
                text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"])
            for match in CITATION.finditer(text):
                found.setdefault(int(match.group(1)), str(path.relative_to(ROOT)))
    return found


def test_every_cited_decision_is_in_the_log():
    log = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    logged = {int(n) for n in re.findall(r"^\| D(\d{1,3})\b", log, flags=re.M)}
    missing = {n: where for n, where in cited_numbers().items() if n not in logged}
    assert not missing, f"cited but not in docs/decisions.md: {missing}"
