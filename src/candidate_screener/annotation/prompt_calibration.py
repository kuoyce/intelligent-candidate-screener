"""Prompt calibration harness for the a1-judge.

Runs candidate system prompts against a small probe set (the ~30 pairs A1 labels
`Good Fit`, where the production judge has 0.067 agreement) and reports agreement
vs A1 and human labels — without running the full 100-pair instrument.

**Not a data collection tool.** Output is printed, never written to
`llm-recheck.csv`. No provenance hashes. No sandbox isolation. The separation
is intentional: the production instrument (dispatch -> judge -> collect) is the
answer key; this is the scratchpad for choosing which prompt to give it.

Uses `claude -p --system-prompt <text> --model haiku` via subprocess — no
`anthropic` package required, no agent file, same transport as the production judge.

Usage:
    # Current production guide, Good Fit probe only
    uv run python -m candidate_screener.annotation.prompt_calibration --run 1

    # Modified guide file
    uv run python -m candidate_screener.annotation.prompt_calibration \\
        --guide plan/2026-08-30-expand-annotation/hr-specialist-prompt.md --run 1

    # All 100 recheck pairs
    uv run python -m candidate_screener.annotation.prompt_calibration --run 1 --all
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from candidate_screener.annotation import llm_recheck as llm
from candidate_screener.annotation import queue as judging
from candidate_screener.annotation import session
from candidate_screener.config import DOCS

DEFAULT_MODEL = "haiku"
CONCURRENCY = 10
TIMEOUT_S = 120


# --- system prompt rendering ---------------------------------------------------

def render_system_prompt(guide_path: Path | None = None) -> str:
    """Body of the a1-judge agent definition, frontmatter stripped.

    `render_agent_definition()` emits "---\\n<frontmatter>\\n---\\n\\n<body>".
    `--system-prompt` needs only the body; the frontmatter is the agent-file
    format, not the system prompt content.
    """
    guide_text = (guide_path.read_text(encoding="utf-8")
                  if guide_path is not None else None)
    defn = llm.render_agent_definition(guide_text)
    # Split on the closing "---\n\n" that separates frontmatter from body.
    parts = defn.split("---\n\n", maxsplit=1)
    return parts[-1] if len(parts) == 2 else defn


# --- probe set loading --------------------------------------------------------

def load_probe_pairs(run: int, good_fit_only: bool = True,
                     seed: int = 0) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    """Return (pair_records, a1_labels, human_labels).

    pair_records — list of {"pair_id", "prompt"} from the dispatched prompts JSONL.
    a1_labels    — {pair_id: label} for all 100 recheck pairs.
    human_labels — {pair_id: label} from judgements.csv (corpus == "a1").

    When good_fit_only=True, pair_records is filtered to the ~30 pairs A1 labels
    "Good Fit" — the hardest cases for the current judge.
    """
    prompts = llm.read_jsonl(llm.PROMPTS)
    if not prompts:
        raise FileNotFoundError(
            f"{llm.PROMPTS} is empty — run "
            "`uv run python -m candidate_screener.annotation.llm_recheck "
            f"--dispatch --run {run}` first")

    prompts = [r for r in prompts if r["run"] == run]
    if not prompts:
        raise FileNotFoundError(f"no run {run} in {llm.PROMPTS}")

    # A1 labels: from the stratified draw (same as dispatch uses)
    draw = judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed)
    draw["pair_id"] = draw.query_id.astype(str) + "__" + draw.doc_id.astype(str)
    a1_labels: dict[str, str] = dict(zip(draw["pair_id"], draw["a1_label"]))

    if good_fit_only:
        good_fit_ids = {pid for pid, lbl in a1_labels.items() if lbl == "Good Fit"}
        prompts = [r for r in prompts if r["pair_id"] in good_fit_ids]

    # Human labels from judgements.csv
    human_labels: dict[str, str] = {}
    judged = session.load_judgements()
    human_a1 = judged[(judged.corpus == "a1") & judged.label.notna()]
    human_labels = (human_a1.drop_duplicates("pair_id")
                    .set_index("pair_id").label.to_dict())

    return prompts, a1_labels, human_labels


# --- calibration --------------------------------------------------------------

def calibrate_one(pair_prompt: str, system_prompt: str,
                  model: str = DEFAULT_MODEL) -> str:
    """One pair, one subprocess call, one raw return string."""
    proc = subprocess.run(
        [shutil.which("claude") or "claude", "-p",
         "--system-prompt", system_prompt,
         "--model", model],
        input=pair_prompt, capture_output=True, text=True,
        timeout=TIMEOUT_S, check=False)
    raw = proc.stdout.strip()
    if proc.returncode != 0 and not raw:
        raw = f"<exited {proc.returncode}: {proc.stderr.strip()[:200]}>"
    return raw


def calibrate(guide_path: Path | None,
              pair_records: list[dict],
              a1_labels: dict[str, str],
              human_labels: dict[str, str],
              concurrency: int = CONCURRENCY,
              model: str = DEFAULT_MODEL) -> dict:
    """Run the candidate prompt against the probe set; return agreement stats."""
    system_prompt = render_system_prompt(guide_path)
    prompt_preview = system_prompt[:200].replace("\n", " ")

    results: dict[str, str] = {}   # pair_id -> label (or "")
    bad: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futures = {
            pool.submit(calibrate_one, r["prompt"], system_prompt, model): r["pair_id"]
            for r in pair_records
        }
        for future in concurrent.futures.as_completed(futures):
            pair_id = futures[future]
            try:
                raw = future.result()
                parsed = llm.parse_return(raw)
                results[pair_id] = parsed["label"]
            except (llm.BadReturn, Exception):  # noqa: BLE001
                results[pair_id] = ""
                bad.append(pair_id)

    # Agreement vs A1
    a1_agree = _pairwise_agreement(results, a1_labels)
    human_agree = _pairwise_agreement(results, human_labels)

    # Per-class breakdown vs A1
    per_class = {}
    for cls in session.LABELS:
        cls_ids = [pid for pid, lbl in a1_labels.items()
                   if lbl == cls and pid in results]
        if not cls_ids:
            per_class[cls] = {"n": 0, "probe_agrees": float("nan"),
                               "probe_distribution": {}}
            continue
        probe_lbls = [results[pid] for pid in cls_ids if results[pid]]
        agrees = sum(1 for pid in cls_ids if results.get(pid) == cls)
        from collections import Counter
        per_class[cls] = {
            "n": len(cls_ids),
            "probe_agrees": agrees / len(cls_ids) if cls_ids else float("nan"),
            "probe_distribution": dict(Counter(probe_lbls)),
        }

    return {
        "model": model,
        "guide": str(guide_path) if guide_path else str(llm.GUIDE),
        "prompt_preview": prompt_preview,
        "pairs_run": len(pair_records),
        "parsed_ok": len(results) - len(bad),
        "unparsed": len(bad),
        "unparsed_pairs": bad[:5],
        "agreement_vs_a1": a1_agree,
        "agreement_vs_human": human_agree,
        "per_class_vs_a1": per_class,
    }


def _pairwise_agreement(results: dict[str, str],
                        reference: dict[str, str]) -> dict:
    common = [(results[pid], reference[pid])
              for pid in results
              if pid in reference and results[pid] and reference[pid]]
    if not common:
        return {"n": 0, "agreement": float("nan")}
    agree = sum(1 for a, b in common if a == b)
    return {"n": len(common), "agreement": agree / len(common)}


# --- cli ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=int, default=1,
                    help="which dispatch run to load prompts from")
    ap.add_argument("--guide", type=Path, default=None,
                    help="modified guide .md to test; defaults to docs/annotation-guide.md")
    ap.add_argument("--all", dest="all_pairs", action="store_true",
                    help="use all 100 recheck pairs, not just Good Fit")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="claude model alias for the probe (default: haiku)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pair_records, a1_labels, human_labels = load_probe_pairs(
        args.run, good_fit_only=not args.all_pairs, seed=args.seed)

    print(f"probe: {len(pair_records)} pairs, model={args.model}, "
          f"guide={args.guide or 'docs/annotation-guide.md'}")

    out = calibrate(args.guide, pair_records, a1_labels, human_labels,
                    concurrency=args.concurrency, model=args.model)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
