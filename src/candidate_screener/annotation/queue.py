"""One dispatch queue for one session *(decision D25)*.

Two objectives, one physical act. Task 3.4b needs 200 Djinni pairs rated; the A1 recheck
needs ~50 already-judged A1 pairs rated to test **A13** — the assumption that A1's own
labels are correct, which every ceiling figure in `plan/2026-08-29-pool-precision-bias/`
rests on. **D17 already made the label identical** (`Good`/`Potential`/`No Fit` doubles as
nDCG's graded relevance), so the two differ only in which pairs are selected, and one
`selection_reason` column lets a single queue serve both.

**The split between what is dispatched and what is committed is deliberate.**

| | Contents | Where | Why |
|---|---|---|---|
| Dispatch | redacted text, opaque `queue_id`, nothing else | `data/processed/indomain/` — git-ignored | It carries real document text. `AGENTS.md` forbids committing that |
| Key | ids, corpus, `selection_reason`, position | `docs/data/manifests/judging-queue.csv` — committed | The audit trail: which pair was which, and at what position |

Blinding is a property of the **dispatch file**, not of the repository. An annotator reads
the dispatch file; if it carried `selection_reason`, they would know which pairs A1 had
already labelled and could anchor on the expected answer, which is the one thing the
recheck cannot survive. The key exists so the session is reconstructible afterwards.

Usage:
    uv run python -m candidate_screener.annotation.queue --build --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from candidate_screener.annotation import redact
from candidate_screener.annotation.sample import PAIRS_MANIFEST, load_english
from candidate_screener.config import PROCESSED
from candidate_screener.data.fit_split import FIT_OUT, MANIFESTS

QUEUE_MANIFEST = MANIFESTS / "judging-queue.csv"
JUDGEMENTS = MANIFESTS / "judgements.csv"
QUEUE_REPORT = MANIFESTS / "judging-queue-report.json"
DISPATCH_OUT = PROCESSED / "indomain"

#: The committed judgement layer *(decision D20)*. Our labels never enter `pools.csv`:
#: that file is a pure function of (`data/raw/`, seed) and `verify --derived` check 6
#: asserts it reproduces byte-for-byte. A human judgement is not a function of a seed, so
#: merging them would have forced a choice between deleting the determinism check and
#: never re-running the builder. The schema is written before any label exists — a schema
#: settled afterwards is a migration.
JUDGEMENT_COLUMNS = ("pair_id", "corpus", "query_id", "doc_id", "selection_reason",
                     "annotator", "label", "shortlist_pick", "notes")

#: Columns an annotator sees. Anything else is anchoring material.
DISPATCH_COLUMNS = ("queue_id", "query_text", "candidate_text")

#: ~50 A1 pairs, spread across A1's three classes so agreement is measurable per class
#: rather than only in aggregate. `none` gets the largest share because it is the
#: majority class and a recheck that only samples positives cannot detect a
#: false-positive bias in A1's labelling.
RECHECK_STRATA = {"Good Fit": 15, "Potential Fit": 15, "No Fit": 20}


def queue_id(corpus: str, query: str, doc: str) -> str:
    """Opaque and stable. Not sequential — a sequential id leaks the build order, and
    the build groups by corpus."""
    digest = hashlib.sha256(f"{corpus}|{query}|{doc}".encode()).hexdigest()
    return "q_" + digest[:12]


def a1_recheck(strata: dict[str, int], seed: int) -> pd.DataFrame:
    """Already-judged A1 test pairs, stratified by A1's own label *(tests A13)*."""
    test = pd.read_parquet(FIT_OUT / "test.parquet")
    seeds = np.random.SeedSequence(seed).spawn(len(strata))
    picked = []
    for (label, n), stratum_seed in zip(sorted(strata.items()), seeds):
        pool = test[test.label == label]
        order = np.random.default_rng(stratum_seed).permutation(len(pool))
        picked.append(pool.iloc[order[:n]])
    out = pd.concat(picked)
    return pd.DataFrame({
        "corpus": "a1",
        "query_id": out.jd_id.to_numpy(),
        "doc_id": out.resume_id.to_numpy(),
        "selection_reason": "a1_recheck",
        "query_text": out.job_description_text.to_numpy(),
        "candidate_text": out.resume_text.to_numpy(),
        "a1_label": out.label.to_numpy()})


def indomain_rows() -> pd.DataFrame:
    """Task 3.4b's 200 pairs, joined to text."""
    if not PAIRS_MANIFEST.exists():
        raise FileNotFoundError(
            f"{PAIRS_MANIFEST} missing — run `uv run python -m "
            "candidate_screener.annotation.sample --build --seed 0` first")
    pairs = pd.read_csv(PAIRS_MANIFEST)
    jd, cv = load_english()
    jd_text = jd.set_index("id").jd_text
    cv_text = cv.set_index("id").cv_text
    return pd.DataFrame({
        "corpus": "a2",
        "query_id": pairs.jd_id.to_numpy(),
        "doc_id": pairs.cv_id.to_numpy(),
        "selection_reason": "indomain_banded",
        "query_text": pairs.jd_id.map(jd_text).to_numpy(),
        "candidate_text": pairs.cv_id.map(cv_text).to_numpy(),
        "a1_label": None})


def build_queue(seed: int, strata: dict[str, int] | None = None) -> pd.DataFrame:
    """Concatenate, redact, shuffle. The shuffle is the blinding."""
    rows = pd.concat([indomain_rows(), a1_recheck(strata or RECHECK_STRATA, seed)],
                     ignore_index=True)

    missing = rows[rows.query_text.isna() | rows.candidate_text.isna()]
    if len(missing):
        raise AssertionError(f"{len(missing)} queue rows have no document text — an "
                             "annotator cannot judge a pair they cannot read")

    rows["query_text"] = redact.redact(rows.query_text)
    rows["candidate_text"] = redact.redact(rows.candidate_text)
    redact.assert_clean(rows.query_text, "queue query text")
    redact.assert_clean(rows.candidate_text, "queue candidate text")

    rows["queue_id"] = [queue_id(c, q, d) for c, q, d in
                        zip(rows.corpus, rows.query_id, rows.doc_id)]
    if rows.queue_id.duplicated().any():
        raise AssertionError("duplicate queue_id — the same pair is dispatched twice")

    order = np.random.default_rng(seed).permutation(len(rows))
    return rows.iloc[order].reset_index(drop=True).assign(position=range(len(rows)))


def build(seed: int, strata: dict[str, int] | None = None) -> dict:
    queue = build_queue(seed, strata)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    queue[["queue_id", "corpus", "query_id", "doc_id", "selection_reason", "position"]] \
        .to_csv(QUEUE_MANIFEST, index=False, lineterminator="\n")
    if not JUDGEMENTS.exists():
        JUDGEMENTS.write_text(",".join(JUDGEMENT_COLUMNS) + "\n", encoding="utf-8")

    DISPATCH_OUT.mkdir(parents=True, exist_ok=True)
    queue[list(DISPATCH_COLUMNS)].to_csv(
        DISPATCH_OUT / "judging-queue.csv", index=False, lineterminator="\n")
    queue.to_parquet(DISPATCH_OUT / "judging-queue-full.parquet", index=False)

    report = {
        "seed": seed,
        "total": int(len(queue)),
        "by_selection_reason": queue.selection_reason.value_counts().to_dict(),
        "a1_recheck_strata": {k: int(v) for k, v in
                              queue[queue.selection_reason == "a1_recheck"]
                              .a1_label.value_counts().items()},
        "double_label_target": 60,
        "note_blinding": (
            "The dispatch file carries only " + ", ".join(DISPATCH_COLUMNS) + ". "
            "selection_reason and a1_label are in the committed key and the git-ignored "
            "full parquet, never in what an annotator opens."),
        "note_effort": (
            "~250 judgements, ~2.5 team-days, one session (D25). The judging wave over "
            "system top-k (D22/D23, 4-6 team-days) is deferred; Q18 closes as a standing "
            "limitation under D26 instead."),
    }
    QUEUE_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_report(r: dict) -> None:
    print(f"\n=== Judging queue — {r['total']} pairs, one session (seed={r['seed']})")
    for reason, n in r["by_selection_reason"].items():
        print(f"  {reason:<20} {n}")
    print(f"  A1 recheck strata   {r['a1_recheck_strata']}  <-- tests A13")
    print(f"  double-label target {r['double_label_target']} of 200 in-domain pairs (30%)")
    print(f"  key      -> {QUEUE_MANIFEST.name} (committed, no text)")
    print(f"  dispatch -> {DISPATCH_OUT / 'judging-queue.csv'} (git-ignored, redacted)")
    print(f"  labels   -> {JUDGEMENTS.name} (committed, header only until the session runs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.build:
        ap.error("pass --build")
    print_report(build(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
