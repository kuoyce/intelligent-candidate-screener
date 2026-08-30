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

**Resuming a half-finished session** *(decision D28)*. The queue is rebuilt from what is
*not yet judged*: any `pair_id` carrying a row in `judgements.csv` is dropped before the
shuffle. Granularity is the pair, not the JD or the batch, so a session that stops at pair
130 of 250 resumes at 120 remaining with no bookkeeping and no state file. Re-running
`--build` at any point is therefore always safe — it is the resume command.

The one thing it cannot do is put a JD back together: the top-1 shortlist pick is asked
once per JD after all its candidates have been seen, so a JD split across two sittings
needs its pick recorded in the second. `--report-progress` prints which JDs are partial.

Usage:
    uv run python -m candidate_screener.annotation.queue --build --seed 0
    uv run python -m candidate_screener.annotation.queue --progress
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
#: `stratum` is recorded alongside `batch` on purpose *(D29)*. Batch is the operational
#: record; stratum is the unit every figure is quoted in, and stamping it at labelling
#: time means editing a batch spec later cannot silently re-stratify judgements already
#: collected. `annotator` carries who labelled the pair — with two annotators both
#: covering every title, that is what makes the 30% double-labelled overlap computable.
JUDGEMENT_COLUMNS = ("pair_id", "batch", "stratum", "corpus", "query_id", "doc_id",
                     "selection_reason", "annotator", "label", "shortlist_pick", "notes")

#: Columns an annotator sees. Anything else is anchoring material.
DISPATCH_COLUMNS = ("queue_id", "query_text", "candidate_text")

#: ~50 A1 pairs, spread across A1's three classes so agreement is measurable per class
#: rather than only in aggregate. `none` gets the largest share because it is the
#: majority class and a recheck that only samples positives cannot detect a
#: false-positive bias in A1's labelling.
RECHECK_STRATA = {"Good Fit": 15, "Potential Fit": 15, "No Fit": 20}

#: The **LLM leg only**, at twice the depth. Machine judging costs no annotator time, so
#: the leg that does not consume the D25 budget is the one worth widening: it takes
#: kappa(A1, llm) from n=50 to n=100 and — the reason it matters — A1's `Good Fit` cell,
#: where the D33 finding actually lives, from n=15 to n=30.
#:
#: **`RECHECK_STRATA` is deliberately left at 50.** `session.load_units` defaults to it, so
#: the human queue is untouched and the 9 outstanding human pairs stay 9.
#:
#: **The draw is nested by construction and asserted anyway.** `a1_recheck` permutes each
#: label's pool at a stratum seed and takes `order[:n]`, so raising `n` appends and never
#: reshuffles. `assert_recheck_nests` holds it: if it ever broke, the published n=50
#: figures would stop being a subset of the n=100 ones and the two could not be quoted
#: side by side.
LLM_RECHECK_STRATA = {"Good Fit": 30, "Potential Fit": 30, "No Fit": 40}


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
        "batch": 0,          # the recheck is not part of the in-domain campaign
        "stratum": "a1_recheck",   # never pooled with either in-domain stratum
        "query_id": out.jd_id.to_numpy(),
        "doc_id": out.resume_id.to_numpy(),
        "selection_reason": "a1_recheck",
        "query_text": out.job_description_text.to_numpy(),
        "candidate_text": out.resume_text.to_numpy(),
        "a1_label": out.label.to_numpy()})


def judged_pair_ids() -> set[str]:
    """Everything already labelled. The resume mechanism, and the whole of it.

    A pair is done when it has a row here — no separate progress file, nothing to keep in
    sync, and no way for the two to disagree. `judgements.csv` is append-only, so this set
    only grows.
    """
    if not JUDGEMENTS.exists():
        return set()
    judged = pd.read_csv(JUDGEMENTS)
    if judged.empty or "pair_id" not in judged:
        return set()
    return set(judged.pair_id.dropna().astype(str))


def indomain_rows() -> pd.DataFrame:
    """Task 3.4b's pairs, every batch, joined to text."""
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
        "batch": pairs.batch.to_numpy(),
        "stratum": pairs.stratum.to_numpy(),
        "query_id": pairs.jd_id.to_numpy(),
        "doc_id": pairs.cv_id.to_numpy(),
        "selection_reason": "indomain_banded",
        "query_text": pairs.jd_id.map(jd_text).to_numpy(),
        "candidate_text": pairs.cv_id.map(cv_text).to_numpy(),
        "a1_label": None})


def build_queue(seed: int, strata: dict[str, int] | None = None) -> pd.DataFrame:
    """Concatenate, drop what is judged, redact, shuffle. The shuffle is the blinding."""
    rows = pd.concat([indomain_rows(), a1_recheck(strata or RECHECK_STRATA, seed)],
                     ignore_index=True)
    rows["pair_id"] = rows.query_id.astype(str) + "__" + rows.doc_id.astype(str)

    done = judged_pair_ids()
    rows = rows[~rows.pair_id.isin(done)].reset_index(drop=True)
    if rows.empty:
        raise AssertionError(
            f"every pair in the campaign is already judged ({len(done)} labels). Add a "
            "batch with `annotation.sample --add-batch` before rebuilding the queue.")

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

    # Reshuffled on every rebuild, at a seed that includes how much is already done, so a
    # resumed sitting does not re-present the remainder in its original relative order —
    # which would leak that the skipped pairs were the ones judged first.
    order = np.random.default_rng([seed, len(rows)]).permutation(len(rows))
    return rows.iloc[order].reset_index(drop=True).assign(position=range(len(rows)))


def progress() -> dict:
    """What is done, what is left, and which JDs are split across sittings."""
    done = judged_pair_ids()
    pairs = pd.read_csv(PAIRS_MANIFEST)
    pairs["pair_id"] = pairs.jd_id.astype(str) + "__" + pairs.cv_id.astype(str)
    pairs["done"] = pairs.pair_id.isin(done)

    by_jd = pairs.groupby("jd_id").done.agg(["sum", "size"])
    partial = by_jd[(by_jd["sum"] > 0) & (by_jd["sum"] < by_jd["size"])]
    return {
        "judged": len(done), "in_domain_total": int(len(pairs)),
        "in_domain_done": int(pairs.done.sum()),
        "remaining": int((~pairs.done).sum()),
        "by_batch": {str(b): {"done": int(g.done.sum()), "total": int(len(g))}
                     for b, g in pairs.groupby("batch")},
        "by_stratum": {str(k): {"done": int(g.done.sum()), "total": int(len(g))}
                       for k, g in pairs.groupby("stratum")},
        # A JD split across two sittings needs its top-1 shortlist pick recorded in the
        # second, because the question is asked once per JD after all its candidates.
        "partial_jds": partial.index.tolist(),
    }


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

    already = judged_pair_ids()
    report = {
        "seed": seed,
        "total": int(len(queue)),
        "already_judged_and_skipped": len(already),
        "by_batch": {str(b): int(n) for b, n in queue.batch.value_counts().items()},
        "by_stratum": {str(k): int(n) for k, n in queue.stratum.value_counts().items()},
        "by_selection_reason": queue.selection_reason.value_counts().to_dict(),
        "a1_recheck_strata": {k: int(v) for k, v in
                              queue[queue.selection_reason == "a1_recheck"]
                              .a1_label.value_counts().items()},
        "double_label_target": int(round(0.30 * len(queue[queue.corpus == "a2"]))),
        "annotators": 2,
        "note_protocol": (
            "Two annotators, both covering every title, working from this one queue split "
            "by hand (D29). 30% of the in-domain pairs are labelled by both, for kappa; the "
            "rest are single-labelled. Because both cover all titles, the double-labelled "
            "subset can be drawn at random — under an expertise split it could not, since "
            "a random draw would rarely land on a pair two people had both seen."),
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
    print(f"\n=== Judging queue — {r['total']} pairs to judge (seed={r['seed']})")
    if r["already_judged_and_skipped"]:
        print(f"  resuming: {r['already_judged_and_skipped']} already judged, skipped")
    for reason, n in r["by_selection_reason"].items():
        print(f"  {reason:<20} {n}")
    print(f"  A1 recheck strata   {r['a1_recheck_strata']}  <-- tests A13")
    print(f"  strata              {r['by_stratum']}")
    print(f"  double-label target {r['double_label_target']} in-domain pairs (30%), "
          f"{r['annotators']} annotators, one queue split by hand")
    print(f"  key      -> {QUEUE_MANIFEST.name} (committed, no text)")
    print(f"  dispatch -> {DISPATCH_OUT / 'judging-queue.csv'} (git-ignored, redacted)")
    print(f"  labels   -> {JUDGEMENTS.name} (committed, header only until the session runs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="write the queue of everything not yet judged — also the "
                         "resume command, safe to run at any point")
    ap.add_argument("--progress", action="store_true",
                    help="what is done, what is left, which JDs are split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.build or args.progress):
        ap.error("pass --build or --progress")

    if args.progress:
        p = progress()
        print(f"\n=== Progress — {p['in_domain_done']}/{p['in_domain_total']} in-domain "
              f"pairs judged, {p['remaining']} remaining")
        for name, st in p["by_stratum"].items():
            print(f"  {name:<10} {st['done']}/{st['total']}")
        for batch, b in p["by_batch"].items():
            print(f"    batch {batch}   {b['done']}/{b['total']}")
        if p["partial_jds"]:
            print(f"  partial JDs ({len(p['partial_jds'])}) — each needs its top-1 "
                  f"shortlist pick recorded when it is finished:")
            for j in p["partial_jds"][:10]:
                print(f"      {j}")
        return 0

    print_report(build(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
