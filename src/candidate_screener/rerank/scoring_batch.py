"""The D50 LLM scoring batch: the one dispatch D42's closure allows *(D50, D51)*.

Plan: `plan/2026-10-06-extraction-and-rerank/04-llm-scoring-batch.md`; record: `15-p6-batch.md`.

1,000 new 0-100 scores from the **unchanged** instrument (`5d986c82`, `claude-haiku-4-5`),
one call per (JD, CV) pair, on 100 JDs x 10 CVs from the A2 **train** region, to train the
stage-2 reranker. Nothing here is a label for the final set and nothing goes near
`judgements.csv`.

**Why this module and not `annotation/`** *(Q-A3)*. `annotation/` is frozen (D42) and
`llm_recheck.DISPATCH_CLOSED` stays `True`: every one of its entry points still raises
`DispatchClosed`. This module imports the instrument from it — `render_prompt`, the
committed agent and its hash check, the tool-free sandbox, `judge_argv`, the strict
`parse_return` — and runs its own short judging loop, because flipping the frozen module's
gate (even in-process) would reopen *every* path behind it, and the frozen loop banks into
`llm-recheck-*.jsonl`, whose `collect()` refuses any draw it does not know: one train-batch
record there would break `llm_recheck --collect` for the whole repository. So the batch
banks into its own append-only files under `data/processed/rerank/`.

**The exemption is checked in code, on every record, at every step** (`assert_in_scope`):
dispatch, before judging, and again inside each spawn. A record is in scope only if

* its instrument hash starts `5d986c82` and is the committed agent's,
* its model is `claude-haiku-4-5`,
* its draw is `a2_train_batch` and its run is 11 (or 12, the replicate, only when the
  caller passed `replicate=True` — Q-A1b is open, so it is off by default),
* its pair is in the committed batch manifest `llm-train-batch.csv` with the same ids
  (and, for run 12, in the manifest's replicate subset),
* neither document is in the final set (`a2-final-labels.csv`, `indomain-pairs.csv`) nor
  in the eval region (`a2-partition.csv`, D19), and both are in the train region,
* the run holds at most 1,000 pairs (100 for the replicate).

Anything else raises `OutOfScope`, a `DispatchClosed`.

**The text sent is the text runs 9/10 sent** (so the instrument is unchanged): JD =
`Long Description`, stripped; CV = `annotation.sample.cv_text`, i.e. `Position + CV +
Highlights + Moreinfo + Looking For` (Position **included**: this is the instrument's form,
not the matching rungs' `text_v2`); both through `redact.redact_text`, then
`llm_recheck.render_prompt`. `--verify-text-form` rebuilds run-9 prompts this way and
compares their `prompt_sha256` with the frozen bank.

Usage:
    uv run python -m candidate_screener.rerank.scoring_batch --build     # write the manifest
    uv run python -m candidate_screener.rerank.scoring_batch --check     # rebuild and compare
    uv run python -m candidate_screener.rerank.scoring_batch --verify-text-form
    uv run python -m candidate_screener.rerank.scoring_batch --dry-run   # spawns nothing
    uv run python -m candidate_screener.rerank.scoring_batch --dispatch
    uv run python -m candidate_screener.rerank.scoring_batch --judge --limit 70
    uv run python -m candidate_screener.rerank.scoring_batch --coverage
    uv run python -m candidate_screener.rerank.scoring_batch --collect
    (add --replicate to dispatch/judge/coverage run 12, the 100-pair replicate; Q-A1b)
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from candidate_screener.annotation import llm_recheck as llm
from candidate_screener.annotation import redact, sample
from candidate_screener.annotation.queue import JUDGEMENTS
from candidate_screener.config import PROCESSED
from candidate_screener.data.a2_finetune import (JD_EXP_TO_BAND, PARTITION_MANIFEST,
                                                 cv_experience_band)
from candidate_screener.data.fit_split import MANIFESTS

# --- the scope D50 grants, and nothing wider ------------------------------------

INSTRUMENT_PREFIX = "5d986c82"
MODEL = "claude-haiku-4-5"
DRAW = "a2_train_batch"

#: Runs 1-10 are taken (`llm-recheck-prompts.jsonl`: 1-4 A1, 5-10 A2). A run is one sitting
#: of one draw (D35), and the replicate is a second sitting of the same draw.
RUN = 11
REPLICATE_RUN = 12

N_JDS, PER_JD = 100, 10
PAIR_CAP = N_JDS * PER_JD          # 1,000 (Q-A1: option B)
REPLICATE_CAP = 100                # Q-A1b: open; behind `replicate=True`
VALIDATION_FRACTION = 0.2          # 80 / 20 by JD
SEED = 0

#: Measured 8-10 Sep 2026 (memory note `judge-spawns-use-the-session-limit`): 600 pairs in
#: nine sittings. An estimate from one prior run, not a promise.
PAIRS_PER_SITTING = 67

#: The final set's `generic` design (Q-A2): 10 CVs per JD, 8 sharing its Primary Keyword,
#: TF-IDF banded, CVs not reused across JDs. Only the region differs.
BATCH_SPEC = {"batch": 1, "n_jds": N_JDS, "per_jd": PER_JD, "seed": SEED,
              "keywords": None, "same_keyword_min": sample.SAME_KEYWORD_MIN,
              "reuse_jds": False}

BATCH_MANIFEST = MANIFESTS / "llm-train-batch.csv"
BATCH_RECORD = MANIFESTS / "llm-train-batch.json"
SCORES_CSV = MANIFESTS / "llm-train-scores.csv"
FINAL_LABELS = MANIFESTS / "a2-final-labels.csv"
INDOMAIN_PAIRS = sample.PAIRS_MANIFEST

#: Append-only bank, git-ignored like the frozen one, and separate from it (module doc).
BANK = PROCESSED / "rerank"
PROMPTS = BANK / "llm-train-prompts.jsonl"
RAW = BANK / "llm-train-raw.jsonl"
FAILED = BANK / "llm-train-failed.jsonl"

MANIFEST_COLUMNS = ("pair_id", "jd_id", "cv_id", "split", "selection_reason", "replicate")

#: Ids and scores only: no document text, and not the judge's free-text reason (it can
#: quote the CV).
SCORE_COLUMNS = ("agent_sha256", "model", "run", "draw", "pair_id", "jd_id", "cv_id",
                 "split", "label", "score", "parsed_ok", "prompt_sha256", "chars_sent",
                 "judged_at")


class OutOfScope(llm.DispatchClosed):
    """A dispatch or spawn outside what D50 reopened. D42 still holds for it."""


def _refuse(why: str) -> None:
    raise OutOfScope(
        f"{why}. D42 closes dispatch; D50 reopens it for one batch only (instrument "
        f"{INSTRUMENT_PREFIX}, {MODEL}, draw {DRAW!r}, run {RUN} [+{REPLICATE_RUN} with "
        f"--replicate], pairs in {BATCH_MANIFEST.name}, A2 train region, no final-set "
        "or eval-region document, at most "
        f"{PAIR_CAP} + {REPLICATE_CAP} pairs).")


@dataclass(frozen=True)
class Scope:
    """What the guard checks a record against. Built from committed files only."""
    pairs: dict                    # pair_id -> (jd_id, cv_id)
    replicate: frozenset
    train_jds: frozenset
    train_cvs: frozenset
    eval_jds: frozenset
    eval_cvs: frozenset
    final_jds: frozenset
    final_cvs: frozenset

    @classmethod
    def from_frames(cls, manifest: pd.DataFrame, partition: pd.DataFrame,
                    final_pairs: pd.DataFrame) -> "Scope":
        part = partition.astype(str)
        region = lambda r, t: frozenset(  # noqa: E731
            part[(part.region == r) & (part.doc_type == t)].doc_id)
        m = manifest.astype({"pair_id": str, "jd_id": str, "cv_id": str})
        rep = m.replicate.astype(str).str.lower().isin(("true", "1"))
        return cls(pairs=dict(zip(m.pair_id, zip(m.jd_id, m.cv_id))),
                   replicate=frozenset(m.pair_id[rep]),
                   train_jds=region("train", "jd"), train_cvs=region("train", "cv"),
                   eval_jds=region("eval", "jd"), eval_cvs=region("eval", "cv"),
                   final_jds=frozenset(final_pairs.jd_id.astype(str)),
                   final_cvs=frozenset(final_pairs.cv_id.astype(str)))

    @classmethod
    def committed(cls) -> "Scope":
        if not BATCH_MANIFEST.exists():
            _refuse(f"{BATCH_MANIFEST} is missing — `--build` it first")
        return cls.from_frames(pd.read_csv(BATCH_MANIFEST), pd.read_csv(PARTITION_MANIFEST),
                               final_set_pairs())


def final_set_pairs() -> pd.DataFrame:
    """Every pair of the final set and of the in-domain campaign it was drawn from."""
    return pd.concat([pd.read_csv(FINAL_LABELS)[["jd_id", "cv_id"]],
                      pd.read_csv(INDOMAIN_PAIRS)[["jd_id", "cv_id"]]], ignore_index=True)


# --- the guard: one named check per clause of D50 --------------------------------

def check_instrument(record: dict, agent_sha256: str) -> None:
    got = str(record.get("agent_sha256", ""))
    if not got.startswith(INSTRUMENT_PREFIX):
        _refuse(f"pair {record.get('pair_id')}: instrument {got[:8]!r} is not "
                f"{INSTRUMENT_PREFIX}")
    if got != agent_sha256:
        _refuse(f"pair {record.get('pair_id')}: dispatched against {got[:12]}, but the "
                f"committed judge is {agent_sha256[:12]}")


def check_model(record: dict) -> None:
    if record.get("model") != MODEL:
        _refuse(f"pair {record.get('pair_id')}: model {record.get('model')!r} is not {MODEL}")


def check_run(record: dict, allow_replicate: bool) -> None:
    if record.get("draw") != DRAW:
        _refuse(f"pair {record.get('pair_id')}: draw {record.get('draw')!r} is not {DRAW!r}")
    allowed = {RUN} | ({REPLICATE_RUN} if allow_replicate else set())
    if record.get("run") not in allowed:
        hint = (" (the replicate needs the explicit --replicate flag; Q-A1b is open)"
                if record.get("run") == REPLICATE_RUN else "")
        _refuse(f"pair {record.get('pair_id')}: run {record.get('run')!r} is not in "
                f"{sorted(allowed)}{hint}")


def check_manifest(record: dict, scope: Scope) -> None:
    pair = record.get("pair_id")
    if pair not in scope.pairs:
        _refuse(f"pair {pair} is not in the committed batch manifest")
    if scope.pairs[pair] != (record.get("query_id"), record.get("doc_id")):
        _refuse(f"pair {pair}: ids {record.get('query_id')}/{record.get('doc_id')} do not "
                "match the manifest")
    if record.get("run") == REPLICATE_RUN and pair not in scope.replicate:
        _refuse(f"pair {pair} is not in the manifest's replicate subset")


def check_documents(jd_id: str, cv_id: str, scope: Scope) -> None:
    """No final-set document, no eval-region document, and both in the train region."""
    if jd_id in scope.final_jds or cv_id in scope.final_cvs:
        _refuse(f"{jd_id}__{cv_id} carries a final-set document")
    if jd_id in scope.eval_jds or cv_id in scope.eval_cvs:
        _refuse(f"{jd_id}__{cv_id} carries an eval-region document (D19)")
    if jd_id not in scope.train_jds or cv_id not in scope.train_cvs:
        _refuse(f"{jd_id}__{cv_id} carries a document outside the A2 train region")


def check_cap(records: list[dict]) -> None:
    by_run: dict = collections.defaultdict(set)
    for r in records:
        by_run[r.get("run")].add(r.get("pair_id"))
    for run, pairs in by_run.items():
        cap = REPLICATE_CAP if run == REPLICATE_RUN else PAIR_CAP
        if len(pairs) > cap:
            _refuse(f"run {run} would hold {len(pairs)} pairs, over its cap of {cap}")


def assert_manifest_in_scope(scope: Scope) -> None:
    """The manifest itself: within the caps, and every document allowed."""
    if len(scope.pairs) > PAIR_CAP:
        _refuse(f"the batch manifest holds {len(scope.pairs)} pairs, over the cap of "
                f"{PAIR_CAP}")
    if len(scope.replicate) > REPLICATE_CAP:
        _refuse(f"the replicate subset holds {len(scope.replicate)} pairs, over its cap "
                f"of {REPLICATE_CAP}")
    if not scope.replicate <= set(scope.pairs):
        _refuse("the replicate subset names pairs outside the batch")
    for jd_id, cv_id in scope.pairs.values():
        check_documents(jd_id, cv_id, scope)


def assert_in_scope(records: list[dict], scope: Scope, *, agent_sha256: str,
                    allow_replicate: bool = False) -> None:
    """Every clause of D50, on every record. Raises `OutOfScope` on the first breach."""
    assert_manifest_in_scope(scope)
    for record in records:
        check_instrument(record, agent_sha256)
        check_model(record)
        check_run(record, allow_replicate)
        check_manifest(record, scope)
        check_documents(str(record.get("query_id")), str(record.get("doc_id")), scope)
    check_cap(records)


def current_instrument() -> str:
    """The committed judge's hash, which must be the D50 instrument."""
    agent = llm.assert_agent_definition_current()
    if not agent.startswith(INSTRUMENT_PREFIX):
        _refuse(f"the committed judge is {agent[:8]}, not {INSTRUMENT_PREFIX}")
    return agent


def assert_runs_unused_elsewhere() -> None:
    """Runs 11/12 must not exist in the frozen bank or in any committed `llm-*.csv`."""
    ours = {RUN, REPLICATE_RUN}
    clash = {r["run"] for r in llm.read_jsonl(llm.PROMPTS)} & ours
    for path in llm.DRAW_FILES.values():
        if path.exists():
            clash |= set(pd.read_csv(path, usecols=["run"]).run.astype(int)) & ours
    if clash:
        _refuse(f"run(s) {sorted(clash)} are already used by the frozen annotation bank")


# --- the batch (task 6.2) -----------------------------------------------------

def load_train_documents() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`sample.load_english`'s filters, on the A2 **train** region instead of eval.

    English, non-empty, the same text fields and experience bands. The final set's
    documents and the eval region are removed explicitly as well, though the partition
    already keeps them apart, so the exclusion does not rest on one file.
    """
    from candidate_screener.data.a2_finetune import load_train_region
    from candidate_screener.data.profile import load_split

    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    train_jds, train_cvs = load_train_region()
    final = final_set_pairs()
    part = pd.read_csv(PARTITION_MANIFEST)
    eval_ids = set(part[part.region == "eval"].doc_id)
    bad_jds = eval_ids | set(final.jd_id)
    bad_cvs = eval_ids | set(final.cv_id)

    jd = jd[jd["id"].isin(train_jds) & ~jd["id"].isin(bad_jds)
            & (jd["Long Description_lang"] == "en")].copy()
    jd["jd_text"] = jd["Long Description"].fillna("").astype(str).str.strip()
    jd = jd[jd.jd_text.str.len() > 0]
    jd["exp_band"] = jd["Exp Years"].map(JD_EXP_TO_BAND)

    cv = cv[cv["id"].isin(train_cvs) & ~cv["id"].isin(bad_cvs)
            & (cv["CV_lang"] == "en")].copy()
    cv["cv_text"] = sample.cv_text(cv)
    cv = cv[cv.cv_text.str.len() > 0]
    cv["exp_band"] = cv["Experience Years"].map(cv_experience_band)
    return jd, cv


def split_by_jd(pairs: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """80 / 20 by JD, then CV-disjoint: a validation pair whose CV is also under a train
    JD is **dropped** (never reassigned, which would move a JD's pairs across splits).

    The generic design draws each CV once (`draw_batch` keeps `used_cvs`), so the drop is
    expected to remove nothing; it is here so the property does not rest on that.
    """
    jds = sorted(pairs.jd_id.unique())
    n_val = int(round(len(jds) * VALIDATION_FRACTION))
    order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(len(jds))
    validation = {jds[i] for i in order[:n_val]}
    pairs = pairs.assign(split=np.where(pairs.jd_id.isin(validation),
                                        "validation", "train"))
    train_cvs = set(pairs.cv_id[pairs.split == "train"])
    clash = (pairs.split == "validation") & pairs.cv_id.isin(train_cvs)
    return pairs[~clash].reset_index(drop=True), {"validation_pairs_dropped_for_cv_overlap":
                                                  int(clash.sum())}


def build_batch(jd: pd.DataFrame, cv: pd.DataFrame,
                seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """The manifest (ids only) and its summary, from eligible JDs and CVs. Pure."""
    spec = dict(BATCH_SPEC, seed=seed)
    pairs, _ = sample.draw_batch(spec, jd, cv, set(), set(), set())
    pairs, split_info = split_by_jd(pairs, seed)
    pairs["selection_reason"] = (np.where(pairs.keyword_match, "same_keyword",
                                          "other_keyword") + ":" + pairs.lexical_band)
    keep = set(llm.subsample_ids(pairs.pair_id.tolist(), REPLICATE_CAP, seed))
    pairs["replicate"] = pairs.pair_id.isin(keep)
    manifest = pairs.sort_values(["split", "jd_id", "cv_id"], kind="stable")[
        list(MANIFEST_COLUMNS)].reset_index(drop=True)

    per_split = {}
    for name, part in manifest.groupby("split"):
        per_split[name] = {"jds": int(part.jd_id.nunique()), "cvs": int(part.cv_id.nunique()),
                           "pairs": int(len(part)),
                           "replicate_pairs": int(part.replicate.sum())}
    summary = {
        "eligible": {"jds": int(len(jd)), "cvs": int(len(cv)),
                     "jds_by_exp_band": jd.exp_band.value_counts(dropna=False)
                     .rename(index=str).to_dict()},
        "pairs": int(len(manifest)), "jds": int(manifest.jd_id.nunique()),
        "cvs": int(manifest.cv_id.nunique()),
        "per_split": per_split, **split_info,
        "cvs_in_both_splits": int(len(
            set(manifest.cv_id[manifest.split == "train"])
            & set(manifest.cv_id[manifest.split == "validation"]))),
        "by_selection_reason": manifest.selection_reason.value_counts().to_dict(),
        "on_category_rate": round(float(pairs.keyword_match.mean()), 4),
        "candidates_per_jd": manifest.groupby("jd_id").size().value_counts()
        .rename(index=str).to_dict(),
        "jds_by_exp_band": pairs.drop_duplicates("jd_id").exp_band.value_counts()
        .rename(index=str).to_dict(),
    }
    return manifest, summary


def manifest_bytes(manifest: pd.DataFrame) -> bytes:
    return manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")


def build(write: bool = True) -> dict:
    jd, cv = load_train_documents()
    manifest, summary = build_batch(jd, cv, SEED)
    body = manifest_bytes(manifest)
    scope = Scope.from_frames(manifest, pd.read_csv(PARTITION_MANIFEST), final_set_pairs())
    assert_manifest_in_scope(scope)
    record = {
        "decision": "D50", "plan": "plan/2026-10-06-extraction-and-rerank/04-llm-scoring-batch.md",
        "spec": BATCH_SPEC, "region": "train (a2-partition.csv, D19)",
        "design": "final set's generic design (Q-A2): 8 same-Primary-Keyword CVs banded "
                  "high/high/high/mid/mid/mid/low/low + 2 other-keyword banded high/mid, "
                  "TF-IDF cosine within each JD's pools; each CV drawn once",
        "split": "80/20 by JD (seeded), validation CV-disjoint from train",
        "replicate": f"{REPLICATE_CAP} pairs seeded (llm_recheck.subsample_ids); "
                     "dispatched only with --replicate (Q-A1b open)",
        "manifest": BATCH_MANIFEST.name,
        "manifest_sha256": hashlib.sha256(body).hexdigest(),
        "summary": summary}
    if write:
        BATCH_MANIFEST.write_bytes(body)
        BATCH_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check() -> dict:
    """Rebuild in memory; the committed manifest must match it byte for byte."""
    committed = BATCH_MANIFEST.read_bytes()
    record = json.loads(BATCH_RECORD.read_text(encoding="utf-8"))
    if hashlib.sha256(committed).hexdigest() != record["manifest_sha256"]:
        raise AssertionError(f"{BATCH_MANIFEST.name} does not match the digest in "
                             f"{BATCH_RECORD.name}")
    assert_manifest_in_scope(Scope.committed())
    rebuilt = build(write=False)
    if rebuilt["manifest_sha256"] != record["manifest_sha256"]:
        raise AssertionError("the rebuilt batch differs from the committed manifest")
    if rebuilt["summary"] != record["summary"]:
        raise AssertionError("the rebuilt batch summary differs from the committed record")
    return {"ok": True, "manifest_sha256": record["manifest_sha256"],
            "pairs": record["summary"]["pairs"]}


# --- what is sent ----------------------------------------------------------

def document_texts(jd_ids, cv_ids) -> tuple[pd.Series, pd.Series]:
    """The raw texts runs 9/10 used, for any ids (region-agnostic), indexed by id."""
    from candidate_screener.data.profile import load_split

    jd = load_split("djinni-jd", "train")
    cv = load_split("djinni-cv", "train")
    jd = jd[jd["id"].isin(set(jd_ids))]
    cv = cv[cv["id"].isin(set(cv_ids))].copy()
    jd_text = jd.set_index("id")["Long Description"].fillna("").astype(str).str.strip()
    cv_text = pd.Series(sample.cv_text(cv).to_numpy(), index=cv["id"])
    return jd_text, cv_text


def render(jd_text: str, cv_text: str) -> str:
    return llm.render_prompt(redact.redact_text(jd_text), redact.redact_text(cv_text))


def verify_text_form(runs: tuple[int, ...] = (9, 10)) -> dict:
    """Rebuild every prompt of runs 9/10 from scratch; each must hash to what was sent."""
    sent = [r for r in llm.read_jsonl(llm.PROMPTS) if r["run"] in runs]
    if not sent:
        raise FileNotFoundError(f"no runs {runs} in {llm.PROMPTS}")
    jd_text, cv_text = document_texts({r["query_id"] for r in sent},
                                      {r["doc_id"] for r in sent})
    same = sum(llm.sha(render(jd_text[r["query_id"]], cv_text[r["doc_id"]]))
               == r["prompt_sha256"] for r in sent)
    return {"runs": list(runs), "checked": len(sent), "identical_prompt_sha256": int(same),
            "ok": same == len(sent)}


# --- dispatch (task 6.3) ----------------------------------------------------

def run_of(replicate: bool) -> int:
    return REPLICATE_RUN if replicate else RUN


def build_records(scope: Scope, agent: str, replicate: bool, model: str,
                  texts=None) -> list[dict]:
    run = run_of(replicate)
    split = {}
    if BATCH_MANIFEST.exists():
        m = pd.read_csv(BATCH_MANIFEST)
        split = dict(zip(m.pair_id.astype(str), m.split.astype(str)))
    pair_ids = sorted(scope.replicate if replicate else scope.pairs)
    jd_ids = {scope.pairs[p][0] for p in pair_ids}
    cv_ids = {scope.pairs[p][1] for p in pair_ids}
    jd_text, cv_text = (texts or document_texts)(jd_ids, cv_ids)
    records = []
    for pair_id in pair_ids:
        jd_id, cv_id = scope.pairs[pair_id]
        prompt = render(jd_text[jd_id], cv_text[cv_id])
        llm.assert_not_anchoring(prompt, f"pair {pair_id}")
        records.append({"pair_id": pair_id, "query_id": jd_id, "doc_id": cv_id,
                        "run": run, "draw": DRAW, "corpus": "a2", "region": "train",
                        "split": split.get(pair_id, ""),
                        "prompt": prompt, "prompt_sha256": llm.sha(prompt),
                        "chars_sent": len(prompt), "agent_sha256": agent,
                        "model": model, "response_scheme": llm.RESPONSE_SCHEME})
    return records


def _plan(replicate: bool, model: str, scope: Scope | None, texts,
          agent: str | None) -> tuple[list[dict], Scope, str]:
    agent = agent or current_instrument()
    scope = scope or Scope.committed()
    assert_runs_unused_elsewhere()
    records = build_records(scope, agent, replicate, model, texts)
    assert_in_scope(records, scope, agent_sha256=agent, allow_replicate=replicate)
    overlap = llm.guide_example_pairs() & {r["pair_id"] for r in records}
    if overlap:
        raise AssertionError(f"{len(overlap)} batch pairs are worked examples in the guide")
    redact.assert_clean(pd.Series([r["prompt"] for r in records], dtype=str),
                        "D50 batch dispatch")
    return records, scope, agent


def dispatch(replicate: bool = False, model: str = MODEL, scope: Scope | None = None,
             texts=None, agent: str | None = None) -> dict:
    """Write the batch's prompts to the bank. Idempotent per (pair_id, run); append-only."""
    records, _, agent = _plan(replicate, model, scope, texts, agent)
    already = {(r["pair_id"], r["run"]) for r in llm.read_jsonl(PROMPTS)}
    new = [r for r in records if (r["pair_id"], r["run"]) not in already]
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("a", encoding="utf-8") as fh:
        for record in new:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"run": run_of(replicate), "draw": DRAW, "model": model, "agent_sha256": agent,
            "pairs": len(new), "already_dispatched": len(records) - len(new),
            "prompts": str(PROMPTS)}


def dry_run(replicate: bool = False, scope: Scope | None = None, texts=None,
            agent: str | None = None) -> dict:
    """Everything `dispatch` checks, nothing written, nothing spawned."""
    records, scope, agent = _plan(replicate, MODEL, scope, texts, agent)
    run = run_of(replicate)
    answered = {r["pair_id"] for r in llm.read_jsonl(RAW) if r["run"] == run}
    owed = len({r["pair_id"] for r in records} - answered)
    chars = [r["chars_sent"] for r in records]
    return {"run": run, "draw": DRAW, "model": MODEL, "agent_sha256": agent,
            "replicate": replicate,
            "cap": REPLICATE_CAP if replicate else PAIR_CAP,
            "pairs_in_scope": len(records),
            "already_dispatched": len({r["pair_id"] for r in llm.read_jsonl(PROMPTS)
                                       if r["run"] == run}),
            "already_answered": len(answered), "owed": owed,
            "pairs_per_sitting_estimate": PAIRS_PER_SITTING,
            "sittings_estimate": math.ceil(owed / PAIRS_PER_SITTING),
            "chars_sent": {"median": int(np.median(chars)), "mean": int(np.mean(chars)),
                           "max": int(max(chars))},
            "spawned": 0, "written": 0}


# --- judge -----------------------------------------------------------------

def spawn_one(record: dict, cwd, scope: Scope, agent: str, allow_replicate: bool) -> dict:
    """`llm_recheck.judge_one`, behind the D50 guard instead of the D42 gate.

    The guard runs again here, per record, so no path reaches `subprocess.run` with a
    record it has not checked. A non-zero exit is never a return (F1).
    """
    assert_in_scope([record], scope, agent_sha256=agent, allow_replicate=allow_replicate)
    try:
        proc = subprocess.run(llm.judge_argv(record), input=record["prompt"],
                              cwd=str(cwd), capture_output=True, text=True,
                              timeout=llm.JUDGE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise llm.JudgeUnavailable("timeout", f"no answer in {llm.JUDGE_TIMEOUT_S}s") from exc
    except OSError as exc:
        raise llm.JudgeUnavailable("spawn", f"{type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        said = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        raise llm.JudgeUnavailable(
            llm.classify_failure(proc.returncode, proc.stderr),
            f"exit {proc.returncode}: {said[:300]}" if said else
            f"exit {proc.returncode}, both streams empty (a usage limit looks like this)")
    return {"pair_id": record["pair_id"], "run": record["run"],
            "returned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw": proc.stdout.strip()}


def judge(replicate: bool = False, concurrency: int = llm.CONCURRENCY,
          limit: int | None = None, scope: Scope | None = None,
          agent: str | None = None) -> dict:
    """Judge every dispatched pair of the run with no banked return. Idempotent; a lost
    spawn is quarantined and retried by re-running the same command (F1/F2)."""
    run = run_of(replicate)
    dispatched = [r for r in llm.read_jsonl(PROMPTS) if r["run"] == run]
    if not dispatched:
        raise FileNotFoundError(f"no run {run} in {PROMPTS} — `--dispatch` first")
    agent = agent or current_instrument()
    scope = scope or Scope.committed()
    assert_in_scope(dispatched, scope, agent_sha256=agent, allow_replicate=replicate)

    done = {(r["pair_id"], r["run"]) for r in llm.read_jsonl(RAW)}
    todo = [r for r in dispatched if (r["pair_id"], r["run"]) not in done]
    remaining = len(todo)
    if limit is not None:
        todo = todo[:limit]

    returned, kinds = 0, collections.Counter()
    with contextlib.ExitStack() as stack:
        cwd = llm.judge_sandbox(stack)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        fh = stack.enter_context(RAW.open("a", encoding="utf-8"))
        failed_fh = stack.enter_context(FAILED.open("a", encoding="utf-8"))
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = {pool.submit(spawn_one, rec, cwd, scope, agent, replicate): rec
                       for rec in todo}
            for future in concurrent.futures.as_completed(futures):
                rec = futures[future]
                try:
                    got = future.result()
                except llm.JudgeUnavailable as exc:
                    kinds[exc.kind] += 1
                    sink, row = failed_fh, {"pair_id": rec["pair_id"], "run": rec["run"],
                                            "failed_at": _now(), "kind": exc.kind,
                                            "detail": exc.detail}
                except OutOfScope:
                    raise
                except Exception as exc:                        # noqa: BLE001
                    kinds["error"] += 1
                    sink, row = failed_fh, {"pair_id": rec["pair_id"], "run": rec["run"],
                                            "failed_at": _now(), "kind": "error",
                                            "detail": f"{type(exc).__name__}: {exc}"}
                else:
                    sink, row = fh, got
                    returned += 1
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()

    summary = {"run": run, "model": MODEL, "dispatched": len(dispatched),
               "already_judged": len({p for p, r in done if r == run}),
               "attempted": len(todo), "judged_now": returned,
               "failed": sum(kinds.values()), "kinds": dict(kinds),
               "remaining": remaining - returned, "concurrency": concurrency,
               "limit": limit, "raw": str(RAW), "failed_log": str(FAILED)}
    if summary["failed"]:
        raise llm.JudgeFailures(summary)
    return summary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_coverage(run: int) -> dict:
    dispatched = {r["pair_id"] for r in llm.read_jsonl(PROMPTS) if r["run"] == run}
    answered = {r["pair_id"] for r in llm.read_jsonl(RAW) if r["run"] == run}
    quarantined = {r["pair_id"] for r in llm.read_jsonl(FAILED) if r["run"] == run}
    return {"run": run, "n_dispatched": len(dispatched), "n_answered": len(answered),
            "n_quarantined": len(quarantined - answered),
            "n_missing": len(dispatched - answered),
            "sittings_estimate_for_missing": math.ceil(
                len(dispatched - answered) / PAIRS_PER_SITTING),
            "complete": dispatched == answered and bool(dispatched)}


# --- collect (task 6.4) -----------------------------------------------------

def collect(out=None) -> dict:
    """Parse the banked returns into `llm-train-scores.csv`. Refuses a short run.

    Never `judgements.csv` (D43): the target is fixed, and refused if it is that file.
    """
    out = SCORES_CSV if out is None else out
    if out.resolve() == JUDGEMENTS.resolve() or out.name == JUDGEMENTS.name:
        raise ValueError(f"LLM scores never go to {JUDGEMENTS.name} (D43)")
    dispatched = {(r["pair_id"], r["run"]): r for r in llm.read_jsonl(PROMPTS)}
    if not dispatched:
        raise FileNotFoundError(f"{PROMPTS} is empty — `--dispatch` first")
    runs = sorted({run for _, run in dispatched})
    for run in runs:
        cov = run_coverage(run)
        if not cov["complete"]:
            raise llm.JudgeFailures({
                "run": run, "attempted": cov["n_dispatched"],
                "judged_now": cov["n_answered"], "failed": cov["n_missing"],
                "kinds": {"unanswered": cov["n_missing"]},
                "remaining": cov["n_missing"], "raw": str(RAW), "failed_log": str(FAILED)})

    rows, bad = [], []
    for got in llm.read_jsonl(RAW):
        key = (got["pair_id"], got["run"])
        if key not in dispatched:
            raise llm.BadReturn(f"{key} was never dispatched")
        sent = dispatched[key]
        if sent.get("draw") != DRAW or sent.get("model") != MODEL:
            raise llm.BadReturn(f"{key}: not a D50 batch record")
        try:
            parsed = llm.parse_return(got["raw"], require_score=True)
            ok, label, score = True, parsed["label"], parsed["score"]
        except llm.BadReturn:
            ok, label, score = False, "", None
            bad.append(key)
        rows.append({"agent_sha256": sent["agent_sha256"], "model": sent["model"],
                     "run": got["run"], "draw": sent["draw"], "pair_id": sent["pair_id"],
                     "jd_id": sent["query_id"], "cv_id": sent["doc_id"],
                     "split": sent.get("split", ""), "label": label, "score": score,
                     "parsed_ok": ok, "prompt_sha256": sent["prompt_sha256"],
                     "chars_sent": sent["chars_sent"],
                     "judged_at": got.get("returned_at", "")})
    frame = pd.DataFrame(rows, columns=list(SCORE_COLUMNS)).sort_values(
        ["run", "pair_id"], kind="stable")
    frame["score"] = frame.score.astype("Int64")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, lineterminator="\n")
    scored = frame.score.dropna()
    return {"rows": len(frame), "runs": runs, "parsed_ok": int(frame.parsed_ok.sum()),
            "unparsed": len(bad), "unparsed_pairs": [list(k) for k in bad[:10]],
            "all_parsed": not bad,
            "scores_in_range": bool(((scored >= llm.SCORE_MIN)
                                     & (scored <= llm.SCORE_MAX)).all()),
            "written": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, text in (("--build", "write the batch manifest and its digest"),
                       ("--check", "rebuild in memory and compare with the committed one"),
                       ("--verify-text-form", "rebuild run-9 prompts; hashes must match"),
                       ("--dry-run", "every dispatch check, then counts; spawns nothing"),
                       ("--dispatch", "write the prompts to the bank"),
                       ("--judge", "spawn the committed judge on unanswered pairs"),
                       ("--coverage", "dispatched vs answered vs quarantined"),
                       ("--collect", "write llm-train-scores.csv (refuses a short run)"),
                       ("--replicate", "act on run 12, the 100-pair replicate (Q-A1b)")):
        ap.add_argument(flag, action="store_true", help=text)
    ap.add_argument("--limit", type=int, default=None, help="judge at most N pairs")
    ap.add_argument("--concurrency", type=int, default=llm.CONCURRENCY)
    args = ap.parse_args(argv)

    def show(obj) -> None:
        print(json.dumps(obj, indent=2, default=str))

    if args.build:
        show(build())
    if args.check:
        show(check())
    if args.verify_text_form:
        show(verify_text_form())
    if args.dry_run:
        show(dry_run(args.replicate))
    if args.dispatch:
        show(dispatch(args.replicate))
    if args.judge:
        try:
            show(judge(args.replicate, args.concurrency, args.limit))
        except llm.JudgeFailures as exc:
            show(exc.summary)
            print(f"\n{exc}")
            return 1
    if args.coverage:
        show(run_coverage(run_of(args.replicate)))
    if args.collect:
        show(collect())
    if not any((args.build, args.check, args.verify_text_form, args.dry_run,
                args.dispatch, args.judge, args.coverage, args.collect)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
