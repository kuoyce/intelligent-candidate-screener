"""Acceptance test on the *derived* artefacts — `verify --derived`.

`verify --all` asks "are the right bytes on disk?"; this asks "is what we built from
them still sound?". Every check corresponds to a numbered acceptance criterion in
`plan/2026-08-23-derived-artefacts/01-design-spec.md` §4, and each one has a
failure mode that would otherwise be invisible: a leaked document does not raise, it
just quietly inflates a score.

Checks 3.3-3.5 arrive with their tasks; the registry below is deliberately open.

Usage:
    uv run python -m candidate_screener.data.verify --derived
"""
from __future__ import annotations

import json

import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.data import a2_finetune as a2
from candidate_screener.data import fit_split as fs
from candidate_screener.data.ids import normalise


def _probe_c4_overlap(splits: dict[str, pd.DataFrame]) -> tuple[list[dict], str]:
    """Locate the 7 C4 resumes whose text is embedded in an A1 resume.

    These documents are legitimately in A1 — the guard is not against them, it is
    against a future re-import of the rejected `vm-structured-onet` set landing the
    *same* people in a training split while they are also being scored in test.
    Recording which split they fell into is what makes that exclusion applyable
    later rather than merely remembered.
    """
    path = PROCESSED / "vm-structured-onet" / "resumes_raw_structured.json"
    if not path.exists():
        return [], "skipped — rejected C4 derivatives not on disk"

    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = list(records.values())
    probes = {}
    for r in records:
        objective = normalise(r.get("career_objective") or "")
        if len(objective) >= 60:
            probes[objective[:60]] = r.get("resume_id") or r.get("id")

    hits = []
    for name, df in splits.items():
        texts = [(rid, normalise(t)) for rid, t in zip(df.resume_id, df.resume_text)]
        seen = set()
        for probe, c4_id in probes.items():
            for rid, text in texts:
                if rid not in seen and probe in text:
                    hits.append({"doc_id": rid, "split": name, "c4_resume_id": c4_id})
                    seen.add(rid)
                    break
    return hits, f"{len({h['doc_id'] for h in hits})} A1 resumes overlap the rejected C4 set"


def check_fit_split() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    if not fs.SPLIT_MANIFEST.exists():
        return [(False, "fit-split.csv missing — run `build --task fit-split`")]

    manifest = pd.read_csv(fs.SPLIT_MANIFEST)
    yields = json.loads(fs.YIELD_MANIFEST.read_text(encoding="utf-8"))

    # 1. Every manifest ID resolves to exactly one document in data/raw/.
    pairs, _ = fs.load_pooled()
    raw_ids = {"resume": set(pairs.resume_id), "jd": set(pairs.jd_id)}
    unresolved = sum(
        len(set(g.doc_id) - raw_ids[t]) for t, g in manifest.groupby("doc_type"))
    results.append((unresolved == 0 and manifest.doc_id.is_unique,
                    f"identity: {len(manifest)} manifest IDs, {unresolved} unresolved, "
                    f"{'no' if manifest.doc_id.is_unique else 'DUPLICATE'} repeats"))

    # 2. Split disjointness — the guarantee the whole artefact exists to provide.
    multi = int((manifest.groupby("doc_id").split.nunique() > 1).sum())
    built = {n: pd.read_parquet(p) for n in ("train", "val", "test")
             if (p := fs.FIT_OUT / f"{n}.parquet").exists()}
    overlaps = []
    names = sorted(built)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for col in ("resume_id", "jd_id"):
                n = len(set(built[a][col]) & set(built[b][col]))
                if n:
                    overlaps.append(f"{a}/{b} share {n} {col}")
    results.append((multi == 0 and not overlaps,
                    f"disjointness: {multi} documents in >1 split; "
                    f"{'; '.join(overlaps) if overlaps else 'no cross-split document overlap'}"))

    # 3. The discarded pairs are accounted for, not silently lost.
    c = yields["counts"]
    total = sum(c[k]["pairs"] for k in ("train", "val", "test", "discarded"))
    usable = yields["pooling"]["usable_pairs"]
    results.append((total == usable,
                    f"accounting: {c['discarded']['pairs']} discarded + "
                    f"{total - c['discarded']['pairs']} assigned = {total} of {usable} usable"))

    # 4. Byte-reproducibility: rebuild the manifest in memory and diff it.
    assigned, r_split, j_split = fs.split_pairs(pairs, yields["test_frac"],
                                                yields["val_frac"], yields["seed"])
    rebuilt = pd.concat([
        pd.DataFrame({"doc_id": r_split.index, "doc_type": "resume", "split": r_split.to_numpy()}),
        pd.DataFrame({"doc_id": j_split.index, "doc_type": "jd", "split": j_split.to_numpy()}),
    ]).sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True)
    same = rebuilt.equals(manifest.reset_index(drop=True))
    results.append((same, f"determinism: rebuild at seed {yields['seed']} "
                          f"{'reproduces' if same else 'DIFFERS FROM'} the committed manifest"))

    # 5. Contamination guard against the rejected C4 derivatives.
    hits, note = _probe_c4_overlap(built)
    if hits:
        pd.DataFrame(hits).sort_values(["split", "doc_id"]).to_csv(
            fs.MANIFESTS / "fit-c4-overlap.csv", index=False, lineterminator="\n")
    results.append((True, f"contamination guard: {note}"
                          f"{' -> fit-c4-overlap.csv' if hits else ''}"))
    return results


def check_pools() -> list[tuple[bool, str]]:
    from candidate_screener.data import pools as pl
    from candidate_screener.evaluation import metrics as mx

    if not pl.POOL_MANIFEST.exists():
        return [(False, "pools.csv missing — run `build --task pools`")]
    pools = pd.read_csv(pl.POOL_MANIFEST)
    results: list[tuple[bool, str]] = []

    # 1. Every query is in the evaluation split — a pool over a training JD would
    #    score a model on documents it was fitted on.
    manifest = pd.read_csv(fs.SPLIT_MANIFEST)
    test_jds = set(manifest[(manifest.doc_type == "jd") & (manifest.split == "test")].doc_id)
    test_resumes = set(manifest[(manifest.doc_type == "resume") & (manifest.split == "test")].doc_id)
    stray_q = set(pools.query_jd_id) - test_jds
    stray_c = set(pools.candidate_resume_id) - test_resumes
    results.append((not stray_q and not stray_c,
                    f"provenance: {len(stray_q)} queries and {len(stray_c)} candidates "
                    f"outside the test split"))

    # 2. No distractor carries a label against its own query — that would score a
    #    judged document as an assumed non-relevant one.
    judged = pd.read_parquet(fs.FIT_OUT / "test.parquet")
    judged_pairs = set(zip(judged.jd_id, judged.resume_id))
    distractors = pools[pools.source == "distractor"]
    mislabelled = sum((q, c) in judged_pairs for q, c in
                      zip(distractors.query_jd_id, distractors.candidate_resume_id))
    results.append((mislabelled == 0,
                    f"distractors: {len(distractors)} rows, {mislabelled} of them actually judged"))

    # 3. Pool depth matches the variant, allowing the documented overshoot where a
    #    query carries more judgements than the target depth.
    problems = []
    for variant, target in pl.VARIANTS.items():
        size = pools[pools.pool_variant == variant].groupby("query_jd_id").size()
        judged_n = (pools[(pools.pool_variant == variant) & (pools.source == "labelled")]
                    .groupby("query_jd_id").size())
        expected = judged_n.combine(pd.Series(target, index=size.index), max) if target else None
        if target is None:
            ok = size.nunique() == 1
        else:
            ok = bool((size == expected.reindex(size.index)).all())
        if not ok:
            problems.append(f"{variant} sizes {size.min()}-{size.max()}")
    results.append((not problems,
                    f"depth: {'; '.join(problems) if problems else 'every pool matches its variant'}"))

    # 4. Variants are nested, so a sensitivity run differs only by depth.
    sets = {v: set(zip(g.query_jd_id, g.candidate_resume_id))
            for v, g in pools.groupby("pool_variant")}
    nested = sets["N20"] <= sets["N100"] <= sets["Nfull"]
    results.append((nested, f"nesting: N20 subset of N100 subset of Nfull is {nested}"))

    # 5. The metric guards fire — the reporting discipline is code, not convention.
    scores = mx.random_scores(pools, 0)
    guards = []
    try:
        mx.score(pools, scores, "N20", "Recall", 50, "strict")
        guards.append("Recall@50 on N20 was allowed")
    except ValueError:
        pass
    try:
        mx.Figure("x", "strict", "N100", 0.5, 0, 0.1, 0.9)
        guards.append("a figure with n=0 was constructible")
    except ValueError:
        pass
    results.append((not guards,
                    f"metric guards: {'; '.join(guards) if guards else 'k>depth and n=0 both refused'}"))
    return results


def check_a2_partition() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    if not a2.PARTITION_MANIFEST.exists():
        return [(False, "a2-partition.csv missing — run `build --task a2-partition`")]
    manifest = pd.read_csv(a2.PARTITION_MANIFEST)

    # 1. Every doc_id is assigned exactly once and resolves to a real A2 id.
    from candidate_screener.data.profile import load_split
    jd_ids = set(load_split("djinni-jd", "train")["id"])
    cv_ids = set(load_split("djinni-cv", "train")["id"])
    universe = {"jd": jd_ids, "cv": cv_ids}
    unresolved = sum(len(set(g.doc_id) - universe[t]) for t, g in manifest.groupby("doc_type"))
    results.append((unresolved == 0 and manifest.doc_id.is_unique,
                    f"identity: {len(manifest)} manifest ids, {unresolved} unresolved, "
                    f"{'no' if manifest.doc_id.is_unique else 'DUPLICATE'} repeats"))

    # 2. eval/train is the only guarantee this artefact exists to provide (D19/Q16):
    #    zero doc_ids in more than one region.
    multi = int((manifest.groupby("doc_id").region.nunique() > 1).sum())
    results.append((multi == 0, f"disjointness: {multi} documents assigned to >1 region"))

    # 3. Determinism: rebuild in memory at the recorded seed/fraction and diff.
    if not a2.PARTITION_REPORT.exists():
        return results + [(False, "a2-partition-report.json missing — cannot check determinism")]
    report = json.loads(a2.PARTITION_REPORT.read_text(encoding="utf-8"))
    seed, eval_fraction = report["seed"], report["eval_fraction"]
    jd_region, cv_region = a2.partition_ids(sorted(jd_ids), sorted(cv_ids), eval_fraction, seed)
    rebuilt = pd.concat([
        pd.DataFrame({"doc_id": jd_region.index, "doc_type": "jd", "region": jd_region.to_numpy()}),
        pd.DataFrame({"doc_id": cv_region.index, "doc_type": "cv", "region": cv_region.to_numpy()}),
    ]).sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True)
    same = rebuilt.equals(manifest.reset_index(drop=True))
    results.append((same, f"determinism: rebuild at seed={seed} eval_fraction={eval_fraction:.0%} "
                          f"{'reproduces' if same else 'DIFFERS FROM'} the committed manifest"))
    return results


def check_a2_shortlist() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    if not a2.SHORTLIST_MANIFEST.exists():
        return [(False, "a2-datascience-shortlist.csv missing — run `build --task a2-shortlist`")]
    if not a2.PARTITION_MANIFEST.exists():
        return [(False, "a2-partition.csv missing — the shortlist cannot be checked without it")]

    shortlist = pd.read_csv(a2.SHORTLIST_MANIFEST)
    partition = pd.read_csv(a2.PARTITION_MANIFEST)
    train_jds = set(partition[(partition.doc_type == "jd") & (partition.region == "train")].doc_id)
    train_cvs = set(partition[(partition.doc_type == "cv") & (partition.region == "train")].doc_id)

    # Every shortlisted pair must draw from the training region only — this is the
    # entire point of D19: a fine-tuning label must never land in the eval region.
    stray_jd = set(shortlist.jd_id) - train_jds
    stray_cv = set(shortlist.cv_id) - train_cvs
    results.append((not stray_jd and not stray_cv,
                    f"eval-region isolation: {len(stray_jd)} JDs and {len(stray_cv)} CVs "
                    f"outside the training region"))

    results.append((shortlist.primary_keyword.isin(a2.DATASCIENCE_KEYWORDS).all(),
                    f"scope: every pair's primary_keyword is in {a2.DATASCIENCE_KEYWORDS}"))

    results.append((shortlist.pair_id.is_unique, "identity: pair_id has no duplicates"))
    return results


def check_indomain() -> list[tuple[bool, str]]:
    """Task 3.4b's 200-pair set and the D18 holdout region."""
    from candidate_screener.annotation import sample

    if not sample.PAIRS_MANIFEST.exists():
        return [(False, "indomain-pairs.csv missing — run `annotation.sample --build`")]
    if not a2.PARTITION_MANIFEST.exists():
        return [(False, "a2-partition.csv missing — the eval-region check needs it")]

    pairs = pd.read_csv(sample.PAIRS_MANIFEST)
    holdout = pd.read_csv(sample.HOLDOUT_MANIFEST)
    partition = pd.read_csv(a2.PARTITION_MANIFEST)
    eval_jds = set(partition[(partition.doc_type == "jd") & (partition.region == "eval")].doc_id)
    eval_cvs = set(partition[(partition.doc_type == "cv") & (partition.region == "eval")].doc_id)

    # The mirror image of the shortlist check above, and the other half of D19. Together
    # they are what makes "fine-tune on train, evaluate on eval" an assertion rather than
    # an intention.
    stray_jd, stray_cv = set(pairs.jd_id) - eval_jds, set(pairs.cv_id) - eval_cvs
    results = [(not stray_jd and not stray_cv,
                f"train-region isolation: {len(stray_jd)} JDs and {len(stray_cv)} CVs "
                f"outside the eval region")]

    shortlist_ids = set()
    if a2.SHORTLIST_MANIFEST.exists():
        shortlist = pd.read_csv(a2.SHORTLIST_MANIFEST)
        shortlist_ids = set(shortlist.jd_id) | set(shortlist.cv_id)
    overlap = (set(pairs.jd_id) | set(pairs.cv_id)) & shortlist_ids
    results.append((not overlap,
                    f"fine-tuning disjointness: {len(overlap)} documents shared with the "
                    f"data-science shortlist"))

    # D18: the *whole* candidate pool is reserved, not the 200 drawn pairs. Reserving only
    # the pairs would leave a later expansion of this set judging documents that pretraining
    # has already seen.
    covered = set(holdout.doc_id)
    missing = (set(pairs.jd_id) | set(pairs.cv_id)) - covered
    results.append((not missing,
                    f"holdout coverage: {len(missing)} evaluation documents outside "
                    f"indomain-holdout.csv"))
    results.append((len(covered) > len(set(pairs.jd_id) | set(pairs.cv_id)),
                    f"holdout is the pool, not the sample: {len(covered)} documents "
                    f"reserved for {len(pairs)} pairs"))

    results.append((pairs.pair_id.is_unique, "identity: pair_id has no duplicates"))

    # D28: batches are frozen and append-only. A document appearing in two batches means
    # a later batch re-drew what an earlier one took, which is exactly the failure that
    # orphans already-collected labels.
    if sample.BATCHES.exists():
        import json
        specs = json.loads(sample.BATCHES.read_text())["batches"]
        numbers = [b["batch"] for b in specs]
        results.append((len(numbers) == len(set(numbers)),
                        f"campaign: {len(numbers)} batch specs, numbers unique"))
        results.append((set(pairs.batch) <= set(numbers),
                        f"campaign: every pair's batch is a declared spec"))
        cv_batches = pairs.groupby("cv_id").batch.nunique()
        results.append(((cv_batches == 1).all(),
                        f"batch disjointness: {int((cv_batches > 1).sum())} CVs drawn by "
                        f"more than one batch"))
        # D29: stratum is the reporting unit and is stamped on the pair, not looked up
        # from the spec, so editing a spec cannot re-stratify collected judgements.
        declared = {b["batch"]: sample.stratum_of(b) for b in specs}
        mismatched = sum(sample.stratum_of(next(b for b in specs if b["batch"] == row.batch))
                         != row.stratum for row in pairs.itertuples())
        results.append((not mismatched,
                        f"strata: {mismatched} pairs whose recorded stratum disagrees with "
                        f"their batch spec"))
        results.append((set(pairs.stratum) <= set(sample.STRATA),
                        f"strata: every pair is in {sample.STRATA}"))
        counts = pairs.stratum.value_counts().to_dict()
        results.append((True, f"strata: {counts} — batches within a stratum MAY be pooled; "
                              f"the two strata MAY NOT be averaged together (D29). "
                              f"targeted = batches "
                              f"{sorted(b for b, k in declared.items() if k == 'targeted') or 'none'}"))
    return results


def check_judging_queue() -> list[tuple[bool, str]]:
    """The dispatch queue: blinding, PII, and that nothing is judged twice."""
    from candidate_screener.annotation import queue, redact

    if not queue.QUEUE_MANIFEST.exists():
        return [(False, "judging-queue.csv missing — run `annotation.queue --build`")]

    key = pd.read_csv(queue.QUEUE_MANIFEST)
    results = [(key.queue_id.is_unique, "identity: queue_id has no duplicates"),
               (not key.duplicated(["corpus", "query_id", "doc_id"]).any(),
                "no pair is dispatched twice")]

    # Blinding is a property of the dispatch file, not of the repository. The committed
    # key carries selection_reason on purpose — it is the audit trail; what must not carry
    # it is the file an annotator opens.
    dispatch = queue.DISPATCH_OUT / "judging-queue.csv"
    if dispatch.exists():
        sent = pd.read_csv(dispatch)
        leaked = set(sent.columns) - set(queue.DISPATCH_COLUMNS)
        results.append((not leaked, f"blinding: dispatch carries {sorted(leaked) or 'no'} "
                                    f"columns beyond {list(queue.DISPATCH_COLUMNS)}"))
        residual = {}
        for column in ("query_text", "candidate_text"):
            residual |= {f"{column}.{k}": v
                         for k, v in redact.residual_pii(sent[column]).items() if v}
        results.append((not residual, f"de-identification: {residual or 'zero'} detectable "
                                      f"PII in the dispatched text"))
    else:
        results.append((True, f"blinding: {dispatch} not built (git-ignored) — skipped"))

    # D28: the queue is rebuilt from what is *not* judged, so a rebuild must never
    # re-present a labelled pair. This is the resume mechanism's only invariant.
    done = queue.judged_pair_ids()
    if done:
        redispatched = {f"{q}__{d}" for q, d in zip(key.query_id, key.doc_id)} & done
        results.append((not redispatched,
                        f"resume: {len(redispatched)} already-judged pairs re-dispatched"))

    if queue.JUDGEMENTS.exists():
        judged = pd.read_csv(queue.JUDGEMENTS)
        results.append((list(judged.columns) == list(queue.JUDGEMENT_COLUMNS),
                        "judgement layer: schema matches the committed definition"))
        # D20: our labels are a separate layer. pools.csv is a pure function of
        # (data/raw/, seed) and check 6 asserts it reproduces byte-for-byte; a human
        # judgement is not a function of a seed and cannot live there.
        results.append(("relevance" not in judged.columns,
                        "judgement layer: carries no `relevance` column — it overlays "
                        "pools.csv, it does not become it (D20)"))
        results += check_collected_labels(judged)
    return results


def check_collected_labels(judged: pd.DataFrame) -> list[tuple[bool, str]]:
    """What the labelling UI wrote, checked against the manifest that defines the pairs.

    `annotation.ui` appends rows here directly — there is no import step to inspect them
    on the way in, which is the point (an import step is a second copy of the truth that
    can disagree with the first). These checks are what replaces it, and the one that
    earns its place is the stratum agreement: `stratum` is stamped onto the judgement at
    labelling time *(D29)* so that editing a batch spec cannot re-stratify collected
    labels, and the failure mode of stamping is a stale value silently disagreeing with
    the pair it names. Every figure in the report is quoted per stratum, so a row filed
    under the wrong one is not a rounding error.
    """
    from candidate_screener.annotation import session

    results: list[tuple[bool, str]] = []
    judged = judged[judged.pair_id.notna()]
    if judged.empty:
        return [(True, "collected labels: none yet — the session has not run")]

    bad_label = sorted(set(judged.label.dropna()) - set(session.LABELS))
    results.append((not bad_label, f"collected labels: {bad_label or 'no'} labels outside "
                                   f"A1's 3-class scheme {list(session.LABELS)} (D17)"))

    twice = judged.groupby(["pair_id", "annotator"]).size()
    twice = twice[twice > 1]
    results.append((twice.empty, f"collected labels: {len(twice)} pair(s) labelled twice "
                                 "by the same annotator — an overlap of one person with "
                                 "themselves is not an agreement measurement"))

    indomain = judged[judged.corpus == "a2"]
    if not indomain.empty:
        pairs = session.load_pairs().set_index("pair_id")
        unknown = sorted(set(indomain.pair_id) - set(pairs.index))
        results.append((not unknown, f"collected labels: {len(unknown)} in-domain pair(s) "
                                     "name no row in indomain-pairs.csv"))
        known = indomain[indomain.pair_id.isin(pairs.index)]
        for column in ("batch", "stratum"):
            expected = known.pair_id.map(pairs[column]).astype(str)
            drift = known[expected != known[column].astype(str)]
            results.append((drift.empty, f"collected labels: {len(drift)} judgement(s) "
                                         f"whose `{column}` disagrees with the manifest "
                                         "(D29 — it is stamped, never recomputed)"))

        picks = known[known.shortlist_pick.astype(str).isin(("1", "True", "true"))]
        many = picks.groupby(["annotator", "query_id"]).size()
        many = many[many > 1]
        results.append((many.empty, f"collected labels: {len(many)} JD(s) where one "
                                    "annotator recorded more than one top-1 shortlist "
                                    "pick (Q14 asks for exactly one)"))
    return results


def check_llm_recheck() -> list[tuple[bool, str]]:
    """Provenance, never determinism *(D33, closing Q31)*.

    `llm-recheck.csv` is **collected data**: no seed reproduces a subagent run, so the
    byte-comparison every other check in this file performs would be red on every run and
    would train a reader to ignore it. What can be checked is that each row names the
    instrument that produced it — the agent definition and the exact prompt — and that the
    judge's labels never leaked into the human file, where `annotator` is a free string and
    an `llm` row would be silently pooled into the human kappa.
    """
    from candidate_screener.annotation import llm_recheck as llm
    from candidate_screener.annotation import queue, session

    results: list[tuple[bool, str]] = []
    results.append((llm.agent_definition_is_current(),
                    "judge: .claude/agents/a1-judge.md is what docs/annotation-guide.md "
                    "renders — a stale one means a row's agent_sha256 names an "
                    "instrument that never judged it"))

    front = (llm.AGENT_DEF.read_text(encoding="utf-8").split("---")[1]
             if llm.AGENT_DEF.exists() else "")
    results.append(("tools: []" in front,
                    "judge: holds no tools — judging-queue-full.parquet carries a1_label "
                    "for every recheck pair, so blinding has to be structural (D33)"))

    if queue.JUDGEMENTS.exists():
        judged = pd.read_csv(queue.JUDGEMENTS, dtype=str)
        machine = judged[judged.annotator.fillna("").str.startswith("llm:")]
        results.append((machine.empty,
                        f"separation: {len(machine)} machine judgement(s) found in "
                        "judgements.csv — the LLM is a third opinion, never a human "
                        "annotator (D33)"))

    if not llm.RECHECK_CSV.exists():
        results.append((True, f"{llm.RECHECK_CSV.name} not built — skipped"))
        return results

    frame = pd.read_csv(llm.RECHECK_CSV)
    results.append((list(frame.columns) == list(llm.LLM_COLUMNS),
                    "schema: llm-recheck.csv matches LLM_COLUMNS"))

    dispatched = {(r["pair_id"], r["run"]): r for r in llm.read_jsonl(llm.PROMPTS)}
    if dispatched:
        drift = [(r.pair_id, r.run) for r in frame.itertuples()
                 if (r.pair_id, r.run) not in dispatched
                 or dispatched[(r.pair_id, r.run)]["prompt_sha256"] != r.prompt_sha256]
        results.append((not drift,
                        f"provenance: {len(drift)} row(s) whose prompt_sha256 does not "
                        "match the prompt that was dispatched for that pair and run"))
    else:
        results.append((True, "provenance: dispatch file not built (git-ignored) — "
                              "prompt_sha256 unverifiable on this machine"))

    scored = frame[frame.parsed_ok.astype(str).str.lower().isin(("true", "1"))]
    bad = sorted(set(scored.label.dropna()) - set(session.LABELS))
    results.append((not bad, f"scheme: {bad or 'no'} labels outside A1's 3 classes"))

    recheck_ids, human_ids = set(), set()
    ids = lambda f: set(f.query_id.astype(str) + "__" + f.doc_id.astype(str))  # noqa: E731
    try:
        recheck_ids = ids(queue.a1_recheck(queue.LLM_RECHECK_STRATA, 0))
        human_ids = ids(queue.a1_recheck(queue.RECHECK_STRATA, 0))
    except (FileNotFoundError, OSError):
        pass
    if recheck_ids:
        stray = sorted(set(frame.pair_id) - recheck_ids)
        results.append((not stray, f"scope: {len(stray)} judged pair(s) are not among the "
                                   f"{len(recheck_ids)} A1 recheck pairs"))
        # The published n=50 figures are quoted beside figures over the wider draw. That
        # is only honest while the narrow draw is a subset of the wide one.
        outside = sorted(human_ids - recheck_ids)
        results.append((not outside,
                        f"nesting: {len(outside)} of the human {len(human_ids)} are absent "
                        f"from the LLM {len(recheck_ids)} — a figure over one may not be "
                        "quoted beside a figure over the other unless one nests"))
    return results


#: Task -> its acceptance checks. 3.1 and 3.5 register here as they land.
CHECKS = {
    "fit-split": check_fit_split,
    "pools": check_pools,
    "a2-partition": check_a2_partition,
    "a2-shortlist": check_a2_shortlist,
    "indomain": check_indomain,
    "judging-queue": check_judging_queue,
    "llm-recheck": check_llm_recheck,
}


def main(tasks: list[str] | None = None) -> int:
    failed = 0
    for name, fn in CHECKS.items():
        if tasks and name not in tasks:
            continue
        print(f"\n=== {name}")
        for ok, message in fn():
            print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
            failed += not ok
    print(f"\n{'all derived checks passed' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0
