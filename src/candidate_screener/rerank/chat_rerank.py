"""Stage-2 LLM reranking by hand through a chat window (plan 2026-10-17-llm-rerank, D59/D60).

For every one of the 60 final-set JDs, X2 shortlists the top `K` of its 500-CV pool. The shortlist
is shuffled (seed 0), given short ids `C001`..`C100`, and split into batches of 20. The owner pastes
each batch's prompt into a chat (one conversation per JD), pastes the JSON reply back, and the
reply is validated and appended to `REPLIES` (raw text and parsed rows, never document text).

The prompt is a **new listwise instrument**: the annotation guide, verbatim from the `a1-judge`
agent, with its one-CV framing replaced by a numbered list and a JSON-array reply. X2 scores and
ranks are never shown. Every document passes `annotation.redact`.

**Extension (D61 amended, 18 Oct 2026).** Stage 1 became S1's passed set (plan
2026-10-18-classifier-then-rerank), and a few passed CVs lie outside X2's top 100. `--extend-s1`
appends them as one more batch per JD (short ids after `C100`), pasted into the **same chat** as
that JD's earlier batches so the scale carries over. Same instrument, same series.

    uv run python -m candidate_screener.rerank.chat_rerank --build       # the plan (ids only)
    uv run python -m candidate_screener.rerank.chat_rerank --extend-s1   # + S1 passed CVs outside it
    uv run python -m candidate_screener.rerank.chat_rerank --sample      # one batch to a text file
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from candidate_screener.config import DATA, OUTPUT, PROJECT_ROOT

K = 100
BATCH = 20
SEED = 0
MODEL = "claude-haiku-5-5 (claude.ai chat)"
RUN = "chat-1"
#: Step 4a (plan 2026-10-18, D61 amended): fresh chats for S1's passed set at P >= 0.3, outside the
#: pairs already scored, padded with already-scored anchors. Own run label, own short ids
#: (`F001`...) and batches numbered from 101, so no row collides with `chat-1`'s.
FRESH_RUN, FRESH_PREFIX, FRESH_BATCH0, MIN_ANCHORS = "chat-2", "F", 100, 5
AGENT = PROJECT_ROOT / ".claude" / "agents" / "a1-judge.md"
PLAN = DATA / "interim" / "chat-rerank-plan.csv"
REPLIES = OUTPUT / "rerank" / "chat-rerank-replies.csv"
#: D62 (plan 2026-10-21-chat3-holdout): `chat-3`, 48 train-region JDs, same instrument and model.
#: Its own plan and replies, so nothing that reads `chat-1` / `chat-2` ever sees a chat-3 row.
CHAT3_RUN = "chat-3"
CHAT3_PLAN = DATA / "interim" / "chat3-plan.csv"
CHAT3_REPLIES = OUTPUT / "rerank" / "chat3-replies.csv"
#: Plan 2026-10-22-c2-validation: `chat-4`, the pooled unscored top-10 CVs of the 60 final-set
#: JDs, fresh chats with anchors; evaluation only. Its own files, as chat-3.
CHAT4_RUN, CHAT4_PREFIX = "chat-4", "H"
CHAT4_PLAN = DATA / "interim" / "chat4-plan.csv"
CHAT4_REPLIES = OUTPUT / "rerank" / "chat4-replies.csv"
LABELS = ("Good Fit", "Potential Fit", "No Fit")
#: Curly double quotes a phone or chat app substitutes for JSON's straight ones.
SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "″": '"'})
X2 = "extracted rule + TF-IDF"

HEADER = """You are one annotator in a labelling session. You will be shown **one** job description
and a numbered list of candidate CVs (batch {b} of {n} for this job; any earlier batches for the
same job are above in this conversation). Judge **each CV on its own** against the job, using the
guide below, and keep the same scale across every batch of this job.

Two rules override anything else:

1. **Reply with exactly one JSON array and no other text.** No prose before it, no markdown fence
   around it, no explanation after it. One object per CV in this batch, every id exactly once:
   `[{{"cv":"C001","label":"Good Fit"|"Potential Fit"|"No Fit","score":<0-100 integer>,"reason":"<=80 chars"}}, ...]`
2. **Judge only what is in front of you.** You have no tools and nothing to look up. If a CV is
   thin, that is evidence about the CV, not a reason to seek more.

Everything below is the annotation guide, unedited. It is the instrument; apply it as written,
including the tie-breaking rule and the what-not-to-consider list.

---
"""
FOLLOW_UP = """Batch {b} of {n} for the same job. Same instructions and the same scale as before.
Reply with exactly one JSON array covering these {m} ids and nothing else.
"""


def guide() -> str:
    """The annotation guide as the `a1-judge` agent carries it: everything after the first `---` rule."""
    text = AGENT.read_text(encoding="utf-8")
    body = text.split("\n---\n", 2)[2]          # [frontmatter, agent preamble, guide]
    return body.strip() + "\n"


def instrument_sha256() -> str:
    return hashlib.sha256((HEADER + FOLLOW_UP + guide()).encode("utf-8")).hexdigest()


def plan_from(pools: pd.DataFrame, scores: np.ndarray, k: int = K, batch: int = BATCH, seed: int = SEED) -> pd.DataFrame:
    """Per JD: X2's top `k` (ties by pair id), shuffled, short ids, batch numbers. Ids only."""
    d = pools[["jd_id", "cv_id", "pair_id", "stratum"]].assign(x2=np.asarray(scores, dtype=float))
    rows = []
    for j, (jd, g) in enumerate(d.groupby("jd_id", sort=True)):
        top = g.sort_values(["x2", "pair_id"], ascending=[False, True]).head(k).assign(x2_rank=np.arange(1, min(k, len(g)) + 1))
        top = top.sample(frac=1.0, random_state=np.random.default_rng([seed, j]).integers(2**31)).reset_index(drop=True)
        top["short_id"] = [f"C{i + 1:03d}" for i in range(len(top))]
        top["batch"] = top.index // batch + 1
        top["jd_order"] = j + 1
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def extend(plan: pd.DataFrame, extra: pd.DataFrame, source: str) -> pd.DataFrame:
    """`plan` plus the `extra` pairs it lacks: per JD one more batch, short ids after the JD's last.

    `extra` carries `jd_id`, `cv_id`, `pair_id`; rows are added in pair-id order. A JD with no new
    pair gets no batch. Ids only, like the plan."""
    plan = plan.assign(source=plan["source"] if "source" in plan else "x2_top100")
    new = extra[~extra.pair_id.isin(set(plan.pair_id))].sort_values("pair_id")
    rows = []
    for jd, g in new.groupby("jd_id", sort=True):
        mine = plan[plan.jd_id == jd]
        if mine.empty:
            raise ValueError(f"{jd} is not in the plan: an extension adds CVs, never a JD")
        last = int(mine.short_id.str[1:].astype(int).max())
        rows.append(pd.DataFrame({
            "jd_id": jd, "cv_id": g.cv_id.to_numpy(), "pair_id": g.pair_id.to_numpy(),
            "stratum": mine.stratum.iloc[0], "short_id": [f"C{last + i + 1:03d}" for i in range(len(g))],
            "batch": int(mine.batch.max()) + 1, "jd_order": int(mine.jd_order.iloc[0]), "source": source}))
    if not rows:
        return plan
    return pd.concat([plan, *rows], ignore_index=True)


def n_anchors(n_new: int, batch: int = BATCH, minimum: int = MIN_ANCHORS) -> int:
    """Anchors that fill `n_new` up to a multiple of `batch`, never fewer than `minimum`."""
    fill = (-n_new) % batch
    while fill < minimum:
        fill += batch
    return fill


def anchors(scored: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    """`k` of one JD's scored pairs, spread evenly over its score order (systematic sample)."""
    order = scored.sort_values(["score", "pair_id"]).reset_index(drop=True)
    if k > len(order):
        raise ValueError(f"{k} anchors asked for {len(order)} scored pairs")
    pos = np.floor((np.arange(k) + rng.uniform()) * len(order) / k).astype(int)
    return order.iloc[pos]


def fresh(plan: pd.DataFrame, scored: pd.DataFrame, new: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """`plan` plus one fresh chat per JD with `new` pairs: the new pairs and `n_anchors` already-scored
    ones (`scored`: `pair_id`, `jd_id`, `cv_id`, `score`), shuffled, ids `F001`..., batches of 20
    numbered from 101. Ids only. Refuses a plan that already has fresh rows."""
    plan = plan.assign(run=plan["run"] if "run" in plan else RUN)
    if (plan.run == FRESH_RUN).any():
        raise ValueError(f"the plan already has {FRESH_RUN} rows: never rebuild a collection")
    new = new[~new.pair_id.isin(set(plan.pair_id))]
    rows = []
    for jd, g in new.sort_values("pair_id").groupby("jd_id", sort=True):
        mine = plan[plan.jd_id == jd]
        if mine.empty:
            raise ValueError(f"{jd} is not in the plan: a fresh chat adds CVs, never a JD")
        rng = np.random.default_rng([seed, int(mine.jd_order.iloc[0])])
        a = anchors(scored[scored.jd_id == jd], n_anchors(len(g)), rng)
        both = pd.concat([g[["jd_id", "cv_id", "pair_id"]].assign(source="s1_t030"),
                          a[["jd_id", "cv_id", "pair_id"]].assign(source="anchor")], ignore_index=True)
        both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
        rows.append(both.assign(stratum=mine.stratum.iloc[0], jd_order=int(mine.jd_order.iloc[0]), run=FRESH_RUN,
                                short_id=[f"{FRESH_PREFIX}{i + 1:03d}" for i in range(len(both))],
                                batch=FRESH_BATCH0 + 1 + both.index // BATCH))
    return pd.concat([plan, *rows], ignore_index=True) if rows else plan


def chat_batches(plan: pd.DataFrame, jd_id: str, b: int) -> tuple[int, int]:
    """(position of batch `b` within its chat, batches in that chat): a run is one chat per JD."""
    g = plan[plan.jd_id == jd_id]
    if "run" in g:
        g = g[g.run.fillna(RUN) == g[g.batch == b].run.fillna(RUN).iloc[0]]
    batches = sorted(g.batch.unique())
    return batches.index(b) + 1, len(batches)


def prompt(jd_text: str, cvs: list[tuple[str, str]], b: int, n: int, first: bool) -> str:
    """The text to paste for one batch: full instrument on a JD's first batch, a short follow-up after."""
    from candidate_screener.annotation.redact import redact_text
    head = (HEADER.format(b=b, n=n) + guide() + "\n---\n\n## Job description\n\n" + redact_text(jd_text).strip() + "\n\n"
            if first else FOLLOW_UP.format(b=b, n=n, m=len(cvs)) + "\n")
    body = "".join(f"## CV {sid}\n\n{redact_text(text).strip()}\n\n" for sid, text in cvs)
    return head + "## CVs in this batch\n\n" + body


def parse_reply(text: str, expected: list[str]) -> pd.DataFrame:
    """Validate one pasted reply against the batch's ids. Raises ValueError with the reason."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)          # tolerate a fence, nothing else
    s = s.translate(SMART_QUOTES)                            # autocorrect on the paste path (17 Oct 2026)
    try:
        items = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    if not isinstance(items, list):
        raise ValueError("the reply is not a JSON array")
    rows = []
    for x in items:
        if not isinstance(x, dict) or not {"cv", "label", "score"} <= set(x):
            raise ValueError(f"an item lacks cv/label/score: {str(x)[:80]}")
        if x["label"] not in LABELS:
            raise ValueError(f"{x['cv']}: label {x['label']!r} is not one of {LABELS}")
        if not isinstance(x["score"], (int, float)) or not 0 <= x["score"] <= 100 or int(x["score"]) != x["score"]:
            raise ValueError(f"{x['cv']}: score {x['score']!r} is not an integer 0-100")
        rows.append({"short_id": str(x["cv"]), "label": x["label"], "score": int(x["score"]),
                     "reason": str(x.get("reason", ""))[:200]})
    got = [r["short_id"] for r in rows]
    if len(got) != len(set(got)):
        raise ValueError(f"duplicate ids: {sorted({g for g in got if got.count(g) > 1})}")
    if set(got) != set(expected):
        missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
        raise ValueError(f"ids do not match the batch: missing {missing}, unexpected {extra}")
    return pd.DataFrame(rows)


def save_reply(jd_id: str, b: int, raw: str, parsed: pd.DataFrame, plan: pd.DataFrame, path=REPLIES) -> None:
    """Append one validated batch: the raw reply to the log first, then one row per CV (latest wins)."""
    b = int(b)                                   # the app passes numpy ints from the plan
    p = plan[(plan.jd_id == jd_id) & (plan.batch == b)].set_index("short_id")
    run = p.run.fillna(RUN).iloc[0] if "run" in p else RUN
    out = parsed.assign(jd_id=jd_id, batch=b, pair_id=parsed.short_id.map(p.pair_id), model=MODEL, run=run,
                        instrument_sha256=instrument_sha256(), raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"jd_id": str(jd_id), "batch": b, "raw": raw, "saved_at": out.saved_at.iloc[0]}) + "\n"
    with path.with_suffix(".raw.jsonl").open("a", encoding="utf-8") as f:
        f.write(line)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)


#: Saved batches whose run label is corrected (owner, 18 Oct 2026): `chat-1` batches re-scored by
#: mistake during the `chat-2` session are the run `chat-1r` (a retest), so `chat-1` stays as
#: committed. Keyed by (jd_id, batch, saved_at); read wherever replies are read, i.e. here.
RELABEL = OUTPUT / "rerank" / "chat-rerank-relabel.csv"


def relabel(d: pd.DataFrame, path=RELABEL) -> pd.DataFrame:
    """`d` with the run of every listed (jd_id, batch, saved_at) replaced by the list's `run`."""
    if not Path(path).exists() or d.empty:
        return d
    m = pd.read_csv(path)
    key = pd.MultiIndex.from_frame(d[["jd_id", "batch", "saved_at"]].astype(str))
    to = pd.Series(m.run.to_numpy(), index=pd.MultiIndex.from_frame(m[["jd_id", "batch", "saved_at"]].astype(str)))
    new = to.reindex(key).to_numpy()
    return d.assign(run=np.where(pd.isna(new), d.run.to_numpy(), new))


def latest(path=REPLIES, relabel_path=RELABEL) -> pd.DataFrame:
    """The last saved reply per (run, JD, short id), after `relabel`."""
    if not path.exists():
        return pd.DataFrame(columns=["jd_id", "batch", "short_id", "pair_id", "label", "score", "run"])
    d = relabel(pd.read_csv(path), relabel_path)
    return d.drop_duplicates(["run", "jd_id", "short_id"], keep="last")


def build(seed: int = SEED) -> pd.DataFrame:
    from candidate_screener.evaluation import retrieval_pilot, stage1
    pools = retrieval_pilot.build_pools(seed, stage1.TEXT_VARIANT)
    spec = next(s for s in stage1.specs() if s.factory().name == X2)
    system = spec.factory()
    system.fit(None)
    s = system.score(pools)
    if stage1._sha(s) != json.loads(stage1.RECORD.read_text())["systems"][X2]["pool_scores_sha256"]:
        raise ValueError("X2 pool scores differ from the committed stage-1 record")
    from candidate_screener.matching import corpus
    plan = plan_from(pools, s.to_numpy())
    plan["judged"] = plan.pair_id.isin(set(corpus.a2_pairs().pair_id))   # the pair list, no labels
    return plan


def paths(chat3: bool = False, chat4: bool = False) -> tuple[Path, Path]:
    """(plan, replies) of a collection: `chat-3`'s or `chat-4`'s own files, or `chat-1` / `chat-2`'s."""
    if chat3 and chat4:
        raise ValueError("one collection at a time")
    return (CHAT3_PLAN, CHAT3_REPLIES) if chat3 else (CHAT4_PLAN, CHAT4_REPLIES) if chat4 else (PLAN, REPLIES)


def fresh_plan(scored: pd.DataFrame, new: pd.DataFrame, run: str, prefix: str, order: pd.Series,
               stratum: pd.Series, seed: int = SEED) -> pd.DataFrame:
    """A collection of fresh chats, one per JD with `new` pairs: those pairs plus `n_anchors` of the
    JD's `scored` ones (`pair_id`, `jd_id`, `cv_id`, `score`), shuffled, ids `{prefix}001`...,
    batches of `BATCH` from 1. `order` and `stratum` map JD -> `jd_order` / stratum. Ids only."""
    rows = []
    for jd, g in new.sort_values("pair_id").groupby("jd_id", sort=True):
        rng = np.random.default_rng([seed, int(order[jd])])
        a = anchors(scored[scored.jd_id == jd], n_anchors(len(g)), rng)
        both = pd.concat([g[["jd_id", "cv_id", "pair_id"]].assign(source="pooled"),
                          a[["jd_id", "cv_id", "pair_id"]].assign(source="anchor")], ignore_index=True)
        both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
        rows.append(both.assign(stratum=stratum[jd], jd_order=int(order[jd]), run=run,
                                short_id=[f"{prefix}{i + 1:03d}" for i in range(len(both))],
                                batch=1 + both.index // BATCH))
    return pd.concat(rows, ignore_index=True)


def texts(plan: pd.DataFrame) -> pd.DataFrame:
    if "run" in plan and plan.run.eq(CHAT3_RUN).any():
        if not plan.run.eq(CHAT3_RUN).all():
            raise ValueError("a plan mixes chat-3 with other runs")
        from candidate_screener.evaluation import chat3
        pools = chat3.pairs().set_index("pair_id")
    else:
        from candidate_screener.evaluation import retrieval_pilot, stage1
        pools = retrieval_pilot.build_pools(SEED, stage1.TEXT_VARIANT).set_index("pair_id")
    return plan.assign(jd_text=plan.pair_id.map(pools.jd_text), cv_text=plan.pair_id.map(pools.cv_text))


def batch_prompt(plan_t: pd.DataFrame, jd_id: str, b: int) -> tuple[str, list[str]]:
    g = plan_t[plan_t.jd_id == jd_id]
    rows = g[g.batch == b].sort_values("short_id")
    i, n = chat_batches(g, jd_id, b)
    return prompt(g.jd_text.iloc[0], list(zip(rows.short_id, rows.cv_text)), i, n, first=i == 1), list(rows.short_id)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sample", action="store_true", help="write the first JD's batch 1 and 2 prompts to data/interim")
    ap.add_argument("--fresh-s1", type=float, metavar="T",
                    help="step 4a: fresh chats (chat-2) for S1's passed set at P >= T outside the scored pairs")
    ap.add_argument("--extend-s1", action="store_true",
                    help="append S1's passed CVs outside X2's top 100 (needs ~4 min of features)")
    args = ap.parse_args(argv)
    if args.build:
        plan = build()
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(PLAN, index=False)
        g = plan.groupby("jd_id")
        print(PLAN, f"{plan.jd_id.nunique()} JDs, {len(plan)} CVs, judged in shortlist per JD "
              f"{g.judged.sum().mean():.1f}, unjudged {(~plan.judged).groupby(plan.jd_id).sum().mean():.1f}")
    if args.extend_s1:
        from candidate_screener.evaluation import stage1_classifier
        from candidate_screener.matching import corpus
        plan = pd.read_csv(PLAN, float_precision="round_trip")      # x2 back bit for bit
        out = extend(plan, stage1_classifier.passed_set(SEED), "s1_passed")
        out["x2_rank"] = out.x2_rank.astype("Int64")                # blank for the extension
        out["judged"] = out.pair_id.isin(set(corpus.a2_pairs().pair_id))   # the pair list, no labels
        out.to_csv(PLAN, index=False)
        added = out[out.source == "s1_passed"]
        print(PLAN, f"{len(added)} S1 passed CVs outside X2's top 100, {added.jd_id.nunique()} JDs, "
              f"{added.judged.sum()} of them judged")
    if args.fresh_s1 is not None:
        from candidate_screener.evaluation import stage1_classifier
        from candidate_screener.matching import corpus
        from candidate_screener.rerank import chat_targets
        plan = pd.read_csv(PLAN, float_precision="round_trip")
        out = fresh(plan, chat_targets.load(), stage1_classifier.passed_set(SEED, threshold=args.fresh_s1), SEED)
        out["judged"] = out.pair_id.isin(set(corpus.a2_pairs().pair_id))   # the pair list, no labels
        out["x2_rank"] = out.x2_rank.astype("Int64")
        out.to_csv(PLAN, index=False)
        f = out[out.run == FRESH_RUN]
        print(PLAN, f"{FRESH_RUN}: {f.jd_id.nunique()} JDs, {(f.source == 's1_t030').sum()} new + "
              f"{(f.source == 'anchor').sum()} anchors = {len(f)} CVs, "
              f"{f.groupby('jd_id').batch.nunique().sum()} batches")
    if args.sample:
        plan_t = texts(pd.read_csv(PLAN))
        jd = plan_t.sort_values("jd_order").jd_id.iloc[0]
        for b in (1, 2):
            text, ids = batch_prompt(plan_t, jd, b)
            out = DATA / "interim" / f"chat-rerank-sample-batch{b}.txt"
            out.write_text(text, encoding="utf-8")
            print(out, f"{len(text):,} chars, ids {ids[0]}..{ids[-1]}")


if __name__ == "__main__":
    main()
