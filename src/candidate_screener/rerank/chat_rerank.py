"""Stage-2 LLM reranking by hand through a chat window (plan 2026-10-17-llm-rerank, D59/D60).

For every one of the 60 final-set JDs, X2 shortlists the top `K` of its 500-CV pool. The shortlist
is shuffled (seed 0), given short ids `C001`..`C100`, and split into batches of 20. The owner pastes
each batch's prompt into a chat (one conversation per JD), pastes the JSON reply back, and the
reply is validated and appended to `REPLIES` (raw text and parsed rows, never document text).

The prompt is a **new listwise instrument**: the annotation guide, verbatim from the `a1-judge`
agent, with its one-CV framing replaced by a numbered list and a JSON-array reply. X2 scores and
ranks are never shown. Every document passes `annotation.redact`.

    uv run python -m candidate_screener.rerank.chat_rerank --build     # the plan (ids only)
    uv run python -m candidate_screener.rerank.chat_rerank --sample    # one batch to a text file
"""
from __future__ import annotations

import argparse
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
AGENT = PROJECT_ROOT / ".claude" / "agents" / "a1-judge.md"
PLAN = DATA / "interim" / "chat-rerank-plan.csv"
REPLIES = OUTPUT / "rerank" / "chat-rerank-replies.csv"
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
    out = parsed.assign(jd_id=jd_id, batch=b, pair_id=parsed.short_id.map(p.pair_id), model=MODEL, run=RUN,
                        instrument_sha256=instrument_sha256(), raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"jd_id": str(jd_id), "batch": b, "raw": raw, "saved_at": out.saved_at.iloc[0]}) + "\n"
    with path.with_suffix(".raw.jsonl").open("a", encoding="utf-8") as f:
        f.write(line)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)


def latest(path=REPLIES) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["jd_id", "batch", "short_id", "pair_id", "label", "score"])
    d = pd.read_csv(path)
    return d.drop_duplicates(["jd_id", "short_id"], keep="last")


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


def texts(plan: pd.DataFrame) -> pd.DataFrame:
    from candidate_screener.evaluation import retrieval_pilot, stage1
    pools = retrieval_pilot.build_pools(SEED, stage1.TEXT_VARIANT).set_index("pair_id")
    return plan.assign(jd_text=plan.pair_id.map(pools.jd_text), cv_text=plan.pair_id.map(pools.cv_text))


def batch_prompt(plan_t: pd.DataFrame, jd_id: str, b: int) -> tuple[str, list[str]]:
    g = plan_t[plan_t.jd_id == jd_id]
    rows = g[g.batch == b].sort_values("short_id")
    n = int(g.batch.max())
    return prompt(g.jd_text.iloc[0], list(zip(rows.short_id, rows.cv_text)), b, n, first=b == 1), list(rows.short_id)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sample", action="store_true", help="write the first JD's batch 1 and 2 prompts to data/interim")
    args = ap.parse_args(argv)
    if args.build:
        plan = build()
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(PLAN, index=False)
        g = plan.groupby("jd_id")
        print(PLAN, f"{plan.jd_id.nunique()} JDs, {len(plan)} CVs, judged in shortlist per JD "
              f"{g.judged.sum().mean():.1f}, unjudged {(~plan.judged).groupby(plan.jd_id).sum().mean():.1f}")
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
