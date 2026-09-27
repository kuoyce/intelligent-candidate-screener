"""Build the skill vocabulary `data/vocab/skills.csv` (task 2.1, the never-done task 3.5).

Three sources, one row per alias:

- **ESCO v1.2.1, ICT scope**: every skill in the *digital skills collection*, or below
  the skill group `S5` *working with computers*, or below the knowledge field `06`
  *information and communication technologies* (via `broaderRelationsSkillPillar`).
  Preferred label = canonical; preferred + alternative labels (+ each label with a
  trailing parenthetical qualifier removed) = aliases. Hidden labels are not used.
- **`data_jobs`** (D2): its 252 posting skills, mapped onto an ESCO canonical when the
  alias already exists there, else kept as a project-local canonical.
- **The 8 regex families** of `baselines/skills.py`, expanded to literal aliases.

Aliases that match the same token pattern are one row; the canonical comes from the
highest-priority contribution (a canonical's own label, then ESCO > data_jobs > regex,
then alphabetical). `FILTERS` removes aliases that make matching noisy; its values were
chosen by `tune()` (statistics on SkillSpan train, selection on validation, never test)
and `evaluate --check` fails if a re-tune would choose differently.

    uv run python -m candidate_screener.extraction.vocab --build
    uv run python -m candidate_screener.extraction.vocab --check
    uv run python -m candidate_screener.extraction.vocab --tune
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from candidate_screener.baselines.skills import SKILL_PATTERNS
from candidate_screener.config import RAW
from candidate_screener.extraction.rules import VOCAB_CSV, SkillMatcher, alias_key

ESCO_DIR = RAW / "esco_dataset-v1.2.1-classification"
DATA_JOBS_DIR = RAW / "data-jobs"

#: ESCO skill-group codes whose descendants form the ICT scope.
ICT_GROUP_CODES = ("S5", "06")
SOURCE_ORDER = ("esco", "data_jobs", "regex")
COLUMNS = ["canonical", "alias", "source", "skill_type", "esco_uri", "data_jobs_postings"]

#: `data_jobs` spellings of one tool, folded onto the spelling that carries it.
DATA_JOBS_SYNONYMS = {
    "mongo": "mongodb", "golang": "go", "no-sql": "nosql", "powerbi": "power bi",
    "react.js": "react", "vue.js": "vue", "angular.js": "angular", "node": "node.js",
    "sqlserver": "sql server", "asp.netcore": "asp.net core", "huggingface": "hugging face",
    "msaccess": "ms access", "rubyon rails": "ruby on rails", "visualbasic": "visual basic",
}
REGEX_SYNONYMS = {"nodejs": "node.js", "postgres": "postgresql"}
#: ESCO labels that name a different tool from the one the other sources mean. ESCO's
#: `SPARK` is the Java micro-framework; `data_jobs`' `spark` (114,609 postings) is Apache
#: Spark. The ESCO label is withheld as an alias so the common sense wins the pattern.
ESCO_HOMONYMS = ("spark",)


@dataclass(frozen=True)
class Filters:
    """Alias filters. `min_chars` spares aliases carrying punctuation (`C#`, `F#`)."""
    min_chars: int = 1
    max_tokens: int | None = None
    stopwords: bool = False
    blacklist: tuple[str, ...] = field(default=())


# --- chosen by `tune()` on 26 Sep 2026; see plan/2026-09-26-top-down-reset/08-extraction.md -
#: Only `stopwords` moves validation strict F1 materially (0.2115 -> 0.2202, it removes
#: `it` -> *computer technology*); the blacklist adds 0.0005 and is kept because the
#: pre-set selection rule chose it, not because the difference is resolved.
BLACKLIST_RULE: tuple | None = ("partial", 10, 0.25)
BLACKLIST: tuple[str, ...] = ("email", "it")
FILTERS = Filters(min_chars=1, max_tokens=None, stopwords=True, blacklist=BLACKLIST)
# ----------------------------------------------------------------------------------------


@dataclass
class Sources:
    """Everything the build reads, so tests can pass synthetic frames."""
    skills: pd.DataFrame          # conceptUri, preferredLabel, altLabels, skillType
    digital: pd.DataFrame         # conceptUri
    broader: pd.DataFrame         # conceptUri, broaderUri
    groups: pd.DataFrame          # conceptUri, code
    data_jobs: dict[str, int]     # skill -> postings
    regex_terms: list[str] = field(default_factory=list)


# --- sources ------------------------------------------------------------------------------

def regex_terms(patterns: list[str] = SKILL_PATTERNS) -> list[str]:
    """Literal aliases of the baseline families; each is checked against its pattern."""
    out = []
    for pat in patterns:
        body = re.fullmatch(r"\\b\((.*)\)\\b", pat)
        if body is None:
            raise ValueError(f"unexpected pattern shape: {pat}")
        for alt in body.group(1).split("|"):
            variants = [alt.replace(r"\.?", r"\."), alt.replace(r"\.?", "")] \
                if r"\.?" in alt else [alt]
            for v in variants:
                term = re.sub(r"\\(.)", r"\1", v)
                # Checked against the alternative without its `\b` anchors: `\bc\+\+\b`
                # cannot match `c++` followed by a space or the end of the text.
                if not re.fullmatch(alt, term) or re.search(r"[\\()\[\]?*|]", term):
                    raise ValueError(f"{alt!r} did not expand to a literal its pattern matches")
                out.append(term)
    return list(dict.fromkeys(out))


def load_sources() -> Sources:
    from candidate_screener.data.profile import parse_skill_cell

    d = ESCO_DIR
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found; see data/README.md §3")
    freq: collections.Counter = collections.Counter()
    for f in sorted(DATA_JOBS_DIR.glob("*-train-*.parquet")):
        for cell in pd.read_parquet(f, columns=["job_skills"]).job_skills.dropna():
            freq.update(parse_skill_cell(cell))
    return Sources(
        skills=pd.read_csv(d / "skills_en.csv", dtype=str, keep_default_na=False),
        digital=pd.read_csv(d / "digitalSkillsCollection_en.csv", dtype=str,
                            keep_default_na=False),
        broader=pd.read_csv(d / "broaderRelationsSkillPillar_en.csv", dtype=str,
                            keep_default_na=False),
        groups=pd.read_csv(d / "skillGroups_en.csv", dtype=str, keep_default_na=False),
        data_jobs=dict(freq),
        regex_terms=regex_terms(),
    )


def esco_scope(src: Sources) -> set[str]:
    """URIs in the digital collection or below an `ICT_GROUP_CODES` group."""
    children: dict[str, set[str]] = collections.defaultdict(set)
    for c, b in zip(src.broader.conceptUri, src.broader.broaderUri):
        children[b].add(c)
    frontier = [u for u, code in zip(src.groups.conceptUri, src.groups.code)
                if any(code == g or code.startswith(g) for g in ICT_GROUP_CODES)]
    below: set[str] = set()
    while frontier:
        for child in children[frontier.pop()]:
            if child not in below:
                below.add(child)
                frontier.append(child)
    skills = set(src.skills.conceptUri)
    return (set(src.digital.conceptUri) | below) & skills


# --- build --------------------------------------------------------------------------------

_PAREN = re.compile(r"^(.*\S)\s*\([^()]*\)$")


def _contributions(src: Sources) -> list[dict]:
    """Every (alias, canonical) offer before merging, with its priority rank.

    `own = 0` marks a concept's own name (an ESCO preferred label, with or without its
    parenthetical qualifier; a `data_jobs` or regex term kept as its own canonical).
    A `data_jobs` or regex term joins an ESCO canonical only through an `own` label:
    ESCO alternative labels often list narrower tools (`tensorflow` under *computer
    vision*, `git` under *tools for software configuration management*), and folding a
    named tool into a broad competence would erase what a recruiter screens on.
    """
    rows = []
    homonyms = {alias_key(h) for h in ESCO_HOMONYMS}
    scope = esco_scope(src)
    sk = src.skills[src.skills.conceptUri.isin(scope)]
    for uri, pref, alts, stype in zip(sk.conceptUri, sk.preferredLabel, sk.altLabels,
                                      sk.skillType):
        labels = [(pref.strip(), 0)] + [(a.strip(), 1) for a in alts.split("\n") if a.strip()]
        for label, own in list(labels):
            m = _PAREN.match(label)
            if m:
                labels.append((m.group(1).strip(), own))
        for label, own in labels:
            if alias_key(label) in homonyms:
                continue
            rows.append(dict(alias=label, canonical=pref.strip(), source="esco", own=own,
                             skill_type=stype, esco_uri=uri, postings=""))
    esco_own = _resolve([r for r in rows if r["own"] == 0])

    def canonical_for(term: str, synonyms: dict, known: dict) -> str:
        for cand in (term, synonyms.get(term)):
            if cand is not None and alias_key(cand) in known:
                return known[alias_key(cand)]
        return synonyms.get(term, term)

    dj_rows = []
    for term, n in sorted(src.data_jobs.items()):
        canon = canonical_for(term, DATA_JOBS_SYNONYMS, esco_own)
        dj_rows.append(dict(alias=term, canonical=canon, source="data_jobs",
                            own=0 if canon == term else 1, skill_type="", esco_uri="",
                            postings=str(n)))
    known = {**_resolve(dj_rows), **esco_own}
    rx_rows = []
    for term in src.regex_terms:
        canon = canonical_for(term, REGEX_SYNONYMS, known)
        rx_rows.append(dict(alias=term, canonical=canon, source="regex",
                            own=0 if canon == term else 1, skill_type="", esco_uri="",
                            postings=""))
    return rows + dj_rows + rx_rows


def _rank(r: dict) -> tuple:
    return (r["own"], SOURCE_ORDER.index(r["source"]), r["canonical"].lower(), r["canonical"])


def _resolve(rows: list[dict]) -> dict[tuple, str]:
    best: dict[tuple, dict] = {}
    for r in rows:
        k = alias_key(r["alias"])
        if k and (k not in best or _rank(r) < _rank(best[k])):
            best[k] = r
    return {k: r["canonical"] for k, r in best.items()}


def merge(src: Sources) -> pd.DataFrame:
    """One row per alias pattern, before any filter."""
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in _contributions(src):
        k = alias_key(r["alias"])
        if k:
            groups[k].append(r)
    out = []
    for rows in groups.values():
        win = min(rows, key=_rank)
        esco = min((r for r in rows if r["source"] == "esco"), key=_rank, default=None)
        sources = {r["source"] for r in rows}
        out.append(dict(
            canonical=win["canonical"], alias=win["alias"],
            source="|".join(s for s in SOURCE_ORDER if s in sources),
            skill_type=esco["skill_type"] if esco and esco["canonical"] == win["canonical"] else "",
            esco_uri=esco["esco_uri"] if esco and esco["canonical"] == win["canonical"] else "",
            data_jobs_postings=next((r["postings"] for r in rows if r["source"] == "data_jobs"), ""),
            n_canonicals=len({r["canonical"] for r in rows}),
        ))
    return pd.DataFrame(out)


def _stopwords() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS)


def apply_filters(merged: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    a = merged.alias
    keep = pd.Series(True, index=merged.index)
    keep &= ~((a.str.len() < filters.min_chars) & a.map(lambda s: s.replace(" ", "").isalnum()))
    if filters.max_tokens is not None:
        keep &= a.str.split().str.len() <= filters.max_tokens
    if filters.stopwords:
        keep &= ~a.str.lower().isin(_stopwords())
    if filters.blacklist:
        banned = {alias_key(b) for b in filters.blacklist}
        keep &= ~a.map(lambda s: alias_key(s) in banned)
    return merged[keep]


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    df["_k"] = list(zip(df.canonical.str.lower(), df.canonical, df.alias.str.lower(), df.alias))
    return df.sort_values("_k", kind="mergesort").drop(columns="_k").reset_index(drop=True)


def build(src: Sources, filters: Filters = FILTERS) -> pd.DataFrame:
    return finalise(apply_filters(merge(src), filters))


def render(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def digest(path: Path = VOCAB_CSV) -> str:
    """sha256 of the vocabulary file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(df: pd.DataFrame) -> dict:
    src = collections.Counter()
    for s in df.source:
        for part in s.split("|"):
            src[part] += 1
    return {"rows": int(len(df)), "canonicals": int(df.canonical.nunique()),
            "aliases_by_source": {k: int(src[k]) for k in SOURCE_ORDER},
            "esco_concepts": int(df.loc[df.esco_uri != "", "esco_uri"].nunique())}


# --- tuning (SkillSpan train for statistics, validation for selection) --------------------

_RULES = ((3, 0.1), (3, 0.25), (3, 0.5), (5, 0.1), (5, 0.25), (10, 0.25))
GRID = {
    "min_chars": (1, 2, 3),
    "max_tokens": (None, 6, 4, 3, 2),
    "stopwords": (False, True),
    "blacklist_rule": (None,) + tuple((stat, *r) for stat in ("strict", "partial") for r in _RULES),
}
#: Selection objective. Partial F1 was tried first and rejected on validation: overlap
#: credit rewards common words that fall inside long soft-skill spans (`it`), so under it
#: the unfiltered vocabulary won. Strict F1 is also what the tagger early-stops on.
TUNE_OBJECTIVE = ("strict", "f1")


def _matched_alias_stats(matcher: SkillMatcher, split: str) -> dict[tuple, list[int]]:
    """Per alias pattern: [times matched, exact-boundary hits, overlap hits] vs SKILL gold."""
    from candidate_screener.extraction import skillspan as ss
    from candidate_screener.extraction.metrics import overlaps

    out: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for text, offs, gold in ss.prepared(split, "union"):
        for m in matcher.find(text):
            tok = ss.snap([(m.start, m.end)], offs)
            k = alias_key(m.text)
            out[k][0] += 1
            out[k][1] += int(bool(tok & gold))
            out[k][2] += int(any(overlaps(t, g) for t in tok for g in gold))
    return out


def blacklist_for(stats_: dict[tuple, list[int]], merged: pd.DataFrame,
                  rule: tuple | None) -> tuple[str, ...]:
    """Aliases matched at least `min_support` times whose hit rate is below the bar."""
    if rule is None:
        return ()
    stat, min_support, max_precision = rule
    col = {"strict": 1, "partial": 2}[stat]
    by_key = {alias_key(a): a.lower() for a in merged.alias}
    bad = [by_key[k] for k, c in stats_.items()
           if k in by_key and c[0] >= min_support and c[col] / c[0] < max_precision]
    return tuple(sorted(set(bad)))


def rules_counts(matcher: SkillMatcher, split: str, scope: str = "union"):
    """Per-sentence count vectors of the rules matcher on a SkillSpan split."""
    import numpy as np

    from candidate_screener.extraction import skillspan as ss
    from candidate_screener.extraction.metrics import unit_counts

    rows = []
    for text, offs, gold in ss.prepared(split, scope):
        pred = ss.snap([(m.start, m.end) for m in matcher.find(text)], offs)
        rows.append(unit_counts(pred, gold))
    return np.array(rows)


def tune(src: Sources | None = None, verbose: bool = True) -> dict:
    """Grid-search `Filters` on SkillSpan validation; per-alias statistics from train."""
    from candidate_screener.extraction.metrics import prf

    src = src or load_sources()
    merged = merge(src)
    stats_train = _matched_alias_stats(SkillMatcher(finalise(merged)), "train")
    results = []
    for mc, mt, sw, rule in itertools.product(*GRID.values()):
        f = Filters(mc, mt, sw, blacklist_for(stats_train, merged, rule))
        vocab = finalise(apply_filters(merged, f))
        m = prf(rules_counts(SkillMatcher(vocab), "validation").sum(axis=0))
        results.append(dict(min_chars=mc, max_tokens=mt, stopwords=sw,
                            blacklist_rule=list(rule) if rule else None,
                            blacklist_size=len(f.blacklist), aliases=int(len(vocab)),
                            strict_precision=round(float(m["strict"][0]), 4),
                            strict_recall=round(float(m["strict"][1]), 4),
                            strict_f1=round(float(m["strict"][2]), 4),
                            partial_f1=round(float(m["partial"][2]), 4),
                            partial_precision=round(float(m["partial"][0]), 4),
                            partial_recall=round(float(m["partial"][1]), 4)))
    # Highest strict F1; ties -> higher partial F1 -> first in grid order (least aggressive).
    best = max(enumerate(results),
               key=lambda ir: (ir[1]["strict_f1"], ir[1]["partial_f1"], -ir[0]))[1]
    rule = tuple(best["blacklist_rule"]) if best["blacklist_rule"] else None
    chosen = Filters(best["min_chars"], best["max_tokens"], best["stopwords"],
                     blacklist_for(stats_train, merged, rule))
    if verbose:
        top = sorted(results, key=lambda r: (-r["strict_f1"], -r["partial_f1"]))[:10]
        for r in top:
            print(r)
        print("chosen:", best)
    return {"objective": "validation " + " ".join(TUNE_OBJECTIVE) + " (SKILL union)",
            "statistics_split": "train", "selection_split": "validation",
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals]
                     for k, vals in GRID.items()},
            "chosen": {**{k: best[k] for k in ("min_chars", "max_tokens", "stopwords",
                                                "blacklist_rule", "blacklist_size")},
                       "validation_partial_f1": best["partial_f1"],
                       "validation_strict_f1": best["strict_f1"]},
            "blacklist": list(chosen.blacklist),
            "baseline_no_filters": next(r for r in results
                                        if r["min_chars"] == 1 and r["max_tokens"] is None
                                        and not r["stopwords"] and r["blacklist_rule"] is None),
            "_filters": chosen}


def committed_matches_tuning(tuned: dict) -> bool:
    """True when the committed `FILTERS` / `BLACKLIST_RULE` are what a re-tune selects."""
    rule = tuned["chosen"]["blacklist_rule"]
    return FILTERS == tuned["_filters"] and BLACKLIST_RULE == (tuple(rule) if rule else None)


# --- CLI ----------------------------------------------------------------------------------

def check(path: Path = VOCAB_CSV) -> bool:
    """Rebuild in memory and compare digests with the file on disk."""
    fresh = hashlib.sha256(render(build(load_sources()))).hexdigest()
    ok = Path(path).is_file() and fresh == digest(path)
    print(f"vocab {'OK' if ok else 'MISMATCH'}: built {fresh[:12]}, "
          f"on disk {digest(path)[:12] if Path(path).is_file() else 'missing'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--tune", action="store_true")
    a = ap.parse_args(argv)
    if a.tune:
        t = tune()
        print(f"BLACKLIST_RULE = {tuple(t['chosen']['blacklist_rule'] or ()) or None}")
        print(f"BLACKLIST = {tuple(t['blacklist'])!r}")
        return 0
    if a.check:
        return 0 if check() else 1
    df = build(load_sources())
    VOCAB_CSV.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_CSV.write_bytes(render(df))
    print(f"wrote {VOCAB_CSV} {stats(df)} sha256={digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
