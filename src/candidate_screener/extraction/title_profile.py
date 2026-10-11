"""P2 data checks for the role-family extractor (E2): the evidence for Q-B1.

Three measurements, written to `output/extraction/title-profile.json` (plan
`2026-10-06-extraction-and-rerank`, 03-work-plan P2, 02-design-spec §3.2-3.3):

1. **Title sources** (task 2.1). For ESCO occupations, `data_jobs`, livecareer and
   resume-atlas: fill rate, distinct values, a few example values (titles and categories
   only, never document text), how many are IT roles, and a PII scan of the title field.
2. **The A2 train region** (task 2.2, D19: the `eval` region is filtered out before any
   field is read). `Primary Keyword` class sizes (T-A); `Position` normalised by
   `normalise_title` and kept as a family when it has at least *m* train JDs **and** *m*
   train CVs (T-B); keyword -> family fan-out; coverage of the ESCO + `data_jobs` alias
   lookup on normalised train `Position` strings.
3. **Title-in-text** (task 2.3). On a seeded sample of train JDs (`Long Description`) and
   train CVs with `Position` removed (`CV`, `Highlights`, `Moreinfo`, `Looking For`, the
   `text_v2` fields of 02 §2): is the document's own normalised `Position` in the text, is a
   title of the same T-B family in it, is any multi-token alias-table title in it, and
   where (first 200 characters or later). Wilson 95% intervals over documents.

The normalisation rule (one rule, used for titles, alias entries and texts alike):

- lower-case; `.net` -> `dotnet`, `x.js` -> `xjs`; tokens are `[a-z0-9]+` with an optional
  `++` or `#` suffix (so `c++`, `c#` survive);
- phrase merges: `front end` -> `frontend`, `back end`, `full stack`, `ui ux` / `ux ui` ->
  `uiux`, `react native`, `quality assurance` -> `qa`, `machine learning` -> `ml`;
  token synonyms: `golang` -> `go`, `js` -> `javascript`, `reactjs` -> `react`, ...;
- `engineer` / `programmer` / `dev` after a technology or a development qualifier
  (`frontend`, `software`, `web`, `mobile`, ...) becomes `developer`, so `Java Engineer` and
  `Java Developer` are one title. `data engineer`, `qa engineer`, `devops engineer` keep
  their noun;
- **titles only:** cut at the first `,` `|` ` - ` `(` `[` `@` ` / ` (what follows is
  stack, location or company), then drop seniority and level tokens: junior, jr, middle,
  mid, senior, sr, strong, lead (not before `generation`), team/tech lead, head (of),
  principal, chief, trainee, intern, internship, and trailing level markers (`i`-`iv`,
  `1`-`5`, `l1`-`l5`, `level`).

T-B1 keeps language and framework tokens (`python developer`); T-B2 (`role_function`) also
drops them (`developer`), to show what a function-only taxonomy would look like.

    uv run python -m candidate_screener.extraction.title_profile
    uv run python -m candidate_screener.extraction.title_profile --check
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, RAW

RECORD = OUTPUT / "extraction" / "title-profile.json"

#: The CV fields with `Position` removed: `text_v2` of 02-design-spec §2.
CV_FIELDS_V2 = ("CV", "Highlights", "Moreinfo", "Looking For")
JD_FIELD = "Long Description"

SAMPLE_N = 300
FIRST_CHARS = 200
#: Minimum train JDs and train CVs a normalised title needs to be a T-B family.
MIN_CLASS = (10, 20, 50)
HEADLINE_M = 20
#: Alias titles of fewer tokens are not searched for in text: a lone `developer` or
#: `manager` is a word, not evidence of a title.
MIN_ALIAS_TOKENS = 2
MAX_NGRAM = 6
#: A `data_jobs` title enters the alias table used for text search only if this many
#: postings carry it: the long tail of 94k normalised titles is posting noise.
DJ_MIN_POSTINGS = 5
#: ISCO-08 groups counted as ICT roles in ESCO: 25 (ICT professionals), 35 (ICT
#: technicians), 133 (ICT service managers).
ICT_ISCO_PREFIXES = ("25", "35", "133")
#: Keywords the fan-out table always shows, beside the largest ones.
FANOUT_EXTRA = ("Python", "Other", "Lead", "Data Science", "Data Engineer", "Data Analyst")
FANOUT_TOP = 12

# --- normalisation -------------------------------------------------------------------

SENIORITY = frozenset({
    "junior", "jr", "middle", "mid", "senior", "sr", "strong", "lead", "head",
    "principal", "chief", "trainee", "intern", "internship", "teamlead", "techlead",
})
_LEVEL_TRAILING = re.compile(r"^(?:i{1,3}|iv|[1-5]|l[1-5]|level)$")
_CUT = re.compile(r",|\||\s[-–—]\s|\(|\[|@|\s/\s")
_TOKEN = re.compile(r"[a-z0-9]+(?:\+\+|#)?")
_PRE = ((re.compile(r"\.net\b"), " dotnet"), (re.compile(r"(?<=[a-z0-9])\.js\b"), "js"))

PHRASES: Mapping[tuple[str, ...], str] = {
    ("front", "end"): "frontend", ("back", "end"): "backend", ("full", "stack"): "fullstack",
    ("ui", "ux"): "uiux", ("ux", "ui"): "uiux", ("react", "native"): "reactnative",
    ("quality", "assurance"): "qa", ("machine", "learning"): "ml", ("dot", "net"): "dotnet",
    ("team", "lead"): "teamlead", ("tech", "lead"): "techlead",
    ("technical", "lead"): "techlead", ("head", "of"): "head",
}
SYNONYMS: Mapping[str, str] = {
    "golang": "go", "js": "javascript", "reactjs": "react", "vuejs": "vue",
    "angularjs": "angular", "nodejs": "nodejs", "node": "nodejs", "net": "dotnet",
    "csharp": "c#", "cpp": "c++", "programmer": "developer",
    "dev": "developer", "sw": "software", "fullstackdeveloper": "fullstack",
}
#: Language and framework tokens: kept by T-B1, dropped by T-B2.
TECH = frozenset({
    "python", "java", "javascript", "typescript", "nodejs", "dotnet", "c#", "c++", "c",
    "php", "ruby", "rails", "go", "scala", "rust", "kotlin", "swift", "flutter", "dart",
    "unity", "react", "reactnative", "angular", "vue", "django", "laravel", "symfony",
    "wordpress", "magento", "drupal", "salesforce", "sap", "sql", "oracle", "1c", "delphi",
    "perl", "elixir", "erlang", "haskell", "clojure", "objective", "unreal", "shopify",
    "bitrix", "odoo", "abap", "xamarin", "spring", "nextjs", "nestjs", "html", "css",
})
DEV_QUALIFIERS = TECH | {"frontend", "backend", "fullstack", "software", "web", "mobile",
                         "game", "ios", "android", "embedded", "blockchain"}
DEV_NOUNS = frozenset({"engineer", "developer", "programmer", "dev"})


def canonical_tokens(text: object) -> list[tuple[str, int]]:
    """Tokens of `text` under the shared rule, each with its character offset.

    Offsets index the lower-cased, pre-substituted text, which differs from the original
    only where `.net` / `.js` were rewritten, so "first 200 characters" is approximate by
    at most a few characters.
    """
    s = str(text).lower() if text is not None else ""
    for pat, rep in _PRE:
        s = pat.sub(rep, s)
    raw = [(m.group(), m.start()) for m in _TOKEN.finditer(s)]
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(raw):
        if i + 1 < len(raw) and (raw[i][0], raw[i + 1][0]) in PHRASES:
            out.append((PHRASES[(raw[i][0], raw[i + 1][0])], raw[i][1]))
            i += 2
            continue
        out.append((SYNONYMS.get(raw[i][0], raw[i][0]), raw[i][1]))
        i += 1
    for j in range(1, len(out)):
        if out[j][0] in DEV_NOUNS and out[j - 1][0] in DEV_QUALIFIERS:
            out[j] = ("developer", out[j][1])
    return out


def normalise_title(title: object) -> str:
    """A job title with seniority, level and trailing qualifiers removed ('' if none left)."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    head = _CUT.split(str(title), maxsplit=1)[0]
    toks = [t for t, _ in canonical_tokens(head)]
    kept = [t for k, t in enumerate(toks)
            if t not in SENIORITY or (t == "lead" and k + 1 < len(toks)
                                      and toks[k + 1] in ("generation", "gen"))]
    while kept and _LEVEL_TRAILING.match(kept[-1]):
        kept.pop()
    while kept and kept[0] in ("of", "and", "&"):
        kept.pop(0)
    return " ".join(kept)


def role_function(norm: str) -> str:
    """T-B2: a normalised title with language and framework tokens dropped."""
    return " ".join(t for t in norm.split() if t not in TECH)


# --- find in text ----------------------------------------------------------------------

def ngram_offsets(tokens: Sequence[tuple[str, int]], max_len: int = MAX_NGRAM
                  ) -> dict[str, int]:
    """Every token n-gram (n <= `max_len`) of a text, mapped to its first offset."""
    words = [t for t, _ in tokens]
    first: dict[str, int] = {}
    for n in range(1, max_len + 1):
        for i in range(len(words) - n + 1):
            first.setdefault(" ".join(words[i:i + n]), tokens[i][1])
    return first


def find_title(grams: Mapping[str, int], phrase: str) -> int | None:
    """Offset of the first occurrence of a normalised `phrase`, or None."""
    return grams.get(phrase) if phrase else None


def find_any(grams: Mapping[str, int], aliases: Iterable[str] | set[str]) -> int | None:
    """Offset of the earliest alias found in the text, or None."""
    alias_set = aliases if isinstance(aliases, (set, frozenset)) else set(aliases)
    hits = [off for g, off in grams.items() if g in alias_set]
    return min(hits) if hits else None


def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    """Share with its Wilson 95% interval; one document is one unit."""
    if n == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": [None, None]}
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "share": round(p, 4),
            "ci95": [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]}


# --- 2.1 title sources -----------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,14}\d")


def pii_scan(values: pd.Series) -> dict:
    s = values.dropna().astype(str)
    return {"email_like": int(s.str.contains(_EMAIL).sum()),
            "phone_like": int(s.str.contains(_PHONE).sum())}


def field_stats(values: pd.Series, top: int = 5) -> dict:
    filled = values.dropna().astype(str).str.strip()
    filled = filled[filled.str.len() > 0]
    return {"rows": int(len(values)), "fill_rate": round(len(filled) / max(len(values), 1), 4),
            "distinct": int(filled.nunique()),
            "examples": [str(v) for v in filled.value_counts().head(top).index],
            "pii_in_field": pii_scan(filled)}


def load_esco_occupations() -> pd.DataFrame:
    d = RAW / "esco_dataset-v1.2.1-classification"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found — see data/README.md §3")
    return pd.read_csv(d / "occupations_en.csv", low_memory=False)


def esco_labels(occ: pd.DataFrame, ict_only: bool = False) -> dict[str, str]:
    """Normalised preferred + alternative label -> occupation preferred label."""
    if ict_only:
        occ = occ[occ.iscoGroup.astype(str).str.startswith(ICT_ISCO_PREFIXES)]
    out: dict[str, str] = {}
    for pref, alts in zip(occ.preferredLabel, occ.altLabels):
        for lab in [pref, *(str(alts).split("\n") if isinstance(alts, str) else [])]:
            n = normalise_title(lab)
            if n:
                out.setdefault(n, str(pref))
    return out


def data_jobs_labels(dj: pd.DataFrame, min_count: int = 1) -> dict[str, str]:
    """Normalised `job_title` and `job_title_short` -> the modal `job_title_short`.

    `min_count` keeps only normalised titles seen in that many postings (the 10
    `job_title_short` classes are always kept).
    """
    pairs = pd.DataFrame({"n": dj.job_title.map(normalise_title), "c": dj.job_title_short})
    pairs = pairs[pairs.n.str.len() > 0]
    size = pairs.groupby("n").size()
    pairs = pairs[pairs.n.isin(size[size >= min_count].index)]
    pairs = pd.concat([pairs, pd.DataFrame({"n": dj.job_title_short.map(normalise_title),
                                            "c": dj.job_title_short}).drop_duplicates()])
    modal = pairs.groupby("n").c.agg(lambda s: s.value_counts().index[0])
    return modal.to_dict()


def profile_sources(occ: pd.DataFrame, dj: pd.DataFrame, lc: pd.DataFrame,
                    ra: pd.DataFrame) -> dict:
    alts = occ.altLabels.dropna().astype(str).str.split("\n")
    ict = occ[occ.iscoGroup.astype(str).str.startswith(ICT_ISCO_PREFIXES)]
    ict_alts = ict.altLabels.dropna().astype(str).str.split("\n")
    lc_it = sorted(c for c in lc.Category.dropna().unique() if "INFORMATION" in str(c))
    return {
        "esco_occupations": {
            "preferredLabel": field_stats(occ.preferredLabel),
            "altLabels_fill_rate": round(float(occ.altLabels.notna().mean()), 4),
            "altLabels_total": int(alts.map(len).sum()),
            "ict_isco_prefixes": list(ICT_ISCO_PREFIXES),
            "ict_occupations": int(len(ict)),
            "ict_altLabels_total": int(ict_alts.map(len).sum()),
            "ict_examples": [str(v) for v in ict.preferredLabel.head(5)],
            "normalised_alias_strings": len(esco_labels(occ)),
            "normalised_alias_strings_ict": len(esco_labels(occ, ict_only=True)),
        },
        "data_jobs": {
            "job_title_short": {**field_stats(dj.job_title_short, top=10),
                                "classes": {str(k): int(v) for k, v in
                                            dj.job_title_short.value_counts().items()}},
            "job_title": field_stats(dj.job_title),
            "job_title_normalised_distinct": int(
                dj.job_title.dropna().map(normalise_title).replace("", np.nan).nunique()),
        },
        "livecareer": {
            "Category": {**field_stats(lc.Category),
                         "classes": {str(k): int(v) for k, v in lc.Category.value_counts().items()}},
            "it_categories": lc_it,
            "it_rows": int(lc.Category.isin(lc_it).sum()),
        },
        "resume_atlas": {
            "Category": {**field_stats(ra.Category),
                         "classes": {str(k): int(v) for k, v in ra.Category.value_counts().items()}},
        },
    }


def load_sources() -> tuple[pd.DataFrame, ...]:
    from candidate_screener.data.profile import load_split
    return (load_esco_occupations(), load_split("data-jobs", "train"),
            load_split("livecareer", "train"), load_split("resume-atlas", "train"))


# --- 2.2 T-A vs T-B --------------------------------------------------------------------

def class_sizes(counts: pd.Series) -> dict:
    return {"classes": int(len(counts)), "min": int(counts.min()),
            "median": float(counts.median()), "max": int(counts.max()),
            "docs": int(counts.sum())}


def tb_families(jd_norm: pd.Series, cv_norm: pd.Series, m: int) -> set[str]:
    """Normalised titles with at least `m` train JDs and `m` train CVs."""
    j, c = jd_norm[jd_norm != ""].value_counts(), cv_norm[cv_norm != ""].value_counts()
    return set(j[j >= m].index) & set(c[c >= m].index)


def fanout(kw: pd.Series, fam: pd.Series, keywords: Sequence[str]) -> dict:
    """Per keyword: distinct normalised titles, kept families, families covering 80%."""
    out = {}
    for k in keywords:
        f = fam[kw == k]
        if f.empty:
            continue
        kept = f[f != ""].value_counts()
        out[k] = {
            "docs": int(len(f)),
            "share_in_a_family": round(float((f != "").mean()), 4),
            "families": int(len(kept)),
            "families_for_80pct_of_mapped": int(np.searchsorted(
                kept.cumsum().to_numpy() / max(kept.sum(), 1), 0.8) + 1) if len(kept) else 0,
            "top_families": {str(a): round(float(b) / len(f), 3) for a, b in kept.head(4).items()},
        }
    return out


def purity(kw: pd.Series, fam: pd.Series) -> dict:
    """How often a family's documents share its modal keyword (1.0 = nested in T-A)."""
    df = pd.DataFrame({"kw": kw, "fam": fam})
    df = df[df.fam != ""]
    if df.empty:
        return {"doc_weighted": None, "families_below_0.8": 0}
    modal = df.groupby("fam").kw.agg(lambda s: s.value_counts().iloc[0] / len(s))
    size = df.groupby("fam").size()
    return {"doc_weighted": round(float((modal * size).sum() / size.sum()), 4),
            "families_below_0.8": int((modal < 0.8).sum()),
            "least_pure": {str(k): round(float(v), 3)
                           for k, v in modal[size >= 50].sort_values().head(5).items()}}


def coverage(norm: pd.Series, table: Mapping[str, str] | set[str]) -> dict:
    s = norm[norm != ""]
    hit = s.isin(set(table))
    distinct = s.drop_duplicates()
    return {"docs": wilson(int(hit.sum()), len(s)),
            "distinct_strings": {"k": int(distinct.isin(set(table)).sum()),
                                 "n": int(len(distinct)),
                                 "share": round(float(distinct.isin(set(table)).mean()), 4)}}


def profile_taxonomy(jd: pd.DataFrame, cv: pd.DataFrame, esco: Mapping[str, str],
                     esco_ict: Mapping[str, str], dj: Mapping[str, str],
                     dj5: Mapping[str, str]) -> dict:
    jd_norm, cv_norm = jd.Position.map(normalise_title), cv.Position.map(normalise_title)
    jd_kw, cv_kw = jd["Primary Keyword"], cv["Primary Keyword"]
    out: dict = {"n_train": {"jd": int(len(jd)), "cv": int(len(cv))}}

    ta = {}
    for side, kw in (("jd", jd_kw), ("cv", cv_kw)):
        vc = kw.value_counts()
        ta[side] = {**class_sizes(vc), "classes_under_10": int((vc < 10).sum()),
                    "other_share": round(float((kw == "Other").mean()), 4),
                    "sizes": {str(k): int(v) for k, v in vc.items()}}
    out["t_a"] = ta

    out["normalisation"] = {
        side: {"raw_distinct": int(s.Position.str.lower().str.strip().nunique()),
               "normalised_distinct": int(n[n != ""].nunique()),
               "empty_after_normalisation": int((n == "").sum()),
               "had_seniority_token": round(float(s.Position.map(
                   lambda t: any(x in SENIORITY for x, _ in canonical_tokens(t))).mean()), 4),
               "function_distinct": int(n.map(role_function).replace("", np.nan).nunique())}
        for side, s, n in (("jd", jd, jd_norm), ("cv", cv, cv_norm))}

    tb = {}
    for m in MIN_CLASS:
        fams = tb_families(jd_norm, cv_norm, m)
        jf, cf = jd_norm.where(jd_norm.isin(fams), ""), cv_norm.where(cv_norm.isin(fams), "")
        fn_j, fn_c = jd_norm.map(role_function), cv_norm.map(role_function)
        f2 = tb_families(fn_j, fn_c, m)
        tb[str(m)] = {
            "t_b1": {"classes": len(fams),
                     "jd": {**class_sizes(jf[jf != ""].value_counts()),
                            "coverage": wilson(int((jf != "").sum()), len(jf))},
                     "cv": {**class_sizes(cf[cf != ""].value_counts()),
                            "coverage": wilson(int((cf != "").sum()), len(cf))}},
            "t_b2_function": {"classes": len(f2),
                              "jd_coverage": round(float(fn_j.isin(f2).mean()), 4),
                              "cv_coverage": round(float(fn_c.isin(f2).mean()), 4)},
        }
    out["t_b"] = tb

    fams = tb_families(jd_norm, cv_norm, HEADLINE_M)
    jf, cf = jd_norm.where(jd_norm.isin(fams), ""), cv_norm.where(cv_norm.isin(fams), "")
    keys = list(jd_kw.value_counts().head(FANOUT_TOP).index)
    keys += [k for k in FANOUT_EXTRA if k not in keys]
    out["fanout_m"] = HEADLINE_M
    out["fanout"] = {"jd": fanout(jd_kw, jf, keys), "cv": fanout(cv_kw, cf, keys)}
    out["fanout_all_keywords"] = {
        side: {"families_per_keyword_median": float(pd.Series(
                   [len(set(f[(kw == k) & (f != "")])) for k in kw.unique()]).median()),
               "keywords_with_5plus_families": int(sum(
                   len(set(f[(kw == k) & (f != "")])) >= 5 for k in kw.unique()))}
        for side, kw, f in (("jd", jd_kw, jf), ("cv", cv_kw, cf))}
    out["purity"] = {"jd": purity(jd_kw, jf), "cv": purity(cv_kw, cf)}

    out["alias_coverage"] = {
        side: {"esco_all": coverage(n, esco), "esco_ict": coverage(n, esco_ict),
               "data_jobs_all": coverage(n, dj),
               f"data_jobs_min{DJ_MIN_POSTINGS}": coverage(n, dj5),
               "esco_or_data_jobs_all": coverage(n, {**esco, **dj}),
               f"esco_or_data_jobs_min{DJ_MIN_POSTINGS}": coverage(n, {**esco, **dj5})}
        for side, n in (("jd", jd_norm), ("cv", cv_norm))}
    return out


# --- 2.3 title in text -----------------------------------------------------------------

def seeded_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """`n` rows, drawn after sorting by id so the draw does not depend on file order."""
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    idx = np.random.default_rng(seed).permutation(len(df))[:n]
    return df.iloc[np.sort(idx)]


def v2_cv_text(cv: pd.DataFrame) -> pd.Series:
    parts = [cv[f].fillna("").astype(str).str.strip() for f in CV_FIELDS_V2]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep="\n\n").str.strip()
    return joined


def _where(off: int | None) -> str:
    return "none" if off is None else ("first" if off < FIRST_CHARS else "later")


def first_alias(grams: Mapping[str, int], aliases: set[str]) -> str | None:
    """The earliest alias in the text (longest wins at the same offset), or None."""
    hits = [(off, -len(g), g) for g, off in grams.items() if g in aliases]
    return min(hits)[2] if hits else None


def title_in_text(texts: Sequence[str], positions: Sequence[str], families: set[str],
                  aliases: Mapping[str, set[str]]) -> dict:
    """Find rates (with position) over one side's sample.

    `aliases` maps a source name to its alias strings; `any` is their union.
    """
    exact, same_fam, first_ok = [], [], []
    alias: dict[str, list[str]] = {k: [] for k in aliases}
    fam_multi = {f for f in families if len(f.split()) >= MIN_ALIAS_TOKENS}
    for text, pos in zip(texts, positions):
        grams = ngram_offsets(canonical_tokens(text))
        norm = normalise_title(pos)
        exact.append(_where(find_title(grams, norm)) if norm else "none")
        fam = norm if norm in families else ""
        same_fam.append(_where(find_title(grams, fam)) if fam else "n/a")
        if fam in fam_multi:
            first = first_alias(grams, fam_multi)
            first_ok.append("n/a" if first is None else ("first" if first == fam else "none"))
        else:
            first_ok.append("n/a")
        for k, a in aliases.items():
            alias[k].append(_where(find_any(grams, a)))
    n = len(texts)

    def block(vals: list[str]) -> dict:
        eligible = [v for v in vals if v != "n/a"]
        c = collections.Counter(eligible)
        return {"found": wilson(c["first"] + c["later"], len(eligible)),
                "found_in_first_200": wilson(c["first"], len(eligible)),
                "found_only_later": wilson(c["later"], len(eligible))}

    return {"n": n, "own_position_exact": block(exact),
            "own_family_title": {**block(same_fam),
                                 "note": "only documents whose Position is a T-B1 family"},
            "any_alias_title": {k: block(v) for k, v in alias.items()},
            "earliest_family_title_is_own_family": {
                **wilson(first_ok.count("first"), first_ok.count("first") + first_ok.count("none")),
                "note": ("documents whose Position is a multi-token T-B1 family and whose text "
                         "contains any multi-token T-B1 family title: is the earliest one the "
                         "document's own?")}}


def profile_title_in_text(jd: pd.DataFrame, cv: pd.DataFrame, families: set[str],
                          aliases: Mapping[str, set[str]], n: int, seed: int) -> dict:
    sj, sc = seeded_sample(jd, n, seed), seeded_sample(cv, n, seed + 1)
    return {"seed": seed, "sample_n": n, "first_chars": FIRST_CHARS,
            "min_alias_tokens": MIN_ALIAS_TOKENS,
            "alias_strings": {k: len(v) for k, v in aliases.items()},
            "jd": title_in_text(list(sj[JD_FIELD].fillna("").astype(str)),
                                list(sj.Position), families, aliases),
            "cv_v2": title_in_text(list(v2_cv_text(sc)), list(sc.Position), families,
                                   aliases)}


# --- record ----------------------------------------------------------------------------

def load_a2_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A2 JDs and CVs of the `train` region only (D19)."""
    from candidate_screener.data.a2_finetune import load_train_region
    from candidate_screener.data.profile import load_split
    jt, ct = load_train_region()
    jd = load_split("djinni-jd", "train")
    cv = load_split("djinni-cv", "train")
    return (jd[jd["id"].isin(jt)].reset_index(drop=True),
            cv[cv["id"].isin(ct)].reset_index(drop=True))


def build_record(seed: int = 0, n: int = SAMPLE_N) -> dict:
    occ, dj_df, lc, ra = load_sources()
    esco, esco_ict = esco_labels(occ), esco_labels(occ, ict_only=True)
    dj, dj5 = data_jobs_labels(dj_df), data_jobs_labels(dj_df, DJ_MIN_POSTINGS)
    jd, cv = load_a2_train()
    families = tb_families(jd.Position.map(normalise_title), cv.Position.map(normalise_title),
                           HEADLINE_M)
    def multi(xs: Iterable[str]) -> set[str]:
        return {a for a in xs if len(a.split()) >= MIN_ALIAS_TOKENS}
    aliases = {"esco": multi(esco), "data_jobs": multi(dj5), "a2_train_families": multi(families)}
    aliases["any"] = set().union(*aliases.values())
    return {
        "plan": "plan/2026-10-06-extraction-and-rerank (P2, tasks 2.1-2.3)",
        "region": "A2 train only (D19); eval-region documents are filtered out before use",
        "normalisation": {"seniority": sorted(SENIORITY), "phrases": {
            " ".join(k): v for k, v in PHRASES.items()}, "synonyms": dict(SYNONYMS),
            "tech_tokens_dropped_by_t_b2": sorted(TECH)},
        "sources": profile_sources(occ, dj_df, lc, ra),
        "taxonomy": profile_taxonomy(jd, cv, esco, esco_ict, dj, dj5),
        "title_in_text": {**profile_title_in_text(jd, cv, families, aliases, n, seed),
                          "alias_table": (f"ESCO occupation labels; data_jobs titles in >= "
                                          f"{DJ_MIN_POSTINGS} postings; T-B1 families "
                                          f"(m={HEADLINE_M}); each >= {MIN_ALIAS_TOKENS} tokens")},
    }


def render(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=SAMPLE_N, help="title-in-text sample per side")
    ap.add_argument("--check", action="store_true", help="rebuild and diff the record")
    a = ap.parse_args(argv)
    fresh = render(build_record(a.seed, a.n))
    if a.check:
        old = RECORD.read_text(encoding="utf-8")
        ok = old == fresh
        print(f"title profile {'OK' if ok else 'MISMATCH'} ({RECORD.name})")
        return 0 if ok else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(fresh, encoding="utf-8")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
