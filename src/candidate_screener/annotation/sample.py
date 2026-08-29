"""Task 3.4b — the in-domain evaluation set, as a **batch campaign** *(D25, D28)*.

40 Djinni JDs, 5 candidate CVs each, drawn from A2's **eval** region so nothing here can
be fine-tuned on later (D19). This is the set the proposal's "200+ in-domain pairs" and
its ≥85% agreement target are measured on.

**Why batches** *(decision D28)*. The first version of this module was one-shot: a pure
function of (corpus, `n_jds`, seed), rebuilt from scratch each time — the same discipline
`pools.py` uses, where byte-equality across rebuilds is asserted. That is right for a
derived artefact and **wrong for an annotation campaign**, and measurably so: growing
`n_jds` from 40 to 60 kept the same 40 JDs but re-drew the candidates for 38 of them,
changing **190 of 200 `pair_id`s**. Any label already collected would have pointed at a
pair that no longer existed.

Three things fix it. They are not interchangeable, and it is worth being exact about
which one does what:

1. **The campaign is a list of frozen batch specs** (`indomain-batches.json`). Batch *N*
   draws from what batches 1..*N*-1 did not take, so it cannot disturb them. **This alone
   is what keeps collected labels valid** — growing the set now means appending a spec,
   never re-running an existing batch with a bigger `n_jds`.
2. **Each JD's candidate draw is seeded from its `jd_id`, not its position** in the
   sorted list (`document_seed`). This does *not* protect earlier batches — (1) does that
   — it makes a single batch's draw independent of how its own JD list is ordered, so a
   batch re-derived under a filter or a different sort reproduces the same pairs.
3. **`choose_jds` sorts by id before permuting.** Otherwise the draw depends on the order
   `load_split` returns rows in, and a publisher re-upload that shuffled the parquet would
   select a different 40 JDs while every seed, count and config stayed identical. Same
   class of invisible drift as (2), arriving through the corpus instead of the code.

Four sampling choices are load-bearing, and three decide whether the set is worth labelling:

1. **CV text is `Position + CV + Highlights + Moreinfo + Looking For`**, not the `CV`
   column. Q12 measured the median rising 751 -> 1,525 characters from that alone.
2. **JDs are banded by `Exp Years`**, so the set spans seniority.
3. **The 5 CVs per JD are banded high / mid / low by a cheap lexical score.** A random
   draw of 5 CVs from 210,250 returns 5 `No Fit`, P@5 is then 0 for every system, and 200
   human judgements have measured nothing. See `_LEXICAL_NOTE`.
4. **Batch 1 has no role-family stratification** (D14). Later batches *may* scope to
   keywords — see `_STRATUM_NOTE` for the reporting obligation that creates.

Usage:
    uv run python -m candidate_screener.annotation.sample --build --seed 0
    uv run python -m candidate_screener.annotation.sample --add-batch --n-jds 20
    uv run python -m candidate_screener.annotation.sample --add-batch --n-jds 10 \\
        --keywords "Data Science" "Data Engineer"
    uv run python -m candidate_screener.annotation.sample --add-batch --reuse-jds --per-jd 5
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from candidate_screener.data.a2_finetune import (JD_EXP_TO_BAND, PARTITION_MANIFEST,
                                                 cv_experience_band)
from candidate_screener.data.fit_split import MANIFESTS
from candidate_screener.data.profile import load_split

PAIRS_MANIFEST = MANIFESTS / "indomain-pairs.csv"
HOLDOUT_MANIFEST = MANIFESTS / "indomain-holdout.csv"
BATCHES = MANIFESTS / "indomain-batches.json"
PAIRS_REPORT = MANIFESTS / "indomain-report.json"

#: Q12's finding, as the definition rather than as prose.
CV_FIELDS = ("Position", "CV", "Highlights", "Moreinfo", "Looking For")

#: Bands are quantiles of the lexical score *within the JD's candidate pool*, so "high"
#: means high relative to the candidates this JD could plausibly draw.
BANDS = ("high", "high", "mid", "mid", "low")

#: The candidate pool each JD is banded over. The **whole** pool is held out from
#: pretraining, not just the 5 drawn from it (D18).
CANDIDATES_PER_JD = 200

_LEXICAL_NOTE = (
    "The 5 CVs per JD are banded by TF-IDF cosine against the JD, not drawn uniformly. "
    "Consequence, and it must be stated wherever an in-domain precision figure appears: "
    "this pool is NOT a uniform sample of the corpus, so in-domain absolute precision is "
    "not an unbiased estimate of production precision. It is a comparison instrument "
    "between systems. The alternative — a uniform draw — returns ~5 No Fit per JD and "
    "makes P@5 identically zero for every system, which measures nothing at all."
)

#: The two reporting groups *(decision D29)*. **Coarser than a batch, on purpose.**
#:
#: A batch is the operational unit — which titles, drawn under which spec, added when. A
#: stratum is the reporting unit, and there are exactly two. Every targeted batch is drawn
#: the same way and differs only in which titles it covers, so pooling them describes a
#: real population: *"the job families we deliberately chose to cover."* That is a
#: defensible sentence, which is the whole test a figure has to pass.
#:
#: What stays forbidden is pooling ACROSS the two. `generic` answers "how does this system
#: do on a typical posting from this board"; `targeted` answers "how does it do on the
#: families we care about". Averaging them yields a number describing a mixture that exists
#: nowhere and that was created by an accident of how much of each we happened to label.
STRATA = ("generic", "targeted")

_STRATUM_NOTE = (
    "Two reporting strata, never pooled together (D29). `generic` is the unstratified "
    "sample (D14) and answers 'how does this system do on a typical posting'. `targeted` "
    "pools every keyword-scoped batch and answers 'how does it do on the families we chose "
    "to cover'. Both are true statements about a real population; their average is not. "
    "Batches within a stratum MAY be pooled — they differ only in which titles they cover."
)

#: Batch 1, frozen 29 Aug 2026. `keywords: null` is D14 — unstratified by role family.
DEFAULT_CAMPAIGN: tuple[dict, ...] = (
    {"batch": 1, "n_jds": 40, "per_jd": 5, "seed": 0, "keywords": None,
     "reuse_jds": False},
)


# --- text ------------------------------------------------------------------

def cv_text(cv: pd.DataFrame) -> pd.Series:
    """Concatenate the five fields, dropping empties so blanks do not become ' nan '."""
    parts = [cv[field].fillna("").astype(str).str.strip() for field in CV_FIELDS]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="\n\n").str.strip()
    return joined.str.replace(r"\n{3,}", "\n\n", regex=True)


def load_eval_region() -> tuple[set[str], set[str]]:
    """The half of A2 task 3.4 may draw from *(D19)*."""
    if not PARTITION_MANIFEST.exists():
        raise FileNotFoundError(
            f"{PARTITION_MANIFEST} missing — run `uv run python -m "
            "candidate_screener.data.a2_finetune --partition --seed 0` first. Sampling "
            "before the partition exists is exactly the leak D19 was written to prevent.")
    manifest = pd.read_csv(PARTITION_MANIFEST)
    region = manifest[manifest.region == "eval"]
    return (set(region[region.doc_type == "jd"].doc_id),
            set(region[region.doc_type == "cv"].doc_id))


def load_english() -> tuple[pd.DataFrame, pd.DataFrame]:
    """English-language, non-empty JDs and CVs from the eval region, with text built."""
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    jd_ids, cv_ids = load_eval_region()

    jd = jd[jd["id"].isin(jd_ids) & (jd["Long Description_lang"] == "en")].copy()
    jd["jd_text"] = jd["Long Description"].fillna("").astype(str).str.strip()
    jd = jd[jd.jd_text.str.len() > 0]
    jd["exp_band"] = jd["Exp Years"].map(JD_EXP_TO_BAND)

    cv = cv[cv["id"].isin(cv_ids) & (cv["CV_lang"] == "en")].copy()
    cv["cv_text"] = cv_text(cv)
    cv = cv[cv.cv_text.str.len() > 0]
    cv["exp_band"] = cv["Experience Years"].map(cv_experience_band)
    return jd, cv


def available_titles() -> pd.DataFrame:
    """Every `Primary Keyword` reachable in the eval region, with its JD supply.

    The 45 values in the raw corpus are **not** the 22 that batch 1 happens to cover:
    batch 1 is an unstratified draw of 40 JDs (D14), so its title list is an observation
    about what fell out, never a designed scope. 41 of the 45 survive the eval-region and
    English filters. Anything a report needs covered has to arrive as a targeted batch,
    and this is the list such a batch may name.
    """
    jd, _ = load_english()
    counts = jd.groupby("Primary Keyword").size().sort_values(ascending=False)
    return counts.rename("jds").reset_index().rename(
        columns={"Primary Keyword": "title"})


def validate_keywords(keywords: list[str] | None, known) -> None:
    """Fail loudly on a title that does not exist, before a batch is frozen.

    Without this an unmatched keyword scopes the batch to zero JDs, `choose_jds` returns
    an empty frame, and the batch is appended to `indomain-batches.json` contributing
    **no pairs at all** — no exception, no warning, a config that reads as though it
    worked. That is the same signature as the three bugs that shipped in the classical
    baseline, and the reason `AGENTS.md` requires a named invariant test for each.
    """
    if not keywords:
        return
    valid = sorted({k for k in pd.Series(list(known)).dropna()})
    unknown = sorted(set(keywords) - set(valid))
    if unknown:
        raise ValueError(
            f"unknown Primary Keyword(s) {unknown} — a batch scoped to a title that "
            "does not exist draws zero JDs and appends a batch with no pairs in it. "
            f"Valid titles: {valid}")


# --- seeding ---------------------------------------------------------------

def document_seed(doc_id: str, seed: int) -> np.random.SeedSequence:
    """A seed derived from the document's own id — **never from its position**.

    This is the whole of fix (1) in the module docstring. `SeedSequence(seed).spawn(n)`
    zipped positionally with a sorted list gives each JD a seed that depends on how many
    JDs precede it, so inserting one JD re-seeds every later one and silently re-draws
    its candidates. Keying on `doc_id` makes each JD's draw a pure function of
    (`jd_id`, seed), stable under any change to the rest of the campaign.
    """
    key = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.SeedSequence(entropy=[int(seed), key])


# --- one batch -------------------------------------------------------------

def choose_jds(jd: pd.DataFrame, n_jds: int, seed: int,
               exclude: set[str] | None = None) -> pd.DataFrame:
    """`n_jds` JDs spread evenly across experience bands, skipping `exclude`."""
    available = jd[~jd["id"].isin(exclude or set())]
    bands = sorted({b for b in available.exp_band if isinstance(b, str)})
    if not bands:
        return available.head(0)
    per_band = int(np.ceil(n_jds / len(bands)))
    seeds = np.random.SeedSequence(seed).spawn(len(bands))
    picked = []
    for band, band_seed in zip(bands, seeds):
        # Sorted by id *before* permuting. Without this the draw depends on the order
        # `load_split` happens to return rows in, so a re-download that shuffled the
        # parquet would silently select a different 40 JDs while every seed and count
        # stayed identical — the same class of invisible drift the position-derived
        # seeds caused, arriving through the corpus instead of through the code.
        pool = available[available.exp_band == band].sort_values("id", kind="stable")
        order = np.random.default_rng(band_seed).permutation(len(pool))
        picked.append(pool.iloc[order[:per_band]])
    return pd.concat(picked).sort_values("id", kind="stable").head(n_jds)


def band_candidates(jd_row: pd.Series, candidates: pd.DataFrame, seed,
                    per_jd: int) -> pd.DataFrame:
    """Score `candidates` against one JD and take one CV per band, `per_jd` in total.

    The vectoriser is fitted on this JD's candidate pool plus the JD, and exists only to
    spread the draw. It is not a model, is not committed, and no figure is computed from
    it — it is a sampling instrument inside the package, where it can be tested.
    """
    corpus = pd.concat([candidates.cv_text, pd.Series([jd_row.jd_text])])
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", max_features=20_000)
    matrix = vectorizer.fit_transform(corpus)
    scores = np.asarray((matrix[:-1] @ matrix[-1].T).todense()).ravel()

    ranked = candidates.assign(lexical_score=scores).sort_values(
        ["lexical_score", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    thirds = np.array_split(np.arange(len(ranked)), 3)
    by_band = {"high": thirds[0], "mid": thirds[1], "low": thirds[2]}

    wanted = [BANDS[i % len(BANDS)] for i in range(per_jd)]
    rng = np.random.default_rng(seed)
    taken: list[int] = []
    for band in wanted:
        available = [i for i in by_band[band] if i not in taken]
        if not available:                      # tiny band — fall back rather than skip
            available = [i for i in range(len(ranked)) if i not in taken]
        if not available:
            break
        taken.append(int(rng.choice(available)))
    return ranked.iloc[taken].assign(band=wanted[:len(taken)])


def stratum_of(spec: dict) -> str:
    """`generic` if the batch imposed no title filter, `targeted` otherwise.

    Recorded onto every pair rather than looked up from the spec at report time, so that
    editing a batch's `keywords` later cannot silently re-stratify judgements already
    collected under the old definition.
    """
    return "targeted" if spec.get("keywords") else "generic"


def draw_batch(spec: dict, jd: pd.DataFrame, cv: pd.DataFrame,
               used_jds: set[str], used_cvs: set[str],
               used_pairs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One frozen batch. Draws only from what earlier batches left behind.

    `reuse_jds` is the difference between growing the set **wider** (new JDs, the
    default) and growing it **deeper** (more candidates for the JDs already in it, which
    is Q14's wave-2 shape). Depth is what Precision@10 would need; breadth is what
    tightens a confidence interval.
    """
    stratum = stratum_of(spec)
    validate_keywords(spec.get("keywords"), jd["Primary Keyword"].unique())
    scoped = jd if not spec.get("keywords") else jd[jd["Primary Keyword"].isin(spec["keywords"])]

    if spec.get("reuse_jds"):
        chosen = scoped[scoped["id"].isin(used_jds)].sort_values("id", kind="stable")
        if spec.get("n_jds"):
            chosen = chosen.head(spec["n_jds"])
    else:
        chosen = choose_jds(scoped, spec["n_jds"], spec["seed"], exclude=used_jds)

    per_jd, batch = spec["per_jd"], spec["batch"]
    pair_rows, holdout_rows = [], []
    for _, jd_row in chosen.iterrows():
        pool_seed, band_seed = document_seed(jd_row.id, spec["seed"]).spawn(2)
        eligible = cv[~cv["id"].isin(used_cvs)]
        pool = eligible.iloc[np.random.default_rng(pool_seed).permutation(len(eligible))[
            :CANDIDATES_PER_JD]]
        if pool.empty:
            continue
        holdout_rows += [(c, "cv", f"candidate pool for jd {jd_row.id}") for c in pool["id"]]
        holdout_rows.append((jd_row.id, "jd", "in-domain evaluation query"))

        for _, candidate in band_candidates(jd_row, pool, band_seed, per_jd).iterrows():
            pair_id = f"{jd_row.id}__{candidate['id']}"
            if pair_id in used_pairs:
                continue
            used_pairs.add(pair_id)
            used_cvs.add(candidate["id"])
            pair_rows.append((
                pair_id, batch, stratum, jd_row.id, candidate["id"],
                jd_row["Primary Keyword"], jd_row.exp_band,
                candidate["Primary Keyword"], candidate.band,
                round(float(candidate.lexical_score), 6)))
        used_jds.add(jd_row.id)

    pairs = pd.DataFrame(pair_rows, columns=[
        "pair_id", "batch", "stratum", "jd_id", "cv_id", "primary_keyword", "exp_band",
        "cv_primary_keyword", "lexical_band", "lexical_score"])
    holdout = pd.DataFrame(holdout_rows, columns=["doc_id", "doc_type", "reason"])
    return pairs, holdout


# --- the campaign ----------------------------------------------------------

def load_campaign() -> list[dict]:
    if not BATCHES.exists():
        return [dict(spec) for spec in DEFAULT_CAMPAIGN]
    return json.loads(BATCHES.read_text(encoding="utf-8"))["batches"]


def build_campaign(specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every batch in order. Batch *N* cannot disturb batches 1..*N*-1.

    That is the property the whole design rests on: the manifest is byte-reproducible
    from `indomain-batches.json` alone, *and* appending a batch leaves every existing
    `pair_id` untouched — so labels already collected stay valid.
    """
    jd, cv = load_english()
    used_jds: set[str] = set()
    used_cvs: set[str] = set()
    used_pairs: set[str] = set()

    all_pairs, all_holdout = [], []
    for spec in sorted(specs, key=lambda s: s["batch"]):
        pairs, holdout = draw_batch(spec, jd, cv, used_jds, used_cvs, used_pairs)
        all_pairs.append(pairs)
        all_holdout.append(holdout)

    pairs = pd.concat(all_pairs, ignore_index=True).sort_values(
        ["batch", "jd_id", "cv_id"], kind="stable").reset_index(drop=True)
    holdout = pd.concat(all_holdout, ignore_index=True).drop_duplicates(
        "doc_id").sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True)
    return pairs, holdout


def summarise(pairs: pd.DataFrame, holdout: pd.DataFrame, specs: list[dict]) -> dict:
    by_batch = {}
    for spec in sorted(specs, key=lambda s: s["batch"]):
        rows = pairs[pairs.batch == spec["batch"]]
        by_batch[str(spec["batch"])] = {
            "pairs": int(len(rows)), "jds": int(rows.jd_id.nunique()),
            "cvs": int(rows.cv_id.nunique()),
            "keywords": spec.get("keywords"),
            "reuse_jds": bool(spec.get("reuse_jds")),
            "stratum": stratum_of(spec)}

    # The reporting view. Batches are the operational record; figures are quoted per
    # stratum, and the two strata are never averaged together (D29).
    by_stratum = {}
    for name in STRATA:
        rows = pairs[pairs.stratum == name]
        if rows.empty:
            continue
        by_stratum[name] = {
            "pairs": int(len(rows)), "jds": int(rows.jd_id.nunique()),
            "batches": sorted(int(b) for b in rows.batch.unique()),
            "titles": sorted(rows.primary_keyword.unique().tolist()),
            "title_count": int(rows.primary_keyword.nunique())}

    return {
        "pairs": int(len(pairs)), "jds": int(pairs.jd_id.nunique()),
        "cvs": int(pairs.cv_id.nunique()),
        "strata": by_stratum, "batches": by_batch,
        "by_lexical_band": pairs.lexical_band.value_counts().to_dict(),
        "by_exp_band": pairs.exp_band.value_counts().to_dict(),
        # D14 removed the sampling constraint but not the measurement — this is the
        # realised family mix, reported as an observation.
        "realised_family_mix": pairs.primary_keyword.value_counts().head(15).to_dict(),
        "holdout_documents": {"jd": int((holdout.doc_type == "jd").sum()),
                              "cv": int((holdout.doc_type == "cv").sum())},
    }


def build(specs: list[dict] | None = None) -> dict:
    specs = specs or load_campaign()
    pairs, holdout = build_campaign(specs)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(PAIRS_MANIFEST, index=False, lineterminator="\n")
    holdout.to_csv(HOLDOUT_MANIFEST, index=False, lineterminator="\n")
    BATCHES.write_text(json.dumps(
        {"note": "Frozen, append-only (D28). Batch N draws from what 1..N-1 left, so "
                 "appending a batch cannot change an existing pair_id — labels already "
                 "collected stay valid. Edit an existing batch only if nothing from it "
                 "has been labelled.",
         "note_stratum": _STRATUM_NOTE,
         "batches": sorted(specs, key=lambda s: s["batch"])}, indent=2) + "\n",
        encoding="utf-8")

    report = {"cv_text_fields": list(CV_FIELDS), "candidates_per_jd": CANDIDATES_PER_JD,
              "note_banding": _LEXICAL_NOTE, "note_stratum": _STRATUM_NOTE,
              "note_region": "Drawn only from region == eval in a2-partition.csv (D19).",
              "summary": summarise(pairs, holdout, specs)}
    PAIRS_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def add_batch(n_jds: int, per_jd: int, seed: int | None, keywords: list[str] | None,
              reuse_jds: bool) -> dict:
    """Append a batch and rebuild. Existing batches are untouched by construction."""
    specs = load_campaign()
    number = max((s["batch"] for s in specs), default=0) + 1
    specs.append({"batch": number, "n_jds": n_jds, "per_jd": per_jd,
                  "seed": number if seed is None else seed,
                  "keywords": keywords, "reuse_jds": reuse_jds})
    return build(specs)


def print_report(r: dict) -> None:
    s = r["summary"]
    print(f"\n=== In-domain set — {s['pairs']} pairs, {s['jds']} JDs, {s['cvs']} CVs "
          f"(eval region only)")
    print("  --- strata (the reporting unit; never averaged together, D29)")
    for name, st in s["strata"].items():
        print(f"      {name:<9} {st['pairs']:>4} pairs  {st['jds']:>3} JDs  "
              f"{st['title_count']:>2} titles  batches {st['batches']}")
    print("  --- batches (the operational record)")
    for number, b in s["batches"].items():
        scope = ", ".join(b["keywords"]) if b["keywords"] else "all families"
        print(f"      batch {number}  {b['pairs']:>4} pairs  {b['jds']:>3} JDs  "
              f"{b['stratum']:<9} [{scope}]"
              + ("  (deeper, same JDs)" if b["reuse_jds"] else ""))
    print(f"  lexical bands   {s['by_lexical_band']}")
    print(f"  experience      {s['by_exp_band']}")
    print(f"  realised family mix (D14 — observed, not imposed):")
    for family, n in list(s["realised_family_mix"].items())[:6]:
        print(f"      {family:<28} {n}")
    print(f"  holdout (D18)   {s['holdout_documents']['jd']} JDs, "
          f"{s['holdout_documents']['cv']} CVs — the whole candidate pool")
    print(f"  manifests -> {PAIRS_MANIFEST.name}, {HOLDOUT_MANIFEST.name}, {BATCHES.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="rebuild every batch in indomain-batches.json")
    ap.add_argument("--add-batch", action="store_true", help="append a batch and rebuild")
    ap.add_argument("--n-jds", type=int, default=20)
    ap.add_argument("--per-jd", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to the batch number, so batches differ by default")
    ap.add_argument("--keywords", nargs="+", default=None,
                    help="scope this batch to these Primary Keywords — makes it a "
                         "TARGETED stratum that must be reported separately")
    ap.add_argument("--reuse-jds", action="store_true",
                    help="draw more candidates for JDs already in the set (deeper), "
                         "rather than new JDs (wider)")
    args = ap.parse_args()
    if not (args.build or args.add_batch):
        ap.error("pass --build or --add-batch")

    if args.add_batch:
        print_report(add_batch(args.n_jds, args.per_jd, args.seed,
                               args.keywords, args.reuse_jds))
    else:
        print_report(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
