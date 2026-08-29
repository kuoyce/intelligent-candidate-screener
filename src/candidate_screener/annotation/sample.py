"""Task 3.4b — the 200-pair in-domain evaluation set *(D3, D4, D11, D12, D14, D18, D19)*.

40 Djinni JDs, 5 candidate CVs each, drawn from A2's **eval** region so nothing here can
be fine-tuned on later (D19). This is the set the proposal's "200+ in-domain pairs" and
its ≥85% agreement target are measured on, and it is the only human-gated artefact left
on the critical path *(decision D25)*.

Four choices are load-bearing, and three of them are the difference between a usable set
and a degenerate one:

1. **CV text is `Position + CV + Highlights + Moreinfo + Looking For`, not the `CV`
   column.** Q12 measured the median rising 751 -> 1,525 characters from that alone,
   which halves the length gap against A1's 5,134 at zero cost.
2. **JDs are banded by `Exp Years`, not sampled flat**, so the set spans seniority.
3. **The 5 CVs per JD are banded by a cheap lexical score — high / mid / low** — and this
   is the one that decides whether the set is worth labelling at all. A random draw of 5
   CVs from 210,250 returns 5 `No Fit`s, P@5 is then 0 for every system, and 200 human
   judgements have measured nothing. See `_LEXICAL_NOTE` for what that costs.
4. **No role-family stratification** (D14). `primary_keyword` is still *recorded* on every
   pair — the constraint D14 removes is on sampling, not on measurement.

Usage:
    uv run python -m candidate_screener.annotation.sample --build --seed 0
"""
from __future__ import annotations

import argparse
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
PAIRS_REPORT = MANIFESTS / "indomain-report.json"

#: Q12's finding, as the definition rather than as prose. Order matters only for
#: readability — the annotator reads this top to bottom.
CV_FIELDS = ("Position", "CV", "Highlights", "Moreinfo", "Looking For")

#: One CV per band per JD, plus two more from the middle. Bands are quantiles of the
#: lexical score *within the JD's candidate pool*, so "high" means high relative to the
#: candidates this JD could plausibly draw, not high in absolute cosine.
BANDS = ("high", "high", "mid", "mid", "low")

_LEXICAL_NOTE = (
    "The 5 CVs per JD are banded by TF-IDF cosine against the JD, not drawn uniformly. "
    "Consequence, and it must be stated wherever an in-domain precision figure appears: "
    "this pool is NOT a uniform sample of the corpus, so in-domain absolute precision is "
    "not an unbiased estimate of production precision. It is a comparison instrument "
    "between systems. The alternative — a uniform draw — returns ~5 No Fit per JD and "
    "makes P@5 identically zero for every system, which measures nothing at all."
)

#: The candidate pool each JD is banded over. The **whole** pool is held out from
#: pretraining, not just the 5 drawn from it (D18) — otherwise a later expansion of this
#: set would judge documents the model has already been pretrained on.
CANDIDATES_PER_JD = 200


def cv_text(cv: pd.DataFrame) -> pd.Series:
    """Concatenate the five fields, dropping empties so blanks do not become ' nan '."""
    parts = [cv[field].fillna("").astype(str).str.strip() for field in CV_FIELDS]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="\n\n").str.strip()
    return joined.str.replace(r"\n{3,}", "\n\n", regex=True)


def load_eval_region() -> tuple[set[str], set[str]]:
    """The half of A2 task 3.4 may draw from *(D19)*.

    Mirrors `a2_finetune.load_train_region` and inverts it: the shortlist takes `train`,
    this takes `eval`, and `verify --derived` asserts they never meet.
    """
    if not PARTITION_MANIFEST.exists():
        raise FileNotFoundError(
            f"{PARTITION_MANIFEST} missing — run `uv run python -m "
            "candidate_screener.data.a2_finetune --partition --seed 0` first. Sampling "
            "before the partition exists is exactly the leak D19 was written to prevent.")
    manifest = pd.read_csv(PARTITION_MANIFEST)
    region = manifest[manifest.region == "eval"]
    return (set(region[region.doc_type == "jd"].doc_id),
            set(region[region.doc_type == "cv"].doc_id))


def load_english(eval_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """English-language, non-empty JDs and CVs from the eval region, with text built."""
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    jd_ids, cv_ids = load_eval_region() if eval_only else (set(jd["id"]), set(cv["id"]))

    jd = jd[jd["id"].isin(jd_ids) & (jd["Long Description_lang"] == "en")].copy()
    jd["jd_text"] = jd["Long Description"].fillna("").astype(str).str.strip()
    jd = jd[jd.jd_text.str.len() > 0]
    jd["exp_band"] = jd["Exp Years"].map(JD_EXP_TO_BAND)

    cv = cv[cv["id"].isin(cv_ids) & (cv["CV_lang"] == "en")].copy()
    cv["cv_text"] = cv_text(cv)
    cv = cv[cv.cv_text.str.len() > 0]
    cv["exp_band"] = cv["Experience Years"].map(cv_experience_band)
    return jd, cv


def choose_jds(jd: pd.DataFrame, n_jds: int, seed: int) -> pd.DataFrame:
    """`n_jds` JDs spread evenly across experience bands, unstratified by family (D14)."""
    bands = sorted({b for b in jd.exp_band if isinstance(b, str)})
    per_band = int(np.ceil(n_jds / len(bands)))
    seeds = np.random.SeedSequence(seed).spawn(len(bands))
    picked = []
    for band, band_seed in zip(bands, seeds):
        pool = jd[jd.exp_band == band]
        order = np.random.default_rng(band_seed).permutation(len(pool))
        picked.append(pool.iloc[order[:per_band]])
    return pd.concat(picked).sort_values("id", kind="stable").head(n_jds)


def band_candidates(jd_row: pd.Series, candidates: pd.DataFrame, seed) -> pd.DataFrame:
    """Score `candidates` against one JD and take one CV per band in `BANDS`.

    The vectoriser is fitted on **this JD's candidate pool plus the JD**, and exists only
    to spread the draw. It is not a model, is not committed, and no figure is computed
    from it — `test_notebook_is_thin.py`'s rule is about notebooks defining models, and
    this is a sampling instrument inside the package where it can be tested.
    """
    corpus = pd.concat([candidates.cv_text, pd.Series([jd_row.jd_text])])
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", max_features=20_000)
    matrix = vectorizer.fit_transform(corpus)
    scores = np.asarray((matrix[:-1] @ matrix[-1].T).todense()).ravel()

    ranked = candidates.assign(lexical_score=scores).sort_values(
        ["lexical_score", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    thirds = np.array_split(np.arange(len(ranked)), 3)
    by_band = {"high": thirds[0], "mid": thirds[1], "low": thirds[2]}

    rng = np.random.default_rng(seed)
    taken: list[int] = []
    for band in BANDS:
        available = [i for i in by_band[band] if i not in taken]
        if not available:                      # tiny band — fall back rather than skip
            available = [i for i in range(len(ranked)) if i not in taken]
        taken.append(int(rng.choice(available)))
    return ranked.iloc[taken].assign(band=list(BANDS))


def build_pairs(n_jds: int, per_jd: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pairs, holdout). `holdout` covers the whole candidate pool, not the 5."""
    jd, cv = load_english()
    chosen = choose_jds(jd, n_jds, seed)
    seeds = np.random.SeedSequence(seed + 1).spawn(len(chosen))

    pair_rows, holdout_rows = [], []
    for (_, jd_row), jd_seed in zip(chosen.iterrows(), seeds):
        pool_seed, band_seed = jd_seed.spawn(2)
        # Candidates are drawn from the whole eval region, not from the JD's own
        # Primary Keyword — D14 removes family stratification on the JD side, and
        # imposing it on the candidate side would reintroduce it through the back door.
        pool = cv.iloc[np.random.default_rng(pool_seed).permutation(len(cv))[
            :CANDIDATES_PER_JD]]
        holdout_rows += [(c, "cv", f"candidate pool for jd {jd_row.id}") for c in pool["id"]]
        holdout_rows.append((jd_row.id, "jd", "in-domain evaluation query"))

        for _, candidate in band_candidates(jd_row, pool, band_seed).head(per_jd).iterrows():
            pair_rows.append((
                f"{jd_row.id}__{candidate['id']}", jd_row.id, candidate["id"],
                jd_row["Primary Keyword"], jd_row.exp_band, candidate["Primary Keyword"],
                candidate.band, round(float(candidate.lexical_score), 6)))

    pairs = pd.DataFrame(pair_rows, columns=[
        "pair_id", "jd_id", "cv_id", "primary_keyword", "exp_band",
        "cv_primary_keyword", "lexical_band", "lexical_score"]).drop_duplicates("pair_id")
    holdout = pd.DataFrame(holdout_rows, columns=["doc_id", "doc_type", "reason"]) \
        .drop_duplicates("doc_id")
    return (pairs.sort_values(["jd_id", "cv_id"], kind="stable").reset_index(drop=True),
            holdout.sort_values(["doc_type", "doc_id"], kind="stable").reset_index(drop=True))


def summarise(pairs: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    return {
        "pairs": int(len(pairs)), "jds": int(pairs.jd_id.nunique()),
        "cvs": int(pairs.cv_id.nunique()),
        "by_lexical_band": pairs.lexical_band.value_counts().to_dict(),
        "by_exp_band": pairs.exp_band.value_counts().to_dict(),
        # D14 removed the sampling constraint but not the measurement — this is the
        # realised family mix, reported as an observation.
        "realised_family_mix": pairs.primary_keyword.value_counts().head(15).to_dict(),
        "holdout_documents": {"jd": int((holdout.doc_type == "jd").sum()),
                              "cv": int((holdout.doc_type == "cv").sum())},
    }


def build(n_jds: int, per_jd: int, seed: int) -> dict:
    pairs, holdout = build_pairs(n_jds, per_jd, seed)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(PAIRS_MANIFEST, index=False, lineterminator="\n")
    holdout.to_csv(HOLDOUT_MANIFEST, index=False, lineterminator="\n")
    report = {"seed": seed, "n_jds": n_jds, "per_jd": per_jd,
              "cv_text_fields": list(CV_FIELDS), "candidates_per_jd": CANDIDATES_PER_JD,
              "note_banding": _LEXICAL_NOTE,
              "note_region": "Drawn only from region == eval in a2-partition.csv (D19).",
              "summary": summarise(pairs, holdout)}
    PAIRS_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_report(r: dict) -> None:
    s = r["summary"]
    print(f"\n=== In-domain set — {s['pairs']} pairs, {s['jds']} JDs, {s['cvs']} CVs "
          f"(seed={r['seed']}, eval region only)")
    print(f"  lexical bands   {s['by_lexical_band']}")
    print(f"  experience      {s['by_exp_band']}")
    print(f"  realised family mix (D14 — observed, not imposed):")
    for family, n in list(s["realised_family_mix"].items())[:8]:
        print(f"      {family:<28} {n}")
    print(f"  holdout (D18)   {s['holdout_documents']['jd']} JDs, "
          f"{s['holdout_documents']['cv']} CVs — the whole candidate pool, not the "
          f"{s['pairs']} pairs")
    print(f"  manifests -> {PAIRS_MANIFEST.name}, {HOLDOUT_MANIFEST.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n-jds", type=int, default=40)
    ap.add_argument("--per-jd", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.build:
        ap.error("pass --build")
    print_report(build(args.n_jds, args.per_jd, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
