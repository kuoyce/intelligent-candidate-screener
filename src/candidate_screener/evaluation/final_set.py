"""The final annotation set for A2: one Fit / No Fit label per pair (D41, D47).

Rule v1, approved by the user on 26 Sep 2026:

    1. both humans agree  -> their label          tier `humans_agree`
    2. both humans split  -> the LLM's label      tier `llm_breaks_tie`
    3. one human          -> that human's label   tier `single_human`
    4. no human           -> the LLM's label      tier `llm_only`

`Fit` is `Good Fit` or `Potential Fit`, and every raw label is collapsed before the rule
runs. The LLM is one series only, the committed instrument (D43). The set is built from
annotator labels alone: no system output is read (D46). It is frozen once committed, and
a changed rule is a new `RULE_VERSION` and a new file.

Usage:
    uv run python -m candidate_screener.evaluation.final_set --build
    uv run python -m candidate_screener.evaluation.final_set --check
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from candidate_screener.config import DOCS_DATA, OUTPUT
from candidate_screener.evaluation import resample

RULE_VERSION = 1
RULE = (
    "both humans agree -> their label",
    "both humans split -> the LLM's label",
    "one human -> that human's label",
    "no human -> the LLM's label",
)

#: The LLM annotator: instrument `5d986c82` under haiku, runs 9 (generic) and 10
#: (targeted), which is the only series that labelled all 600 pairs (D39, D43).
LLM_AGENT_SHA256 = "5d986c8232ea3c985ebf8e3044049d3dd7ba709ad7037043d726aa9590bc77a3"
LLM_MODEL = "claude-haiku-4-5"

HUMANS = ("yc", "van")
FIT, NO_FIT = "Fit", "No Fit"
RAW_LABELS = ("Good Fit", "Potential Fit", "No Fit")
FIT_CLASSES = frozenset({"Good Fit", "Potential Fit"})
TIERS = ("humans_agree", "llm_breaks_tie", "single_human", "llm_only")
LLM_FREE_TIERS = ("humans_agree", "single_human")
STRATA = ("generic", "targeted")
SCOPES = (*STRATA, "total")

MANIFESTS = DOCS_DATA / "manifests"
PAIRS_CSV = MANIFESTS / "indomain-pairs.csv"
JUDGEMENTS_CSV = MANIFESTS / "judgements.csv"
LLM_CSVS = (MANIFESTS / "llm-a2-generic.csv", MANIFESTS / "llm-a2-targeted.csv")
FINAL_CSV = MANIFESTS / "a2-final-labels.csv"
REPORT_JSON = OUTPUT / "annotation" / "a2-final-report.json"

COLUMNS = ["pair_id", "stratum", "batch", "jd_id", "cv_id",
           "yc_raw", "van_raw", "llm_raw", "yc", "van", "llm",
           "final_label", "tier", "n_humans",
           "rule_version", "llm_agent_sha256", "llm_model", "llm_run"]


# --- the rule ------------------------------------------------------------------

def to_fit(label: object) -> str | None:
    """Collapse a raw three-class label to Fit / No Fit; `None` for a missing label."""
    if label is None or pd.isna(label):
        return None
    if label not in RAW_LABELS:
        raise ValueError(f"{label!r} is not one of {RAW_LABELS}")
    return FIT if label in FIT_CLASSES else NO_FIT


def _missing(value: object) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def resolve(humans: Sequence[str | None], llm: str | None) -> tuple[str, str]:
    """Rule v1 on collapsed labels. Returns `(final_label, tier)`.

    A missing label may arrive as `None` or as NaN (pandas fills an absent human with
    NaN). Both mean "no label": counting a NaN as a second human would read every
    single-human pair as a split and hand it to the LLM.
    """
    present = [h for h in humans if not _missing(h)]
    llm = None if _missing(llm) else llm
    if len(present) > 2:
        raise ValueError(f"rule v1 is defined for at most two humans, got {len(present)}")
    if len(present) == 2:
        if present[0] == present[1]:
            return present[0], "humans_agree"
        if llm is None:
            raise ValueError("the humans split and there is no LLM label to break the tie")
        return llm, "llm_breaks_tie"
    if len(present) == 1:
        return present[0], "single_human"
    if llm is None:
        raise ValueError("a pair with no human label and no LLM label cannot be labelled")
    return llm, "llm_only"


# --- inputs ----------------------------------------------------------------------

def load_humans(judgements: pd.DataFrame) -> pd.DataFrame:
    """Wide raw human labels for A2: one row per pair, a `<name>_raw` column per human."""
    a2 = judgements[(judgements.corpus == "a2") & judgements.label.notna()]
    unknown = set(a2.annotator.astype(str)) - set(HUMANS)
    if unknown:
        raise ValueError(f"annotator(s) {sorted(unknown)} are not in HUMANS={HUMANS}; "
                         "rule v1 names its humans, so a new one is a new rule version")
    if a2.duplicated(["pair_id", "annotator"]).any():
        raise ValueError("a human labelled the same pair twice")
    wide = a2.pivot(index="pair_id", columns="annotator", values="label")
    return wide.reindex(columns=list(HUMANS)).add_suffix("_raw")


def load_llm(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """The one LLM series (D43): committed instrument, one model, parsed rows only."""
    llm = pd.concat(frames, ignore_index=True)
    llm = llm[(llm.agent_sha256 == LLM_AGENT_SHA256) & (llm.model == LLM_MODEL)
              & llm.parsed_ok.astype(bool)]
    if llm.pair_id.duplicated().any():
        raise ValueError("the LLM series holds two labels for one pair — two runs of one "
                         "instrument would be two votes from one judge")
    return llm.set_index("pair_id")[["label", "run"]].rename(
        columns={"label": "llm_raw", "run": "llm_run"})


def build_frame(pairs: pd.DataFrame, humans: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Apply rule v1 to every pair. Pure: frames in, frame out."""
    frame = (pairs.set_index("pair_id")[["stratum", "batch", "jd_id", "cv_id"]]
             .join(humans, how="left").join(llm, how="left"))
    for who in (*HUMANS, "llm"):
        frame[who] = frame[f"{who}_raw"].map(to_fit)
    resolved = [resolve([row[h] for h in HUMANS], row["llm"]) for _, row in frame.iterrows()]
    frame["final_label"] = [label for label, _ in resolved]
    frame["tier"] = [tier for _, tier in resolved]
    frame["n_humans"] = frame[list(HUMANS)].notna().sum(axis=1).astype(int)
    frame["rule_version"] = RULE_VERSION
    frame["llm_agent_sha256"] = LLM_AGENT_SHA256
    frame["llm_model"] = LLM_MODEL
    frame["llm_run"] = frame["llm_run"].astype("Int64")
    return frame.reset_index()[COLUMNS].sort_values("pair_id", kind="stable").reset_index(drop=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(PAIRS_CSV)
    humans = load_humans(pd.read_csv(JUDGEMENTS_CSV))
    llm = load_llm([pd.read_csv(p) for p in LLM_CSVS])
    return pairs, humans, llm


def build() -> pd.DataFrame:
    return build_frame(*load_inputs())


def load() -> pd.DataFrame:
    """The committed final set — what every evaluation reads."""
    return pd.read_csv(FINAL_CSV)


def llm_free(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairs whose label needed no LLM input (tiers 1 and 3) — the sensitivity subset."""
    return frame[frame.tier.isin(LLM_FREE_TIERS)]


def in_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    return frame if scope == "total" else frame[frame.stratum == scope]


# --- invariants --------------------------------------------------------------------

def check_invariants(frame: pd.DataFrame) -> list[tuple[bool, str]]:
    """The rule, restated as checks on the built frame (02 §3.4)."""
    humans = frame[list(HUMANS)]
    both = humans.notna().all(axis=1)
    agree = both & (frame.yc == frame.van)
    split = both & (frame.yc != frame.van)
    single = humans.notna().sum(axis=1) == 1
    none = humans.isna().all(axis=1)
    single_value = frame.yc.where(frame.yc.notna(), frame.van)
    checks = [
        (bool((frame.final_label[agree] == frame.yc[agree]).all()),
         f"rule 1: {int(agree.sum())} pair(s) where the humans agree carry their label"),
        (bool((frame.final_label[split] == frame.llm[split]).all()),
         f"rule 2: {int(split.sum())} split pair(s) carry the LLM's label"),
        (bool((frame.final_label[single] == single_value[single]).all()),
         f"rule 3: {int(single.sum())} single-human pair(s) carry the human's label"),
        (bool((frame.final_label[none] == frame.llm[none]).all()
              and (frame.tier[none] == "llm_only").all()),
         f"rule 4: {int(none.sum())} no-human pair(s) carry the LLM's label"),
        (bool(frame.final_label.isin([FIT, NO_FIT]).all()),
         "every final label is Fit or No Fit (D47)"),
        (bool(frame.tier.isin(TIERS).all()), "every tier is one of the four"),
        (bool(frame.pair_id.is_unique), "one row per pair"),
        (bool((frame.llm_agent_sha256 == LLM_AGENT_SHA256).all()
              and (frame.llm_model == LLM_MODEL).all()),
         "the LLM column names exactly one (instrument, model)"),
    ]
    return checks


def check_provenance(frame: pd.DataFrame, humans: pd.DataFrame, llm: pd.DataFrame
                     ) -> list[tuple[bool, str]]:
    """Every raw value in the set equals its source row."""
    f = frame.set_index("pair_id")
    out = []
    for who in HUMANS:
        src = humans[f"{who}_raw"].reindex(f.index)
        same = (f[f"{who}_raw"].fillna("") == src.fillna("")).all()
        out.append((bool(same), f"{who}: every raw label matches judgements.csv"))
    src = llm.llm_raw.reindex(f.index)
    out.append((bool((f.llm_raw.fillna("") == src.fillna("")).all()),
                "llm: every raw label matches its row in llm-a2-*.csv"))
    return out


# --- report ------------------------------------------------------------------------

def _agreement_block(frame: pd.DataFrame, a: str, b: str, seed: int) -> dict:
    both = frame.dropna(subset=[a, b])
    if both.empty:
        return {"n_pairs": 0}
    both = both.assign(_a=(both[a] == FIT).astype(int), _b=(both[b] == FIT).astype(int))
    fa, fb = both._a.to_numpy(), both._b.to_numpy()
    agree_ci = resample.pair_stat_ci(both, resample.binary_agreement, ["_a", "_b"], seed=seed)
    kappa_ci = resample.pair_stat_ci(both, resample.binary_kappa, ["_a", "_b"], seed=seed)
    return {"n_pairs": int(len(both)), "n_jds": int(both.jd_id.nunique()),
            "agreement": round(resample.binary_agreement(fa, fb), 6),
            "agreement_ci": [round(x, 6) for x in agree_ci],
            "kappa": round(resample.binary_kappa(fa, fb), 6),
            "kappa_ci": [round(x, 6) for x in kappa_ci]}


def report(frame: pd.DataFrame, seed: int = 0) -> dict:
    scopes = {}
    for scope in SCOPES:
        sub = in_scope(frame, scope)
        scopes[scope] = {
            "n_pairs": int(len(sub)), "n_jds": int(sub.jd_id.nunique()),
            "tiers": {t: int((sub.tier == t).sum()) for t in TIERS},
            "fit_rate": {who: round(float((sub[who].dropna() == FIT).mean()), 6)
                         for who in ("final_label", *HUMANS, "llm")},
            "inter_annotator": {f"{a}_vs_{b}": _agreement_block(sub, a, b, seed)
                                for a, b in (("yc", "van"), ("yc", "llm"), ("van", "llm"))},
            "final_vs_annotator": {who: _agreement_block(sub.assign(final=sub.final_label),
                                                         "final", who, seed)
                                   for who in (*HUMANS, "llm")},
        }
    return {
        "rule_version": RULE_VERSION,
        "rule": list(RULE),
        "labels": {"Fit": sorted(FIT_CLASSES), "No Fit": ["No Fit"]},
        "llm_series": {"agent_sha256": LLM_AGENT_SHA256, "model": LLM_MODEL,
                       "runs": sorted(int(r) for r in frame.llm_run.dropna().unique())},
        "note_inter_annotator": "Reported with no target (D44). CIs resample JDs.",
        "note_final_vs_annotator": (
            "Descriptive only. Each annotator contributed to the final set, so its "
            "agreement with it is inflated by construction and is not a human ceiling."),
        "scopes": scopes,
    }


# --- IO ------------------------------------------------------------------------------

def render_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def render_report(rep: dict) -> str:
    return json.dumps(rep, indent=2) + "\n"


def write(frame: pd.DataFrame, rep: dict) -> None:
    FINAL_CSV.write_text(render_csv(frame), encoding="utf-8")
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(render_report(rep), encoding="utf-8")


def check(seed: int = 0) -> list[tuple[bool, str]]:
    """Rebuild in memory: invariants, provenance, and byte-identity with the committed files."""
    pairs, humans, llm = load_inputs()
    frame = build_frame(pairs, humans, llm)
    results = check_invariants(frame) + check_provenance(frame, humans, llm)
    results.append((len(frame) == len(pairs), f"{len(frame)} rows for {len(pairs)} pairs"))
    for path, text in ((FINAL_CSV, render_csv(frame)),
                       (REPORT_JSON, render_report(report(frame, seed)))):
        same = path.exists() and path.read_text(encoding="utf-8") == text
        results.append((same, f"determinism: rebuild reproduces {path.name} byte-for-byte"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="write the manifest and report")
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.build:
        frame = build()
        bad = [m for ok, m in check_invariants(frame) if not ok]
        if bad:
            raise SystemExit(f"invariant(s) failed, nothing written: {bad}")
        rep = report(frame, args.seed)
        write(frame, rep)
        total = rep["scopes"]["total"]
        print(f"wrote {FINAL_CSV.name} ({len(frame)} pairs) and {REPORT_JSON.name}")
        print(f"tiers {total['tiers']}  fit rate {total['fit_rate']['final_label']:.3f}")
    if args.check:
        results = check(args.seed)
        for ok, message in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        return 0 if all(ok for ok, _ in results) else 1
    if not (args.build or args.check):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
