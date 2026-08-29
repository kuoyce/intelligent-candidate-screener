"""The labelling session as data — everything the UI needs, with no HTTP in it.

`ui.py` is a thin transport over this module. The split is the same one `queue.py`
makes between the dispatch file and the key: the rules about *what an annotator may
see* and *what counts as done* are decisions, and decisions belong somewhere they can
be unit-tested on a synthetic corpus. A rule that only exists inside a request handler
is a rule nobody ever runs a test against.

Three properties this module is responsible for, each of which the UI would otherwise
be free to break:

1. **Nothing reaches the screen unredacted.** `serve_group` runs `redact.redact` and
   then `redact.assert_clean` on its own output, so a UI bug cannot bypass it — the
   check is on the way out, not on the way in.
2. **Nothing anchoring reaches the screen.** The group carries `jd_text` and the
   candidates' text and nothing else. No `lexical_band`, no `lexical_score`, no
   `selection_reason`, no existing label. `lexical_band` is the one to watch: it is the
   band the pair was *sampled* from, and an annotator who saw it would be told which
   candidates the sampler already thought were good.
3. **`judgements.csv` is append-only.** `record` opens in append mode and writes whole
   lines; it has no code path that rewrites an existing row. Re-labelling a pair means
   a second row, and the later row does not delete the earlier one.

**Double labelling is decided per pair, by hash** *(the same discipline as D28)*. 30% of
in-domain pairs are labelled by both annotators so a kappa is computable. Which 30% is a
pure function of `pair_id`, so appending a batch cannot re-designate a pair that has
already been labelled once — the failure that made the one-shot sampler unusable.

**The A1 recheck is deliberately not in this UI.** It stays on the flat dispatch file
`queue.py` builds. The reason is measured, not stylistic: the in-domain set is uniformly
5 candidates per JD, while the 50 recheck pairs spread over 32 A1 JDs as 1-4 each. On a
screen that shows one query and its candidates, a group of two is visibly not an
in-domain group, and the recheck's whole value is that an annotator cannot tell a
rechecked pair from a fresh one (**A13**). A UI that grouped both would leak that
distinction through its layout.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field

import pandas as pd

from candidate_screener.annotation import redact
from candidate_screener.annotation.queue import JUDGEMENT_COLUMNS, JUDGEMENTS
from candidate_screener.annotation.sample import BATCHES, PAIRS_MANIFEST, load_campaign

#: A1's 3-class scheme *(D17)*, verbatim. The in-domain labels use A1's exact strings
#: rather than a parallel vocabulary, which is what lets one `judgements.csv` hold both
#: corpora and what makes the recheck's agreement rate computable without a mapping.
LABELS = ("Good Fit", "Potential Fit", "No Fit")

#: The labels that make a candidate eligible for the top-1 shortlist pick.
SHORTLISTABLE = ("Good Fit", "Potential Fit")

#: `Q14`'s per-JD question. Recorded once per group, on the row of the candidate picked;
#: `"none"` when nothing was Good or Potential, which is itself an answer.
NO_PICK = "none"

DOUBLE_LABEL_FRACTION = 0.30
_HASH_SCALE = 10_000


@dataclass(frozen=True)
class Candidate:
    cv_id: str
    text: str


@dataclass(frozen=True)
class Group:
    """One screen: a job description and the candidates still owed a label.

    `complete` is whether this annotator is seeing **every** candidate the manifest has
    for this JD. It is normally true; it is false on the second pass over a
    double-labelled JD, where the first annotator has already taken the pairs that are
    only wanted once and just the shared subset is left.

    That distinction decides whether the shortlist question may be asked. *"Of the
    candidates you marked Good or Potential, which would you shortlist first?"* is a
    statement about a field of five. Asked over the two the second annotator happens to
    have been given, it produces a pick that looks identical in `judgements.csv` to one
    made over the full set and means something different — a disagreement between the
    two annotators' picks would then be unattributable between the people and the
    truncation. So on a partial group the question is not asked.
    """
    jd_id: str
    batch: int
    stratum: str
    jd_text: str
    candidates: list[Candidate] = field(default_factory=list)
    complete: bool = True

    def as_dict(self) -> dict:
        return {"jd_id": self.jd_id, "batch": int(self.batch), "stratum": self.stratum,
                "jd_text": self.jd_text, "complete": self.complete,
                "candidates": [{"cv_id": c.cv_id, "text": c.text} for c in self.candidates]}


# --- what needs labelling --------------------------------------------------

def needs_two_labels(pair_id: str, fraction: float = DOUBLE_LABEL_FRACTION,
                     seed: int = 0) -> bool:
    """Is this one of the double-labelled pairs? A pure function of `pair_id`.

    Deliberately not a random sample drawn over the current pair list: that would be a
    function of how many pairs exist, so appending a batch would re-roll the designation
    for pairs already labelled once and silently strand half of the kappa sample.
    """
    digest = hashlib.sha256(f"{seed}|{pair_id}".encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) % _HASH_SCALE) < fraction * _HASH_SCALE


def load_pairs() -> pd.DataFrame:
    if not PAIRS_MANIFEST.exists():
        raise FileNotFoundError(
            f"{PAIRS_MANIFEST} missing — run `uv run python -m "
            "candidate_screener.annotation.sample --build --seed 0` first")
    pairs = pd.read_csv(PAIRS_MANIFEST)
    pairs["pair_id"] = pairs.jd_id.astype(str) + "__" + pairs.cv_id.astype(str)
    return pairs


def load_judgements() -> pd.DataFrame:
    if not JUDGEMENTS.exists():
        return pd.DataFrame(columns=list(JUDGEMENT_COLUMNS))
    judged = pd.read_csv(JUDGEMENTS, dtype=str)
    return judged if len(judged) else pd.DataFrame(columns=list(JUDGEMENT_COLUMNS))


def outstanding(pairs: pd.DataFrame, judgements: pd.DataFrame, annotator: str,
                fraction: float = DOUBLE_LABEL_FRACTION, seed: int = 0) -> pd.DataFrame:
    """The pairs `annotator` still owes a label on.

    A pair is owed when it still needs labels *and* this annotator has not already given
    one. Both conditions are needed: the first alone would keep serving a finished pair,
    and the second alone would let one person label the whole kappa sample twice.
    """
    counts = (judgements.groupby("pair_id").size() if len(judgements)
              else pd.Series(dtype=int))
    mine = (set(judgements[judgements.annotator == annotator].pair_id)
            if len(judgements) else set())

    wanted = pairs.pair_id.map(
        lambda p: 2 if needs_two_labels(p, fraction, seed) else 1)
    have = pairs.pair_id.map(counts).fillna(0).astype(int)
    return pairs[(have < wanted) & ~pairs.pair_id.isin(mine)]


def group_order(jd_id: str, seed: int = 0) -> int:
    """Stable, arbitrary ordering key — used to shuffle candidates inside a screen.

    The manifest is sorted by `cv_id`, which is a UUID and so already unrelated to the
    lexical band a candidate was drawn from. This makes that independence deliberate
    rather than incidental, so a future change to the manifest's sort order cannot start
    presenting the five candidates in high-to-low sampled-similarity order.
    """
    return int(hashlib.sha256(f"{seed}|{jd_id}".encode("utf-8")).hexdigest()[:12], 16)


# --- serving a screen ------------------------------------------------------

def serve_group(pairs: pd.DataFrame, judgements: pd.DataFrame, annotator: str,
                jd_text: pd.Series, cv_text: pd.Series, batch: int | None = None,
                fraction: float = DOUBLE_LABEL_FRACTION, seed: int = 0) -> Group | None:
    """The next screen for `annotator`, or `None` when there is nothing left.

    Groups are served whole and in manifest order, and a partially-labelled JD is
    finished before a fresh one is started. That is not tidiness: the top-1 shortlist
    question is asked once per JD *after* all its candidates have been seen, so a JD
    split across two sittings is the one thing the resume path cannot reconstruct
    (`queue.progress` reports those as `partial_jds`).
    """
    owed = outstanding(pairs, judgements, annotator, fraction, seed)
    if batch is not None:
        owed = owed[owed.batch == batch]
    if owed.empty:
        return None

    # JDs are visited in an order keyed by the annotator's name, so two people working at
    # the same time start in different places instead of both being handed JD 1 and
    # duplicating each other for an afternoon. It is a collision-avoidance measure, not a
    # lock: if they do meet in the middle the cost is a wasted screen, never a bad row —
    # `record` only appends, and `verify --derived` refuses a pair labelled twice by the
    # same person. This is the "one queue split by hand" of D29, done by the tool.
    owed = owed.assign(_seat=owed.jd_id.map(
        lambda j: group_order(f"{annotator}|{j}", seed))).sort_values(
            "_seat", kind="stable")
    jd_id = owed.jd_id.iloc[0]
    rows = owed[owed.jd_id == jd_id].copy()
    rows["order"] = rows.cv_id.map(lambda c: group_order(f"{jd_id}|{c}", seed))
    rows = rows.sort_values("order", kind="stable")

    texts = pd.Series([jd_text.get(jd_id, "")] + [cv_text.get(c, "") for c in rows.cv_id])
    if texts.isna().any() or (texts.astype(str).str.len() == 0).any():
        raise AssertionError(
            f"jd {jd_id}: a document in this group has no text — an annotator cannot "
            "judge a pair they cannot read")
    texts = redact.redact(texts)
    redact.assert_clean(texts, f"group for jd {jd_id}")

    return Group(jd_id=str(jd_id), batch=int(rows.batch.iloc[0]),
                 stratum=str(rows.stratum.iloc[0]), jd_text=texts.iloc[0],
                 complete=len(rows) == int((pairs.jd_id == jd_id).sum()),
                 candidates=[Candidate(str(c), t) for c, t
                             in zip(rows.cv_id, texts.iloc[1:])])


# --- recording -------------------------------------------------------------

def record(annotator: str, group: Group, labels: dict[str, str],
           shortlist_pick: str = NO_PICK, notes: str = "") -> int:
    """Append one row per candidate. Never rewrites; a re-label is a second row.

    Validation is here rather than in the UI because the UI is not the only caller that
    will ever exist, and an invalid label reaching `judgements.csv` is unrecoverable
    without knowing which of the two annotators meant what.
    """
    if not annotator.strip():
        raise ValueError("every judgement carries who made it — annotator is required")

    served = [c.cv_id for c in group.candidates]
    if set(labels) != set(served):
        raise ValueError(
            f"expected a label for each of {len(served)} candidates, got "
            f"{sorted(labels)} against {sorted(served)}. A group is recorded whole so "
            "that the shortlist pick refers to a complete set.")
    bad = {c: v for c, v in labels.items() if v not in LABELS}
    if bad:
        raise ValueError(f"not in A1's 3-class scheme {LABELS}: {bad}")

    if not group.complete and shortlist_pick != NO_PICK:
        raise ValueError(
            "this group is a partial re-serve of a double-labelled JD, so the shortlist "
            "question was not asked — a pick made over a subset of the candidates is not "
            "comparable to one made over the full set. Record it as 'none'.")

    eligible = {c for c, v in labels.items() if v in SHORTLISTABLE}
    if shortlist_pick != NO_PICK and shortlist_pick not in eligible:
        raise ValueError(
            f"shortlist pick {shortlist_pick!r} is not among the candidates labelled "
            f"{' or '.join(SHORTLISTABLE)} ({sorted(eligible) or 'none'}). The pick "
            "breaks ties inside the relevant set; a pick outside it means nothing.")
    if shortlist_pick == NO_PICK and eligible and group.complete:
        raise ValueError(
            f"{len(eligible)} candidate(s) were labelled relevant, so 'none' is not an "
            "answer to which you would shortlist first — pick one.")

    JUDGEMENTS.parent.mkdir(parents=True, exist_ok=True)
    if not JUDGEMENTS.exists():
        JUDGEMENTS.write_text(",".join(JUDGEMENT_COLUMNS) + "\n", encoding="utf-8")

    with JUDGEMENTS.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for cv_id in served:
            writer.writerow([
                f"{group.jd_id}__{cv_id}", group.batch, group.stratum, "a2",
                group.jd_id, cv_id, "indomain_banded", annotator, labels[cv_id],
                "1" if cv_id == shortlist_pick else "0", notes])
    return len(served)


# --- the first screen ------------------------------------------------------

def batch_overview(annotator: str = "", fraction: float = DOUBLE_LABEL_FRACTION,
                   seed: int = 0) -> dict:
    """Every batch, what it covers, and how much of it is done."""
    pairs, judgements = load_pairs(), load_judgements()
    specs = {int(s["batch"]): s for s in load_campaign()}
    counts = (judgements.groupby("pair_id").size() if len(judgements)
              else pd.Series(dtype=int))
    pairs = pairs.assign(
        wanted=pairs.pair_id.map(lambda p: 2 if needs_two_labels(p, fraction, seed) else 1),
        have=pairs.pair_id.map(counts).fillna(0).astype(int))
    owed = outstanding(pairs, judgements, annotator, fraction, seed) if annotator \
        else pairs.head(0)

    batches = []
    for number, rows in pairs.groupby("batch"):
        spec = specs.get(int(number), {})
        batches.append({
            "batch": int(number),
            "stratum": str(rows.stratum.iloc[0]),
            "keywords": spec.get("keywords"),
            "titles": sorted(rows.primary_keyword.dropna().unique().tolist()),
            "pairs": int(len(rows)), "jds": int(rows.jd_id.nunique()),
            "labels_wanted": int(rows.wanted.sum()),
            "labels_have": int(rows.have.sum()),
            "pairs_done": int((rows.have >= rows.wanted).sum()),
            "owed_by_me": int((owed.batch == int(number)).sum()) if len(owed) else 0})

    return {"annotator": annotator,
            "double_label_fraction": fraction,
            "batches": sorted(batches, key=lambda b: b["batch"]),
            "totals": {"pairs": int(len(pairs)),
                       "labels_wanted": int(pairs.wanted.sum()),
                       "labels_have": int(pairs.have.sum()),
                       "owed_by_me": int(len(owed))},
            "annotators_seen": sorted(judgements.annotator.dropna().unique().tolist())
            if len(judgements) else []}


def campaign_path() -> str:
    return str(BATCHES)
