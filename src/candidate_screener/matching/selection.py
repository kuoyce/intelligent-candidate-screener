"""The registered decision rule, applied mechanically and exhaustively (D48).

Registration §2 gives a paired comparison B vs A its verdict. §4 adopts the **cheapest
rung that no justified, more expensive rung beats**, in the fixed cost order below. "No
justified rung beats it" is checked against **every** more expensive rung in the context,
not only the comparisons §3 listed. A missing comparison is not an acquittal.

Two contexts are selected separately:
- `metadata`: both sides carry a `Primary Keyword` (Djinni-style input);
- `text_only`: they do not (uploaded files).

`indomain.evaluate` runs the selection where the scores exist, and writes the full trail
into the record. This module's CLI prints it.

Usage:
    uv run python -m candidate_screener.matching.selection
"""
from __future__ import annotations

import json

#: Registration §4, cheapest first. A metadata hybrid costs what its text rung costs.
COST = {
    "random": 1, "keyword match": 2, "metadata rules": 2,
    "skill overlap": 3, "metadata + skill overlap": 3,
    "TF-IDF": 4, "metadata + TF-IDF": 4,
    "BM25": 5, "metadata + BM25": 5,
    "SBERT (zero-shot)": 6, "metadata + SBERT (zero-shot)": 6,
    "SBERT (weak fine-tuned)": 7, "metadata + SBERT (weak fine-tuned)": 7,
    "cross-encoder (A1)": 8, "cross-encoder (weak)": 8,
    "metadata + cross-encoder (A1)": 8, "metadata + cross-encoder (weak)": 8,
    "metadata + learned text combination": 9,
}
USES_METADATA = {"keyword match", "metadata rules"} | {n for n in COST if n.startswith("metadata +")}
CELLS = (("total", "all"), ("generic", "all"), ("targeted", "all"), ("total", "llm_free"))


def verdict_of(d: dict) -> str:
    """Registration §2 on a {(scope, subset): paired_delta} dict."""
    tot = d[("total", "all")]
    if tot["ci"] is None:
        return "not scoreable"
    if tot["ci"][1] < 0:
        return "harmful"
    if tot["ci"][0] > 0:
        if d[("generic", "all")]["delta"] <= 0 or d[("targeted", "all")]["delta"] <= 0:
            return "justified, one stratum"
        free = d[("total", "llm_free")]["delta"]
        if free is not None and free <= 0:
            return "justified, LLM-dependent"
        return "justified"
    return "not resolved"


def select(scores: dict, auc_total: dict, delta, context: str) -> dict:
    """`delta(b, a, scope, subset)` returns a paired-delta dict (indomain.paired_delta)."""
    names = [n for n in scores if n in COST]
    if context == "text_only":
        names = [n for n in names if n not in USES_METADATA]
    trail = []
    for name in sorted(names, key=lambda n: (COST[n], -auc_total[n], n)):
        checks = []
        for other in sorted((o for o in names if COST[o] > COST[name]), key=lambda o: (COST[o], o)):
            d = {cell: delta(other, name, *cell) for cell in CELLS}
            checks.append({"b": other, "delta_total": d[("total", "all")]["delta"],
                           "ci_total": d[("total", "all")]["ci"], "verdict": verdict_of(d)})
        beaten = [c["b"] for c in checks if c["verdict"] == "justified"]
        trail.append({"rung": name, "cost": COST[name], "auc_total": round(auc_total[name], 6),
                      "beaten_by": beaten, "checks": checks})
        if not beaten:
            return {"context": context, "selected": name, "trail": trail}
    return {"context": context, "selected": None, "trail": trail}


def main() -> int:
    from candidate_screener.evaluation import indomain
    record = json.loads(indomain.RECORD.read_text(encoding="utf-8"))
    for context, s in record["selection"].items():
        print(f"[{context}] selected: {s['selected']}")
        for step in s["trail"]:
            print(f"   cost {step['cost']} {step['rung']:38} AUC {step['auc_total']:.3f}  "
                  f"beaten by: {step['beaten_by'] or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
