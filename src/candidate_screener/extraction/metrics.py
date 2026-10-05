"""Exact original-span/label scoring for independently annotated adapters."""
from collections import Counter


def score_exact_spans(gold: list[dict], predicted: list[dict]) -> dict:
    """Score by document, field, half-open span and label; no fuzzy matching.

    Adapters must preserve original gold granularity and validate provenance,
    offsets, splits and layer identities before calling this function.
    """
    def key(row):
        return (row.get("source"), row["doc_id"], row["field"], row.get("layer"),
                row["start"], row["end"], row["label"])

    gold_keys, predicted_keys = Counter(map(key, gold)), Counter(map(key, predicted))
    def metrics(g, p):
        tp = sum((g & p).values())
        support, count = sum(g.values()), sum(p.values())
        precision = tp / count if count else 0.0
        recall = tp / support if support else 0.0
        return {"precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "support": support, "predicted": count, "true_positive": tp}
    labels = sorted({k[-1] for k in gold_keys | predicted_keys})
    return {"micro": metrics(gold_keys, predicted_keys), "per_label": {
        label: metrics(Counter({k:v for k,v in gold_keys.items() if k[-1] == label}),
                       Counter({k:v for k,v in predicted_keys.items() if k[-1] == label}))
        for label in labels}}
