"""Strict entity scoring against exhaustively reviewed document gold."""
from __future__ import annotations

import hashlib
import random


def _metrics(tp, fp, fn):
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score_predictions(gold: list[dict], predictions: list[dict], *, prediction_documents: dict,
                      bootstrap_samples=1000, seed=0) -> dict:
    if not gold or any(row.get("reviewed_all") is not True for row in gold):
        raise ValueError("Scoring requires complete reviewed document gold")
    if any(row.get("assistance") or row.get("annotation_mode", "blind") != "blind" for row in gold):
        raise ValueError("Blind accuracy evaluation cannot use model-assisted annotations")
    if bootstrap_samples < 1:
        raise ValueError("Positive bootstrap sample count required")
    ids = [row["doc_id"] for row in gold]
    for row in gold:
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if row.get("source_sha256") != digest:
            raise ValueError("Gold source digest mismatch")
        metadata = prediction_documents.get(row["doc_id"], {})
        if metadata.get("source_sha256") != digest or metadata.get("status") not in {"complete", "complete_zero"}:
            raise ValueError("Predictions require complete document coverage and matching source digest")
    if len(set(ids)) != len(ids):
        raise ValueError("Score one annotator/revision per document")
    categories = set(gold[0]["categories"])
    for key in ("split", "annotator", "campaign_id", "guideline_version"):
        if len({row.get(key) for row in gold}) != 1:
            raise ValueError("Gold must share one split, annotator, campaign and guideline")
    if any(set(row["categories"]) != categories for row in gold):
        raise ValueError("Gold ontology differs across documents")
    prediction_sets = {doc_id: set() for doc_id in ids}
    sources = {row["doc_id"]: row["text"] for row in gold}
    for prediction in predictions:
        if prediction["doc_id"] not in prediction_sets:
            raise ValueError("Prediction document outside evaluated gold")
        if prediction["label"] in categories:
            start, end = prediction["start"], prediction["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sources[prediction["doc_id"]]):
                raise ValueError("Invalid prediction source offsets")
            if prediction.get("surface", sources[prediction["doc_id"]][start:end]) != sources[prediction["doc_id"]][start:end]:
                raise ValueError("Prediction surface differs from source")
            prediction_sets[prediction["doc_id"]].add((prediction["start"], prediction["end"], prediction["label"]))
    documents = []
    for row in gold:
        expected = {(span["start"], span["end"], span["label"]) for span in row["spans"]}
        if len(expected) != len(row["spans"]):
            raise ValueError("Duplicate gold spans")
        for span in row["spans"]:
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row["text"]) or
                    span["surface"] != row["text"][start:end]):
                raise ValueError("Invalid gold source offsets/surface")
        if any(span[2] not in categories for span in expected):
            raise ValueError("Gold span outside selected ontology")
        documents.append((row["doc_type"], expected, prediction_sets[row["doc_id"]]))

    def aggregate(selected, label=None):
        tp = fp = fn = 0
        for _, expected, observed in selected:
            if label is not None:
                expected = {x for x in expected if x[2] == label}
                observed = {x for x in observed if x[2] == label}
            tp += len(expected & observed)
            fp += len(observed - expected)
            fn += len(expected - observed)
        return _metrics(tp, fp, fn)

    def report(selected, label=None):
        result = aggregate(selected, label)
        result["documents"] = len(selected)
        result["gold_spans"] = result["true_positive"] + result["false_negative"]
        randomizer = random.Random(seed)
        draws = {metric: [] for metric in ("precision", "recall", "f1")}
        for _ in range(bootstrap_samples):
            sample = [randomizer.choice(selected) for _ in selected]
            metrics = aggregate(sample, label)
            for metric in draws:
                if metrics[metric] is not None:
                    draws[metric].append(metrics[metric])
        result["ci95"] = {}
        result["defined_bootstrap_draws"] = {metric: len(values) for metric, values in draws.items()}
        for metric, values in draws.items():
            values.sort()
            result["ci95"][metric] = ([values[int((len(values)-1)*.025)], values[int((len(values)-1)*.975)]] if values else None)
        return result

    return {"metric": "strict-boundary-and-label", "bootstrap_unit": "document",
            "bootstrap_samples": bootstrap_samples, "seed": seed,
            "total": report(documents),
            "categories": {label: report(documents, label) for label in sorted(categories)},
            "document_types": {kind: report([row for row in documents if row[0] == kind])
                               for kind in sorted({row[0] for row in documents})}}
