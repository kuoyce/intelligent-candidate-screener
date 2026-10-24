"""Fixed exploratory extractive adaptation on authored train families only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from candidate_screener.config import PROJECT_ROOT
from .qa_assets import ROOT, file_digest, verify_manifest

CONFIG = {"seed": 0, "learning_rate": 2e-5, "epochs": 3, "accumulation": 8,
          "threads": 4, "max_length": 384, "stride": 128, "weight_decay": .01,
          "max_gradient_norm": 1., "no_answer_threshold": 0.}


def training_examples(families):
    examples, skipped = [], {}
    for family in families:
        if family["split"] != "train":
            continue
        documents = {doc["document_id"]: doc for doc in family["documents"]}
        for turn in family["turns"]:
            compatible = turn["task"] in {"fact", "long", "followup", "unanswerable"}
            if not compatible or turn.get("must_clarify") or (not turn["is_impossible"] and len(turn["answers"]) != 1):
                skipped[turn["task"]] = skipped.get(turn["task"], 0) + 1
                continue
            if turn["is_impossible"]:
                candidates = turn["expected_candidate_ids"]
                selected = [doc for doc in documents.values() if doc["candidate_id"] in candidates]
                if len(selected) != 1:
                    raise ValueError("Impossible QA requires one explicitly identified source document")
                doc, answer = selected[0], None
            else:
                answer = turn["answers"][0]
                doc = documents[answer["document_id"]]
                if doc["text"][answer["start"]:answer["end"]] != answer["text"]:
                    raise ValueError("QA training answer source mismatch")
            examples.append({"family_id": family["family_id"], "turn_id": turn["turn_id"],
                             "question": turn["question"], "context": doc["text"], "answer": answer})
    return examples, skipped


def window_target(offsets, sequence_ids, answer, cls_index):
    """Exact boundaries only; a complete answer cannot become a false negative."""
    context = [i for i, identity in enumerate(sequence_ids) if identity == 1 and offsets[i][1] > offsets[i][0]]
    if answer is None or not context:
        return cls_index, cls_index
    start, end = answer["start"], answer["end"]
    if offsets[context[0]][0] <= start and end <= offsets[context[-1]][1]:
        starts = [i for i in context if offsets[i][0] == start]
        ends = [i for i in context if offsets[i][1] == end]
        if not starts or not ends or starts[0] > ends[-1]:
            raise ValueError("QA answer character boundary is unrepresentable; no rounding allowed")
        return starts[0], ends[-1]
    return cls_index, cls_index


def prepare_windows(examples, tokenizer):
    records = []
    for example in examples:
        encoded = tokenizer(example["question"], example["context"], truncation="only_second",
                            max_length=CONFIG["max_length"], stride=CONFIG["stride"],
                            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        positives = 0
        for index, ids in enumerate(encoded["input_ids"]):
            cls = ids.index(tokenizer.cls_token_id)
            start, end = window_target(encoded["offset_mapping"][index], encoded.sequence_ids(index), example["answer"], cls)
            positives += start != cls
            record = {key: encoded[key][index] for key in tokenizer.model_input_names if key in encoded}
            record.update(start_positions=start, end_positions=end)
            records.append(record)
        if example["answer"] is not None and not positives:
            raise ValueError("No overflow window represents the complete QA answer")
    return records


def run_training(model, records, *, pilot=False):
    import torch
    if not records:
        raise ValueError("No compatible QA training windows")
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    model.train()
    order = list(range(len(records)))
    randomizer = random.Random(CONFIG["seed"])
    losses, steps = [], 0
    started = time.monotonic()
    for _ in range(1 if pilot else CONFIG["epochs"]):
        randomizer.shuffle(order)
        for offset in range(0, len(order), CONFIG["accumulation"]):
            group = order[offset:offset+CONFIG["accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                batch = {key: torch.tensor([value]) for key, value in records[index].items()}
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite QA training loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            if pilot:
                break
    elapsed = time.monotonic()-started
    model.eval()
    return {"windows": len(records), "optimizer_steps": steps, "microbatches": len(losses),
            "elapsed_seconds": elapsed, "loss_mean": sum(losses)/len(losses),
            "estimated_full_seconds": elapsed / steps * math.ceil(len(records)/CONFIG["accumulation"]) * CONFIG["epochs"]}


def save_model(model, tokenizer, destination, base, provenance):
    destination = Path(destination).resolve()
    if not destination.is_relative_to((PROJECT_ROOT / "data").resolve()) or destination.exists():
        raise ValueError("Choose a new adapted model revision under ignored data/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".qa-adapted-", dir=destination.parent))
    try:
        checkpoint = staging / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        files = {path.relative_to(checkpoint).as_posix(): file_digest(path) for path in checkpoint.rglob("*") if path.is_file()}
        manifest = {"model_id": base["model_id"], "revision": base["revision"], "license": base.get("license"),
                    "artifact_sha256": files, "runtime": base["runtime"], "training": provenance,
                    "base_artifact_sha256": base["artifact_sha256"]}
        manifest["adaptation_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        os.rename(staging, destination)
        verify_manifest(destination / "manifest.json")
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "manifest.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.train and args.destination is None:
        parser.error("--train requires --destination")
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from candidate_screener.app.rag_evaluation import build_scenarios
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    base = verify_manifest(args.base_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.base_manifest.parent/"checkpoint", local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Exact QA training requires a fast tokenizer")
    examples, skipped = training_examples(build_scenarios())
    records = prepare_windows(examples, tokenizer)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_manifest.parent/"checkpoint", local_files_only=True)
    result = run_training(model, records, pilot=args.pilot)
    provenance = {"config": CONFIG, "examples": len(examples), "families": len({row["family_id"] for row in examples}),
                  "skipped_tasks": skipped, "training_split": "train", "result": result,
                  "source_sha256": hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest(),
                  "code_sha256": file_digest(Path(__file__)), "scenario_code_sha256": file_digest(PROJECT_ROOT/"src/candidate_screener/app/rag_evaluation.py")}
    if args.train:
        provenance["adaptation_id"] = save_model(model, tokenizer, args.destination, base, provenance)["adaptation_id"]
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
